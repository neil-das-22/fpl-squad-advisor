"""
Multi-gameweek fixture-run analysis, originally built for GK/DEF and
extended to cover MID (and, later, FWD).

WHY THIS EXISTS
Everything in xp_model.py is single-gameweek: "what's this player worth
next match." Neil wants selection to also weigh a rolling window (default:
next 4 gameweeks) of upcoming opposition, combined with the player's own
output -- not just next week's fixture in isolation. A player with modest
week-1 numbers but four straight favourable matchups is a different pick
than one with a great week 1 followed by three brutal fixtures. This was
originally scoped to defensive assets only; when the MID review reached
the same "look 4 gameweeks ahead" request, the fixture-run machinery
itself didn't need to change -- `fixture_context()` already returns both
halves (`clean_sheet_prob` for defenders, `attack_multiplier` for
attackers) from the same team-strength model, so `team_fixture_run()` now
reports both instead of just the defensive one.

This reuses `xp_model.fixture_context()` (the same team-strength /
difficulty-rating model that drives single-gameweek clean-sheet
probability and attacking output) rather than inventing a second fixture
model -- the run score and the single-gameweek xP stay internally
consistent with each other.

WHAT IT DOES NOT DO
This only scores the fixture list itself (opponent strength, home/away,
difficulty rating). It does not know about new-manager tactical changes,
European-competition rotation risk, or other qualitative context -- that
stays the fixtures research agent's job. Team-level scoring here is the
quantifiable half of "look at the run of games"; the agent supplies the
half that isn't in any data table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import xp_model as m

DEFAULT_RUN_LENGTH = 4


def _zscore(s: pd.Series) -> pd.Series:
    """Standard z-score, robust to the degenerate cases this module hits
    in practice: zero variance (every value identical -- e.g. a position
    group where nobody has any prior-season data), and a column that's
    entirely NaN (e.g. team_possession_pct when the synthetic test pool
    has no possession column at all). Both cases return all-zero
    contributions rather than propagating NaN into run_score for every
    row -- "no information" should mean "this term doesn't move the
    score," not "silently poison the total."
    """
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _team_fixtures_by_gameweek(fixtures_df: pd.DataFrame, team_name: str,
                                start_gw: int, n_gw: int) -> dict[int, list[tuple[Any, bool]]]:
    """{gameweek: [(fixture_row, is_home), ...]} for one team's run window.

    A gameweek maps to zero fixtures (blank), one (normal), or two+ (double).
    """
    out: dict[int, list[tuple[Any, bool]]] = {gw: [] for gw in range(start_gw, start_gw + n_gw)}
    window = fixtures_df[
        pd.to_numeric(fixtures_df["gameweek"], errors="coerce").between(start_gw, start_gw + n_gw - 1)
    ]
    for _, frow in window.iterrows():
        gw = int(frow["gameweek"])
        if str(frow["home_team"]) == team_name:
            out[gw].append((frow, True))
        elif str(frow["away_team"]) == team_name:
            out[gw].append((frow, False))
    return out


def team_fixture_run(team_name: str, teams_df: pd.DataFrame, fixtures_df: pd.DataFrame,
                     start_gw: int, n_gw: int = DEFAULT_RUN_LENGTH) -> dict[str, Any]:
    """One team's fixture-run summary over `n_gw` gameweeks from `start_gw`.

    Returns mean clean-sheet probability AND mean attack multiplier across
    every fixture in the window (directly comparable to their
    single-gameweek counterparts) -- the first is what GK/DEF selection
    cares about, the second is the same thing for MID/FWD (a kind run of
    fixtures against weak defences, not weak attacks). Also returns the
    count of actual fixtures faced (a double gameweek counts twice, a
    blank counts zero -- this matters as much as the difficulty itself),
    and a gameweek-by-gameweek breakdown for showing the actual run, not
    just a summary statistic.
    """
    teams_by_name = {str(r["name"]): r for _, r in teams_df.iterrows()}
    own_row = teams_by_name.get(team_name)
    if own_row is None:
        return {"team_name": team_name, "n_fixtures": 0, "mean_clean_sheet_prob": None,
                "mean_attack_multiplier": None, "mean_difficulty": None, "breakdown": []}

    by_gw = _team_fixtures_by_gameweek(fixtures_df, team_name, start_gw, n_gw)

    breakdown = []
    cs_probs, attack_mults, difficulties = [], [], []
    for gw in range(start_gw, start_gw + n_gw):
        fixtures_this_gw = by_gw[gw]
        if not fixtures_this_gw:
            breakdown.append({"gameweek": gw, "opponent": None, "is_home": None,
                              "clean_sheet_prob": None, "attack_multiplier": None,
                              "difficulty": None, "blank": True})
            continue
        for frow, is_home in fixtures_this_gw:
            opp_name = str(frow["away_team"] if is_home else frow["home_team"])
            opp_row = teams_by_name.get(opp_name)
            difficulty = frow["team_h_difficulty"] if is_home else frow["team_a_difficulty"]
            ctx = m.fixture_context(own_row, opp_row, is_home, difficulty)
            breakdown.append({
                "gameweek": gw, "opponent": opp_name, "is_home": is_home,
                "clean_sheet_prob": ctx["clean_sheet_prob"],
                "attack_multiplier": ctx["attack_multiplier"],
                "difficulty": ctx["difficulty"], "blank": False,
            })
            cs_probs.append(ctx["clean_sheet_prob"])
            attack_mults.append(ctx["attack_multiplier"])
            difficulties.append(ctx["difficulty"])

    return {
        "team_name": team_name,
        "n_fixtures": len(cs_probs),
        "n_blanks": sum(1 for b in breakdown if b["blank"]),
        "n_doubles": sum(1 for gw in by_gw if len(by_gw[gw]) > 1),
        "mean_clean_sheet_prob": (sum(cs_probs) / len(cs_probs)) if cs_probs else None,
        "mean_attack_multiplier": (sum(attack_mults) / len(attack_mults)) if attack_mults else None,
        "mean_difficulty": (sum(difficulties) / len(difficulties)) if difficulties else None,
        "breakdown": breakdown,
    }


def all_teams_fixture_run(teams_df: pd.DataFrame, fixtures_df: pd.DataFrame,
                          start_gw: int, n_gw: int = DEFAULT_RUN_LENGTH,
                          rank_by: str = "mean_clean_sheet_prob") -> pd.DataFrame:
    """Every team's fixture run, ranked best-to-worst.

    Default ranks by `mean_clean_sheet_prob` -- the table to hand a GK/DEF
    research pass: "who has the kindest run of fixtures over the next N
    gameweeks," independent of any individual player. Pass
    `rank_by="mean_attack_multiplier"` for the MID/FWD equivalent: kindest
    run against weak defences rather than weak attacks. Both columns are
    always returned regardless of which one is ranked on, since either
    caller may want to inspect the other.
    """
    rows = [team_fixture_run(name, teams_df, fixtures_df, start_gw, n_gw)
            for name in teams_df["name"]]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "breakdown"} for r in rows])
    df = df.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "run_rank", range(1, len(df) + 1))
    return df


def team_attack_strength(team_name: str, teams_df: pd.DataFrame) -> float:
    """A team's own attacking strength, independent of any specific fixture.

    WHY THIS EXISTS: Neil's point on midfielders -- "the attackers on the
    same team help the midfielders get assists/goals, so their passing,
    assist, and goal stats should also be taken into consideration." The
    literal version of that (aggregate this specific midfielder's specific
    strike-partners' stats) means picking an arbitrary "front line" cutoff
    with no clean definition in the data. This uses FPL's own published
    team-attack-strength rating instead (`strength_attack_home/away`, the
    same numbers that already drive `fixture_context()`'s attack_multiplier)
    averaged across home/away -- a real, always-current, club-wide measure
    of how dangerous this team's attackers collectively are, which is what
    actually determines how many of the good chances in a match a
    midfielder gets to be involved in. It is a team-level proxy, not a
    hand-picked "top 3 attackers" sum -- flagged here so that distinction
    isn't lost downstream.

    USES `xp_model._strength()`, NOT A DIRECT COLUMN READ -- verified
    against the live pre-season data: `strength_attack_home/away` come back
    as 0 for all 20 clubs right now (FPL hasn't published attack/defence
    sub-ratings yet this early in the season), which would silently make
    every team look identical if read raw. `_strength()` already has the
    correct fallback for exactly this case (drops to `strength_overall_home`,
    which IS populated -- real integers 2-5 differentiating clubs -- then to
    a flat default only if that's missing too), the same fallback
    `fixture_context()` already relies on for clean-sheet/attack-multiplier
    math. Reusing it keeps this number honest and internally consistent
    rather than inventing a second, silently-broken read path.
    """
    row = teams_df[teams_df["name"] == team_name]
    if row.empty:
        return float("nan")
    row = row.iloc[0]
    home = m._strength(row, "strength_attack_home")
    away = m._strength(row, "strength_attack_away")
    vals = [v for v in (home, away) if v == v and v > 0]
    return sum(vals) / len(vals) if vals else float("nan")


def team_possession_pct(team_name: str, teams_df: pd.DataFrame) -> float:
    """A team's real prior-season average ball possession %.

    Per Neil: "possession of the team is important for both midfielders
    and attackers" -- more time on the ball means more of the match spent
    creating and taking chances, for both positions, not just defensively
    (see the GK agent's use of the same idea as a defensive proxy).

    Reads `possession_pct_prev_season` directly off `teams_df` -- the
    CALLER is responsible for having attached it first via
    `fpl_client.attach_team_prior_season_stats()`, same division of
    responsibility as prior-season player columns throughout this module.
    Returns NaN if the column isn't present at all (older/synthetic
    teams_df) or the specific team has no row in the source (this
    season's 3 promoted clubs -- see fpl_client.py module comment for why
    that's real missing data, not a bug).
    """
    if "possession_pct_prev_season" not in teams_df.columns:
        return float("nan")
    row = teams_df[teams_df["name"] == team_name]
    if row.empty:
        return float("nan")
    val = row.iloc[0]["possession_pct_prev_season"]
    return float(val) if pd.notna(val) else float("nan")


DEFAULT_CREATIVE_SUPPLY_TOP_N = 3


def _mid_creative_output(row: Any) -> float:
    """One midfielder's shrunk per-90 SUPPLY output (assists + xA +
    creativity) -- deliberately excludes goals, which is scoring output,
    not service. Used both to rank a team's own midfielders and, summed
    across a team's best creators, to score a striker's service quality
    in `team_creative_supply()`.
    """
    minutes = row.get("minutes_prev_season")
    minutes = 0.0 if pd.isna(minutes) else float(minutes)
    assists = row.get("assists_prev_season")
    assists = 0.0 if pd.isna(assists) else float(assists)
    xa = row.get("expected_assists_prev_season")
    xa = 0.0 if pd.isna(xa) else float(xa)
    creativity = row.get("creativity_prev_season")
    creativity = 0.0 if pd.isna(creativity) else float(creativity)
    supply_p90 = m.shrunk_per90_rate(assists + xa, minutes, prior_rate=0.15)
    creativity_p90 = m.shrunk_per90_rate(creativity, minutes, prior_rate=25.0)
    return supply_p90 + (creativity_p90 / 50.0)


def team_creative_supply(team_name: str, players_df: pd.DataFrame,
                         top_n: int = DEFAULT_CREATIVE_SUPPLY_TOP_N) -> float:
    """How much real service a striker at this club can expect from his
    attacking midfielders/wingers -- Neil's specific ask for the FWD
    rebuild: "I want to consider the attacking midfielder and wingers in
    their team, and their chances created, passing, assist stats."

    WHY TOP-N SUM, NOT A TEAM-WIDE AVERAGE: averaging in every MID on the
    books (reserves, holding midfielders with near-zero creative output)
    would dilute a club with 2-3 genuine creative outlets down to the same
    number as a club with one. Summing the top `top_n` creators instead
    rewards actual creative depth without being diluted by players who
    structurally aren't service providers. `top_n=3` is a judgment call,
    not derived from data -- change it if 3 feels like the wrong cutoff
    for "front line."

    WHY NOT A FORMAL "WINGER" FILTER: same limitation as
    `midfielder_shortlist()` -- FPL's data has no winger/attacking-mid
    classification, so this pulls from ALL of a club's MID-classified
    players and lets `_mid_creative_output`'s own per-90 shrinkage do the
    filtering implicitly: a genuine holding midfielder with low
    assists/xA/creativity contributes little regardless of formally being
    "MID" too, so he simply doesn't make the top-N cut.

    ONLY COUNTS PLAYERS WITH REAL PRIOR-SEASON DATA -- same guess-vs-
    evidence concern as `midfielder_shortlist`'s no-prior-data gate: a
    club full of unproven MIDs with flat-prior supply numbers should not
    look like a creative team on paper. A club with fewer than `top_n`
    data-backed midfielders is scored on however many it actually has,
    not padded out with guesses.
    """
    mids = players_df[(players_df["team_name"] == team_name) & (players_df["position"] == "MID")]
    if "minutes_prev_season" not in mids.columns or mids.empty:
        return float("nan")
    has_data = mids["minutes_prev_season"].apply(lambda v: pd.notna(v) and float(v) > 0)
    mids = mids[has_data]
    if mids.empty:
        return float("nan")
    outputs = sorted((_mid_creative_output(row) for _, row in mids.iterrows()), reverse=True)
    return sum(outputs[:top_n])


MIN_START_PROBABILITY_FOR_SHORTLIST = 0.40


def defensive_shortlist(players_df: pd.DataFrame, teams_df: pd.DataFrame,
                        fixtures_df: pd.DataFrame, start_gw: int,
                        n_gw: int = DEFAULT_RUN_LENGTH,
                        positions: tuple[str, ...] = ("GKP", "DEF"),
                        min_start_probability: float = MIN_START_PROBABILITY_FOR_SHORTLIST,
                        matches_played: float = 0,
                        ) -> pd.DataFrame:
    """GK/DEF players ranked by fixture run combined with their own defensive
    stats -- the actual thing a person picking a squad wants to look at,
    not just the team-level run in isolation.

    `run_score` blends team fixture-run quality with the player's own
    defensive stats (prior-season clean sheets, CBIT/CBIRT rate), so a
    good run behind a genuinely weak defensive player doesn't outrank a
    good run behind a real defensive performer. Both halves are z-scored
    within the position group before combining so goalkeepers and
    defenders aren't compared on incompatible raw scales.

    IMPORTANT: a great fixture run is worthless for a player who doesn't
    play. Every player on a good-defense team shares that team's run score,
    which without a start-probability gate would surface a nailed starter
    and his third-choice backup as equally good picks. This filters out
    anyone below `min_start_probability` (via
    `xp_model.estimate_start_probability`, so it uses the same prior-season
    signal as the rest of the model) before ranking, and reports p_start
    alongside the score so a human can see exactly why a name is or isn't
    on the list.
    """
    run_df = all_teams_fixture_run(teams_df, fixtures_df, start_gw, n_gw)
    run_by_team = run_df.set_index("team_name")[["mean_clean_sheet_prob", "n_fixtures", "n_blanks", "n_doubles"]]

    pool = players_df[players_df["position"].isin(positions)].copy()
    pool = pool.join(run_by_team, on="team_name")

    p_start, p_start_grounded = [], []
    for _, row in pool.iterrows():
        p, flags = m.estimate_start_probability(row, matches_played=matches_played)
        p_start.append(p)
        # "Grounded" = backed by an actual data signal (this season's form,
        # last season's starts, or a real zero). "Not grounded" = the flat
        # default fired, i.e. we have no idea and are guessing 65%. A shaky
        # p_start clearing the bar purely on the guess is exactly the kind
        # of thing that should be visible, not hidden inside a passing score.
        p_start_grounded.append("start_prob_default" not in flags)
    pool["p_start"] = p_start
    pool["p_start_grounded"] = p_start_grounded
    pool = pool[pool["p_start"] >= min_start_probability].copy()

    def _own_defense_score(row) -> float:
        """Player's own defensive output, position-appropriate.

        GKP: prior-season clean sheets (there is no CBIT/CBIRT threshold
        for keepers in this model -- see DEFCON_THRESHOLD["GKP"] = None).
        DEF: shrunk CBIT rate (tackles + clearances/blocks/interceptions),
        matching how the model's own DefCon term is built, so this stays
        consistent with calculate_xp() rather than inventing a second
        definition of "good defensively."

        IMPORTANT: reads the `_prev_season`-suffixed columns directly,
        NOT `m._defcon_count_for_position()` -- that helper reads
        *current*-season CBIT columns, which is right for calculate_xp()
        mid-season but wrong here: this module exists for the pre-season /
        thin-current-data case, where the signal that actually exists is
        last season's counts.
        """
        cs = row.get("clean_sheets_prev_season")
        cs = 0.0 if pd.isna(cs) else float(cs)
        if row["position"] == "GKP":
            return cs
        cbi = row.get("clearances_blocks_interceptions_prev_season")
        cbi = 0.0 if pd.isna(cbi) else float(cbi)
        tackles = row.get("tackles_prev_season")
        tackles = 0.0 if pd.isna(tackles) else float(tackles)
        cbit = cbi + tackles
        minutes = row.get("minutes_prev_season")
        minutes = 0.0 if pd.isna(minutes) else float(minutes)
        cbit_per90 = m.shrunk_per90_rate(cbit, minutes, m.DEFCON_PER90_PRIOR["DEF"])
        return cbit_per90

    pool["own_defense_score"] = pool.apply(_own_defense_score, axis=1)

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std and std > 0 else s * 0.0

    pool["run_score"] = 0.0
    for pos in positions:
        mask = pool["position"] == pos
        if mask.sum() == 0:
            continue
        cs_z = _zscore(pool.loc[mask, "mean_clean_sheet_prob"].fillna(pool.loc[mask, "mean_clean_sheet_prob"].mean()))
        def_z = _zscore(pool.loc[mask, "own_defense_score"])
        pool.loc[mask, "run_score"] = (cs_z + def_z) / 2.0

    keep_cols = ["id", "web_name", "team_name", "position", "price_m",
                "p_start", "p_start_grounded",
                "mean_clean_sheet_prob", "n_fixtures", "n_blanks", "n_doubles",
                "own_defense_score", "run_score"]
    return pool[keep_cols].sort_values("run_score", ascending=False).reset_index(drop=True)


MIN_START_PROBABILITY_FOR_MID_SHORTLIST = 0.40

# Yellow/red weighting for the discipline penalty. Not an official FPL
# points ratio -- reds cost a start as much as they cost a card, so this
# weights them heavily as a "this player is a booking/suspension risk"
# signal, not a literal points translation (that's -1/-3 in the game's own
# scoring, which calculate_xp() already applies separately; this shortlist
# is about who to shortlist in the first place, not the final xP number).
CARD_WEIGHT = {"yellow": 1.0, "red": 3.0}


def midfielder_shortlist(players_df: pd.DataFrame, teams_df: pd.DataFrame,
                         fixtures_df: pd.DataFrame, start_gw: int,
                         n_gw: int = DEFAULT_RUN_LENGTH,
                         min_start_probability: float = MIN_START_PROBABILITY_FOR_MID_SHORTLIST,
                         matches_played: float = 0,
                         ) -> pd.DataFrame:
    """Midfielders ranked by a blended attacking/defensive profile, next-N
    fixture run, team attacking context, discipline, and set-piece duty.

    WHY A BLENDED PROFILE INSTEAD OF ONE MIDFIELDER SCORE: a defensive
    midfielder and a winger both play "MID" in FPL's own classification,
    but they earn points completely differently -- one from tackles/
    interceptions/DefCon, the other from goals/assists/chance creation.
    Scoring everyone on the same axis structurally punishes whichever
    archetype the axis doesn't favour. Per player this computes BOTH an
    `attacking_score` (goals, assists, expected goal involvement,
    creativity -- all shrunk per-90 off prior-season data, same treatment
    as everywhere else in this model) and a `defensive_score` (tackles +
    clearances/blocks/interceptions, i.e. the same CBIT count that feeds
    the model's own MID DefCon term), z-scored within the MID pool, and
    takes `profile_score = max(attacking_z, defensive_z)`. A player is
    ranked on whichever half of the job he actually does -- this is also
    what naturally surfaces wingers as strong picks (per Neil: "wingers
    should be in high regard as midfielders as they are usually a lot more
    involved in attacking play") without needing a "is this a winger" flag
    that doesn't exist anywhere in FPL's data; a genuine winger simply
    scores high on the attacking half and that's what wins.

    DATA HONESTLY NOT AVAILABLE: FPL's public API has no pass-completion
    counts and no standalone "big chances created" stat. `creativity` (an
    ICT sub-index built from chance-creation events) is used as the closest
    real proxy and is labelled as such everywhere it appears -- this is not
    silently treated as equivalent to the real thing.

    OTHER FACTORS, EACH REPORTED AS ITS OWN COLUMN RATHER THAN FOLDED
    INVISIBLY INTO ONE NUMBER (Neil's DEF-agent feedback was that he wanted
    to see the actual stats behind a recommendation, not just a score):
      - `team_attack_strength`: the player's own team's attack rating
        (see `team_attack_strength()`) -- "the attackers on the same team
        help the midfielder score/assist."
      - `mean_attack_multiplier`: the 4-gameweek fixture run, attacking
        framing (favourable = facing weak defences).
      - `set_piece_duty`: penalties/corners/free-kicks order, straight from
        live FPL data (1 = primary taker) -- no longer purely a research-
        agent web-search item, see fpl_client.py.
      - `discipline_score`: cards per 90, prior season -- a NEGATIVE
        contributor to run_score, per Neil's request that a bad foul/card
        record count against a player rather than being ignored.

    Same start-probability gate as `defensive_shortlist()`, same reasoning:
    a great attacking/defensive profile behind a bench spot is worthless.
    """
    run_df = all_teams_fixture_run(teams_df, fixtures_df, start_gw, n_gw,
                                   rank_by="mean_attack_multiplier")
    run_by_team = run_df.set_index("team_name")[["mean_attack_multiplier", "n_fixtures", "n_blanks", "n_doubles"]]

    pool = players_df[players_df["position"] == "MID"].copy()
    pool = pool.join(run_by_team, on="team_name")
    pool["team_attack_strength"] = pool["team_name"].apply(lambda t: team_attack_strength(t, teams_df))
    # Added after the fact, per Neil: "possession of the team is important
    # for both midfielders and attackers" -- see team_possession_pct().
    pool["team_possession_pct"] = pool["team_name"].apply(lambda t: team_possession_pct(t, teams_df))

    p_start, p_start_grounded = [], []
    for _, row in pool.iterrows():
        p, flags = m.estimate_start_probability(row, matches_played=matches_played)
        p_start.append(p)
        p_start_grounded.append("start_prob_default" not in flags)
    pool["p_start"] = p_start
    pool["p_start_grounded"] = p_start_grounded
    pool = pool[pool["p_start"] >= min_start_probability].copy()

    # Second gate, separate from p_start: a player with literally no
    # prior-season row (no `code` match -- promoted-team debuts, brand new
    # arrivals, academy names that happen to appear in the live squad list)
    # has EVERY input to attacking_score/defensive_score computed off the
    # flat priors, not real data. That's fine for calculate_xp(), where the
    # prior is meant as "assume average until proven otherwise" -- but it's
    # wrong for a *ranked* shortlist: DEFCON_PER90_PRIOR["MID"] (9.3,
    # calibrated to the population-wide average) sits ABOVE most genuine
    # attacking midfielders' real, LOWER defensive output (attacking
    # players structurally rack up fewer tackles/CBI than average -- that's
    # not a flaw, that's their job). Left ungated, `profile_score`'s max()
    # would let a total unknown's flat "average" defensive prior outrank
    # proven attacking stars purely because the guess landed on the
    # generous side, which is exactly the kind of guess-dressed-as-evidence
    # problem `p_start_grounded` exists to catch on the start-probability
    # side. These players are excluded from the ranked list rather than
    # scored on priors dressed up as data -- same "flat fallback is honest,
    # not a bug" philosophy documented for the ~20% of the pool with no PL
    # history at all in position_agent_specs.md.
    minutes_prev = pool["minutes_prev_season"] if "minutes_prev_season" in pool.columns \
        else pd.Series(pd.NA, index=pool.index)
    pool["has_prior_data"] = minutes_prev.apply(lambda v: pd.notna(v) and float(v) > 0)
    excluded_no_data = pool[~pool["has_prior_data"]][["id", "web_name", "team_name", "p_start"]].copy()
    pool = pool[pool["has_prior_data"]].copy()

    def _get0(row, col: str) -> float:
        v = row.get(col)
        return 0.0 if pd.isna(v) else float(v)

    def _attacking_score(row) -> float:
        """Shrunk per-90 blend of goals + assists + xGI + creativity,
        prior season. `expected_goal_involvements` isn't always populated
        in the archive, so xG+xA is used directly as the underlying-quality
        half, alongside actual goals+assists as the realised-output half."""
        minutes = _get0(row, "minutes_prev_season")
        goals = _get0(row, "goals_scored_prev_season")
        assists = _get0(row, "assists_prev_season")
        xg = _get0(row, "expected_goals_prev_season")
        xa = _get0(row, "expected_assists_prev_season")
        creativity = _get0(row, "creativity_prev_season")
        realised_p90 = m.shrunk_per90_rate(goals + assists, minutes, prior_rate=0.15)
        underlying_p90 = m.shrunk_per90_rate(xg + xa, minutes, prior_rate=0.15)
        creativity_p90 = m.shrunk_per90_rate(creativity, minutes, prior_rate=25.0)
        # creativity is on a much larger raw scale (season totals often in
        # the hundreds) than goals/assists rates -- scale it down before
        # blending so it doesn't just drown the other two out.
        return realised_p90 + underlying_p90 + (creativity_p90 / 50.0)

    def _defensive_score(row) -> float:
        """Same CBIT definition the model's own MID DefCon term uses
        (clearances_blocks_interceptions + tackles), shrunk per-90 off
        prior season -- consistent with `defensive_shortlist`'s DEF version
        and with `calculate_xp()`'s own MID DefCon component."""
        minutes = _get0(row, "minutes_prev_season")
        cbi = _get0(row, "clearances_blocks_interceptions_prev_season")
        tackles = _get0(row, "tackles_prev_season")
        return m.shrunk_per90_rate(cbi + tackles, minutes, m.DEFCON_PER90_PRIOR["MID"])

    def _discipline_score(row) -> float:
        """Cards per 90, prior season -- higher is WORSE, applied as a
        negative contribution to run_score below."""
        minutes = _get0(row, "minutes_prev_season")
        yellow = _get0(row, "yellow_cards_prev_season")
        red = _get0(row, "red_cards_prev_season")
        weighted = yellow * CARD_WEIGHT["yellow"] + red * CARD_WEIGHT["red"]
        return m.shrunk_per90_rate(weighted, minutes, prior_rate=0.25)

    def _set_piece_duty(row) -> str:
        """Human-readable summary of penalties/corners/free-kicks order.
        1 in any column = primary taker for that duty."""
        pens = row.get("penalties_order")
        corners = row.get("corners_and_indirect_freekicks_order")
        dfk = row.get("direct_freekicks_order")
        tags = []
        if pd.notna(pens) and float(pens) == 1:
            tags.append("penalties")
        if pd.notna(corners) and float(corners) == 1:
            tags.append("corners/IFK")
        if pd.notna(dfk) and float(dfk) == 1:
            tags.append("direct FK")
        if tags:
            return "+".join(tags)
        # Secondary in the pecking order is still worth flagging.
        if any(pd.notna(v) and float(v) == 2 for v in (pens, corners, dfk)):
            return "backup taker"
        return "none"

    pool["attacking_score"] = pool.apply(_attacking_score, axis=1)
    pool["defensive_score"] = pool.apply(_defensive_score, axis=1)
    pool["discipline_score"] = pool.apply(_discipline_score, axis=1)
    pool["set_piece_duty"] = pool.apply(_set_piece_duty, axis=1)

    att_z = _zscore(pool["attacking_score"])
    def_z = _zscore(pool["defensive_score"])
    pool["profile_score"] = pd.concat([att_z, def_z], axis=1).max(axis=1)

    fixture_z = _zscore(pool["mean_attack_multiplier"].fillna(pool["mean_attack_multiplier"].mean()))
    team_z = _zscore(pool["team_attack_strength"].fillna(pool["team_attack_strength"].mean()))
    # NaN-safe: possession data only covers 17/20 clubs (see fpl_client.py);
    # missing values fall back to the pool mean so they don't distort
    # everyone else's z-score, same treatment as the other team-level
    # columns above.
    poss_z = _zscore(pool["team_possession_pct"].fillna(pool["team_possession_pct"].mean()))
    discipline_z = _zscore(pool["discipline_score"])
    set_piece_bonus = pool["set_piece_duty"].apply(
        lambda s: 0.5 if s not in ("none",) and "backup" not in s else (0.15 if s == "backup taker" else 0.0)
    )

    pool["run_score"] = (
        pool["profile_score"] * 1.5   # the player's own output is the main driver
        + fixture_z * 0.5
        + team_z * 0.5
        + poss_z * 0.25
        + set_piece_bonus
        - discipline_z * 0.25
    )

    keep_cols = ["id", "web_name", "team_name", "position", "price_m",
                "p_start", "p_start_grounded",
                "attacking_score", "defensive_score", "profile_score",
                "mean_attack_multiplier", "team_attack_strength", "team_possession_pct",
                "set_piece_duty", "discipline_score", "run_score"]
    result = pool[keep_cols].sort_values("run_score", ascending=False).reset_index(drop=True)
    # Not silently dropped: attached as metadata so a caller can still see
    # who was excluded and why (`result.attrs["excluded_no_prior_data"]`),
    # same transparency principle as `p_start_grounded` -- these players
    # clear the start-probability bar and may well be worth a pick, they
    # just can't be RANKED against real data honestly, so they're surfaced
    # separately rather than either hidden or scored on a guess.
    result.attrs["excluded_no_prior_data"] = excluded_no_data.reset_index(drop=True)
    return result


def forward_shortlist(players_df: pd.DataFrame, teams_df: pd.DataFrame,
                      fixtures_df: pd.DataFrame, start_gw: int,
                      n_gw: int = DEFAULT_RUN_LENGTH,
                      min_start_probability: float = MIN_START_PROBABILITY_FOR_MID_SHORTLIST,
                      matches_played: float = 0,
                      ) -> pd.DataFrame:
    """Forwards ranked by own attacking output, service quality from the
    team's creative midfielders/wingers, team possession, discipline,
    penalty duty, and the 4-gameweek attacking fixture run.

    Unlike `midfielder_shortlist()`, this does NOT need a blended
    attacking/defensive profile -- there's no real "defensive forward"
    archetype the way there's a defensive-midfielder one, and Neil didn't
    ask for one here. `attacking_score` alone (goals, assists, xG+xA,
    threat -- shrunk per-90 off prior season, same treatment as
    everywhere else) is the core signal.

    WHAT'S NEW VERSUS THE OLD FWD AGENT: two of the three things Neil
    specifically asked for did not exist anywhere in this project before
    today:
      - `team_creative_supply`: the combined shrunk per-90 output of the
        team's best creative midfielders (see `team_creative_supply()`)
        -- "I want to consider the attacking midfielder and wingers in
        their team, and their chances created, passing, assist stats."
        This is literally a striker's supply line, quantified.
      - `team_possession_pct`: real 2025/26 team possession %, sourced
        from Sofascore (FPL's API and the vaastav archive have no
        possession stat at all -- confirmed by grep, not assumed) --
        "possession of the team is important for both midfielders and
        attackers." Also applied to `midfielder_shortlist()` retroactively
        for the same reason. Covers 17 of the current season's 20 clubs;
        this season's 3 promoted teams have no top-flight possession
        history and are left NaN, not guessed.

    Same two gates as `midfielder_shortlist()`, same reasoning: excluded
    below `min_start_probability`, and separately excluded (not scored on
    a guess) if there's no real prior-season row at all --
    `result.attrs["excluded_no_prior_data"]`.
    """
    run_df = all_teams_fixture_run(teams_df, fixtures_df, start_gw, n_gw,
                                   rank_by="mean_attack_multiplier")
    run_by_team = run_df.set_index("team_name")[["mean_attack_multiplier", "n_fixtures", "n_blanks", "n_doubles"]]

    pool = players_df[players_df["position"] == "FWD"].copy()
    pool = pool.join(run_by_team, on="team_name")
    pool["team_attack_strength"] = pool["team_name"].apply(lambda t: team_attack_strength(t, teams_df))
    pool["team_possession_pct"] = pool["team_name"].apply(lambda t: team_possession_pct(t, teams_df))
    pool["team_creative_supply"] = pool["team_name"].apply(lambda t: team_creative_supply(t, players_df))

    p_start, p_start_grounded = [], []
    for _, row in pool.iterrows():
        p, flags = m.estimate_start_probability(row, matches_played=matches_played)
        p_start.append(p)
        p_start_grounded.append("start_prob_default" not in flags)
    pool["p_start"] = p_start
    pool["p_start_grounded"] = p_start_grounded
    pool = pool[pool["p_start"] >= min_start_probability].copy()

    # Same no-prior-data gate as midfielder_shortlist -- see that
    # function's docstring for the full reasoning (flat priors dressed up
    # as evidence would silently win over real proven output).
    minutes_prev = pool["minutes_prev_season"] if "minutes_prev_season" in pool.columns \
        else pd.Series(pd.NA, index=pool.index)
    pool["has_prior_data"] = minutes_prev.apply(lambda v: pd.notna(v) and float(v) > 0)
    excluded_no_data = pool[~pool["has_prior_data"]][["id", "web_name", "team_name", "p_start"]].copy()
    pool = pool[pool["has_prior_data"]].copy()

    def _get0(row, col: str) -> float:
        v = row.get(col)
        return 0.0 if pd.isna(v) else float(v)

    def _attacking_score(row) -> float:
        """Shrunk per-90 blend of goals + assists, xG + xA, and threat
        (ICT's goal-threat sub-index -- the forward-specific counterpart
        to creativity, same documented-proxy treatment)."""
        minutes = _get0(row, "minutes_prev_season")
        goals = _get0(row, "goals_scored_prev_season")
        assists = _get0(row, "assists_prev_season")
        xg = _get0(row, "expected_goals_prev_season")
        xa = _get0(row, "expected_assists_prev_season")
        threat = _get0(row, "threat_prev_season")
        realised_p90 = m.shrunk_per90_rate(goals + assists, minutes, prior_rate=0.35)
        underlying_p90 = m.shrunk_per90_rate(xg + xa, minutes, prior_rate=0.35)
        threat_p90 = m.shrunk_per90_rate(threat, minutes, prior_rate=25.0)
        return realised_p90 + underlying_p90 + (threat_p90 / 50.0)

    def _discipline_score(row) -> float:
        minutes = _get0(row, "minutes_prev_season")
        yellow = _get0(row, "yellow_cards_prev_season")
        red = _get0(row, "red_cards_prev_season")
        weighted = yellow * CARD_WEIGHT["yellow"] + red * CARD_WEIGHT["red"]
        return m.shrunk_per90_rate(weighted, minutes, prior_rate=0.25)

    def _set_piece_duty(row) -> str:
        """Penalty duty specifically -- the single biggest scoring swing
        factor for forwards, per the original FWD spec, and now real
        structured data instead of pure research (see fpl_client.py)."""
        pens = row.get("penalties_order")
        if pd.notna(pens) and float(pens) == 1:
            return "penalties"
        if pd.notna(pens) and float(pens) == 2:
            return "backup taker"
        return "none"

    pool["attacking_score"] = pool.apply(_attacking_score, axis=1)
    pool["discipline_score"] = pool.apply(_discipline_score, axis=1)
    pool["set_piece_duty"] = pool.apply(_set_piece_duty, axis=1)

    att_z = _zscore(pool["attacking_score"])
    fixture_z = _zscore(pool["mean_attack_multiplier"].fillna(pool["mean_attack_multiplier"].mean()))
    team_z = _zscore(pool["team_attack_strength"].fillna(pool["team_attack_strength"].mean()))
    poss_z = _zscore(pool["team_possession_pct"].fillna(pool["team_possession_pct"].mean()))
    supply_z = _zscore(pool["team_creative_supply"].fillna(pool["team_creative_supply"].mean()))
    discipline_z = _zscore(pool["discipline_score"])
    penalty_bonus = pool["set_piece_duty"].apply(
        lambda s: 0.75 if s == "penalties" else (0.2 if s == "backup taker" else 0.0)
    )

    pool["run_score"] = (
        att_z * 1.5              # own output is still the main driver
        + fixture_z * 0.5
        + team_z * 0.4
        + poss_z * 0.25
        + supply_z * 0.35        # service from the team's creative mids
        + penalty_bonus
        - discipline_z * 0.25
    )

    keep_cols = ["id", "web_name", "team_name", "position", "price_m",
                "p_start", "p_start_grounded",
                "attacking_score", "mean_attack_multiplier", "team_attack_strength",
                "team_possession_pct", "team_creative_supply",
                "set_piece_duty", "discipline_score", "run_score"]
    result = pool[keep_cols].sort_values("run_score", ascending=False).reset_index(drop=True)
    result.attrs["excluded_no_prior_data"] = excluded_no_data.reset_index(drop=True)
    return result


__all__ = ["team_fixture_run", "all_teams_fixture_run", "team_attack_strength",
           "team_possession_pct", "team_creative_supply", "forward_shortlist",
           "defensive_shortlist", "midfielder_shortlist", "DEFAULT_RUN_LENGTH"]
