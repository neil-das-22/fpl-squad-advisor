"""
Multi-gameweek fixture-run analysis for defensive assets (GK/DEF).

WHY THIS EXISTS
Everything in xp_model.py is single-gameweek: "what's this player worth
next match." For goalkeepers and defenders specifically, Neil wants
selection to also weigh a rolling window (default: next 4 gameweeks) of
upcoming opposition, combined with the player's own defensive stats --
not just next week's fixture in isolation. A defender with modest week-1
numbers but four straight favourable matchups is a different pick than one
with a great week 1 followed by three brutal fixtures.

This reuses `xp_model.fixture_context()` (the same team-strength /
difficulty-rating model that drives single-gameweek clean-sheet
probability) rather than inventing a second fixture model -- the run score
and the single-gameweek xP stay internally consistent with each other.

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
    """One team's defensive fixture-run summary over `n_gw` gameweeks from `start_gw`.

    Returns mean clean-sheet probability across every fixture in the window
    (the headline number -- directly comparable to a single-gameweek
    clean_sheet_prob), the count of actual fixtures faced (a double
    gameweek counts twice, a blank counts zero -- this matters as much as
    the difficulty itself), and a gameweek-by-gameweek breakdown for
    showing the actual run, not just a summary statistic.
    """
    teams_by_name = {str(r["name"]): r for _, r in teams_df.iterrows()}
    own_row = teams_by_name.get(team_name)
    if own_row is None:
        return {"team_name": team_name, "n_fixtures": 0, "mean_clean_sheet_prob": None,
                "mean_difficulty": None, "breakdown": []}

    by_gw = _team_fixtures_by_gameweek(fixtures_df, team_name, start_gw, n_gw)

    breakdown = []
    cs_probs, difficulties = [], []
    for gw in range(start_gw, start_gw + n_gw):
        fixtures_this_gw = by_gw[gw]
        if not fixtures_this_gw:
            breakdown.append({"gameweek": gw, "opponent": None, "is_home": None,
                              "clean_sheet_prob": None, "difficulty": None, "blank": True})
            continue
        for frow, is_home in fixtures_this_gw:
            opp_name = str(frow["away_team"] if is_home else frow["home_team"])
            opp_row = teams_by_name.get(opp_name)
            difficulty = frow["team_h_difficulty"] if is_home else frow["team_a_difficulty"]
            ctx = m.fixture_context(own_row, opp_row, is_home, difficulty)
            breakdown.append({
                "gameweek": gw, "opponent": opp_name, "is_home": is_home,
                "clean_sheet_prob": ctx["clean_sheet_prob"],
                "difficulty": ctx["difficulty"], "blank": False,
            })
            cs_probs.append(ctx["clean_sheet_prob"])
            difficulties.append(ctx["difficulty"])

    return {
        "team_name": team_name,
        "n_fixtures": len(cs_probs),
        "n_blanks": sum(1 for b in breakdown if b["blank"]),
        "n_doubles": sum(1 for gw in by_gw if len(by_gw[gw]) > 1),
        "mean_clean_sheet_prob": (sum(cs_probs) / len(cs_probs)) if cs_probs else None,
        "mean_difficulty": (sum(difficulties) / len(difficulties)) if difficulties else None,
        "breakdown": breakdown,
    }


def all_teams_fixture_run(teams_df: pd.DataFrame, fixtures_df: pd.DataFrame,
                          start_gw: int, n_gw: int = DEFAULT_RUN_LENGTH) -> pd.DataFrame:
    """Every team's fixture run, ranked best-to-worst for defensive assets.

    This is the table to hand a GK/DEF research pass: "who has the kindest
    run of fixtures over the next N gameweeks," independent of any
    individual player.
    """
    rows = [team_fixture_run(name, teams_df, fixtures_df, start_gw, n_gw)
            for name in teams_df["name"]]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "breakdown"} for r in rows])
    df = df.sort_values("mean_clean_sheet_prob", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "run_rank", range(1, len(df) + 1))
    return df


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


__all__ = ["team_fixture_run", "all_teams_fixture_run", "defensive_shortlist", "DEFAULT_RUN_LENGTH"]
