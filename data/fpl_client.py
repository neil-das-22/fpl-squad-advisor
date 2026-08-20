"""
FPL API client — fetches and parses data from the official Fantasy Premier
League API.

Endpoints used (no auth required, public API):
  - https://fantasy.premierleague.com/api/bootstrap-static/
      -> players ("elements"), teams, positions ("element_types"), gameweeks ("events")
  - https://fantasy.premierleague.com/api/fixtures/
      -> full season fixture list with difficulty ratings
  - https://fantasy.premierleague.com/api/element-summary/{player_id}/
      -> one player's gameweek-by-gameweek history + past-season summaries

NOTE on running this: this file was developed inside a sandboxed dev session
whose outbound network is restricted to an allowlist that does NOT include
fantasy.premierleague.com. That means it could not be live-tested against
the real API from that sandbox. Field names below reflect the well-documented,
stable public FPL API schema. Run `validate_schema()` the first time you use
this on a machine with real internet access (local machine, GitHub Actions,
etc.) — it will loudly flag anything that's drifted rather than silently
producing bad data downstream.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

# Clubs promoted into the 2026/27 Premier League with no current top-flight
# data. The model layer needs fallback handling for these — see
# docs/football_domain_knowledge.md, "Promoted & newly-assembled teams".
PROMOTED_TEAMS_2026_27 = {"Coventry City", "Ipswich Town", "Hull City"}

# Maps FPL's "status" field to plain meaning.
STATUS_MEANINGS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
}

EXPECTED_BOOTSTRAP_KEYS = {"elements", "teams", "element_types", "events"}
EXPECTED_PLAYER_FIELDS = {
    "id", "code", "first_name", "second_name", "web_name", "team", "element_type",
    "now_cost", "total_points", "minutes", "goals_scored", "assists",
    "clean_sheets", "expected_goals", "expected_assists", "status",
    "chance_of_playing_next_round", "selected_by_percent", "form",
}
EXPECTED_FIXTURE_FIELDS = {
    "id", "event", "team_h", "team_a", "team_h_difficulty",
    "team_a_difficulty", "kickoff_time", "finished",
}


def _get(url: str) -> dict | list:
    resp = requests.get(url, headers={"User-Agent": "fpl-squad-advisor/1.0"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap_static() -> dict:
    """Players, teams, positions, gameweeks — the core reference data."""
    return _get(f"{BASE_URL}/bootstrap-static/")


def fetch_fixtures() -> list:
    """Full season fixture list with difficulty ratings."""
    return _get(f"{BASE_URL}/fixtures/")


def fetch_player_summary(player_id: int) -> dict:
    """One player's gameweek history + past-season summaries."""
    return _get(f"{BASE_URL}/element-summary/{player_id}/")


# ---------------------------------------------------------------------------
# Manager (a real person's FPL team) endpoints -- added for the public
# website's "look up my team" feature.
# ---------------------------------------------------------------------------
#
# Both endpoints are PUBLIC and read-only: no login, password, or API key
# required. A manager's team ID is the number in the URL when they view
# their own team on fantasy.premierleague.com (e.g.
# .../entry/1234567/event/1/ -> team ID 1234567) -- it is not a secret, it's
# how the official site itself links to a public team. This is deliberately
# the ONLY way the website asks for account information; it never asks a
# user for their FPL email/password (entering those into a third-party site
# would be a real credential-phishing pattern, not something this project
# does).


def fetch_manager_entry(entry_id: int) -> dict:
    """A manager's public profile: display name, team name, overall rank,
    total points, etc. Raises `requests.HTTPError` (404) for a team ID that
    doesn't exist -- the caller should catch this and show a friendly
    "team not found" message rather than a stack trace.
    """
    return _get(f"{BASE_URL}/entry/{entry_id}/")


def fetch_manager_picks(entry_id: int, event: int) -> dict:
    """A manager's 15-player squad for one gameweek: `picks` (each with
    `element` = player id, `position` 1-15, `is_captain`, `is_vice_captain`,
    `multiplier`), plus `entry_history` (bank, squad value, points that
    gameweek). Raises `requests.HTTPError` if the manager hasn't set a
    squad for that gameweek yet (e.g. asking for a future, unplayed
    gameweek) or the team ID doesn't exist.
    """
    return _get(f"{BASE_URL}/entry/{entry_id}/event/{event}/picks/")


def parse_manager_squad(picks_payload: dict, players_df: pd.DataFrame) -> pd.DataFrame:
    """Join a manager's raw `picks` list onto our own players table by id,
    so the rest of the app can treat "my squad" exactly like any other
    slice of `players_df` -- same columns, same downstream code.

    Adds `squad_position` (1-15, FPL's own slot order: 1-11 starting XI in
    formation order, 12-15 bench), `is_captain`, `is_vice_captain`, and
    `multiplier` (2 for captain, 3 for a Triple-Captained pick, 0 for an
    unused bench spot in some historical payload shapes -- callers should
    not assume it's always 0 or 1 for non-captains).
    """
    picks = pd.DataFrame(picks_payload["picks"])
    picks = picks.rename(columns={"element": "id", "position": "squad_position"})
    merged = players_df.merge(
        picks[["id", "squad_position", "is_captain", "is_vice_captain", "multiplier"]],
        on="id", how="inner",
    )
    if len(merged) != len(picks):
        missing = set(picks["id"]) - set(merged["id"])
        raise ValueError(
            f"{len(missing)} of {len(picks)} squad picks did not match our player "
            f"table (ids: {missing}) -- our cached player data may be stale."
        )
    return merged.sort_values("squad_position").reset_index(drop=True)


def save_raw(data, filename: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def validate_schema(bootstrap: dict, fixtures: list) -> list[str]:
    """Returns a list of warning strings. Empty list = schema looks as expected."""
    warnings = []
    missing_top = EXPECTED_BOOTSTRAP_KEYS - set(bootstrap.keys())
    if missing_top:
        warnings.append(f"bootstrap-static missing expected top-level keys: {missing_top}")

    elements = bootstrap.get("elements", [])
    if elements:
        missing_player = EXPECTED_PLAYER_FIELDS - set(elements[0].keys())
        if missing_player:
            warnings.append(f"player objects missing expected fields: {missing_player}")
    else:
        warnings.append("bootstrap-static has no elements (players) — fetch likely failed")

    if fixtures:
        missing_fixture = EXPECTED_FIXTURE_FIELDS - set(fixtures[0].keys())
        if missing_fixture:
            warnings.append(f"fixture objects missing expected fields: {missing_fixture}")
    else:
        warnings.append("fixtures list is empty — fetch likely failed")

    return warnings


def load_teams_df(bootstrap: dict) -> pd.DataFrame:
    teams = pd.DataFrame(bootstrap["teams"])[
        ["id", "code", "name", "short_name", "strength",
         "strength_overall_home", "strength_overall_away",
         "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ]
    teams["is_promoted"] = teams["name"].isin(PROMOTED_TEAMS_2026_27)
    return teams


def load_positions_df(bootstrap: dict) -> pd.DataFrame:
    return pd.DataFrame(bootstrap["element_types"])[
        ["id", "singular_name", "singular_name_short", "squad_select",
         "squad_min_play", "squad_max_play"]
    ]


def load_players_df(bootstrap: dict) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    # NOTE: `players` already carries its own `team_code` field straight from
    # the API (each element duplicates its club's stable code onto the row) --
    # do NOT also pull `code` off `teams` here and rename it to the same
    # name. That previously caused a silent pandas merge collision
    # (`team_code_x`/`team_code_y`, neither literally named `team_code`)
    # that made the crest-badge URL builder quietly get no column to read.
    teams = load_teams_df(bootstrap)[["id", "name", "short_name", "is_promoted"]].rename(
        columns={"id": "team", "name": "team_name", "short_name": "team_short"}
    )
    positions = load_positions_df(bootstrap)[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )

    df = players.merge(teams, on="team", how="left").merge(positions, on="element_type", how="left")
    df["price_m"] = df["now_cost"] / 10.0
    df["status_meaning"] = df["status"].map(STATUS_MEANINGS)
    df["full_name"] = df["first_name"] + " " + df["second_name"]

    keep_cols = [
        # `code` is FPL's STABLE player identifier: it is minted once per human
        # being and never changes, so it survives season rollovers and summer
        # transfers. `id` does NOT -- it is season/context-scoped and gets
        # reissued, so joining historical data on `id` silently matches the
        # wrong players (measured against the 2025/26 archive: 568 of 573
        # current `id`s now belong to a different `code` than they did last
        # season). Everything that joins across seasons must use `code`.
        "id", "code", "full_name", "web_name", "team_name", "team_short", "team_code", "is_promoted",
        "position", "price_m", "total_points", "points_per_game", "form",
        "selected_by_percent", "minutes", "starts", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "bonus", "bps",
        "ict_index", "status", "status_meaning", "chance_of_playing_next_round",
        "news",
        # Raw defensive-contribution counting stats (CBIT for DEF, CBIRT for
        # MID/FWD). The API returns these but they were previously dropped,
        # forcing xp_model to guess every player's DefCon rate from a flat
        # positional average. Backtesting against real 2025/26 data showed
        # this costs a 66% hit in DefCon-scorer identification -- see
        # backtest/results_2025_26.md section 8. season-cumulative totals,
        # converted to a per-90 rate in xp_model via shrunk_per90_rate().
        "clearances_blocks_interceptions", "tackles", "recoveries",
        # Chance-creation / attacking-output proxies and discipline record.
        # FPL's public API does NOT expose pass-completion counts or a
        # standalone "big chances created" stat (that's Opta-tier data FPL
        # doesn't publish) -- `creativity` (an ICT sub-index built from
        # passing/crossing/chance-creation events) and `threat` are the
        # closest real substitutes, and are used as documented proxies, not
        # a silent stand-in. `yellow_cards`/`red_cards` are real counts, used
        # as a negative factor for the defensive-midfielder profile.
        "creativity", "threat", "yellow_cards", "red_cards",
        # Real, structured set-piece pecking order -- previously the MID/FWD
        # research agents had to find this via web search each week even
        # though the API has published it directly all along. 1 = primary
        # taker, higher numbers = further down the list, null = not on it.
        "penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Prior-season reference data (closes xp_model KNOWN DATA GAP #1)
# ---------------------------------------------------------------------------
#
# WHY THIS IS A SEPARATE, OPT-IN STEP
# -----------------------------------
# At GW1 of a new season a player's *current-season* minutes/starts carry no
# information, so the xP model has nothing to base a start probability on and
# falls back to a flat prior for everyone -- nailed-on starter and third-choice
# backup alike. The fix is to feed it last season's minutes/starts explicitly.
#
# The obvious place to do that would be inside `load_players_df()`, but that
# would give the live API call a hard dependency on a historical CSV existing
# on disk, and `load_players_df()` has to keep working in environments where it
# doesn't (CI, a fresh clone, someone else's machine). So this is deliberately
# a separate function the caller opts into, and `attach_prior_season_stats()`
# is a pure left-join that leaves NaN wherever there is no match.
#
# NaN IS THE POINT. A player with no row in the archive did not appear in the
# Premier League last season at all (promoted clubs, arrivals from abroad).
# That is "we have no information", which must keep routing him to the flat
# prior / promoted-team fallback. It is emphatically NOT the same as a player
# who WAS in the league and played zero minutes (Illan Meslier, 2025/26) --
# that is strong evidence he won't start. Filling missing rows with zeros would
# conflate the two and reintroduce almost exactly the bug fixed in
# backtest/results_2025_26.md section 1b (see NEVER_APPEARED_MATCHES_THRESHOLD
# in models/xp_model.py). Do not "helpfully" fillna(0) downstream.

# Default source: the season-end 2025/26 player table bulk-downloaded from the
# vaastav/Fantasy-Premier-League archive for backtesting. Same underlying
# numbers as the per-player `history_past` array on the live
# /api/element-summary/{id}/ endpoint, but all ~700 players in one file and no
# network round-trip per player.
PRIOR_SEASON_CSV_DEFAULT = os.path.join(RAW_DIR, "historical_2025_26", "players_raw.csv")

# source column -> renamed column. The `_prev_season` suffix is not decoration:
# these frames get merged alongside identically-named current-season columns
# and then round-tripped through CSV, and an ambiguous `minutes` would be a
# silent-corruption bug waiting to happen.
PRIOR_SEASON_COLUMN_MAP = {
    "minutes": "minutes_prev_season",
    "starts": "starts_prev_season",
    "total_points": "total_points_prev_season",
    "clean_sheets": "clean_sheets_prev_season",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions_prev_season",
    "tackles": "tackles_prev_season",
    "recoveries": "recoveries_prev_season",
    # Not in the original spec for this function, added deliberately: pre-season
    # the live bootstrap-static `expected_goals`/`expected_assists` are still
    # last season's carryover numbers, and the moment FPL rolls the season over
    # they reset to 0. Without these two columns every player's xG/xA per 90
    # would collapse to the flat positional prior at exactly the gameweek this
    # model exists to predict. See xp_model.calculate_xp().
    "expected_goals": "expected_goals_prev_season",
    "expected_assists": "expected_assists_prev_season",
    # Added for the MID-agent rebuild: attacking output (goals/assists),
    # creativity (chance-creation proxy -- see keep_cols comment in
    # load_players_df for why this substitutes for "big chances created",
    # which FPL's API doesn't publish), and cards (discipline record for the
    # defensive-midfielder profile). Same carryover logic as xG/xA above:
    # pre-rollover these duplicate the live bootstrap-static numbers, but
    # once the new season starts accumulating fresh data this is what keeps
    # last season's baseline available instead of it silently vanishing.
    "goals_scored": "goals_scored_prev_season",
    "assists": "assists_prev_season",
    "creativity": "creativity_prev_season",
    "threat": "threat_prev_season",
    "yellow_cards": "yellow_cards_prev_season",
    "red_cards": "red_cards_prev_season",
}

PRIOR_SEASON_COLUMNS = [
    "code",
    "minutes_prev_season",
    "starts_prev_season",
    "total_points_prev_season",
    "price_prev_season_m",
    "points_per_million_prev_season",
    "clean_sheets_prev_season",
    "clearances_blocks_interceptions_prev_season",
    "tackles_prev_season",
    "recoveries_prev_season",
    "expected_goals_prev_season",
    "expected_assists_prev_season",
    "goals_scored_prev_season",
    "assists_prev_season",
    "creativity_prev_season",
    "threat_prev_season",
    "yellow_cards_prev_season",
    "red_cards_prev_season",
]


def load_prior_season_reference(csv_path: str = PRIOR_SEASON_CSV_DEFAULT) -> pd.DataFrame:
    """Slim, unambiguously-named prior-season player table keyed on `code`.

    Args:
        csv_path: a season-end player table shaped like the vaastav archive's
            `players_raw.csv` (or the equivalent assembled from the live API's
            `history_past` arrays). Must contain a `code` column.

    Returns:
        One row per player `code`, columns per PRIOR_SEASON_COLUMNS. Every
        stat column is numeric; unparseable values become NaN rather than 0.

    Notes:
        * `now_cost` in a season-end archive is the player's END-of-season
          price, so it is exposed as `price_prev_season_m` (= now_cost / 10) --
          not to be confused with the live `price_m`.
        * `points_per_million_prev_season` is NaN, not inf, when the price is
          zero or missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"prior-season reference CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if "code" not in raw.columns:
        raise ValueError(
            f"{csv_path} has no `code` column. Joining prior-season data on `id` "
            "is not an acceptable substitute -- ids are reissued between seasons."
        )

    out = pd.DataFrame()
    out["code"] = pd.to_numeric(raw["code"], errors="coerce").astype("Int64")

    for src, dest in PRIOR_SEASON_COLUMN_MAP.items():
        out[dest] = (pd.to_numeric(raw[src], errors="coerce")
                     if src in raw.columns else pd.NA)

    # Season-end price. `now_cost` is what this archive calls it; `end_cost` is
    # what the live element-summary `history_past` calls the same number.
    cost_col = next((c for c in ("now_cost", "end_cost") if c in raw.columns), None)
    out["price_prev_season_m"] = (
        pd.to_numeric(raw[cost_col], errors="coerce") / 10.0 if cost_col else pd.NA
    )

    price = pd.to_numeric(out["price_prev_season_m"], errors="coerce")
    points = pd.to_numeric(out["total_points_prev_season"], errors="coerce")
    # Divide-by-zero guard: a 0.0m price is meaningless, so value is unknown
    # (NaN), not infinite.
    out["points_per_million_prev_season"] = points.divide(price.where(price > 0))

    out = out.dropna(subset=["code"])
    if out["code"].duplicated().any():
        # Shouldn't happen in a clean season-end export, but if a code appears
        # twice keep the row with the most minutes rather than an arbitrary one.
        out = (out.sort_values("minutes_prev_season", ascending=False)
                  .drop_duplicates(subset=["code"], keep="first"))

    return out[PRIOR_SEASON_COLUMNS].reset_index(drop=True)


def attach_prior_season_stats(players_df: pd.DataFrame,
                              prior_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join prior-season stats onto a `load_players_df()` result by `code`.

    Unmatched players (no Premier League appearance last season) get NaN in
    every `*_prev_season` column. That is the correct signal and downstream code
    depends on it -- see the module comment above.

    Passing `prior_df=None` still returns the full column set, all-NaN, so the
    schema does not change depending on whether the archive happened to be on
    disk.
    """
    if "code" not in players_df.columns:
        raise ValueError(
            "players_df has no `code` column -- rebuild it with load_players_df(). "
            "Do not fall back to joining on `id`; ids are reissued across seasons."
        )

    out = players_df.copy()
    added = [c for c in PRIOR_SEASON_COLUMNS if c != "code"]

    if prior_df is None or len(prior_df) == 0:
        for col in added:
            out[col] = pd.NA
        return out

    # Join on a temporary normalised key so the caller's own `code` dtype is
    # preserved exactly as load_players_df() produced it.
    left = out.assign(_code_key=pd.to_numeric(out["code"], errors="coerce").astype("Int64"))
    right = prior_df.copy()
    right["_code_key"] = pd.to_numeric(right["code"], errors="coerce").astype("Int64")
    right = right.drop(columns=["code"])

    merged = left.merge(right, on="_code_key", how="left").drop(columns=["_code_key"])
    return merged


# ---------------------------------------------------------------------------
# Team-level prior-season reference data (possession, big chances created,
# pass accuracy -- NOT published by the FPL API or the vaastav archive at all)
# ---------------------------------------------------------------------------
#
# WHY THIS IS A SEPARATE FILE FROM EVERYTHING ELSE IN THIS MODULE
# Every other prior-season number in this file comes from the FPL API or the
# vaastav archive, which mirrors it. Team possession %, team big-chances-
# created, and team pass-accuracy % do not exist in either source -- checked
# directly (grepped every column header in players_raw.csv, teams.csv,
# merged_gw.csv, fixtures.csv, and the live bootstrap-static.json; none of
# them have it). This is the one dataset in the whole project sourced from
# somewhere else: Sofascore's team-statistics API for the 2025/26 Premier
# League season (unique-tournament 17, season 76986), pulled via a live
# Chrome fetch in-session (fbref.com blocked the same request behind bot
# detection; premierleague.com's stats pages returned dead links).
#
# COVERAGE: 17 of this season's 20 clubs. The 2025/26 Premier League's 20
# clubs and this season's (2026/27) 20 clubs overlap on exactly 17 -- three
# clubs relegated at the end of 2025/26 (not needed here) were replaced by
# this season's three promoted clubs (Coventry City, Ipswich Town, Hull
# City), who were playing Championship football last season and have no
# top-flight possession/chance-creation data to pull. Same "NaN is the
# point, not a bug" principle as PRIOR_SEASON_COLUMNS above -- a promoted
# club's missing row is real information (no Premier League history),
# not a gap to paper over with a league-average guess.
TEAM_PRIOR_SEASON_CSV_DEFAULT = os.path.join(RAW_DIR, "historical_2025_26", "team_possession.csv")

TEAM_PRIOR_SEASON_COLUMNS = [
    "team_name",
    "possession_pct_prev_season",
    "big_chances_created_prev_season",
    "pass_accuracy_pct_prev_season",
]


def load_team_prior_season_reference(csv_path: str = TEAM_PRIOR_SEASON_CSV_DEFAULT) -> pd.DataFrame:
    """Team-level prior-season possession/chance-creation/passing reference.

    Returns one row per club (17 of the current season's 20 -- see module
    comment above for why 3 are structurally missing), keyed on `team_name`
    exactly as `load_teams_df()` spells it (already hand-mapped from
    Sofascore's naming when this CSV was built -- "Manchester City" ->
    "Man City", "Nottingham Forest" -> "Nott'm Forest", etc. -- so this
    joins on team_name directly with no further name-matching needed).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"team prior-season reference CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    for col in TEAM_PRIOR_SEASON_COLUMNS[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw[TEAM_PRIOR_SEASON_COLUMNS]


def attach_team_prior_season_stats(teams_df: pd.DataFrame,
                                   team_prior_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join team-level prior-season stats onto a `load_teams_df()` result.

    Unmatched clubs (this season's 3 promoted teams) get NaN, same
    contract as `attach_prior_season_stats()`. `team_prior_df=None` still
    returns the full column set, all-NaN.
    """
    out = teams_df.copy()
    added = [c for c in TEAM_PRIOR_SEASON_COLUMNS if c != "team_name"]
    if team_prior_df is None or len(team_prior_df) == 0:
        for col in added:
            out[col] = pd.NA
        return out
    return out.merge(team_prior_df, left_on="name", right_on="team_name", how="left").drop(columns=["team_name"])


def load_fixtures_df(fixtures: list, bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(fixtures)
    teams = load_teams_df(bootstrap)[["id", "name", "short_name"]]
    df = df.merge(teams.rename(columns={"id": "team_h", "name": "home_team", "short_name": "home_short"}),
                   on="team_h", how="left")
    df = df.merge(teams.rename(columns={"id": "team_a", "name": "away_team", "short_name": "away_short"}),
                   on="team_a", how="left")
    keep_cols = [
        "id", "event", "kickoff_time", "home_team", "home_short", "away_team",
        "away_short", "team_h_difficulty", "team_a_difficulty", "finished",
        "team_h_score", "team_a_score",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].rename(columns={"event": "gameweek"})


def load_players_with_prior_season(bootstrap: dict,
                                   csv_path: str = PRIOR_SEASON_CSV_DEFAULT,
                                   verbose: bool = True) -> pd.DataFrame:
    """`load_players_df()` + prior-season columns, degrading gracefully.

    This is the convenience wrapper the live pipeline uses. If the archive CSV
    isn't on disk (different machine, fresh clone, CI) it prints a note and
    returns the frame with all-NaN prior-season columns rather than crashing --
    the model treats that exactly like "player has no PL history", i.e. flat
    prior, which is the pre-fix behaviour and is safe.
    """
    players_df = load_players_df(bootstrap)
    try:
        prior_df = load_prior_season_reference(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        if verbose:
            print(f"[PRIOR SEASON] unavailable ({exc}); falling back to flat "
                  f"start-probability priors for all players.")
        return attach_prior_season_stats(players_df, None)

    out = attach_prior_season_stats(players_df, prior_df)
    if verbose:
        matched = int(out["minutes_prev_season"].notna().sum())
        print(f"[PRIOR SEASON] matched {matched}/{len(out)} players on `code` "
              f"({len(out) - matched} with no 2025/26 PL history -> NaN, flat prior)")
    return out


def run_pipeline() -> dict:
    """Fetch, validate, save raw + processed data. Returns dict of DataFrames."""
    bootstrap = fetch_bootstrap_static()
    fixtures = fetch_fixtures()

    warnings = validate_schema(bootstrap, fixtures)
    for w in warnings:
        print(f"[SCHEMA WARNING] {w}")

    save_raw(bootstrap, "bootstrap_static.json")
    save_raw(fixtures, "fixtures.json")

    players_df = load_players_with_prior_season(bootstrap)
    teams_df = load_teams_df(bootstrap)
    fixtures_df = load_fixtures_df(fixtures, bootstrap)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    players_df.to_csv(os.path.join(PROCESSED_DIR, "players.csv"), index=False)
    teams_df.to_csv(os.path.join(PROCESSED_DIR, "teams.csv"), index=False)
    fixtures_df.to_csv(os.path.join(PROCESSED_DIR, "fixtures.csv"), index=False)

    print(f"Pulled {len(players_df)} players, {len(teams_df)} teams, "
          f"{len(fixtures_df)} fixtures at {datetime.now(timezone.utc).isoformat()}")

    return {"players": players_df, "teams": teams_df, "fixtures": fixtures_df}


if __name__ == "__main__":
    run_pipeline()
