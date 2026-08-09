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
    "id", "first_name", "second_name", "web_name", "team", "element_type",
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
        ["id", "name", "short_name", "strength",
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
        "id", "full_name", "web_name", "team_name", "team_short", "is_promoted",
        "position", "price_m", "total_points", "points_per_game", "form",
        "selected_by_percent", "minutes", "starts", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "bonus", "bps",
        "ict_index", "status", "status_meaning", "chance_of_playing_next_round",
        "news",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


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


def run_pipeline() -> dict:
    """Fetch, validate, save raw + processed data. Returns dict of DataFrames."""
    bootstrap = fetch_bootstrap_static()
    fixtures = fetch_fixtures()

    warnings = validate_schema(bootstrap, fixtures)
    for w in warnings:
        print(f"[SCHEMA WARNING] {w}")

    save_raw(bootstrap, "bootstrap_static.json")
    save_raw(fixtures, "fixtures.json")

    players_df = load_players_df(bootstrap)
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
