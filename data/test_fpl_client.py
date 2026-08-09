"""
Offline test for the parsing logic in fpl_client.py, using a small synthetic
bootstrap-static/fixtures payload shaped like the real API. This sandbox can't
reach the live FPL API (see fpl_client.py docstring), so this is how the
parsing functions get validated until this runs somewhere with real network
access — at which point run_pipeline() itself is the real test.

Run with: python3 data/test_fpl_client.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import fpl_client  # noqa: E402

SAMPLE_BOOTSTRAP = {
    "elements": [
        {
            "id": 1, "first_name": "Erling", "second_name": "Haaland",
            "web_name": "Haaland", "team": 1, "element_type": 4, "now_cost": 150,
            "total_points": 0, "points_per_game": "0.0", "form": "0.0",
            "selected_by_percent": "45.0", "minutes": 0, "starts": 0,
            "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
            "expected_goals": "0.00", "expected_assists": "0.00",
            "expected_goal_involvements": "0.00", "expected_goals_conceded": "0.00",
            "bonus": 0, "bps": 0, "ict_index": "0.0", "status": "a",
            "status_meaning_placeholder": None,
            "chance_of_playing_next_round": 100, "news": "",
        },
        {
            "id": 2, "first_name": "Haji", "second_name": "Wright",
            "web_name": "Wright", "team": 2, "element_type": 3, "now_cost": 55,
            "total_points": 0, "points_per_game": "0.0", "form": "0.0",
            "selected_by_percent": "0.5", "minutes": 0, "starts": 0,
            "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
            "expected_goals": "0.00", "expected_assists": "0.00",
            "expected_goal_involvements": "0.00", "expected_goals_conceded": "0.00",
            "bonus": 0, "bps": 0, "ict_index": "0.0", "status": "a",
            "chance_of_playing_next_round": 100, "news": "",
        },
    ],
    "teams": [
        {
            "id": 1, "name": "Man City", "short_name": "MCI", "strength": 5,
            "strength_overall_home": 1350, "strength_overall_away": 1370,
            "strength_attack_home": 1360, "strength_attack_away": 1380,
            "strength_defence_home": 1340, "strength_defence_away": 1360,
        },
        {
            "id": 2, "name": "Coventry City", "short_name": "COV", "strength": 2,
            "strength_overall_home": 1050, "strength_overall_away": 1030,
            "strength_attack_home": 1040, "strength_attack_away": 1020,
            "strength_defence_home": 1050, "strength_defence_away": 1030,
        },
    ],
    "element_types": [
        {"id": 1, "singular_name": "Goalkeeper", "singular_name_short": "GKP",
         "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
        {"id": 2, "singular_name": "Defender", "singular_name_short": "DEF",
         "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
        {"id": 3, "singular_name": "Midfielder", "singular_name_short": "MID",
         "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
        {"id": 4, "singular_name": "Forward", "singular_name_short": "FWD",
         "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
    ],
    "events": [],
}

SAMPLE_FIXTURES = [
    {
        "id": 1, "event": 1, "kickoff_time": "2026-08-21T19:00:00Z",
        "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 5,
        "finished": False, "team_h_score": None, "team_a_score": None,
    },
]


def run_tests():
    warnings = fpl_client.validate_schema(SAMPLE_BOOTSTRAP, SAMPLE_FIXTURES)
    # We expect no warnings for required fields on this synthetic payload.
    assert warnings == [], f"Unexpected schema warnings: {warnings}"

    players = fpl_client.load_players_df(SAMPLE_BOOTSTRAP)
    assert len(players) == 2
    assert set(players["position"]) == {"FWD", "MID"}
    assert players.loc[players["web_name"] == "Haaland", "price_m"].iloc[0] == 15.0
    assert bool(players.loc[players["web_name"] == "Wright", "is_promoted"].iloc[0]) is True
    assert bool(players.loc[players["web_name"] == "Haaland", "is_promoted"].iloc[0]) is False

    teams = fpl_client.load_teams_df(SAMPLE_BOOTSTRAP)
    assert len(teams) == 2
    assert teams.loc[teams["name"] == "Coventry City", "is_promoted"].iloc[0]

    fixtures = fpl_client.load_fixtures_df(SAMPLE_FIXTURES, SAMPLE_BOOTSTRAP)
    assert len(fixtures) == 1
    assert fixtures.iloc[0]["home_team"] == "Man City"
    assert fixtures.iloc[0]["away_team"] == "Coventry City"
    assert fixtures.iloc[0]["gameweek"] == 1

    print("All parsing tests passed.")


if __name__ == "__main__":
    run_tests()
