"""
Offline test for the parsing logic in fpl_client.py, using a small synthetic
bootstrap-static/fixtures payload shaped like the real API. This sandbox can't
reach the live FPL API (see fpl_client.py docstring), so this is how the
parsing functions get validated until this runs somewhere with real network
access — at which point run_pipeline() itself is the real test.

Run with: python3 data/test_fpl_client.py
"""

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fpl_client  # noqa: E402

SAMPLE_BOOTSTRAP = {
    "elements": [
        {
            # `code` and `id` are deliberately unrelated numbers here, exactly
            # as they are in the real payload -- see test_prior_season_join().
            "id": 1, "code": 900001, "first_name": "Erling", "second_name": "Haaland",
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
            "id": 2, "code": 900002, "first_name": "Haji", "second_name": "Wright",
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


# A season-end player table shaped like
# data/raw/historical_2025_26/players_raw.csv, cut down to the columns
# load_prior_season_reference() actually reads.
#
# Note the trap baked into row 2: its `id` is 1, the same id the CURRENT
# payload gives Haaland, but it belongs to a different human being
# (code 555555). Joining prior-season data on `id` would hand Haaland this
# player's season. That is not hypothetical -- against the real 2026/27
# payload, 568 of 573 ids had changed hands since 2025/26.
SAMPLE_PRIOR_SEASON_ROWS = [
    {"code": 900001, "id": 88, "web_name": "Haaland", "element_type": 4,
     "minutes": 2953, "starts": 34, "total_points": 239, "clean_sheets": 0,
     "clearances_blocks_interceptions": 48, "tackles": 15, "recoveries": 41,
     "now_cost": 155, "expected_goals": 25.50, "expected_assists": 2.67},
    {"code": 555555, "id": 1, "web_name": "SomeoneElse", "element_type": 3,
     "minutes": 1200, "starts": 9, "total_points": 40, "clean_sheets": 3,
     "clearances_blocks_interceptions": 30, "tackles": 20, "recoveries": 60,
     "now_cost": 50, "expected_goals": 1.10, "expected_assists": 2.20},
    # Free-transfer filler with a nonsense 0.0m price -- exercises the
    # points-per-million divide-by-zero guard.
    {"code": 777777, "id": 99, "web_name": "Priceless", "element_type": 2,
     "minutes": 90, "starts": 1, "total_points": 2, "clean_sheets": 0,
     "clearances_blocks_interceptions": 4, "tackles": 2, "recoveries": 3,
     "now_cost": 0, "expected_goals": 0.0, "expected_assists": 0.0},
]


def _write_sample_prior_season_csv(directory: str) -> str:
    path = os.path.join(directory, "players_raw.csv")
    pd.DataFrame(SAMPLE_PRIOR_SEASON_ROWS).to_csv(path, index=False)
    return path


def test_prior_season_reference_shape():
    with tempfile.TemporaryDirectory() as tmp:
        prior = fpl_client.load_prior_season_reference(_write_sample_prior_season_csv(tmp))

    assert list(prior.columns) == fpl_client.PRIOR_SEASON_COLUMNS, list(prior.columns)
    assert len(prior) == 3

    haaland = prior[prior["code"] == 900001].iloc[0]
    assert haaland["minutes_prev_season"] == 2953
    assert haaland["starts_prev_season"] == 34
    assert haaland["total_points_prev_season"] == 239
    # now_cost in a season-end archive is the END-of-season price, in tenths.
    assert abs(haaland["price_prev_season_m"] - 15.5) < 1e-9
    assert abs(haaland["points_per_million_prev_season"] - 239 / 15.5) < 1e-9
    assert haaland["clearances_blocks_interceptions_prev_season"] == 48
    assert abs(haaland["expected_goals_prev_season"] - 25.50) < 1e-9

    # Divide-by-zero guard: unknown value, not inf.
    priceless = prior[prior["code"] == 777777].iloc[0]
    assert pd.isna(priceless["points_per_million_prev_season"])

    # An archive with no `code` column is a hard error, not a silent fallback
    # to joining on `id`.
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "no_code.csv")
        pd.DataFrame([{"id": 1, "minutes": 10}]).to_csv(bad, index=False)
        try:
            fpl_client.load_prior_season_reference(bad)
            raise AssertionError("expected a ValueError for a CSV with no `code` column")
        except ValueError as exc:
            assert "code" in str(exc)

    print("  prior-season reference shape ok")


def test_prior_season_join():
    players = fpl_client.load_players_df(SAMPLE_BOOTSTRAP)
    assert "code" in players.columns, "`code` must survive load_players_df()"

    with tempfile.TemporaryDirectory() as tmp:
        prior = fpl_client.load_prior_season_reference(_write_sample_prior_season_csv(tmp))
    joined = fpl_client.attach_prior_season_stats(players, prior)

    # The join must not duplicate or drop players.
    assert len(joined) == len(players)

    haaland = joined[joined["web_name"] == "Haaland"].iloc[0]
    assert haaland["minutes_prev_season"] == 2953
    assert haaland["starts_prev_season"] == 34
    # The id=1 decoy row (code 555555, 1200 minutes) must NOT have been picked
    # up. This is the whole reason the join key is `code`.
    assert haaland["minutes_prev_season"] != 1200

    # Wright has no row in the archive -- he did not appear in the PL last
    # season. Every prior-season field must be NaN, NOT zero. "No history" and
    # "played and recorded zero" are different signals, and the model relies on
    # being able to tell them apart (see NEVER_APPEARED_MATCHES_THRESHOLD in
    # models/xp_model.py).
    wright = joined[joined["web_name"] == "Wright"].iloc[0]
    for col in fpl_client.PRIOR_SEASON_COLUMNS:
        if col == "code":
            continue
        assert pd.isna(wright[col]), f"{col} should be NaN for an unmatched player, got {wright[col]!r}"

    # No prior data at all still yields the full schema, all-NaN, so downstream
    # code doesn't have to branch on which columns exist.
    empty = fpl_client.attach_prior_season_stats(players, None)
    assert set(fpl_client.PRIOR_SEASON_COLUMNS) <= set(empty.columns)
    assert empty["minutes_prev_season"].isna().all()

    # A frame with no `code` column can't be joined safely, and must say so.
    try:
        fpl_client.attach_prior_season_stats(players.drop(columns=["code"]), prior)
        raise AssertionError("expected a ValueError when `code` is missing")
    except ValueError as exc:
        assert "code" in str(exc)

    print("  prior-season join ok (1 matched on code, 1 correctly left NaN)")


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

    test_prior_season_reference_shape()
    test_prior_season_join()

    print("All parsing tests passed.")


if __name__ == "__main__":
    run_tests()
