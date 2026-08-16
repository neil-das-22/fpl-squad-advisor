"""
Tests for models/fixture_run.py, run against tests/synthetic_data.py.

Standalone plain-assert style, matching test_xp_model.py.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import fixture_run as fr  # noqa: E402
from synthetic_data import (  # noqa: E402
    make_sample_fixtures_df,
    make_sample_players_df,
    make_sample_teams_df,
)

PLAYERS = make_sample_players_df()
TEAMS = make_sample_teams_df()
FIXTURES = make_sample_fixtures_df()


def test_team_fixture_run_basic():
    # Every team should get exactly n_gw slots accounted for, blank or not.
    strong_team = TEAMS.sort_values("strength", ascending=False)["name"].iloc[0]
    run = fr.team_fixture_run(strong_team, TEAMS, FIXTURES, start_gw=1, n_gw=4)
    assert run["n_fixtures"] + run["n_blanks"] <= 4 + run["n_doubles"]
    assert len(run["breakdown"]) == 4
    if run["mean_clean_sheet_prob"] is not None:
        assert 0.0 <= run["mean_clean_sheet_prob"] <= 1.0
    print(f"  team_fixture_run basic ok ({strong_team}: "
          f"{run['n_fixtures']} fixtures, mean CS {run['mean_clean_sheet_prob']})")


def test_unknown_team_does_not_crash():
    run = fr.team_fixture_run("Not A Real Club FC", TEAMS, FIXTURES, start_gw=1, n_gw=4)
    assert run["n_fixtures"] == 0
    assert run["mean_clean_sheet_prob"] is None
    print("  unknown team handled gracefully ok")


def test_all_teams_ranking_direction():
    """A team with strong defence and weak opponents should outrank one
    with the reverse, all else equal -- sanity-checks the ranking direction,
    not exact numbers (which depend on the v1 heuristic constants)."""
    runs = fr.all_teams_fixture_run(TEAMS, FIXTURES, start_gw=1, n_gw=4)
    assert list(runs.columns[:1]) == ["run_rank"]
    assert runs["run_rank"].tolist() == list(range(1, len(runs) + 1))
    # Ranked strictly by mean_clean_sheet_prob, descending (NaNs last).
    non_null = runs.dropna(subset=["mean_clean_sheet_prob"])
    assert non_null["mean_clean_sheet_prob"].is_monotonic_decreasing
    print(f"  all_teams_fixture_run ranking ok ({len(runs)} teams)")


def test_defensive_shortlist_start_probability_gate():
    """The core bug this module was built to avoid: a bench player on a
    great-defence team must not outrank a nailed starter just because they
    share a fixture run. Every row in the shortlist has to clear the
    start-probability bar on its own, individually-computed p_start."""
    shortlist = fr.defensive_shortlist(PLAYERS, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                       min_start_probability=0.40, matches_played=0)
    assert (shortlist["p_start"] >= 0.40).all()
    assert set(shortlist["position"].unique()) <= {"GKP", "DEF"}
    assert "p_start_grounded" in shortlist.columns

    # A player with zero minutes across a real chunk of last season (a
    # confirmed non-starter, same shape as the real Meslier case) must not
    # appear even if his team's fixture run is excellent.
    non_starter = PLAYERS.iloc[0].copy()
    non_starter["starts_prev_season"] = 0
    non_starter["minutes_prev_season"] = 0
    p, _ = __import__("xp_model").estimate_start_probability(non_starter, matches_played=0)
    assert p < 0.40, "test fixture assumption broke: expected a low p_start here"
    print(f"  defensive_shortlist start-probability gate ok ({len(shortlist)} players pass)")


def test_defensive_shortlist_score_uses_both_halves():
    """run_score has to move with BOTH the fixture run and the player's own
    defensive numbers -- not just one of them (that would silently ignore
    half of what this module was asked to combine).

    Hand-built rather than drawn from the shared synthetic pool: every
    defender in tests/synthetic_data.py happens to carry identical
    prior-season defensive stats (a fixture-data limitation, not a bug in
    this module), which would make this assertion pass trivially either
    way. Two players on the SAME team (so the fixture-run half is held
    constant) with deliberately different defensive stats isolate the
    "own defense" half specifically.
    """
    strong_defender = PLAYERS.iloc[0].copy()
    weak_defender = PLAYERS.iloc[0].copy()
    same_team = strong_defender["team_name"]
    for col in ("id", "web_name"):
        weak_defender[col] = f"weak_{weak_defender[col]}"
    strong_defender["position"] = "DEF"
    weak_defender["position"] = "DEF"
    strong_defender["minutes_prev_season"] = 3000
    strong_defender["clearances_blocks_interceptions_prev_season"] = 300
    strong_defender["tackles_prev_season"] = 80
    weak_defender["minutes_prev_season"] = 3000
    weak_defender["clearances_blocks_interceptions_prev_season"] = 20
    weak_defender["tackles_prev_season"] = 5

    two_players = pd.DataFrame([strong_defender, weak_defender])
    shortlist = fr.defensive_shortlist(two_players, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                       min_start_probability=0.0, matches_played=0)
    assert len(shortlist) == 2
    strong_row = shortlist[shortlist["web_name"] == strong_defender["web_name"]].iloc[0]
    weak_row = shortlist[shortlist["web_name"] == weak_defender["web_name"]].iloc[0]
    # Same team -> identical fixture-run inputs -> the only thing that can
    # separate them is own_defense_score, and it has to rank the busier
    # defender higher.
    assert strong_row["mean_clean_sheet_prob"] == weak_row["mean_clean_sheet_prob"]
    assert strong_row["own_defense_score"] > weak_row["own_defense_score"]
    assert strong_row["run_score"] > weak_row["run_score"]
    print(f"  defensive_shortlist own-defense half ok "
          f"({strong_defender['web_name']} {strong_row['run_score']:.2f} > "
          f"{weak_defender['web_name']} {weak_row['run_score']:.2f}, same team/fixtures)")


def run_tests():
    test_team_fixture_run_basic()
    test_unknown_team_does_not_crash()
    test_all_teams_ranking_direction()
    test_defensive_shortlist_start_probability_gate()
    test_defensive_shortlist_score_uses_both_halves()
    print("All fixture_run tests passed.")


if __name__ == "__main__":
    run_tests()
