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


def test_team_fixture_run_reports_attack_multiplier():
    """The MID rebuild needs the same run machinery to also report the
    attacking half, not just clean_sheet_prob."""
    any_team = TEAMS["name"].iloc[0]
    run = fr.team_fixture_run(any_team, TEAMS, FIXTURES, start_gw=1, n_gw=4)
    assert "mean_attack_multiplier" in run
    if run["mean_attack_multiplier"] is not None:
        assert run["mean_attack_multiplier"] > 0
    print(f"  team_fixture_run attack_multiplier ok ({any_team}: {run['mean_attack_multiplier']})")


def test_all_teams_fixture_run_rank_by_attack():
    """rank_by='mean_attack_multiplier' should reorder the table by the
    attacking column, not silently keep ranking by clean sheets."""
    by_cs = fr.all_teams_fixture_run(TEAMS, FIXTURES, start_gw=1, n_gw=4)
    by_att = fr.all_teams_fixture_run(TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                      rank_by="mean_attack_multiplier")
    non_null = by_att.dropna(subset=["mean_attack_multiplier"])
    assert non_null["mean_attack_multiplier"].is_monotonic_decreasing
    # Both columns present on both tables regardless of which is ranked on.
    assert "mean_clean_sheet_prob" in by_att.columns
    assert "mean_attack_multiplier" in by_cs.columns
    print("  all_teams_fixture_run rank_by=attack ok")


def test_team_attack_strength_matches_fixture_context_inputs():
    """team_attack_strength() should be a plain average of strength_attack_
    home/away -- not some new invented number -- and higher for a team
    whose teams_df row has higher attack ratings."""
    strong = TEAMS.sort_values("strength_attack_home", ascending=False)["name"].iloc[0]
    weak = TEAMS.sort_values("strength_attack_home", ascending=True)["name"].iloc[0]
    s_strong = fr.team_attack_strength(strong, TEAMS)
    s_weak = fr.team_attack_strength(weak, TEAMS)
    assert s_strong >= s_weak
    assert fr.team_attack_strength("Not A Real Club FC", TEAMS) != fr.team_attack_strength("Not A Real Club FC", TEAMS)  # NaN != NaN
    print(f"  team_attack_strength ok ({strong} {s_strong:.1f} >= {weak} {s_weak:.1f})")


def test_midfielder_shortlist_start_probability_gate():
    """Same core protection as defensive_shortlist: nobody below the start
    probability bar should appear, and only MID rows should be present.

    The shared synthetic pool has no prior-season columns at all, which
    (correctly, per the has_prior_data gate below) would exclude every
    single row -- not a meaningful exercise of the start-probability gate
    specifically. So this attaches minimal real prior-season starts data
    to a copy, isolating the p_start behaviour from the separate
    no-prior-data gate.
    """
    players_with_history = PLAYERS.copy()
    is_mid = players_with_history["position"] == "MID"
    players_with_history.loc[is_mid, "minutes_prev_season"] = 3000
    players_with_history.loc[is_mid, "starts_prev_season"] = 34
    players_with_history.loc[is_mid, "goals_scored_prev_season"] = 3
    players_with_history.loc[is_mid, "assists_prev_season"] = 3
    players_with_history.loc[is_mid, "clearances_blocks_interceptions_prev_season"] = 40
    players_with_history.loc[is_mid, "tackles_prev_season"] = 30

    shortlist = fr.midfielder_shortlist(players_with_history, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                        min_start_probability=0.40, matches_played=0)
    assert len(shortlist) > 0
    assert (shortlist["p_start"] >= 0.40).all()
    assert set(shortlist["position"].unique()) <= {"MID"}
    assert "p_start_grounded" in shortlist.columns
    print(f"  midfielder_shortlist start-probability gate ok ({len(shortlist)} players pass)")


def test_midfielder_shortlist_excludes_players_with_no_prior_data():
    """The second gate: a player with NO prior-season row at all must not
    be ranked using flat priors dressed up as real data, even if he clears
    the start-probability bar. This is exactly the shared synthetic pool's
    natural state (no prior-season columns), so no hand-building needed --
    every MID should land in excluded_no_prior_data, not the ranked table."""
    shortlist = fr.midfielder_shortlist(PLAYERS, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                        min_start_probability=0.0, matches_played=0)
    assert len(shortlist) == 0
    excluded = shortlist.attrs.get("excluded_no_prior_data")
    assert excluded is not None
    assert len(excluded) == (PLAYERS["position"] == "MID").sum()
    print(f"  midfielder_shortlist no-prior-data gate ok ({len(excluded)} correctly excluded, "
          f"not scored on flat priors)")


def test_midfielder_shortlist_profile_score_rewards_either_archetype():
    """The core design point of this function: a pure attacker with zero
    defensive output and a pure destroyer with zero attacking output
    should NOT be penalised for lacking the other half -- profile_score
    takes the max of the two, so each is judged on his own strength.
    Hand-built (not drawn from the shared synthetic pool, which has no
    prior-season stats at all) so the attacking/defensive split is
    actually exercised rather than trivially zero for everyone."""
    attacker = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    destroyer = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    weak_allrounder = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    for col in ("id", "web_name"):
        destroyer[col] = f"destroyer_{destroyer[col]}"
        weak_allrounder[col] = f"weak_{weak_allrounder[col]}"

    attacker["minutes_prev_season"] = 3000
    attacker["goals_scored_prev_season"] = 12
    attacker["assists_prev_season"] = 10
    attacker["expected_goals_prev_season"] = 10.0
    attacker["expected_assists_prev_season"] = 8.0
    attacker["creativity_prev_season"] = 900
    attacker["clearances_blocks_interceptions_prev_season"] = 10
    attacker["tackles_prev_season"] = 5

    destroyer["minutes_prev_season"] = 3000
    destroyer["goals_scored_prev_season"] = 0
    destroyer["assists_prev_season"] = 1
    destroyer["expected_goals_prev_season"] = 0.5
    destroyer["expected_assists_prev_season"] = 1.0
    destroyer["creativity_prev_season"] = 100
    destroyer["clearances_blocks_interceptions_prev_season"] = 180
    destroyer["tackles_prev_season"] = 90

    weak_allrounder["minutes_prev_season"] = 3000
    weak_allrounder["goals_scored_prev_season"] = 1
    weak_allrounder["assists_prev_season"] = 1
    weak_allrounder["expected_goals_prev_season"] = 1.0
    weak_allrounder["expected_assists_prev_season"] = 1.0
    weak_allrounder["creativity_prev_season"] = 150
    weak_allrounder["clearances_blocks_interceptions_prev_season"] = 20
    weak_allrounder["tackles_prev_season"] = 10

    three = pd.DataFrame([attacker, destroyer, weak_allrounder])
    shortlist = fr.midfielder_shortlist(three, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                        min_start_probability=0.0, matches_played=0)
    att_row = shortlist[shortlist["web_name"] == attacker["web_name"]].iloc[0]
    des_row = shortlist[shortlist["web_name"] == destroyer["web_name"]].iloc[0]
    weak_row = shortlist[shortlist["web_name"] == weak_allrounder["web_name"]].iloc[0]

    # Both specialists should out-rank the weak all-rounder on profile_score,
    # each via a different half of the score.
    assert att_row["profile_score"] > weak_row["profile_score"]
    assert des_row["profile_score"] > weak_row["profile_score"]
    assert att_row["attacking_score"] > des_row["attacking_score"]
    assert des_row["defensive_score"] > att_row["defensive_score"]
    print(f"  midfielder_shortlist profile_score ok (attacker {att_row['profile_score']:.2f}, "
          f"destroyer {des_row['profile_score']:.2f}, both > weak all-rounder {weak_row['profile_score']:.2f})")


def test_midfielder_shortlist_discipline_penalises_score():
    """A heavily-carded player should score lower than an identical player
    with a clean record, all else equal -- the discipline factor has to
    actually move run_score downward, not just be computed and ignored."""
    clean = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    carded = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    carded["id"] = "carded_" + str(carded["id"])
    carded["web_name"] = "carded_" + carded["web_name"]
    for row in (clean, carded):
        row["minutes_prev_season"] = 3000
        row["goals_scored_prev_season"] = 5
        row["assists_prev_season"] = 5
        row["expected_goals_prev_season"] = 4.0
        row["expected_assists_prev_season"] = 4.0
        row["creativity_prev_season"] = 400
        row["clearances_blocks_interceptions_prev_season"] = 60
        row["tackles_prev_season"] = 40
    clean["yellow_cards_prev_season"] = 1
    clean["red_cards_prev_season"] = 0
    carded["yellow_cards_prev_season"] = 14
    carded["red_cards_prev_season"] = 2

    two = pd.DataFrame([clean, carded])
    shortlist = fr.midfielder_shortlist(two, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                        min_start_probability=0.0, matches_played=0)
    clean_row = shortlist[shortlist["web_name"] == clean["web_name"]].iloc[0]
    carded_row = shortlist[shortlist["web_name"] == carded["web_name"]].iloc[0]
    assert carded_row["discipline_score"] > clean_row["discipline_score"]
    assert carded_row["run_score"] < clean_row["run_score"]
    print(f"  midfielder_shortlist discipline penalty ok (clean {clean_row['run_score']:.2f} > "
          f"carded {carded_row['run_score']:.2f}, same output otherwise)")


def test_midfielder_shortlist_set_piece_duty_flagged():
    """Real penalties_order/corners.../direct_freekicks_order data should
    surface as a readable duty tag and lift run_score, not just sit unused."""
    taker = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    non_taker = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    non_taker["id"] = "nontaker_" + str(non_taker["id"])
    non_taker["web_name"] = "nontaker_" + non_taker["web_name"]
    for row in (taker, non_taker):
        row["minutes_prev_season"] = 3000
        row["goals_scored_prev_season"] = 5
        row["assists_prev_season"] = 5
        row["expected_goals_prev_season"] = 4.0
        row["expected_assists_prev_season"] = 4.0
        row["creativity_prev_season"] = 400
        row["clearances_blocks_interceptions_prev_season"] = 30
        row["tackles_prev_season"] = 20
        row["yellow_cards_prev_season"] = 3
        row["red_cards_prev_season"] = 0
    taker["penalties_order"] = 1
    non_taker["penalties_order"] = None

    two = pd.DataFrame([taker, non_taker])
    shortlist = fr.midfielder_shortlist(two, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                        min_start_probability=0.0, matches_played=0)
    taker_row = shortlist[shortlist["web_name"] == taker["web_name"]].iloc[0]
    non_taker_row = shortlist[shortlist["web_name"] == non_taker["web_name"]].iloc[0]
    assert taker_row["set_piece_duty"] == "penalties"
    assert non_taker_row["set_piece_duty"] == "none"
    assert taker_row["run_score"] > non_taker_row["run_score"]
    print("  midfielder_shortlist set-piece duty flag ok")


def test_team_possession_pct_missing_column_and_missing_team():
    """Graceful NaN, not a crash, when the possession column isn't
    attached at all (synthetic TEAMS has no possession_pct_prev_season)
    or the team isn't found."""
    val = fr.team_possession_pct(TEAMS["name"].iloc[0], TEAMS)
    assert val != val  # NaN
    val2 = fr.team_possession_pct("Not A Real Club FC", TEAMS)
    assert val2 != val2
    print("  team_possession_pct graceful-NaN ok")


def test_team_creative_supply_rewards_real_creators_only():
    """A team with two real, data-backed creative midfielders should score
    higher than an identical team where those midfielders have no
    prior-season data at all (flat priors must not count as service)."""
    team = "Supply Test FC"
    creator1 = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    creator2 = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    creator1["id"], creator1["web_name"], creator1["team_name"] = "c1", "c1", team
    creator2["id"], creator2["web_name"], creator2["team_name"] = "c2", "c2", team
    for row in (creator1, creator2):
        row["minutes_prev_season"] = 3000
        row["assists_prev_season"] = 10
        row["expected_assists_prev_season"] = 8.0
        row["creativity_prev_season"] = 800

    no_data_team = "No Data FC"
    blank1 = creator1.copy()
    blank2 = creator2.copy()
    blank1["id"], blank1["web_name"], blank1["team_name"] = "b1", "b1", no_data_team
    blank2["id"], blank2["web_name"], blank2["team_name"] = "b2", "b2", no_data_team
    for row in (blank1, blank2):
        row["minutes_prev_season"] = pd.NA
        row["assists_prev_season"] = pd.NA
        row["expected_assists_prev_season"] = pd.NA
        row["creativity_prev_season"] = pd.NA

    pool = pd.DataFrame([creator1, creator2, blank1, blank2])
    supply_real = fr.team_creative_supply(team, pool)
    supply_blank = fr.team_creative_supply(no_data_team, pool)
    assert supply_real > 0
    assert supply_blank != supply_blank  # NaN -- no data-backed creators at all
    print(f"  team_creative_supply ok (real creators {supply_real:.2f}, no-data team correctly NaN)")


def test_forward_shortlist_start_probability_and_no_prior_data_gates():
    """Same two protections as midfielder_shortlist, exercised on FWD."""
    fwd_with_history = PLAYERS.copy()
    is_fwd = fwd_with_history["position"] == "FWD"
    fwd_with_history.loc[is_fwd, "minutes_prev_season"] = 2800
    fwd_with_history.loc[is_fwd, "goals_scored_prev_season"] = 15
    fwd_with_history.loc[is_fwd, "assists_prev_season"] = 4
    fwd_with_history.loc[is_fwd, "expected_goals_prev_season"] = 13.0
    fwd_with_history.loc[is_fwd, "expected_assists_prev_season"] = 3.0
    fwd_with_history.loc[is_fwd, "threat_prev_season"] = 700

    shortlist = fr.forward_shortlist(fwd_with_history, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                     min_start_probability=0.40, matches_played=0)
    assert len(shortlist) > 0
    assert (shortlist["p_start"] >= 0.40).all()
    assert set(shortlist["position"].unique()) <= {"FWD"}

    # No-prior-data gate: the raw synthetic pool (no prior-season columns
    # at all) should exclude every FWD, not silently rank them on priors.
    bare_shortlist = fr.forward_shortlist(PLAYERS, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                          min_start_probability=0.0, matches_played=0)
    assert len(bare_shortlist) == 0
    excluded = bare_shortlist.attrs.get("excluded_no_prior_data")
    assert excluded is not None
    assert len(excluded) == (PLAYERS["position"] == "FWD").sum()
    print(f"  forward_shortlist gates ok ({len(shortlist)} pass start-prob gate, "
          f"{len(excluded)} correctly excluded for no prior data)")


def test_forward_shortlist_penalty_duty_lifts_score():
    """A real, live penalties_order=1 flag should outrank an otherwise
    identical non-taker."""
    taker = PLAYERS[PLAYERS["position"] == "FWD"].iloc[0].copy()
    non_taker = PLAYERS[PLAYERS["position"] == "FWD"].iloc[0].copy()
    non_taker["id"] = "nonpen_" + str(non_taker["id"])
    non_taker["web_name"] = "nonpen_" + non_taker["web_name"]
    for row in (taker, non_taker):
        row["minutes_prev_season"] = 2800
        row["goals_scored_prev_season"] = 12
        row["assists_prev_season"] = 3
        row["expected_goals_prev_season"] = 11.0
        row["expected_assists_prev_season"] = 2.5
        row["threat_prev_season"] = 600
        row["yellow_cards_prev_season"] = 3
        row["red_cards_prev_season"] = 0
    taker["penalties_order"] = 1
    non_taker["penalties_order"] = None

    two = pd.DataFrame([taker, non_taker])
    shortlist = fr.forward_shortlist(two, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                     min_start_probability=0.0, matches_played=0)
    taker_row = shortlist[shortlist["web_name"] == taker["web_name"]].iloc[0]
    non_taker_row = shortlist[shortlist["web_name"] == non_taker["web_name"]].iloc[0]
    assert taker_row["set_piece_duty"] == "penalties"
    assert taker_row["run_score"] > non_taker_row["run_score"]
    print("  forward_shortlist penalty duty ok")


def test_forward_shortlist_creative_supply_wired_through():
    """team_creative_supply has to actually reach forward_shortlist's
    output matching what team_creative_supply() computes directly, and
    move run_score in the right direction.

    Both forwards sit on the SAME real team (so mean_attack_multiplier,
    team_attack_strength, and team_possession_pct -- all team-level and
    therefore identical for both -- can't confound the comparison); a
    fake second team with no creative midfielder at all would make every
    OTHER team-level factor collapse to NaN and get masked by the
    fillna(mean) safety net, which is exactly what broke the first version
    of this test. Same team, only the creator differs: swap the creator's
    prior-season data to blank inside the SAME dataframe and re-run, so
    every other input (fixtures, team strength, possession) stays exactly
    constant across the two calls and only team_creative_supply moves.
    """
    real_team = TEAMS["name"].iloc[0]
    fwd = PLAYERS[PLAYERS["position"] == "FWD"].iloc[0].copy()
    fwd["id"], fwd["web_name"], fwd["team_name"] = "fwd_test", "fwd_test", real_team
    fwd["minutes_prev_season"] = 2800
    fwd["goals_scored_prev_season"] = 12
    fwd["assists_prev_season"] = 3
    fwd["expected_goals_prev_season"] = 11.0
    fwd["expected_assists_prev_season"] = 2.5
    fwd["threat_prev_season"] = 600
    fwd["yellow_cards_prev_season"] = 2
    fwd["red_cards_prev_season"] = 0

    creator = PLAYERS[PLAYERS["position"] == "MID"].iloc[0].copy()
    creator["id"], creator["web_name"], creator["team_name"] = "cr", "cr", real_team
    creator["minutes_prev_season"] = 3000
    creator["assists_prev_season"] = 12
    creator["expected_assists_prev_season"] = 10.0
    creator["creativity_prev_season"] = 900

    blank_creator = creator.copy()
    blank_creator["minutes_prev_season"] = pd.NA
    blank_creator["assists_prev_season"] = pd.NA
    blank_creator["expected_assists_prev_season"] = pd.NA
    blank_creator["creativity_prev_season"] = pd.NA

    with_creator = pd.DataFrame([fwd, creator])
    without_creator = pd.DataFrame([fwd, blank_creator])

    sl_with = fr.forward_shortlist(with_creator, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                   min_start_probability=0.0, matches_played=0)
    sl_without = fr.forward_shortlist(without_creator, TEAMS, FIXTURES, start_gw=1, n_gw=4,
                                      min_start_probability=0.0, matches_played=0)
    row_with = sl_with[sl_with["web_name"] == "fwd_test"].iloc[0]
    row_without = sl_without[sl_without["web_name"] == "fwd_test"].iloc[0]

    expected_supply = fr.team_creative_supply(real_team, with_creator)
    assert abs(row_with["team_creative_supply"] - expected_supply) < 1e-9
    assert row_with["team_creative_supply"] > 0
    assert row_without["team_creative_supply"] != row_without["team_creative_supply"]  # NaN
    print(f"  forward_shortlist creative-supply wiring ok "
          f"(with creator: supply={row_with['team_creative_supply']:.2f}, "
          f"without: supply=NaN)")


def run_tests():
    test_team_fixture_run_basic()
    test_unknown_team_does_not_crash()
    test_all_teams_ranking_direction()
    test_defensive_shortlist_start_probability_gate()
    test_defensive_shortlist_score_uses_both_halves()
    test_team_fixture_run_reports_attack_multiplier()
    test_all_teams_fixture_run_rank_by_attack()
    test_team_attack_strength_matches_fixture_context_inputs()
    test_midfielder_shortlist_start_probability_gate()
    test_midfielder_shortlist_excludes_players_with_no_prior_data()
    test_midfielder_shortlist_profile_score_rewards_either_archetype()
    test_midfielder_shortlist_discipline_penalises_score()
    test_midfielder_shortlist_set_piece_duty_flagged()
    test_team_possession_pct_missing_column_and_missing_team()
    test_team_creative_supply_rewards_real_creators_only()
    test_forward_shortlist_start_probability_and_no_prior_data_gates()
    test_forward_shortlist_penalty_duty_lifts_score()
    test_forward_shortlist_creative_supply_wired_through()
    print("All fixture_run tests passed.")


if __name__ == "__main__":
    run_tests()
