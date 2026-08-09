"""
Tests for models/xp_model.py, run against tests/synthetic_data.py.

Standalone plain-assert style, matching data/test_fpl_client.py, so it runs
with either:

    python3 models/test_xp_model.py
    python3 -m pytest models/test_xp_model.py

These are behavioural tests, not golden-number tests: they assert the model
obeys the FPL scoring rules and reacts in the right direction to its inputs.
Deliberately so -- the v1 heuristic constants are expected to move once
backtesting (build step 11) tunes them, and tests pinned to exact xP values
would just have to be rewritten every time.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import xp_model as m  # noqa: E402
from synthetic_data import (  # noqa: E402
    make_sample_fixtures_df,
    make_sample_overrides_df,
    make_sample_players_df,
    make_sample_teams_df,
)

PLAYERS = make_sample_players_df()
TEAMS = make_sample_teams_df()
FIXTURES = make_sample_fixtures_df()


def _player(web_name):
    return PLAYERS[PLAYERS["web_name"] == web_name].iloc[0]


def _team(name):
    return TEAMS[TEAMS["name"] == name].iloc[0]


def _fixture(fid):
    return FIXTURES[FIXTURES["id"] == fid].iloc[0]


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def test_expected_floor_divide():
    # E[floor(K/2)] for K ~ Poisson(0) is 0.
    assert m.expected_floor_divide(0.0, 2) == 0.0
    # Monotonic in lambda, and strictly between the naive bounds.
    low = m.expected_floor_divide(1.0, 2)
    high = m.expected_floor_divide(3.0, 2)
    assert 0 < low < high
    # For a large lambda, E[floor(K/2)] approaches lambda/2 from below.
    lam = 20.0
    assert lam / 2 - 1.0 < m.expected_floor_divide(lam, 2, k_max=80) < lam / 2
    print("  expected_floor_divide ok")


def test_prob_at_least():
    assert m.prob_at_least(10, 0.0) == 0.0
    assert m.prob_at_least(0, 5.0) == 1.0
    p_poisson = m.prob_at_least(10, 6.0, overdispersion=1.0)
    p_nb = m.prob_at_least(10, 6.0, overdispersion=1.6)
    assert 0.0 < p_poisson < 1.0
    # Over-dispersion fattens the tail, so threshold-hitting must get MORE likely.
    assert p_nb > p_poisson
    print(f"  prob_at_least ok (poisson {p_poisson:.3f} < negbin {p_nb:.3f})")


def test_shrinkage():
    prior = 0.30
    # No sample at all -> exactly the prior.
    assert abs(m.shrunk_per90_rate(0.0, 0.0, prior) - prior) < 1e-12
    # A huge sample washes the prior out.
    big = m.shrunk_per90_rate(90.0, 90 * 90.0, prior)
    assert abs(big - 1.0) < 0.10
    # A tiny hot streak is pulled hard back toward the prior.
    hot = m.shrunk_per90_rate(3.0, 90.0, prior)   # 3 xG in one match
    assert prior < hot < 1.0
    print(f"  shrinkage ok (3 xG in 90 mins -> {hot:.3f}/90, not 3.0/90)")


# ---------------------------------------------------------------------------
# Minutes / availability
# ---------------------------------------------------------------------------

def test_start_probability_and_minutes():
    # Nailed-on starter: 32 starts from 38 matches.
    p, flags = m.estimate_start_probability(_player("Ahlberg"), matches_played=38)
    assert 0.7 < p < 0.99, p
    assert "start_prob_default" not in flags

    # Doubtful player (status 'd', 50% chance) gets halved.
    p_doubt, flags_doubt = m.estimate_start_probability(_player("Novak"), matches_played=38)
    assert any(f.startswith("availability") for f in flags_doubt)
    assert p_doubt < 0.5

    # Genuine no-data case (preseason, nobody has a track record) -> flat
    # prior, loudly flagged.
    p_preseason, flags_preseason = m.estimate_start_probability(_player("Nwosu"), matches_played=0)
    assert "start_prob_default" in flags_preseason
    assert abs(p_preseason - m.DEFAULT_START_PROBABILITY) < 1e-9

    # Too early in the season to tell "hasn't played yet" from "won't play"
    # apart -> still the flat prior.
    p_early, flags_early = m.estimate_start_probability(_player("Nwosu"), matches_played=1)
    assert "start_prob_default" in flags_early
    assert abs(p_early - m.DEFAULT_START_PROBABILITY) < 1e-9

    # Zero minutes across a real chunk of the season -> that's evidence, not
    # a missing value. Regression test for the 2025/26 backtest's single
    # largest error source (backtest/results_2025_26.md section 1b): this
    # used to return the same 65% flat prior as a genuine GW1 unknown.
    p_new, flags_new = m.estimate_start_probability(_player("Nwosu"), matches_played=38)
    assert "start_prob_never_appeared" in flags_new
    assert "start_prob_default" not in flags_new
    assert abs(p_new - m.NEVER_APPEARED_START_PROBABILITY) < 1e-9
    assert p_new < 0.1, "a confirmed non-player should not get a meaningful start chance"

    dist = m.minutes_distribution(0.9)
    assert dist["p_60"] <= dist["p_appear"] <= 1.0
    assert 0 < dist["expected_minutes"] <= 90
    # An unavailable player cannot appear off the bench either.
    dead = m.minutes_distribution(0.0, availability=0.0)
    assert dead["p_appear"] == 0.0 and dead["expected_minutes"] == 0.0
    print(f"  minutes ok (starter p={p:.2f}, doubtful p={p_doubt:.2f})")


# ---------------------------------------------------------------------------
# Fixture model
# ---------------------------------------------------------------------------

def test_fixture_context_directionality():
    mci, cov = _team("Man City"), _team("Coventry")

    strong_home = m.fixture_context(mci, cov, is_home=True, difficulty=2)
    weak_away = m.fixture_context(cov, mci, is_home=False, difficulty=5)

    # A strong side at home against a promoted side should be much likelier to
    # keep a clean sheet, and expect to concede far fewer goals.
    assert strong_home["clean_sheet_prob"] > weak_away["clean_sheet_prob"] * 3
    assert strong_home["lambda_conceded"] < weak_away["lambda_conceded"]
    assert strong_home["attack_multiplier"] > weak_away["attack_multiplier"]

    # Everything stays inside the documented clip ranges.
    for ctx in (strong_home, weak_away):
        lo, hi = m.CLEAN_SHEET_PROB_BOUNDS
        assert lo <= ctx["clean_sheet_prob"] <= hi
        lo, hi = m.LAMBDA_CONCEDED_BOUNDS
        assert lo <= ctx["lambda_conceded"] <= hi
        lo, hi = m.ATTACK_MULTIPLIER_BOUNDS
        assert lo <= ctx["attack_multiplier"] <= hi

    # Clean sheet probability is exp(-lambda), i.e. internally consistent with
    # the goals-conceded term (before clipping).
    import math
    assert abs(strong_home["clean_sheet_prob"] - math.exp(-strong_home["lambda_conceded"])) < 1e-9

    # Home advantage: same two teams, flipped venue.
    home = m.fixture_context(mci, cov, is_home=True, difficulty=3)
    away = m.fixture_context(mci, cov, is_home=False, difficulty=3)
    assert home["clean_sheet_prob"] > away["clean_sheet_prob"]
    print(f"  fixture context ok (MCI home CS {strong_home['clean_sheet_prob']:.2f} "
          f"vs COV away CS {weak_away['clean_sheet_prob']:.2f})")


# ---------------------------------------------------------------------------
# Single-fixture xP
# ---------------------------------------------------------------------------

def test_calculate_xp_components_and_rules():
    fx = _fixture(1)   # Man City (H) v Coventry (A)
    mci, cov = _team("Man City"), _team("Coventry")

    striker = m.calculate_xp(_player("Ahlberg"), fx, cov, mci)
    keeper = m.calculate_xp(_player("Ekdahl"), fx, cov, mci)
    defender = m.calculate_xp(_player("Karlsson"), fx, cov, mci)
    mid = m.calculate_xp(_player("Duarte"), fx, cov, mci)

    # Total must equal the sum of its parts -- the breakdown has to be honest.
    for res in (striker, keeper, defender, mid):
        parts = sum(res[c] for c in m.COMPONENT_COLUMNS)
        assert abs(parts - res["xp_total"]) < 1e-9

    # Position-specific rules.
    assert striker["xp_clean_sheet"] == 0.0            # FWD get no clean sheet pts
    assert striker["xp_saves"] == 0.0                  # outfielders don't save
    assert striker["xp_goals_conceded"] == 0.0         # no conceded penalty for FWD
    assert keeper["xp_saves"] > 0.0                    # GK do
    assert keeper["xp_goals_conceded"] < 0.0           # and get docked for goals
    assert defender["xp_goals_conceded"] < 0.0
    assert keeper["xp_defcon"] == 0.0                  # GK excluded from DefCon here
    assert defender["xp_defcon"] > 0.0
    assert mid["xp_clean_sheet"] > 0.0                 # MID get 1 pt, so > 0
    assert mid["xp_clean_sheet"] < defender["xp_clean_sheet"]   # ...but less than DEF

    # DefCon is capped at 2 points however good the player is.
    assert defender["xp_defcon"] <= m.DEFCON_POINTS + 1e-9
    monster = m.calculate_xp(_player("Karlsson"), fx, cov, mci, defcon_per90=40.0)
    assert monster["xp_defcon"] <= m.DEFCON_POINTS + 1e-9

    # Appearance points sit in the legal 0-2 range.
    for res in (striker, keeper, defender, mid):
        assert 0.0 <= res["xp_appearance"] <= 2.0

    # Cards are a drag, never a bonus.
    for res in (striker, keeper, defender, mid):
        assert res["xp_cards"] <= 0.0

    # A premium striker in a great fixture should land in a believable range.
    assert 4.0 < striker["xp_total"] < 12.0, striker["xp_total"]
    print(f"  components ok (striker {striker['xp_total']:.2f}, "
          f"GK {keeper['xp_total']:.2f}, DEF {defender['xp_total']:.2f})")


def test_goal_points_scale_by_position():
    """Same expected goals must be worth 6/6/5/4 points by position."""
    fx = _fixture(1)
    mci, cov = _team("Man City"), _team("Coventry")
    base = _player("Karlsson").copy()

    ratios = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        row = base.copy()
        row["position"] = pos
        res = m.calculate_xp(row, fx, cov, mci)
        if res["expected_goals"] > 0:
            ratios[pos] = res["xp_goals"] / res["expected_goals"]
    for pos, expected in m.GOAL_POINTS.items():
        if pos in ratios:
            assert abs(ratios[pos] - expected) < 1e-9, (pos, ratios[pos])
    print(f"  goal point scaling ok {  {k: round(v, 1) for k, v in ratios.items()} }")


def test_promoted_team_fallback():
    """A promoted club must be handled conservatively AND flagged, not guessed."""
    fx = _fixture(1)
    mci, cov = _team("Man City"), _team("Coventry")

    prom = m.calculate_xp(_player("Asante"), fx, mci, cov, is_home=False)
    assert "promoted_team_fallback" in prom["flags"]
    assert "no_minutes_history" in prom["flags"]
    assert prom["clean_sheet_prob"] <= m.PROMOTED_CLEAN_SHEET_CAP + 1e-9

    # Same player, same fixture, but pretend the club is established: the
    # promoted handling must be strictly more conservative on attack.
    established = _player("Asante").copy()
    established["is_promoted"] = False
    cov_not_promoted = cov.copy()
    cov_not_promoted["is_promoted"] = False
    normal = m.calculate_xp(established, fx, mci, cov_not_promoted, is_home=False)
    assert prom["expected_goals"] < normal["expected_goals"]
    assert prom["xp_total"] < normal["xp_total"]
    print(f"  promoted fallback ok ({prom['xp_total']:.2f} vs "
          f"{normal['xp_total']:.2f} if treated as established)")


def test_overrides():
    fx = _fixture(1)
    mci, cov = _team("Man City"), _team("Coventry")
    p = _player("Duarte")

    benched = m.calculate_xp(p, fx, cov, mci, start_probability=0.05)
    nailed = m.calculate_xp(p, fx, cov, mci, start_probability=0.99)
    assert benched["xp_total"] < nailed["xp_total"]
    assert "start_prob_override" in nailed["flags"]

    # Penalty duty is inert by default and only adds a miss-risk drag when set
    # (the goal itself is already inside xG, so it must not be double counted).
    no_pens = m.calculate_xp(p, fx, cov, mci)
    with_pens = m.calculate_xp(p, fx, cov, mci, penalty_share=1.0)
    assert no_pens["xp_penalties"] == 0.0
    assert with_pens["xp_penalties"] < 0.0
    print("  overrides ok")


# ---------------------------------------------------------------------------
# Gameweek batch: blanks and doubles
# ---------------------------------------------------------------------------

def test_gameweek_batch():
    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)

    # Every player is present, nobody silently dropped.
    assert len(xp) == len(PLAYERS)
    assert {"id", "web_name", "position", "price_m", "xp"} <= set(xp.columns)

    # BLANK gameweek: Brighton have no GW1 fixture -> exactly 0 xP.
    brighton = xp[xp["team_name"] == "Brighton"]
    assert len(brighton) > 0
    assert (brighton["n_fixtures"] == 0).all()
    assert (brighton["xp"] == 0.0).all()
    assert brighton["flags"].str.contains("blank_gameweek").all()

    # DOUBLE gameweek: Everton and Fulham play twice.
    for club in ("Everton", "Fulham"):
        rows = xp[xp["team_name"] == club]
        assert (rows["n_fixtures"] == 2).all(), club
        assert rows["flags"].str.contains("double_gameweek").all()
        assert (rows["opponents"].str.count(",") == 1).all()

    # Single gameweek clubs.
    assert (xp[xp["team_name"] == "Man City"]["n_fixtures"] == 1).all()

    print(f"  batch ok ({len(xp)} players; "
          f"{int((xp['n_fixtures'] == 0).sum())} blanking, "
          f"{int((xp['n_fixtures'] == 2).sum())} doubling)")


def test_double_gameweek_sums_both_fixtures():
    """A DGW total must be the exact sum of the two single-fixture totals."""
    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)
    matches_played = m.infer_matches_played(PLAYERS)

    player = _player("Okoro")             # Everton MID, plays twice in GW1
    fx_away = _fixture(2)                 # Arsenal (H) v Everton (A)
    fx_home = _fixture(5)                 # Everton (H) v Fulham (A)
    eve, ars, ful = _team("Everton"), _team("Arsenal"), _team("Fulham")

    a = m.calculate_xp(player, fx_away, ars, eve, is_home=False,
                       matches_played=matches_played)
    b = m.calculate_xp(player, fx_home, ful, eve, is_home=True,
                       matches_played=matches_played)

    batched = float(xp.loc[xp["web_name"] == "Okoro", "xp"].iloc[0])
    assert abs(batched - (a["xp_total"] + b["xp_total"])) < 1e-9

    # And a DGW player must out-score his own single-fixture value.
    assert batched > a["xp_total"]
    print(f"  double gameweek sums ok ({a['xp_total']:.2f} + {b['xp_total']:.2f} "
          f"= {batched:.2f})")


def test_blank_beats_nothing():
    """A blanking player must never be preferred to an identical playing one."""
    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)
    blanking_max = xp[xp["n_fixtures"] == 0]["xp"].max()
    assert blanking_max == 0.0
    # Every AVAILABLE player with a fixture must score above a blanking player.
    # (A player on status 'i' with a 0% chance of playing correctly scores 0 too
    #  -- he is not in the squad, so a fixture is worth nothing to him.)
    playing = xp[(xp["n_fixtures"] > 0) & (xp["status"] == "a")]
    assert playing["xp"].min() > 0.0
    assert float(xp[xp["status"] == "i"]["xp"].max()) == 0.0
    print("  blank gameweek ordering ok")


def test_gameweek_2_runs():
    xp2 = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=2)
    assert len(xp2) == len(PLAYERS)
    # Coventry blank in GW2.
    assert (xp2[xp2["team_name"] == "Coventry"]["xp"] == 0.0).all()
    # Brighton play in GW2 having blanked in GW1.
    assert (xp2[xp2["team_name"] == "Brighton"]["n_fixtures"] == 1).all()
    print("  gameweek 2 ok")


# ---------------------------------------------------------------------------
# Manual adjustment hook
# ---------------------------------------------------------------------------

def test_apply_manual_adjustments():
    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)
    overrides = make_sample_overrides_df()
    adjusted = m.apply_manual_adjustments(xp, overrides)

    def before(name):
        return float(xp.loc[xp["web_name"] == name, "xp"].iloc[0])

    def after(name):
        return float(adjusted.loc[adjusted["web_name"] == name, "xp"].iloc[0])

    # upgrade: +1.0
    assert abs(after("Ahlberg") - (before("Ahlberg") + 1.0)) < 1e-9
    # flat_override: hard set to 0
    assert after("Lanzini") == 0.0
    # multiplier: halved
    assert abs(after("Nwosu") - before("Nwosu") * 0.5) < 1e-9
    # untouched player unchanged
    assert abs(after("Duarte") - before("Duarte")) < 1e-9

    # The pre-adjustment number is preserved for auditability.
    assert abs(float(adjusted.loc[adjusted["web_name"] == "Ahlberg", "xp_model"].iloc[0])
               - before("Ahlberg")) < 1e-9
    assert adjusted.loc[adjusted["web_name"] == "Ahlberg", "adjustment_notes"].iloc[0] != ""

    # A name that matches nobody is reported, not silently swallowed.
    unmatched = adjusted.attrs["unmatched_overrides"]
    assert len(unmatched) == 1
    assert "NotARealPlayer" in str(unmatched[0]["row"])

    # Downgrades floor at zero rather than going negative.
    hard = pd.DataFrame([{"web_name": "Duarte", "adjustment_type": "downgrade",
                          "value": 999.0}])
    floored = m.apply_manual_adjustments(xp, hard)
    assert float(floored.loc[floored["web_name"] == "Duarte", "xp"].iloc[0]) == 0.0

    # No overrides at all is a clean no-op.
    noop = m.apply_manual_adjustments(xp, None)
    assert abs(noop["xp"].sum() - xp["xp"].sum()) < 1e-9
    print("  manual adjustments ok")


def test_string_typed_inputs_are_coerced():
    """The live API returns numbers as strings; a CSV round-trip does too."""
    assert isinstance(PLAYERS["expected_goals"].iloc[0], str)
    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)
    assert xp["xp"].notna().all()
    assert xp["xp"].max() > 1.0     # would be ~0 if xG had been read as zero
    print("  string coercion ok")


def run_tests():
    test_expected_floor_divide()
    test_prob_at_least()
    test_shrinkage()
    test_start_probability_and_minutes()
    test_fixture_context_directionality()
    test_calculate_xp_components_and_rules()
    test_goal_points_scale_by_position()
    test_promoted_team_fallback()
    test_overrides()
    test_gameweek_batch()
    test_double_gameweek_sums_both_fixtures()
    test_blank_beats_nothing()
    test_gameweek_2_runs()
    test_apply_manual_adjustments()
    test_string_typed_inputs_are_coerced()
    print("All xP model tests passed.")


if __name__ == "__main__":
    run_tests()
