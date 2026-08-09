"""
Tests for backtest/backtest.py.

Standalone plain-assert style, matching models/test_xp_model.py, so it runs with
either:

    python3 backtest/test_backtest.py
    python3 -m pytest backtest/test_backtest.py

Three things are being proved here, in order of importance:

  1. NO LOOKAHEAD. The features the model sees for gameweek N are provably
     independent of everything that happens in gameweek N and later. This is
     tested destructively: corrupt the future, rebuild the past, assert nothing
     moved. If this test ever fails the whole backtest is worthless, so it is
     the first thing that runs.

  2. CORRECT METRICS. Every metric is checked against a tiny example whose
     answer was worked out by hand, written out in the test so a reader can
     re-derive it without trusting the implementation.

  3. END-TO-END. The full harness runs on the synthetic season without error and
     produces a sane, non-degenerate result frame and a renderable report.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture: one synthetic season, generated once (it is not cheap).
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.mkdtemp(prefix="fpl_backtest_tests_")
bt.make_synthetic_dataset(_TMPDIR, n_gameweeks=10, seed=7)
HIST = bt.load_historical_gw_data(_TMPDIR)
HIST.source = "synthetic"


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. LOOKAHEAD — the property everything else depends on
# ---------------------------------------------------------------------------

def test_no_lookahead_leakage():
    """Destructive proof: corrupt GW >= N, rebuild GW N inputs, assert no change.

    If any feature for gameweek N were computed from a row in gameweek N or
    later, multiplying those rows by 1000 (and blowing out minutes, points,
    goals, xG, price...) would necessarily change the reconstructed state. It
    does not, for every gameweek in the season.
    """
    target_gws = [1, 2, 5, 6, 9]

    for target in target_gws:
        clean = HIST.gw
        poisoned = clean.copy()
        future = poisoned["gw"] >= target
        assert future.sum() > 0, f"no rows at or after GW{target} to poison"

        for col in ("minutes", "starts", "total_points", "goals_scored",
                    "assists", "clean_sheets", "goals_conceded", "saves",
                    "bonus", "bps", "expected_goals", "expected_assists",
                    "expected_goals_conceded", "ict_index", "value",
                    "yellow_cards", "red_cards", "own_goals",
                    # DefCon counting stats feed the data-gap experiment, so they
                    # are inside the firewall too and must be poisoned as well.
                    "defensive_contribution"):
            poisoned.loc[future, col] = poisoned.loc[future, col] * 1000.0 + 999.0
        poisoned.loc[future, "position"] = "FWD"
        poisoned.loc[future, "team"] = "NOT_A_REAL_CLUB"

        prior_clean = bt.aggregate_prior_history(clean, target)
        prior_poisoned = bt.aggregate_prior_history(poisoned, target)

        assert list(prior_clean.index) == list(prior_poisoned.index), (
            f"GW{target}: poisoning the future changed the player universe")
        pd.testing.assert_frame_equal(
            prior_clean.sort_index(), prior_poisoned.sort_index(),
            check_dtype=False,
            obj=f"prior state for GW{target} leaked future data")

    print("  aggregate_prior_history ignores GW >= N for every tested gameweek")

    # The same must hold for the fully-built model input frames, and for the
    # predictions themselves -- not just the aggregation step.
    target = 6
    poisoned = HIST.gw.copy()
    future = poisoned["gw"] >= target
    for col in ("minutes", "starts", "total_points", "goals_scored", "assists",
                "expected_goals", "expected_assists", "bonus", "value"):
        poisoned.loc[future, col] = poisoned.loc[future, col] * 1000.0 + 999.0

    hist_poisoned = bt.HistoricalData(
        gw=poisoned, teams=HIST.teams, fixtures=HIST.fixtures,
        players_raw=HIST.players_raw, source="synthetic", base_dir=None)

    frames_clean = bt.build_players_frame(
        HIST, bt.aggregate_prior_history(HIST.gw, target), target)
    frames_poisoned = bt.build_players_frame(
        hist_poisoned, bt.aggregate_prior_history(poisoned, target), target)
    pd.testing.assert_frame_equal(frames_clean, frames_poisoned, check_dtype=False,
                                  obj="players_df leaked future data")
    print("  build_players_frame is unaffected by poisoned future gameweeks")

    res_clean = bt.run_gameweek(HIST, target).sort_values("id").reset_index(drop=True)
    res_poisoned = bt.run_gameweek(hist_poisoned, target).sort_values("id").reset_index(drop=True)
    np.testing.assert_allclose(
        res_clean["predicted"].to_numpy(), res_poisoned["predicted"].to_numpy(),
        rtol=0, atol=0,
        err_msg="predictions changed when only future gameweeks were altered")
    print("  predictions for GW6 are bit-identical with a poisoned future")


def test_truncation_invariance():
    """Building GW N's state from the full season == building it from GW<N only.

    A second, independent angle on the same property: if the harness only ever
    reads GW < N, then physically deleting GW >= N from the input must be a
    no-op. This catches leaks that a value-corruption test could miss (e.g. code
    that reads the *shape* of the future rather than their values).
    """
    for target in (3, 7, 10):
        full = bt.aggregate_prior_history(HIST.gw, target)
        truncated = bt.aggregate_prior_history(
            HIST.gw[HIST.gw["gw"] < target].copy(), target)
        pd.testing.assert_frame_equal(full.sort_index(), truncated.sort_index(),
                                      check_dtype=False,
                                      obj=f"GW{target} truncation invariance")
    print("  aggregate_prior_history is invariant to deleting future gameweeks")


def test_gw1_has_no_history_at_all():
    """GW1 must be a genuine cold start: zero minutes, zero xG, for everyone.

    This is the situation the 2026/27 build is actually in, so it needs to be
    exercised rather than skipped. Also confirms the roster still exists (the
    universe comes from players_raw, which is legitimately known pre-season).
    """
    prior = bt.aggregate_prior_history(HIST.gw, 1)
    assert len(prior) == 0, "GW1 prior history should be empty"

    players = bt.build_players_frame(HIST, prior, 1)
    assert len(players) > 0, "GW1 player pool should still be populated from the roster"
    assert float(players["minutes"].sum()) == 0.0
    assert float(players["expected_goals"].astype(float).sum()) == 0.0
    assert float(players["total_points"].sum()) == 0.0
    print(f"  GW1 cold start: {len(players)} players, all with zero history")


def test_fixture_frame_strips_results():
    """The model must never be handed a scoreline for the gameweek it predicts."""
    fx = bt.build_fixtures_frame(HIST, 4)
    assert len(fx) > 0
    for banned in ("team_h_score", "team_a_score"):
        assert banned not in fx.columns, f"{banned} leaked into the model's fixtures"
    assert not fx["finished"].any(), "fixtures handed to the model claim to be finished"
    print("  build_fixtures_frame drops scores and the finished flag")


def test_actuals_are_read_separately():
    """Ground truth comes from the gw == N slice and matches the raw archive."""
    actual = bt.get_actual_points(HIST.gw, 4)
    raw = HIST.gw[HIST.gw["gw"] == 4]
    assert _approx(float(actual["total_points"].sum()),
                   float(raw["total_points"].sum()), 1e-6)
    assert len(actual) == raw["element"].nunique()
    print("  get_actual_points reproduces the archive's GW4 totals exactly")


# ---------------------------------------------------------------------------
# 2. METRICS — every number below was worked out by hand
# ---------------------------------------------------------------------------

def test_metrics_on_hand_checked_example():
    """pred = [1,2,3,4], actual = [2,2,2,8].

    errors            = [-1, 0, 1, -4]
    MAE               = (1 + 0 + 1 + 4) / 4                  = 1.5
    RMSE              = sqrt((1 + 0 + 1 + 16) / 4) = sqrt(4.5) = 2.1213203...
    bias  = mean(pred) - mean(actual) = 2.5 - 3.5            = -1.0

    Pearson:
      dev(pred)   = [-1.5, -0.5, 0.5, 1.5]      sd numerator sqrt(5)
      dev(actual) = [-1.5, -1.5, -1.5, 4.5]     sd numerator sqrt(27)
      cov numerator = 2.25 + 0.75 - 0.75 + 6.75 = 9
      r = 9 / (sqrt(5) * sqrt(27)) = 9 / sqrt(135) = 0.7745966...

    Spearman:
      ranks(pred)   = [1, 2, 3, 4]
      ranks(actual) = [2, 2, 2, 4]     (three-way tie on 2 -> average rank 2)
      Pearson of those: cov numerator = 0.75 + 0.25 - 0.25 + 2.25 = 3
      r = 3 / (sqrt(5) * sqrt(3)) = 3 / sqrt(15) = 0.7745966...
    """
    pred = [1.0, 2.0, 3.0, 4.0]
    actual = [2.0, 2.0, 2.0, 8.0]

    assert _approx(bt.mae(pred, actual), 1.5)
    assert _approx(bt.rmse(pred, actual), math.sqrt(4.5))
    assert _approx(bt.bias(pred, actual), -1.0)
    assert _approx(bt.pearson(pred, actual), 9.0 / math.sqrt(135.0), 1e-12)
    assert _approx(bt.spearman(pred, actual), 3.0 / math.sqrt(15.0), 1e-12)
    print("  MAE / RMSE / bias / Pearson / Spearman match the hand calculation")


def test_metrics_edge_cases():
    """Degenerate inputs return NaN rather than exploding or lying."""
    assert math.isnan(bt.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))   # zero variance
    assert math.isnan(bt.mae([], []))
    # NaNs in either series are dropped pairwise, not propagated.
    assert _approx(bt.mae([1.0, float("nan"), 3.0], [2.0, 5.0, 1.0]), 1.5)
    print("  metric edge cases (zero variance, empty, NaN) handled")


def test_spearman_is_rank_only():
    """A monotone transform of the prediction must not change Spearman."""
    pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    actual = [0.0, 3.0, 1.0, 9.0, 6.0]
    squashed = [math.log(p + 1) * 100 for p in pred]
    assert _approx(bt.spearman(pred, actual), bt.spearman(squashed, actual), 1e-12)
    assert not _approx(bt.pearson(pred, actual), bt.pearson(squashed, actual), 1e-6)
    print("  Spearman is invariant to monotone rescaling, Pearson is not")


def test_top_n_precision_hand_checked():
    """Two gameweeks, top-3 picks each, threshold 6.

    GW1 predictions [5,4,3,2,1] -> top 3 are the players with actual [7,0,6].
         hits = 7>=6 and 6>=6 -> 2 of 3.
    GW2 predictions [9,8,7,1,0] -> top 3 are the players with actual [6,6,2].
         hits = 2 of 3.
    Overall precision = 4 / 6 = 0.666...
    Mean actual points over the 6 picks = (7+0+6+6+6+2)/6 = 27/6 = 4.5
    """
    df = pd.DataFrame({
        "gameweek": [1] * 5 + [2] * 5,
        "predicted": [5, 4, 3, 2, 1, 9, 8, 7, 1, 0],
        "actual": [7, 0, 6, 1, 1, 6, 6, 2, 0, 0],
        "blank": [False] * 10,
    })
    out = bt.top_n_precision(df, n=3, threshold=6.0)
    assert _approx(out["precision"], 4.0 / 6.0)
    assert _approx(out["mean_actual_points"], 27.0 / 6.0)
    assert out["picks"] == 6 and out["gameweeks"] == 2
    print("  top-N precision matches the hand calculation (4/6, 4.5 pts/pick)")


def test_top_n_precision_excludes_blanks():
    """A player with no fixture cannot be a pick, even if his xP row exists."""
    df = pd.DataFrame({
        "gameweek": [1] * 4,
        "predicted": [9.0, 5.0, 4.0, 3.0],
        "actual": [0.0, 8.0, 7.0, 0.0],
        "blank": [True, False, False, False],
    })
    out = bt.top_n_precision(df, n=2, threshold=6.0)
    assert _approx(out["precision"], 1.0), "blank-gameweek player was picked"
    print("  top-N precision skips blank-gameweek players")


def test_calibration_table_hand_checked():
    """Ten predictions split into two quantile bins.

    preds 0..9; bottom half [0,1,2,3,4] mean 2, top half [5,6,7,8,9] mean 7.
    actuals chosen so the bottom bin is perfectly calibrated (mean 2) and the
    top bin over-predicts by exactly 3 (mean actual 4).
    """
    pred = list(range(10))
    actual = [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]
    tbl = bt.calibration_table(pred, actual, n_bins=2)
    assert len(tbl) == 2
    assert _approx(float(tbl.loc[0, "mean_pred"]), 2.0)
    assert _approx(float(tbl.loc[0, "mean_actual"]), 2.0)
    assert _approx(float(tbl.loc[0, "bias"]), 0.0)
    assert _approx(float(tbl.loc[1, "mean_pred"]), 7.0)
    assert _approx(float(tbl.loc[1, "mean_actual"]), 4.0)
    assert _approx(float(tbl.loc[1, "bias"]), 3.0)
    assert int(tbl["n"].sum()) == 10
    print("  calibration binning and per-bin bias match the hand calculation")


def test_decompose_actual_points_hand_checked():
    """A defender's real 90 minutes, scored by hand against the FPL rules.

    60+ minutes                     -> 2 appearance
    1 goal as a DEF (6 pts)         -> 6
    1 assist                        -> 3
    clean sheet as a DEF            -> 4
    0 conceded                      -> 0
    1 yellow                        -> -1
    3 bonus                         -> 3
    total                           = 17
    """
    row = pd.DataFrame([{
        "position": "DEF", "minutes": 90, "goals_scored": 1, "assists": 1,
        "clean_sheets": 1, "goals_conceded": 0, "saves": 0, "bonus": 3,
        "yellow_cards": 1, "red_cards": 0, "own_goals": 0,
        "penalties_saved": 0, "penalties_missed": 0, "actual": 17.0,
    }])
    d = bt.decompose_actual_points(row).iloc[0]
    assert _approx(d["appearance"], 2.0)
    assert _approx(d["goals"], 6.0)
    assert _approx(d["assists"], 3.0)
    assert _approx(d["clean_sheet"], 4.0)
    assert _approx(d["cards"], -1.0)
    assert _approx(d["bonus"], 3.0)
    assert _approx(d["residual"], 0.0), "decomposition should fully explain 17 pts"

    # A keeper: 5 saves -> +1 (floor(5/3)), 3 conceded -> -1 (floor(3/2)),
    # sub appearance (30 mins) -> 1 point, no clean sheet.
    gk = pd.DataFrame([{
        "position": "GK", "minutes": 30, "goals_scored": 0, "assists": 0,
        "clean_sheets": 0, "goals_conceded": 3, "saves": 5, "bonus": 0,
        "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
        "penalties_saved": 0, "penalties_missed": 0, "actual": 1.0,
    }])
    dg = bt.decompose_actual_points(gk).iloc[0]
    assert _approx(dg["appearance"], 1.0)
    assert _approx(dg["saves"], 1.0)
    assert _approx(dg["goals_conceded"], -1.0)
    assert _approx(dg["clean_sheet"], 0.0)
    assert _approx(dg["residual"], 1.0 - (1.0 + 1.0 - 1.0))
    print("  actual-points decomposition reproduces the FPL scoring rules")


def test_decomposition_reconciles_on_synthetic_season():
    """Across the whole synthetic season the residual must be ~zero.

    The synthetic ground truth is generated by applying the real scoring rules,
    so if `decompose_actual_points()` is right the two must agree to the point.
    This is the strongest available check that the component attribution driving
    the report's recommendations is not quietly wrong.
    """
    results = bt.run_backtest(HIST, from_gw=1, to_gw=10, verbose=False)
    resid = bt.decompose_actual_points(results)["residual"]
    assert abs(float(resid.mean())) < 1e-9, (
        f"component decomposition does not reconcile: mean residual "
        f"{float(resid.mean()):.6f}")
    print("  decomposition reconciles to total_points across the whole season")


def test_position_normalisation():
    """vaastav's 'GK' and FPL's element_type ints both land on 'GKP'."""
    assert bt.normalise_position("GK") == "GKP"
    assert bt.normalise_position("GKP") == "GKP"
    assert bt.normalise_position(1) == "GKP"
    assert bt.normalise_position("1.0") == "GKP"
    assert bt.normalise_position(4) == "FWD"
    assert bt.normalise_position("fwd") == "FWD"
    assert bt.normalise_position(None) == "MID"
    assert bt.normalise_position("nonsense") == "MID"
    print("  position normalisation handles GK/GKP/int/float/None")


# ---------------------------------------------------------------------------
# 3. END-TO-END
# ---------------------------------------------------------------------------

def test_loader_normalises_synthetic_csvs():
    """The synthetic CSVs go through the same loader the real files will."""
    assert set(["element", "gw", "minutes", "total_points", "was_home"]).issubset(
        HIST.gw.columns)
    assert HIST.gw["element"].dtype.kind in "iu"
    assert HIST.gw["gw"].dtype.kind in "iu"
    assert HIST.gw["was_home"].dtype == bool
    assert set(HIST.gw["position"].unique()).issubset(set(bt.POSITIONS))
    assert HIST.teams["name"].is_unique
    assert {"home_team", "away_team", "team_h_difficulty"}.issubset(HIST.fixtures.columns)
    # Every gameweek row must resolve to a real club name, or the model cannot
    # match players to fixtures at all.
    assert set(HIST.gw["team"].unique()).issubset(set(HIST.teams["name"]))
    print(f"  loader produced {len(HIST.gw):,} normalised gameweek rows")


def test_loader_raises_on_missing_files():
    empty = tempfile.mkdtemp(prefix="fpl_backtest_empty_")
    try:
        bt.load_historical_gw_data(empty)
    except FileNotFoundError as exc:
        assert "merged_gw.csv" in str(exc)
        print("  loader raises a clear FileNotFoundError on an empty folder")
    else:
        raise AssertionError("loader should have raised on an empty folder")
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def test_historical_data_available_flag():
    assert bt.historical_data_available(_TMPDIR) is True
    assert bt.historical_data_available(os.path.join(_TMPDIR, "nope")) is False
    print("  historical_data_available correctly detects present/absent data")


def test_end_to_end_runs_on_synthetic_sample():
    """The whole harness, start to finish, without error."""
    results = bt.run_backtest(HIST, from_gw=1, to_gw=10, verbose=False)

    assert len(results) > 0
    for col in ("gameweek", "predicted", "actual", "position", "is_cold_start",
                "blank", "baseline_zero", "baseline_ppg", "baseline_position"):
        assert col in results.columns, f"missing result column {col}"

    assert results["predicted"].notna().all()
    assert results["actual"].notna().all()
    assert np.isfinite(results["predicted"]).all()
    assert results["gameweek"].nunique() == 10
    assert results["is_cold_start"].sum() > 0 and (~results["is_cold_start"]).sum() > 0

    # Non-degenerate: the model must not be emitting a constant.
    assert results["predicted"].std() > 0.1
    # Sanity band: a per-gameweek xP outside this range means something is badly
    # wrong with the frames, not just with the calibration.
    assert 0.0 <= results["predicted"].mean() <= 12.0

    block = bt.metric_block(results)
    for key in ("mae", "rmse", "bias", "pearson", "spearman"):
        assert np.isfinite(block[key]), f"{key} came out non-finite"

    print(f"  end-to-end: {len(results):,} rows, MAE {block['mae']:.3f}, "
          f"rho {block['spearman']:.3f}")


def test_blank_and_double_gameweeks_are_handled():
    """The synthetic season deliberately contains a blank (GW9) and a double (GW11).

    Blanks must predict exactly 0. Note they then drop out of the scored results
    entirely, and that is correct: the archive has no row for a player whose club
    did not play, so there is no ground truth to score against. The blank branch
    still has to work, because it is the model's own output that the optimizer
    consumes live.

    Doubles must have their actual points summed across both fixtures, matching
    the way the model sums its components, or predicted and actual would be on
    different scales.
    """
    hist14 = bt.load_synthetic_dataset(n_gameweeks=14, seed=11)

    # Blank: check the model's own output for GW9, before the ground-truth join.
    prior9 = bt.aggregate_prior_history(hist14.gw, 9)
    players9 = bt.build_players_frame(hist14, prior9, 9)
    fixtures9 = bt.build_fixtures_frame(hist14, 9)
    teams_with_fixtures = set(fixtures9["home_team"]) | set(fixtures9["away_team"])
    blanking = set(hist14.teams["name"]) - teams_with_fixtures
    assert blanking, "GW9 should leave at least one club without a fixture"

    import xp_model as m
    pred9 = m.calculate_xp_for_gameweek(players9, hist14.teams, fixtures9, gameweek=9)
    blanks = pred9[pred9["n_fixtures"] == 0]
    assert len(blanks) > 0, "GW9 should contain blank-gameweek players"
    assert float(blanks["xp"].abs().max()) == 0.0, "a blank predicted non-zero xP"
    assert set(blanks["team_name"]) == blanking

    scored9 = bt.run_gameweek(hist14, 9)
    assert not scored9["blank"].any(), (
        "blank players should have no ground-truth row and drop out of scoring")

    # Double: both the prediction and the actual must cover two fixtures.
    gw11 = bt.run_gameweek(hist14, 11)
    doubles = gw11[gw11["n_fixtures"] > 1]
    assert len(doubles) > 0, "GW11 should contain double-gameweek players"
    assert (doubles["n_fixtures_actual"] > 1).all(), (
        "double-gameweek actuals were not summed across both fixtures")
    print(f"  blanks ({len(blanks)}) predict 0 and drop out of scoring; "
          f"doubles ({len(doubles)}) sum both fixtures")


def test_baselines_are_history_only():
    """GW1 baselines must be empty of information (no season history exists)."""
    prior = bt.aggregate_prior_history(HIST.gw, 1)
    players = bt.build_players_frame(HIST, prior, 1)
    base = bt.build_baselines(prior, players, HIST.gw, 1)
    assert float(base["baseline_ppg"].abs().sum()) == 0.0
    assert float(base["baseline_position"].abs().sum()) == 0.0

    # By GW6 they should carry real information.
    prior6 = bt.aggregate_prior_history(HIST.gw, 6)
    players6 = bt.build_players_frame(HIST, prior6, 6)
    base6 = bt.build_baselines(prior6, players6, HIST.gw, 6)
    assert float(base6["baseline_ppg"].sum()) > 0.0
    assert base6["baseline_position"].nunique() > 1
    print("  baselines are empty at GW1 and informative by GW6")


def test_report_renders():
    """The markdown report builds, is substantial, and is labelled synthetic."""
    results = bt.run_backtest(HIST, from_gw=1, to_gw=10, verbose=False)
    report = bt.build_report(results, HIST, from_gw=1, to_gw=10)
    assert "SYNTHETIC SELF-TEST MODE" in report
    assert "## 1. Overall accuracy" in report
    assert "## 9. Recommended constant changes" in report
    assert "xp_model.py` was **not** modified" in report
    assert len(report) > 3000
    # Recommendations must name real constants that exist in xp_model.
    recs = bt.derive_recommendations(results, bt.component_comparison(results))
    assert len(recs) > 0
    import xp_model as m
    named = [r.constant for r in recs]
    known = [n for n in named
             if any(hasattr(m, part.strip()) for part in n.replace("/", " ").split())]
    assert known, f"no recommendation named a real xp_model constant: {named}"
    print(f"  report renders ({len(report):,} chars) with {len(recs)} recommendations")


# ---------------------------------------------------------------------------
# 4. DEFCON DATA-GAP EXPERIMENT
# ---------------------------------------------------------------------------

def test_defcon_reimplementation_matches_model():
    """`_defcon_points_from_rate` must reproduce xp_model's own DefCon term.

    The experiment re-scores the DefCon component without re-running the model,
    which is only legitimate if the re-implementation is exact. Feed it the flat
    DEFCON_PER90_PRIOR the model itself used and the two must agree to floating
    point. This is the test that licenses the whole of report section 8.
    """
    import xp_model as m

    results = bt.run_backtest(HIST, from_gw=4, to_gw=10, verbose=False)
    played = results[~results["blank"]].copy()
    played["position"] = played["position"].map(bt.normalise_position)

    # Every backtested player has status 'a', so availability = 1.0 and
    # minutes_distribution() reduces p_60 to exactly p_start * P_60_GIVEN_START.
    p_60 = played["p_start"].astype(float) * m.P_60_GIVEN_START
    recomputed = np.array([
        bt._defcon_points_from_rate(m.DEFCON_PER90_PRIOR.get(pos, 0.0), pos, p, k)
        for pos, p, k in zip(played["position"], p_60,
                             played["n_fixtures"].astype(float))
    ])
    np.testing.assert_allclose(
        recomputed, played["xp_defcon"].astype(float).to_numpy(),
        rtol=0, atol=1e-9,
        err_msg="DefCon re-implementation diverges from xp_model.calculate_xp")

    # And it must actually be doing something, not trivially zero everywhere.
    assert recomputed.sum() > 0
    print(f"  DefCon re-implementation matches model exactly over "
          f"{len(played):,} rows")


def test_rolling_defcon_rates_have_no_lookahead():
    """Rolling CBIT/CBIRT rates must not move when the future is corrupted."""
    target = 6
    poisoned = HIST.gw.copy()
    future = poisoned["gw"] >= target
    for col in ("defensive_contribution", "minutes"):
        poisoned.loc[future, col] = poisoned.loc[future, col] * 1000.0 + 999.0

    prior_clean = bt.aggregate_prior_history(HIST.gw, target)
    prior_poisoned = bt.aggregate_prior_history(poisoned, target)
    players = bt.build_players_frame(HIST, prior_clean, target)

    a = bt.rolling_defcon_rates(prior_clean, players)
    b = bt.rolling_defcon_rates(prior_poisoned, players)
    pd.testing.assert_frame_equal(a, b, check_dtype=False,
                                  obj="rolling DefCon rates leaked future data")
    # Must carry real per-player variation, otherwise the experiment is vacuous.
    assert a["defcon_per90_shrunk"].notna().sum() > 0
    print("  rolling DefCon rates are unchanged by a poisoned future")


def test_rolling_defcon_rate_hand_checked():
    """One player's rolling rate, computed by hand from a 3-gameweek history."""
    import xp_model as m

    prior = pd.DataFrame(
        {"defensive_contribution": [21.0], "minutes": [180.0]},
        index=pd.Index([1], name="element"))
    players = pd.DataFrame([{"id": 1, "position": "DEF"}])
    out = bt.rolling_defcon_rates(prior, players)

    # 21 actions in 180 minutes = 2 full 90s -> 10.5 per 90.
    assert _approx(float(out.loc[1, "defcon_per90_raw"]), 10.5)

    # Shrunk: (21 + 6.5 * 3) / (2 + 3) = (21 + 19.5) / 5 = 8.1
    expected = (21.0 + m.DEFCON_PER90_PRIOR["DEF"]
                * bt.DEFCON_ROLLING_PRIOR_WEIGHT_90S) / (2.0 + bt.DEFCON_ROLLING_PRIOR_WEIGHT_90S)
    assert _approx(expected, 8.1)
    assert _approx(float(out.loc[1, "defcon_per90_shrunk"]), 8.1)
    print("  rolling DefCon rate: 21 actions / 180 mins -> 10.5 raw, 8.1 shrunk")


def test_defcon_experiment_runs_and_is_self_consistent():
    """The experiment produces all three variants against a 0/2 target."""
    import xp_model as m

    results = bt.run_backtest(HIST, from_gw=1, to_gw=10, verbose=False)
    exp = bt.defcon_experiment(results)
    if exp.get("n", 0) == 0:
        print("  (synthetic season too small for the DefCon experiment — skipped)")
        return
    table = exp["table"]
    assert len(table) == 3
    assert set(table["variant"]) == {
        "flat prior (live model today)", "own rolling rate, shrunk",
        "own rolling rate, raw"}
    # All three variants score the SAME target, so mean_actual must be identical.
    assert table["mean_actual"].nunique() == 1
    # The target is the binary DefCon award, so its mean lies in [0, 2].
    assert 0.0 <= float(table["mean_actual"].iloc[0]) <= float(m.DEFCON_POINTS)
    assert _approx(exp["base_rate"],
                   float(table["mean_actual"].iloc[0]) / m.DEFCON_POINTS, 1e-9)
    print(f"  DefCon experiment ran over {exp['n']:,} rows, "
          f"base rate {exp['base_rate']:.1%}")


# ---------------------------------------------------------------------------
# 5. FPL'S OWN xP BASELINE
# ---------------------------------------------------------------------------

def test_gameweeks_with_fpl_xp_rejects_all_zero_rounds():
    """All-zero rounds are missing data and must be excluded, not scored.

    Hand-built: GW1 is fully populated, GW2 is entirely zero (a scraper gap),
    GW3 is half populated (legitimate — FPL projects 0.0 for players it expects
    not to feature). Only GW1 and GW3 are usable.
    """
    gw = pd.DataFrame({
        "gw": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "fpl_xp": [2.5, 1.0, 0.0, 3.0,
                   0.0, 0.0, 0.0, 0.0,
                   4.0, 0.0, 0.0, 0.0],
    })
    assert bt.gameweeks_with_fpl_xp(gw) == [1, 3]
    # No column at all -> nothing usable, and no exception.
    assert bt.gameweeks_with_fpl_xp(pd.DataFrame({"gw": [1, 2]})) == []
    print("  all-zero FPL-xP rounds are correctly treated as missing data")


def test_fpl_official_xp_is_not_treated_as_ground_truth():
    """fpl_xp must never enter the cumulative feature set or the actuals."""
    assert "fpl_xp" not in bt._CUMULATIVE_COLUMNS
    assert "fpl_xp" not in bt._NUMERIC_GW_COLUMNS

    prior = bt.aggregate_prior_history(HIST.gw, 6)
    assert "fpl_xp" not in prior.columns
    players = bt.build_players_frame(HIST, prior, 6)
    assert not [c for c in players.columns if "fpl" in str(c).lower()]

    # Double gameweeks: FPL's xP is summed across fixtures, like everything else.
    gw = pd.DataFrame({
        "gw": [1, 1, 1],
        "element": [10, 10, 11],
        "fpl_xp": [2.0, 3.0, 1.5],
    })
    out = bt.fpl_official_xp(gw, 1)
    assert _approx(float(out.loc[10]), 5.0)
    assert _approx(float(out.loc[11]), 1.5)
    # An unpopulated round yields nothing rather than a column of zeros.
    assert len(bt.fpl_official_xp(
        pd.DataFrame({"gw": [1, 1], "element": [1, 2], "fpl_xp": [0.0, 0.0]}), 1)) == 0
    print("  FPL xP is a baseline only: absent from features, summed over doubles")


# ---------------------------------------------------------------------------
# 6. CREDIBLE POOL
# ---------------------------------------------------------------------------

def test_credible_pool_uses_only_prior_information():
    """The credible-pool filter must not condition on the gameweek's outcome."""
    results = bt.run_backtest(HIST, from_gw=1, to_gw=10, verbose=False)
    cred = bt.credible_pool(results)
    assert len(cred) < len(results), "credible pool filtered nothing"
    # n_appearances is built inside the firewall, so every retained row must have
    # a prior appearance -- and rows may legitimately have 0 minutes THIS week.
    assert (cred["n_appearances"].fillna(0) >= 1).all()
    assert (cred["minutes"].astype(float) == 0).any(), (
        "credible pool looks like it is conditioning on playing this gameweek")
    print(f"  credible pool ({len(cred):,}/{len(results):,}) uses prior info only")


def test_never_appeared_analysis_arithmetic():
    """Hand-checked: the never-appeared summary must add up."""
    results = pd.DataFrame({
        "gameweek": [6, 6, 6, 6],
        "n_gws_available": [5, 5, 5, 5],
        "n_appearances": [0.0, 0.0, 3.0, 2.0],
        "predicted": [2.0, 2.0, 5.0, 4.0],
        "actual": [0.0, 1.0, 6.0, 2.0],
        "p_start": [0.65, 0.65, 0.9, 0.8],
        "minutes": [0.0, 20.0, 90.0, 90.0],
    })
    g = bt.never_appeared_analysis(results)
    assert g["n"] == 2
    assert _approx(g["share_of_pool"], 0.5)
    assert _approx(g["mean_pred"], 2.0)
    assert _approx(g["mean_actual"], 0.5)          # (0 + 1) / 2
    assert _approx(g["mean_p_start"], 0.65)
    assert _approx(g["pct_who_played"], 0.5)       # 1 of 2 got minutes
    assert _approx(g["signed_error"], 3.0)         # (2-0) + (2-1)
    assert _approx(g["pool_signed_error"], 4.0)    # 3 + (5-6) + (4-2)
    print("  never-appeared analysis arithmetic matches the hand calculation")


# ---------------------------------------------------------------------------
# 7. REAL DATA (skipped automatically if the archive is not present)
# ---------------------------------------------------------------------------

def test_real_data_end_to_end():
    """The real 2025/26 archive loads, replays, and scores without error.

    Skipped with a message rather than failed when the CSVs are absent, so the
    suite still passes on a clean checkout.
    """
    if not bt.historical_data_available():
        print("  (real archive not present — skipped)")
        return
    hist = bt.load_historical_gw_data()

    # Schema gotcha this project was explicitly warned about: the archive says
    # "GK", the model says "GKP". If normalisation regressed, every goalkeeper
    # would silently fall through to "MID" and be scored with the wrong rules.
    assert set(hist.gw["position"].unique()) <= set(bt.POSITIONS)
    assert (hist.gw["position"] == "GKP").sum() > 0, "no GKP rows: GK->GKP mapping broke"

    results = bt.run_backtest(hist, from_gw=1, to_gw=6, verbose=False)
    assert len(results) > 3000
    assert results["predicted"].notna().all()
    assert results["actual"].notna().all()
    # Sanity: actual points must reconcile with the archive's own column.
    assert _approx(float(results["actual"].sum()),
                   float(results["total_points"].sum()), 1e-6)
    # The FPL-xP baseline should be present for these early gameweeks.
    assert results["baseline_fpl_xp"].notna().sum() > 0
    report = bt.build_report(results, hist, from_gw=1, to_gw=6)
    assert "SYNTHETIC" not in report
    assert "## 0. Summary" in report
    print(f"  real data: {len(results):,} rows over GW1-6, report renders "
          f"({len(report):,} chars)")


def test_main_synthetic_mode_smoke():
    """`python3 backtest/backtest.py --synthetic` path, writing to a temp report."""
    report_path = os.path.join(_TMPDIR, "smoke_report.md")
    rc = bt.main(["--synthetic", "--to-gw", "6", "--quiet", "--report", report_path])
    assert rc == 0
    assert os.path.isfile(report_path)
    assert os.path.getsize(report_path) > 3000
    print("  main() --synthetic runs end to end and writes a report")


# ---------------------------------------------------------------------------

def run_tests():
    tests = [
        # leakage first: nothing else matters if this fails
        test_no_lookahead_leakage,
        test_truncation_invariance,
        test_gw1_has_no_history_at_all,
        test_fixture_frame_strips_results,
        test_actuals_are_read_separately,
        # metrics
        test_metrics_on_hand_checked_example,
        test_metrics_edge_cases,
        test_spearman_is_rank_only,
        test_top_n_precision_hand_checked,
        test_top_n_precision_excludes_blanks,
        test_calibration_table_hand_checked,
        test_decompose_actual_points_hand_checked,
        test_decomposition_reconciles_on_synthetic_season,
        test_position_normalisation,
        # end to end
        test_loader_normalises_synthetic_csvs,
        test_loader_raises_on_missing_files,
        test_historical_data_available_flag,
        test_end_to_end_runs_on_synthetic_sample,
        test_blank_and_double_gameweeks_are_handled,
        test_baselines_are_history_only,
        # defcon data-gap experiment
        test_defcon_reimplementation_matches_model,
        test_rolling_defcon_rates_have_no_lookahead,
        test_rolling_defcon_rate_hand_checked,
        test_defcon_experiment_runs_and_is_self_consistent,
        # FPL's own xP baseline
        test_gameweeks_with_fpl_xp_rejects_all_zero_rounds,
        test_fpl_official_xp_is_not_treated_as_ground_truth,
        # credible pool
        test_credible_pool_uses_only_prior_information,
        test_never_appeared_analysis_arithmetic,
        # reporting + real data
        test_report_renders,
        test_real_data_end_to_end,
        test_main_synthetic_mode_smoke,
    ]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
    print(f"\nAll {len(tests)} backtest harness tests passed.")


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        shutil.rmtree(_TMPDIR, ignore_errors=True)
