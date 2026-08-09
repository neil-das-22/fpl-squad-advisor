"""
Tests for optimization/squad_optimizer.py, plus an end-to-end demo.

Standalone plain-assert style, matching data/test_fpl_client.py:

    python3 optimization/test_squad_optimizer.py       # tests + demo
    python3 -m pytest optimization/test_squad_optimizer.py

The demo at the bottom (`run_demo`) chains the whole pipeline --
    synthetic data -> xp_model -> pick_squad -> pick_starting_xi -> transfers
-- and prints the squad, so there is visible proof the two modules integrate.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "models"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import squad_optimizer as so  # noqa: E402
import xp_model as m  # noqa: E402
from synthetic_data import (  # noqa: E402
    make_sample_fixtures_df,
    make_sample_players_df,
    make_sample_teams_df,
)

PLAYERS = make_sample_players_df()
TEAMS = make_sample_teams_df()
FIXTURES = make_sample_fixtures_df()

# Gameweek-1 expected points for the whole pool -- the optimizer's only input.
XP = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def assert_valid_squad(squad: pd.DataFrame, budget: float = so.DEFAULT_BUDGET):
    assert len(squad) == so.SQUAD_SIZE, f"squad has {len(squad)} players"
    counts = squad["position"].value_counts().to_dict()
    for pos, need in so.SQUAD_COMPOSITION.items():
        assert counts.get(pos, 0) == need, f"{pos}: {counts.get(pos, 0)} != {need}"
    cost = round(float(squad["price_m"].sum()), 4)
    assert cost <= budget + 1e-6, f"squad costs {cost} > budget {budget}"
    club_counts = squad["team_name"].value_counts()
    assert club_counts.max() <= so.MAX_PER_CLUB, \
        f"club cap broken: {club_counts.to_dict()}"
    assert squad["id"].nunique() == so.SQUAD_SIZE, "duplicate players in squad"


# ---------------------------------------------------------------------------
# pick_squad
# ---------------------------------------------------------------------------

def test_pick_squad_respects_all_constraints():
    result = so.pick_squad(XP, budget=100.0)
    assert_valid_squad(result["squad"], 100.0)
    assert result["optimal"] is True
    assert abs(result["total_xp"] - result["squad"]["xp"].sum()) < 1e-9
    assert abs(result["total_cost"] + result["remaining_budget"] - 100.0) < 1e-6
    print(f"  pick_squad ok  (solver={result['solver']}, "
          f"cost={result['total_cost']}m, xP={result['total_xp']:.2f})")


def test_tighter_budget_is_feasible_and_worse():
    rich = so.pick_squad(XP, budget=100.0)
    poor = so.pick_squad(XP, budget=80.0)
    assert_valid_squad(poor["squad"], 80.0)
    # A smaller budget can never buy a better squad.
    assert poor["total_xp"] <= rich["total_xp"] + 1e-9
    print(f"  budget sensitivity ok (100m -> {rich['total_xp']:.2f} xP, "
          f"80m -> {poor['total_xp']:.2f} xP)")


def test_club_cap_binds():
    """Force the club cap to bite by making one club dominate the xP table."""
    boosted = XP.copy()
    boosted.loc[boosted["team_name"] == "Man City", "xp"] += 20.0
    result = so.pick_squad(boosted, budget=100.0)
    squad = result["squad"]
    assert_valid_squad(squad, 100.0)
    n_city = int((squad["team_name"] == "Man City").sum())
    # It should want every City player it can get -- and be capped at exactly 3.
    assert n_city == so.MAX_PER_CLUB, n_city
    print(f"  club cap ok (wanted all of Man City, took {n_city})")


def test_optimality_no_improving_swap_exists():
    """Local-optimality check: no single swap improves the squad legally.

    A necessary condition for the ILP optimum. It catches whole classes of
    solver/constraint bugs without needing to re-derive the optimum by hand.
    """
    budget = 100.0
    result = so.pick_squad(XP, budget=budget)
    squad = result["squad"]
    pool = XP[pd.to_numeric(XP["price_m"], errors="coerce") > 0]
    held = set(squad["id"].astype(str))

    for _, out_row in squad.iterrows():
        spare = budget - float(squad["price_m"].sum()) + float(out_row["price_m"])
        kept = squad[squad["id"] != out_row["id"]]
        club_counts = kept["team_name"].value_counts().to_dict()
        same_pos = pool[(pool["position"] == out_row["position"])
                        & (~pool["id"].astype(str).isin(held))]
        for _, cand in same_pos.iterrows():
            if float(cand["price_m"]) > spare + 1e-9:
                continue
            if club_counts.get(cand["team_name"], 0) + 1 > so.MAX_PER_CLUB:
                continue
            assert float(cand["xp"]) <= float(out_row["xp"]) + 1e-9, (
                f"improving swap exists: {out_row['web_name']} "
                f"({out_row['xp']:.2f}) -> {cand['web_name']} ({cand['xp']:.2f})")
    print("  optimality ok (no improving single swap)")


def test_matches_brute_force_on_random_instances():
    """Cross-check the solver against exhaustive enumeration.

    Small pools (22 players, 6 clubs) make every legal 15-man squad
    enumerable, so the true optimum is known. This is the test that guards the
    two clever bits of the fallback solver -- the dominance reduction and the
    Lagrangian bound -- because a bug in either would silently return a squad
    that is merely good rather than optimal.
    """
    import itertools
    import random

    need = so.SQUAD_COMPOSITION

    def brute_force(recs, budget_tenths, cap):
        by_pos = {p: [r for r in recs if r[0] == p] for p in need}
        best = None
        pools = [itertools.combinations(by_pos[p], need[p]) for p in so.POSITION_ORDER]
        for combo in itertools.product(*pools):
            flat = [r for group in combo for r in group]
            if sum(r[1] for r in flat) > budget_tenths:
                continue
            counts = {}
            legal = True
            for r in flat:
                counts[r[3]] = counts.get(r[3], 0) + 1
                if counts[r[3]] > cap:
                    legal = False
                    break
            if not legal:
                continue
            value = sum(r[2] for r in flat)
            if best is None or value > best:
                best = value
        return best

    checked = 0
    for seed in range(12):
        random.seed(seed)
        clubs = [f"C{i}" for i in range(6)]
        rows = []
        pid = 0
        for pos, count in (("GKP", 3), ("DEF", 7), ("MID", 7), ("FWD", 5)):
            for _ in range(count):
                pid += 1
                rows.append({"id": pid, "web_name": f"p{pid}", "position": pos,
                             "team_name": random.choice(clubs),
                             "price_m": round(random.uniform(4.0, 12.0), 1),
                             "xp": round(random.uniform(0.0, 8.0), 3)})
        pool = pd.DataFrame(rows)
        budget = random.choice([75.0, 85.0, 95.0, 110.0])

        recs = [(r["position"], int(round(r["price_m"] * 10)), r["xp"], r["team_name"])
                for r in rows]
        reference = brute_force(recs, int(round(budget * 10)), so.MAX_PER_CLUB)

        try:
            got = so.pick_squad(pool, budget=budget)["total_xp"]
        except so.OptimizerError:
            got = None

        assert (reference is None) == (got is None), \
            f"seed {seed}: feasibility disagreement (brute={reference}, solver={got})"
        if reference is not None:
            assert abs(reference - got) < 1e-6, \
                f"seed {seed}: brute force {reference:.4f} != solver {got:.4f}"
        checked += 1
    print(f"  brute-force cross-check ok ({checked} random instances, exact match)")


def test_solver_agreement():
    """Both solvers must reach the same objective value.

    IMPORTANT: this is the check that validates the PuLP path. The environment
    this module was written in had no network access to install PuLP, so the
    ILP branch has only been exercised by review, not execution. The first time
    this suite runs on a machine with `pip install -r requirements.txt` done,
    this test is what proves the two code paths agree -- if it fails there, the
    ILP formulation is wrong, not the fallback.
    """
    if not so.PULP_AVAILABLE:
        print("  solver agreement SKIPPED -- PuLP not installed in this "
              "environment (run this suite again once it is)")
        return
    df = so._prepare(XP)
    budget_tenths = 1000
    pulp_sel, _ = so._solve_squad_pulp(df, budget_tenths, so.MAX_PER_CLUB, [])
    bnb_sel, _ = so._solve_squad_branch_and_bound(df, budget_tenths, so.MAX_PER_CLUB, [])
    pulp_xp = float(df.loc[pulp_sel, "xp"].sum())
    bnb_xp = float(df.loc[bnb_sel, "xp"].sum())
    assert abs(pulp_xp - bnb_xp) < 1e-6, f"PuLP {pulp_xp} != branch-and-bound {bnb_xp}"
    assert len(pulp_sel) == so.SQUAD_SIZE
    print(f"  solver agreement ok (PuLP and branch-and-bound both {pulp_xp:.4f} xP)")


def test_exclude_and_must_include():
    top_id = XP.sort_values("xp", ascending=False)["id"].iloc[0]
    excluded = so.pick_squad(XP, exclude_ids=[top_id])
    assert str(top_id) not in set(excluded["squad"]["id"].astype(str))
    assert_valid_squad(excluded["squad"])

    # Force in a cheap fringe player who would not otherwise be picked.
    cheap = XP[(XP["position"] == "MID")].sort_values("xp").iloc[0]
    forced = so.pick_squad(XP, must_include_ids=[cheap["id"]])
    assert str(cheap["id"]) in set(forced["squad"]["id"].astype(str))
    assert_valid_squad(forced["squad"])
    print("  exclude / must_include ok")


def test_infeasible_budget_raises():
    try:
        so.pick_squad(XP, budget=30.0)
    except so.OptimizerError:
        print("  infeasible budget raises OptimizerError ok")
        return
    raise AssertionError("a 30.0m budget should be infeasible for 15 players")


def test_missing_columns_raise():
    try:
        so.pick_squad(XP.drop(columns=["xp"]))
    except so.OptimizerError as exc:
        assert "xp" in str(exc)
        print("  missing-column validation ok")
        return
    raise AssertionError("missing 'xp' column should raise")


# ---------------------------------------------------------------------------
# pick_starting_xi
# ---------------------------------------------------------------------------

def test_pick_starting_xi():
    squad = so.pick_squad(XP, budget=100.0)["squad"]
    xi_result = so.pick_starting_xi(squad)
    xi, bench = xi_result["starting_xi"], xi_result["bench"]

    assert len(xi) == so.XI_SIZE
    assert len(bench) == so.SQUAD_SIZE - so.XI_SIZE

    counts = xi["position"].value_counts().to_dict()
    assert counts.get("GKP", 0) == 1
    for pos in ("DEF", "MID", "FWD"):
        assert so.XI_MIN[pos] <= counts.get(pos, 0) <= so.XI_MAX[pos], counts

    # XI and bench partition the squad exactly.
    assert set(xi["id"]) | set(bench["id"]) == set(squad["id"])
    assert not (set(xi["id"]) & set(bench["id"]))

    # Bench is in auto-sub priority order.
    assert list(bench["xp"]) == sorted(bench["xp"], reverse=True)
    # ...and the reserve keeper is marked, since he can only replace the GK.
    assert bench["can_only_replace_gk"].sum() == 1

    # Captain / vice are the two best in the XI, and distinct.
    ranked = xi.sort_values("xp", ascending=False)
    assert xi_result["captain"]["id"] == ranked.iloc[0]["id"]
    assert xi_result["vice_captain"]["id"] == ranked.iloc[1]["id"]
    assert xi_result["captain"]["id"] != xi_result["vice_captain"]["id"]

    # Nobody on the bench outscores the worst starter at their own position
    # (an XI that leaves a better same-position player out is not optimal).
    for _, b in bench.iterrows():
        starters = xi[xi["position"] == b["position"]]
        if len(starters) > so.XI_MIN[b["position"]]:
            assert float(b["xp"]) <= float(starters["xp"].min()) + 1e-9

    # Objective bookkeeping: XI total + captain's xP counted twice.
    assert abs(xi_result["total_xp_with_captain"]
               - (xi_result["xi_xp"] + xi_result["captain"]["xp"])) < 1e-9
    print(f"  starting XI ok (formation {xi_result['formation']}, "
          f"captain {xi_result['captain']['web_name']}, "
          f"{xi_result['total_xp_with_captain']:.2f} xP with captaincy)")


def test_xi_is_best_over_all_formations():
    """Brute-force every legal formation and confirm nothing beats the pick."""
    squad = so.pick_squad(XP, budget=100.0)["squad"]
    chosen = so.pick_starting_xi(squad)

    by_pos = {p: squad[squad["position"] == p].sort_values("xp", ascending=False)
              for p in so.POSITION_ORDER}
    best = -1.0
    for d, mid, f in so.LEGAL_FORMATIONS:
        xi = pd.concat([by_pos["GKP"].head(1), by_pos["DEF"].head(d),
                        by_pos["MID"].head(mid), by_pos["FWD"].head(f)])
        best = max(best, float(xi["xp"].sum()) + float(xi["xp"].max()))
    assert abs(best - chosen["total_xp_with_captain"]) < 1e-9
    print(f"  XI optimality ok (best of {len(so.LEGAL_FORMATIONS)} legal formations)")


def test_xi_avoids_blank_gameweek_players():
    """Brighton blank in GW1, so a Brighton player must never start."""
    squad = so.pick_squad(XP, budget=100.0)["squad"]
    xi = so.pick_starting_xi(squad)["starting_xi"]
    assert (xi["xp"] > 0).all(), "a 0 xP (blanking) player was started"
    print("  XI avoids blanking players ok")


# ---------------------------------------------------------------------------
# optimize_transfers
# ---------------------------------------------------------------------------

def _deliberately_weak_squad() -> pd.DataFrame:
    """A legal but poor squad: the cheapest legal 15, ignoring xP.

    Built by maximising *negative* xP under the same constraints, which
    guarantees it is legal and leaves obvious upgrades on the table.
    """
    inverted = XP.copy()
    inverted["xp"] = -inverted["xp"]
    weak = so.pick_squad(inverted, budget=100.0)["squad"]
    # Restore true xP values.
    return weak.drop(columns=["xp"]).merge(XP[["id", "xp"]], on="id", how="left")


def test_transfer_finds_an_upgrade():
    weak = _deliberately_weak_squad()
    result = so.optimize_transfers(weak, XP, free_transfers=1,
                                   max_transfers_considered=2, bank=5.0)
    assert result["recommendation"] == "transfer"
    assert result["net_gain"] > 0
    assert 1 <= result["n_transfers"] <= 2
    assert len(result["transfers"]) == result["n_transfers"]

    for t in result["transfers"]:
        assert t["in"]["position"] == t["out"]["position"], "position must match"
        assert t["in"]["id"] != t["out"]["id"]

    assert_valid_squad(result["new_squad"], budget=100.0 + 5.0)
    assert result["bank_after"] >= -1e-9
    assert abs(result["squad_xp_after"] - result["squad_xp_before"]
               - result["gross_gain"]) < 1e-9
    print(f"  transfer search ok ({result['n_transfers']} transfer(s), "
          f"net +{result['net_gain']:.2f} xP, {result['evaluated']} options checked)")


def test_no_transfer_when_squad_already_optimal():
    """The optimal squad should not be told to churn itself for a -4."""
    best = so.pick_squad(XP, budget=100.0)["squad"]
    result = so.optimize_transfers(best, XP, free_transfers=1, bank=0.0)
    assert result["recommendation"] == "no_transfer", result["transfers"]
    assert result["n_transfers"] == 0
    assert result["net_gain"] == 0.0
    print("  no-transfer recommendation ok")


def test_never_recommends_negative_net_value():
    """Whatever it returns, the net gain must be strictly positive."""
    weak = _deliberately_weak_squad()
    for free in (0, 1, 2):
        for max_t in (1, 2):
            res = so.optimize_transfers(weak, XP, free_transfers=free,
                                        max_transfers_considered=max_t, bank=0.0)
            if res["recommendation"] == "transfer":
                assert res["net_gain"] > 0, (free, max_t, res["net_gain"])
                expected_hit = 4 * max(0, res["n_transfers"] - free)
                assert abs(res["hit_cost_total"] - expected_hit) < 1e-9
                assert abs(res["net_gain"]
                           - (res["gross_gain"] - expected_hit)) < 1e-9
            else:
                assert res["net_gain"] == 0.0
    print("  hit-cost accounting ok (never a negative-value transfer)")


def test_hit_cost_suppresses_marginal_transfers():
    """A big hit cost must turn a marginal upgrade into 'hold'."""
    weak = _deliberately_weak_squad()
    cheap_hits = so.optimize_transfers(weak, XP, free_transfers=1, hit_cost=0)
    dear_hits = so.optimize_transfers(weak, XP, free_transfers=0, hit_cost=1000)
    assert cheap_hits["recommendation"] == "transfer"
    assert dear_hits["recommendation"] == "no_transfer"
    print("  hit-cost sensitivity ok")


def test_unlimited_transfers_chip_hook():
    """The Wildcard/Free Hit hook must zero the hit cost (timing is not our job)."""
    weak = _deliberately_weak_squad()
    limited = so.optimize_transfers(weak, XP, free_transfers=0,
                                    max_transfers_considered=2, hit_cost=4)
    wildcard = so.optimize_transfers(weak, XP, free_transfers=0,
                                     max_transfers_considered=2, hit_cost=4,
                                     unlimited_transfers=True)
    assert wildcard["hit_cost_total"] == 0.0
    assert wildcard["net_gain"] >= limited["net_gain"] - 1e-9
    print("  unlimited_transfers (chip hook) ok")


def test_transfer_respects_budget_and_club_cap():
    weak = _deliberately_weak_squad()
    result = so.optimize_transfers(weak, XP, free_transfers=2,
                                   max_transfers_considered=2, bank=0.0)
    if result["recommendation"] == "transfer":
        new = result["new_squad"]
        assert_valid_squad(new, budget=float(weak["price_m"].sum()) + 1e-6)
        assert new["team_name"].value_counts().max() <= so.MAX_PER_CLUB
    print("  transfer legality ok")


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------

def run_demo():
    """xp_model -> pick_squad -> pick_starting_xi -> optimize_transfers."""
    print("\n" + "=" * 74)
    print("END-TO-END DEMO  --  synthetic data, gameweek 1")
    print("=" * 74)

    xp = m.calculate_xp_for_gameweek(PLAYERS, TEAMS, FIXTURES, gameweek=1)
    print(f"\nxP model: scored {len(xp)} players "
          f"({int((xp['n_fixtures'] == 0).sum())} blank, "
          f"{int((xp['n_fixtures'] == 2).sum())} double gameweek)")
    print("\nTop 8 by expected points:")
    print(xp[["web_name", "team_name", "position", "price_m", "n_fixtures", "xp"]]
          .head(8).to_string(index=False,
                             formatters={"xp": "{:.2f}".format,
                                         "price_m": "{:.1f}".format}))

    result = so.pick_squad(xp, budget=100.0)
    squad = result["squad"]
    print(f"\nOptimal 15  --  solver: {result['solver']}")
    print(f"  cost {result['total_cost']}m of 100.0m "
          f"(bank {result['remaining_budget']}m), total {result['total_xp']:.2f} xP")

    xi_result = so.pick_starting_xi(squad)
    xi = xi_result["starting_xi"]
    bench = xi_result["bench"]
    starting_ids = set(xi["id"])

    print(f"\nStarting XI  --  formation {xi_result['formation']}")
    for pos in so.POSITION_ORDER:
        for _, row in squad[squad["position"] == pos].iterrows():
            if row["id"] not in starting_ids:
                continue
            tag = ""
            if row["id"] == xi_result["captain"]["id"]:
                tag = "  (C)"
            elif row["id"] == xi_result["vice_captain"]["id"]:
                tag = "  (V)"
            print(f"  {pos}  {row['web_name']:<12} {row['team_name']:<11} "
                  f"{row['price_m']:>5.1f}m  {row['xp']:>5.2f} xP{tag}")

    print("\nBench (auto-sub order):")
    for _, row in bench.iterrows():
        note = "  [reserve GK]" if row["can_only_replace_gk"] else ""
        print(f"  {int(row['bench_order'])}. {row['position']}  "
              f"{row['web_name']:<12} {row['team_name']:<11} "
              f"{row['price_m']:>5.1f}m  {row['xp']:>5.2f} xP{note}")

    print(f"\n  XI xP                    {xi_result['xi_xp']:.2f}")
    print(f"  captain ({xi_result['captain']['web_name']}) doubled  "
          f"{xi_result['total_xp_with_captain']:.2f}")
    print(f"  bench xP (dead weight)   {xi_result['bench_xp']:.2f}")

    clubs = squad["team_name"].value_counts()
    print(f"\n  club spread: {clubs.to_dict()} (cap {so.MAX_PER_CLUB})")

    transfer = so.optimize_transfers(squad, xp, free_transfers=1)
    print(f"\nTransfer check on the optimal squad: {transfer['recommendation']}")
    print(f"  {transfer['notes']}")

    weak = _deliberately_weak_squad()
    weak_transfer = so.optimize_transfers(weak, xp, free_transfers=1,
                                          max_transfers_considered=2, bank=5.0)
    print(f"\nTransfer check on a deliberately bad squad "
          f"({float(weak['xp'].sum()):.2f} xP): {weak_transfer['recommendation']}")
    for t in weak_transfer["transfers"]:
        print(f"  OUT {t['out']['web_name']:<12} ({t['out']['xp']:.2f} xP, "
              f"{t['out']['price_m']:.1f}m)   "
              f"IN {t['in']['web_name']:<12} ({t['in']['xp']:.2f} xP, "
              f"{t['in']['price_m']:.1f}m)")
    print(f"  {weak_transfer['notes']}")
    print("=" * 74)


def run_tests():
    test_pick_squad_respects_all_constraints()
    test_tighter_budget_is_feasible_and_worse()
    test_club_cap_binds()
    test_optimality_no_improving_swap_exists()
    test_matches_brute_force_on_random_instances()
    test_solver_agreement()
    test_exclude_and_must_include()
    test_infeasible_budget_raises()
    test_missing_columns_raise()
    test_pick_starting_xi()
    test_xi_is_best_over_all_formations()
    test_xi_avoids_blank_gameweek_players()
    test_transfer_finds_an_upgrade()
    test_no_transfer_when_squad_already_optimal()
    test_never_recommends_negative_net_value()
    test_hit_cost_suppresses_marginal_transfers()
    test_unlimited_transfers_chip_hook()
    test_transfer_respects_budget_and_club_cap()
    print("All squad optimizer tests passed.")


if __name__ == "__main__":
    run_tests()
    run_demo()
