"""
Tests for agents/ai_manager.py. Standalone plain-assert style, matching
the rest of this project's test files.

The real season hadn't started yet as of writing (GW1 deadline still
ahead) and this dev sandbox has no live network access anyway, so the
scoring/auto-sub/chip logic can't be exercised against real results --
these tests build synthetic finished-gameweek data instead (same
approach already used elsewhere in this project, e.g. the OCR pipeline's
synthetic-image test).

`test_advance_end_to_end_with_synthetic_results` is the one integration
test that imports the real webapp Pool (rather than a hand-built one) --
deliberate: ai_manager's real contract is "operates on whatever
POOL.players/POOL.buyable() produce," so testing against that real shape
IS the correct test, not an architecture smell.

Run with: python3 agents/test_ai_manager.py
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in (".", "../optimization", "../data", "../models"):
    sys.path.insert(0, os.path.join(_HERE, _sub))
import ai_manager  # noqa: E402
import squad_optimizer  # noqa: E402


# ---------------------------------------------------------------------------
# _formation_legal
# ---------------------------------------------------------------------------

def test_formation_legal_boundaries():
    assert ai_manager._formation_legal({"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2}) is True  # 3-5-2
    assert ai_manager._formation_legal({"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2}) is True  # 5-3-2
    assert ai_manager._formation_legal({"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}) is True  # 4-4-2
    assert ai_manager._formation_legal({"GKP": 0, "DEF": 4, "MID": 4, "FWD": 2}) is False  # no GK
    assert ai_manager._formation_legal({"GKP": 1, "DEF": 2, "MID": 5, "FWD": 3}) is False  # DEF < 3
    assert ai_manager._formation_legal({"GKP": 1, "DEF": 4, "MID": 1, "FWD": 5}) is False  # MID < 2, FWD > 3
    print("  _formation_legal boundaries ok")


# ---------------------------------------------------------------------------
# _apply_auto_subs
# ---------------------------------------------------------------------------

def _positions_4_4_2():
    """1 GK, 4 DEF, 4 MID, 2 FWD starting XI (ids 1-11), bench = 12 (GK),
    13 (DEF), 14 (MID), 15 (FWD), in that auto-sub order."""
    starting = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    bench = [12, 13, 14, 15]
    positions = {
        1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "DEF",
        6: "MID", 7: "MID", 8: "MID", 9: "MID", 10: "FWD", 11: "FWD",
        12: "GKP", 13: "DEF", 14: "MID", 15: "FWD",
    }
    return starting, bench, positions


def test_auto_sub_replaces_blank_outfield_starter():
    starting, bench, positions = _positions_4_4_2()
    # Player 10 (FWD) blanks. Bench order is [12 GK, 13 DEF, 14 MID, 15 FWD]
    # -- real FPL auto-subs go by BENCH ORDER, not matching position, so the
    # first outfield bench player who keeps the formation legal comes in.
    # Swapping FWD 10 for DEF 13 gives 5-4-1 (DEF 5, MID 4, FWD 1), which is
    # legal, so bench player 13 -- not the same-position 15 -- is the one
    # that's actually subbed on.
    gw_stats = {pid: {"points": 4, "minutes": 90} for pid in starting + bench}
    gw_stats[10] = {"points": 0, "minutes": 0}
    result = ai_manager._apply_auto_subs(starting, bench, gw_stats, positions)
    assert 10 not in result["final_xi_ids"]
    assert 13 in result["final_xi_ids"]
    assert result["subs_applied"] == [{"out": 10, "in": 13}]
    print("  auto-sub replaces a blank outfield starter with the first legal bench player in order ok")


def test_auto_sub_gk_only_replaced_by_bench_gk():
    starting, bench, positions = _positions_4_4_2()
    gw_stats = {pid: {"points": 4, "minutes": 90} for pid in starting + bench}
    gw_stats[1] = {"points": 0, "minutes": 0}  # starting GK blanks
    result = ai_manager._apply_auto_subs(starting, bench, gw_stats, positions)
    assert 1 not in result["final_xi_ids"]
    assert 12 in result["final_xi_ids"]  # bench GK comes in, not an outfield bench player
    print("  auto-sub: starting GK can only be replaced by the bench GK ok")


def test_auto_sub_skips_when_no_legal_replacement():
    starting, bench, positions = _positions_4_4_2()
    gw_stats = {pid: {"points": 4, "minutes": 90} for pid in starting + bench}
    gw_stats[10] = {"points": 0, "minutes": 0}   # FWD blanks
    # None of the outfield bench played -> no candidate is even eligible,
    # regardless of formation legality.
    gw_stats[13] = {"points": 0, "minutes": 0}
    gw_stats[14] = {"points": 0, "minutes": 0}
    gw_stats[15] = {"points": 0, "minutes": 0}
    result = ai_manager._apply_auto_subs(starting, bench, gw_stats, positions)
    assert 10 in result["final_xi_ids"]  # stays in, scores 0 -- real FPL behavior
    assert result["subs_applied"] == []
    print("  auto-sub: leaves a blank starter in place when no legal replacement exists ok")


# ---------------------------------------------------------------------------
# _score_gameweek
# ---------------------------------------------------------------------------

def test_score_gameweek_captain_fallback_to_vice():
    starting, bench, positions = _positions_4_4_2()
    entry = {
        "squad_ids": starting + bench, "starting_xi_ids": starting, "bench_order_ids": bench,
        "captain_id": 10, "vice_captain_id": 6, "chip_played": None,
    }
    gw_stats = {pid: {"points": 4, "minutes": 90} for pid in starting + bench}
    gw_stats[10] = {"points": 0, "minutes": 0}   # captain (FWD) blanks
    gw_stats[6] = {"points": 9, "minutes": 90}   # vice-captain has a big game
    scored = ai_manager._score_gameweek(entry, gw_stats, positions)
    # Player 10 blanking triggers an auto-sub (bench player 13, DEF, is next
    # in bench order and keeps the formation legal as 5-4-1) -- so player 10
    # isn't in the final XI at all, and the captain armband falls to the
    # vice (6), who IS in the final XI and played.
    # Final XI: 1,2,3,4,5,6,7,8,9,11,13 -> 10 players @ 4pts + player 6 @ 9pts = 49,
    # plus the vice-captain's own points counted again for the multiplier -> +9.
    assert 10 not in scored["final_xi_ids"]
    assert 13 in scored["final_xi_ids"]
    assert scored["captain_used_id"] == 6
    assert scored["points"] == 49 + 9
    print("  _score_gameweek: captain multiplier falls back to vice-captain when captain blanks ok")


def test_score_gameweek_bench_boost_counts_all_fifteen():
    starting, bench, positions = _positions_4_4_2()
    entry = {
        "squad_ids": starting + bench, "starting_xi_ids": starting, "bench_order_ids": bench,
        "captain_id": 1, "vice_captain_id": 2, "chip_played": "bench_boost",
    }
    gw_stats = {pid: {"points": 5, "minutes": 90} for pid in starting + bench}
    scored = ai_manager._score_gameweek(entry, gw_stats, positions)
    # 15 players * 5 + captain (1) extra 5 = 80
    assert scored["points"] == 15 * 5 + 5
    print("  _score_gameweek: Bench Boost counts all 15 players' points ok")


def test_score_gameweek_triple_captain_triples_not_doubles():
    starting, bench, positions = _positions_4_4_2()
    entry = {
        "squad_ids": starting + bench, "starting_xi_ids": starting, "bench_order_ids": bench,
        "captain_id": 1, "vice_captain_id": 2, "chip_played": "triple_captain",
    }
    gw_stats = {pid: {"points": 5, "minutes": 90} for pid in starting + bench}
    gw_stats[1] = {"points": 10, "minutes": 90}
    scored = ai_manager._score_gameweek(entry, gw_stats, positions)
    # XI points: 10 (captain) + 10*5 (rest of the 11) = 60, plus captain extra *2 (for x3 total) = 20 -> 80
    xi_total = 10 + 10 * 5
    assert scored["points"] == xi_total + 10 * 2
    print("  _score_gameweek: Triple Captain triples (not doubles) the captain's own points ok")


# ---------------------------------------------------------------------------
# End-to-end: real pool, synthetic gameweek results
# ---------------------------------------------------------------------------

def test_advance_end_to_end_with_synthetic_results():
    webapp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webapp")
    sys.path.insert(0, webapp_dir)
    import app as webapp_app  # noqa: E402  (imports the real Pool-building code)

    # Use an isolated log file so this test never touches the real season log.
    test_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed",
        "ai_manager_log.TEST.json")
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    with patch.object(ai_manager, "LOG_PATH", test_log_path):
        webapp_app.POOL.refresh()
        full_pool = webapp_app.POOL.players
        buyable_pool = webapp_app.POOL.buyable()

        log = ai_manager.seed_log(buyable_pool)
        gw1_ids = log["history"][0]["squad_ids"]
        assert len(gw1_ids) == 15

        # Synthetic GW1 results: everyone plays a solid 90 minutes / 4 points,
        # except the bench goalkeeper (didn't play -- irrelevant to the total)
        # and the designated captain, who blanks -- this exercises the
        # captain -> vice-captain fallback against the REAL seeded squad.
        captain_id = log["history"][0]["captain_id"]
        synthetic_stats = {pid: {"points": 4, "minutes": 90} for pid in gw1_ids}
        synthetic_stats[captain_id] = {"points": 0, "minutes": 0}

        fake_bootstrap = {"events": [
            {"id": 1, "finished": True, "is_current": False, "is_next": False},
            {"id": 2, "finished": False, "is_current": True, "is_next": False},
        ]}

        with patch.object(ai_manager, "_fetch_gw_stats_for_players",
                          return_value=synthetic_stats):
            result_log = ai_manager.advance(full_pool, buyable_pool, fake_bootstrap)

        gw1_entry = result_log["history"][0]
        assert gw1_entry["points"] is not None
        assert gw1_entry["captain_used_id"] != captain_id  # fell back to vice (or stayed None)
        # 15 squad members, 10 outfield starters unaffected + subs/GK nuance --
        # just sanity-check it's a plausible, non-negative, non-huge total.
        assert 0 <= gw1_entry["points"] <= 200

        assert len(result_log["history"]) == 2
        gw2_entry = result_log["history"][1]
        assert gw2_entry["gameweek"] == 2
        assert gw2_entry["points"] is None
        assert len(gw2_entry["squad_ids"]) == 15
        assert gw2_entry["chip_played"] in (None, "wildcard", "free_hit", "bench_boost", "triple_captain")
        assert gw2_entry["free_transfers_after"] in (1, 2)

        # Calling advance() again with nothing new should be a clean no-op.
        result_log_2 = ai_manager.advance(full_pool, buyable_pool, fake_bootstrap)
        assert result_log_2 == result_log

    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    print("  advance(): end-to-end synthetic-results catch-up (real pool) ok")


def run_tests():
    test_formation_legal_boundaries()
    test_auto_sub_replaces_blank_outfield_starter()
    test_auto_sub_gk_only_replaced_by_bench_gk()
    test_auto_sub_skips_when_no_legal_replacement()
    test_score_gameweek_captain_fallback_to_vice()
    test_score_gameweek_bench_boost_counts_all_fifteen()
    test_score_gameweek_triple_captain_triples_not_doubles()
    test_advance_end_to_end_with_synthetic_results()
    print("All ai_manager tests passed.")


if __name__ == "__main__":
    run_tests()
