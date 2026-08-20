"""
Tests for agents/chip_strategy.py. Standalone plain-assert style, matching
the rest of this project's test files.

Run with: python3 agents/test_chip_strategy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chip_strategy as cs  # noqa: E402


def test_current_half_boundaries():
    assert cs.current_half(1) == "first_half"
    assert cs.current_half(19) == "first_half"
    assert cs.current_half(20) == "second_half"
    assert cs.current_half(38) == "second_half"
    try:
        cs.current_half(39)
        raise AssertionError("expected a ValueError for an out-of-range gameweek")
    except ValueError:
        pass
    try:
        cs.current_half(0)
        raise AssertionError("expected a ValueError for gameweek 0")
    except ValueError:
        pass
    print("  current_half boundaries ok (GW1/19 -> first_half, GW20/38 -> second_half)")


def test_chip_status_unused_chip_urgency_escalates():
    """An unused chip's urgency should move from hold -> plan_soon -> urgent
    purely as a function of how close the half's deadline is -- no chips
    used in any of these calls, so this isolates the timing logic."""
    far = cs.chip_status(gameweek=5)  # GW19 deadline, 15 GWs away
    assert far["gameweeks_remaining"] == 15
    assert all(c["urgency"] == "hold" for c in far["chips"].values())

    soon = cs.chip_status(gameweek=15)  # 5 GWs away
    assert soon["gameweeks_remaining"] == 5
    assert all(c["urgency"] == "plan_soon" for c in soon["chips"].values())

    urgent = cs.chip_status(gameweek=18)  # 2 GWs away
    assert urgent["gameweeks_remaining"] == 2
    assert all(c["urgency"] == "urgent" for c in urgent["chips"].values())

    # The deadline gameweek itself is still usable -- 1 gameweek remaining,
    # not zero or negative.
    last_chance = cs.chip_status(gameweek=19)
    assert last_chance["gameweeks_remaining"] == 1
    assert all(c["urgency"] == "urgent" for c in last_chance["chips"].values())

    print(f"  chip_status urgency escalation ok "
          f"(GW5: hold, GW15: plan_soon, GW18/19: urgent)")


def test_chip_status_used_chips_marked_used_not_urgent():
    """A chip already played this half must show used=True regardless of
    how close the deadline is -- it shouldn't also get flagged urgent."""
    status = cs.chip_status(gameweek=18, chips_used_this_half={"wildcard", "bench_boost"})
    assert status["chips"]["wildcard"]["used"] is True
    assert status["chips"]["wildcard"]["urgency"] == "used"
    assert status["chips"]["bench_boost"]["used"] is True
    # The two NOT used should still show urgent, since GW18 is 2 GWs from
    # the GW19 deadline.
    assert status["chips"]["free_hit"]["used"] is False
    assert status["chips"]["free_hit"]["urgency"] == "urgent"
    assert status["chips"]["triple_captain"]["urgency"] == "urgent"
    print("  chip_status used-chip tracking ok (used chips don't also show urgent)")


def test_chip_status_halves_are_independent():
    """Chips used in the first half must not carry over or affect the
    second half's status -- each half's allowance is genuinely separate,
    not a rolling budget."""
    used_in_first_half = {"wildcard"}
    second_half_status = cs.chip_status(gameweek=25, chips_used_this_half=used_in_first_half)
    # Passing first-half usage into a second-half gameweek is a caller
    # error in practice (the caller should track each half separately),
    # but the function's OWN behaviour should still be internally correct
    # for whatever set it's given -- verifying deadline_gameweek itself
    # switches to GW38, not GW19, is the real check here.
    assert second_half_status["half"] == "second_half"
    assert second_half_status["deadline_gameweek"] == 38
    assert second_half_status["gameweeks_remaining"] == 14
    print("  chip_status half-independence ok (GW25 resolves to second_half, deadline GW38)")


def test_chip_status_rejects_unknown_chip_name():
    try:
        cs.chip_status(gameweek=5, chips_used_this_half={"not_a_real_chip"})
        raise AssertionError("expected a ValueError for an unknown chip name")
    except ValueError as exc:
        assert "not_a_real_chip" in str(exc)
    print("  chip_status unknown-chip-name rejection ok")


def run_tests():
    test_current_half_boundaries()
    test_chip_status_unused_chip_urgency_escalates()
    test_chip_status_used_chips_marked_used_not_urgent()
    test_chip_status_halves_are_independent()
    test_chip_status_rejects_unknown_chip_name()
    print("All chip_strategy tests passed.")


if __name__ == "__main__":
    run_tests()
