"""
Chip deadline tracking -- the one genuinely deterministic part of chip
strategy, split out of the chip strategy agent's job so it isn't left to
qualitative judgement.

WHY THIS EXISTS
Per Neil: "I think the chips reset at some point through the season, just
make sure to use all chips before they reset." Confirmed live (see sources
below): 2026/27 gives one Wildcard, one Free Hit, one Bench Boost, and one
Triple Captain for the FIRST half of the season (GW1-19), and a second,
completely fresh set of all four for the second half (GW20-38). Unused
first-half chips are LOST at the GW19 deadline, not carried over -- this is
the one hard, use-it-or-lose-it fact in an otherwise judgement-heavy area,
and exactly the kind of thing that belongs in code rather than being
re-derived by a research agent's memory every week.

WHAT THIS DOES NOT DO
It has no opinion on WHEN in a half to use a chip -- that's a real
football-strategy judgement (a double gameweek, a brutal fixture swing, a
squad crisis) and stays the chip strategy agent's job. This module only
answers "how many gameweeks are left before an unused chip is gone."

Sources: premierleague.com's official 2026/27 chip rules confirm the GW19
first-half deadline (13:30 GMT, Sat 2 Jan) and that first-half chips do not
carry over -- see position_agent_specs.md section 6 for the citation.
"""

from __future__ import annotations

CHIP_NAMES = ("wildcard", "free_hit", "bench_boost", "triple_captain")

# (first gameweek, last gameweek) of each half's chip window. Both halves
# give the same four chips; GW19 is the hard deadline for the first set,
# GW38 for the second (end of season -- nothing to roll forward to).
CHIP_WINDOWS = {
    "first_half": (1, 19),
    "second_half": (20, 38),
}

# Below this many gameweeks remaining in the current half, an unused chip
# is flagged "urgent" rather than just "plan soon". Judgment call, not a
# rule from the game itself -- 3 gameweeks is roughly enough runway to spot
# and act on a good fixture swing without leaving it to the last minute.
URGENT_GAMEWEEKS_REMAINING = 3
PLAN_SOON_GAMEWEEKS_REMAINING = 6


def current_half(gameweek: int) -> str:
    """Which chip half a gameweek falls in. Raises on an out-of-range GW
    rather than silently guessing -- a bad gameweek number here should be
    loud, not produce a confidently wrong deadline."""
    for half, (start, end) in CHIP_WINDOWS.items():
        if start <= gameweek <= end:
            return half
    raise ValueError(f"gameweek {gameweek} is outside both chip windows "
                     f"{CHIP_WINDOWS} -- check the season's actual length")


def chip_status(gameweek: int, chips_used_this_half: set[str] | None = None) -> dict:
    """Status of every chip as of `gameweek`, for the half it falls in.

    Args:
        gameweek: the current (or upcoming) gameweek.
        chips_used_this_half: names (from CHIP_NAMES) already played in
            THIS half. Chips used in the other half don't apply here --
            each half's allowance is independent and doesn't carry over.

    Returns:
        {
            "half": "first_half" | "second_half",
            "deadline_gameweek": last GW of this half (chip must be played
                on or before this GW -- GW19 or GW38),
            "gameweeks_remaining": inclusive count from `gameweek` to the
                deadline,
            "chips": {chip_name: {"used": bool, "urgency": str}},
        }

    `urgency` is one of "used" (already played this half), "hold" (plenty
    of runway left, no reason to force a decision), "plan_soon" (within
    PLAN_SOON_GAMEWEEKS_REMAINING of the deadline -- start actively
    looking for a fixture to use it on), or "urgent" (within
    URGENT_GAMEWEEKS_REMAINING -- use it or lose it).
    """
    chips_used_this_half = chips_used_this_half or set()
    unknown = chips_used_this_half - set(CHIP_NAMES)
    if unknown:
        raise ValueError(f"unknown chip name(s): {unknown} -- expected one of {CHIP_NAMES}")

    half = current_half(gameweek)
    _, deadline_gw = CHIP_WINDOWS[half]
    gameweeks_remaining = deadline_gw - gameweek + 1

    chips = {}
    for name in CHIP_NAMES:
        if name in chips_used_this_half:
            chips[name] = {"used": True, "urgency": "used"}
            continue
        if gameweeks_remaining <= URGENT_GAMEWEEKS_REMAINING:
            urgency = "urgent"
        elif gameweeks_remaining <= PLAN_SOON_GAMEWEEKS_REMAINING:
            urgency = "plan_soon"
        else:
            urgency = "hold"
        chips[name] = {"used": False, "urgency": urgency}

    return {
        "half": half,
        "deadline_gameweek": deadline_gw,
        "gameweeks_remaining": gameweeks_remaining,
        "chips": chips,
    }


__all__ = ["CHIP_NAMES", "CHIP_WINDOWS", "URGENT_GAMEWEEKS_REMAINING",
          "PLAN_SOON_GAMEWEEKS_REMAINING", "current_half", "chip_status"]
