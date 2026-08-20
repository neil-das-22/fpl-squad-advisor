"""
AI Manager -- an autonomous FPL manager built entirely on this project's
own tested model (`squad_optimizer`, `chip_strategy`), making its own
real, committed transfer and chip decisions gameweek by gameweek as the
season actually progresses. Powers the website's "AI Performance" page.

HOW THIS WORKS (real-manager mode, not a live recalculated snapshot)
Its season starts from the exact squad in `reports/gw1_recommendation.md`
-- the one human judgment call in that report (forcing Haaland in) is
baked into the opening squad; every decision from gameweek 2 onward is
the model acting completely alone. State is persisted to
`data/processed/ai_manager_log.json`: one entry per gameweek, each
holding the squad/captain/chip/transfer decision that was ACTUALLY LOCKED
IN for that week (using whatever the model projected at decision time),
plus the real points once that gameweek finishes. Nothing is ever
re-decided in hindsight.

Advancing only happens when `advance()` is called (the AI Performance
page calls it on every load) AND the app's cached FPL data has actually
been refreshed since that gameweek finished (see `_refresh_raw_data()`
in webapp/app.py) -- this module never fetches bootstrap data itself,
only per-player gameweek history for the ~15 players it needs to score.

CHIP TIMING -- HONEST CAVEAT
`squad_optimizer.py` and `chip_strategy.py` deliberately do NOT decide
WHEN to use a chip (see both modules' docstrings) -- that was always left
to a human/agent judgment call. This module adds a small set of
explicit, documented threshold heuristics (see the constants below) so
the AI can act autonomously. These are illustrative rules of thumb, not
competitive-level FPL strategy -- treat the chip decisions here as "a
real automated choice, honestly labelled," not "optimal chip strategy."

AUTO-SUBS AND CAPTAIN FALLBACK
When a gameweek is scored, a starter with 0 minutes is automatically
replaced by the first eligible bench player (in bench order) who kept
the resulting XI legal -- reserve GK can only replace the starting GK,
same as real FPL. If the captain ends up with 0 minutes, the multiplier
falls back to the vice-captain, same as real FPL.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import chip_strategy
import fpl_client
import squad_optimizer

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ai_manager_log.json")

SEASON_END_GW = 38

# The exact GW1 squad from reports/gw1_recommendation.md. Matched onto the
# live pool by `web_name` at seed time -- these are FPL's own web_names,
# not full names (e.g. Van Dijk's is "Virgil", already a known gotcha in
# this project -- see fpl_client.py's DEF override comment).
GW1_STARTING_XI_WEB_NAMES = [
    "Petrović",
    "Virgil", "Lacroix", "Guéhi",
    "Enzo", "Gakpo", "Szoboszlai", "Schade",
    "Haaland", "Thiago", "João Pedro",
]
GW1_BENCH_WEB_NAMES = ["Shaw", "Amad", "Kayode", "Dubravka"]  # auto-sub order
GW1_CAPTAIN_WEB_NAME = "Haaland"
# FPL's own web_name for Enzo Fernández is just "Enzo" -- it changed
# (shortened) since reports/gw1_recommendation.md was written, caught by
# `_match_by_web_name`'s loud-failure check rather than silently mismatching.
GW1_VICE_CAPTAIN_WEB_NAME = "Enzo"
GW1_NOTE = (
    "Season-opening squad -- the model's own GW1 recommendation "
    "(reports/gw1_recommendation.md). One human judgment call is baked "
    "in here: Haaland was force-included over the pure optimizer's pick "
    "(cost: ~4.65 projected points over GW1-4, per the report). Every "
    "decision from gameweek 2 onward is the model acting alone."
)

# Chips are mechanically blocked in GW1 (can't Wildcard/Free Hit a squad
# you haven't picked yet) and the report itself found no signal yet to
# justify Bench Boost/Triple Captain that early -- gating all four
# uniformly from GW2 is a deliberate, documented simplification.
CHIP_ELIGIBLE_FROM_GW = 2

# Single-gameweek (not cumulative) projected xP thresholds for playing a
# chip on its own merits, before the chip-status "urgent" deadline forces
# a use-it-or-lose-it decision on whatever's best available at the time.
TRIPLE_CAPTAIN_XP_THRESHOLD = 8.0
BENCH_BOOST_XP_THRESHOLD = 10.0
WILDCARD_GAIN_THRESHOLD = 15.0   # cumulative 4-GW xP gain from a full rebuild
FREE_HIT_DEFICIT_THRESHOLD = 10.0  # single-GW xP shortfall vs. a fresh one-week squad


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _to_native(v: Any) -> Any:
    """numpy/pandas scalars -> plain Python, so json.dump never chokes."""
    return v.item() if hasattr(v, "item") else v


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _load_log() -> dict | None:
    if not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH) as f:
        return json.load(f)


def _save_log(log: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Seeding the season
# ---------------------------------------------------------------------------

def _projected_points_for_selection(pool_df: pd.DataFrame, starting_ids: list[int],
                                    bench_ids: list[int], captain_id: int,
                                    chip: str | None) -> float:
    """Single-gameweek (xp_gw1) projected total for a starting XI plus the
    captain multiplier and any chip effect -- the number the AI is
    "predicting" for that gameweek, stored on the entry at decision time so
    it can be compared against the real points once the gameweek finishes
    (see the "accuracy" card on the AI Performance page)."""
    def xp_gw1_sum(ids: list[int]) -> float:
        sub = pool_df[pool_df["id"].isin(ids)]
        return float(pd.to_numeric(sub["xp_gw1"], errors="coerce").fillna(0.0).sum())

    captain_row = pool_df[pool_df["id"] == captain_id]
    captain_xp = (
        float(pd.to_numeric(captain_row["xp_gw1"], errors="coerce").fillna(0.0).iloc[0])
        if not captain_row.empty else 0.0
    )

    total = xp_gw1_sum(starting_ids) + captain_xp  # captain's xp counted twice
    if chip == "triple_captain":
        total += captain_xp  # ...three times total
    if chip == "bench_boost":
        total += xp_gw1_sum(bench_ids)
    return round(total, 2)


def _match_by_web_name(pool_df: pd.DataFrame, names: list[str]) -> list[int]:
    ids = []
    missing = []
    for name in names:
        match = pool_df[pool_df["web_name"] == name]
        if match.empty:
            missing.append(name)
        else:
            ids.append(_to_native(match.iloc[0]["id"]))
    if missing:
        raise ValueError(
            f"AI manager seed squad: couldn't find {missing} in the live pool by "
            "web_name -- FPL's web_name for a player can change (transfer, "
            "spelling fix); check reports/gw1_recommendation.md against the "
            "current pool and update GW1_*_WEB_NAMES in ai_manager.py."
        )
    return ids


def seed_log(buyable_pool_df: pd.DataFrame) -> dict:
    """Build and persist the season-opening log entry from the fixed GW1
    squad above. Only ever called once -- if a log already exists,
    `advance()` loads it instead."""
    starting_ids = _match_by_web_name(buyable_pool_df, GW1_STARTING_XI_WEB_NAMES)
    bench_ids = _match_by_web_name(buyable_pool_df, GW1_BENCH_WEB_NAMES)
    captain_id = _match_by_web_name(buyable_pool_df, [GW1_CAPTAIN_WEB_NAME])[0]
    vice_id = _match_by_web_name(buyable_pool_df, [GW1_VICE_CAPTAIN_WEB_NAME])[0]
    projected_points = _projected_points_for_selection(
        buyable_pool_df, starting_ids, bench_ids, captain_id, None)

    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chips_used": {"first_half": [], "second_half": []},
        "history": [
            {
                "gameweek": 1,
                "squad_ids": starting_ids + bench_ids,
                "starting_xi_ids": starting_ids,
                "bench_order_ids": bench_ids,
                "captain_id": captain_id,
                "vice_captain_id": vice_id,
                "chip_played": None,
                "transfers": [],
                "bank_after": 0.0,
                "free_transfers_after": 1,
                "points": None,
                "projected_points": projected_points,
                "note": GW1_NOTE,
            }
        ],
    }
    _save_log(log)
    return log


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def _squad_df_from_ids(full_pool_df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """Looks up HELD players against the full pool (not the buyable/
    research-filtered one) -- a player the AI already owns should stay
    trackable even if a later research pass flags them "avoid"; that
    filter only matters for who's a legal NEW signing."""
    sub = full_pool_df[full_pool_df["id"].isin(ids)].copy()
    missing = set(ids) - set(sub["id"].tolist())
    if missing:
        raise ValueError(
            f"AI manager: squad references player id(s) no longer in the pool: "
            f"{missing} (likely removed from the Premier League / bad data)."
        )
    return sub


def _positions_by_id(full_pool_df: pd.DataFrame, ids: list[int]) -> dict[int, str]:
    sub = _squad_df_from_ids(full_pool_df, ids)
    return dict(zip(sub["id"], sub["position"]))


# ---------------------------------------------------------------------------
# Scoring a finished gameweek
# ---------------------------------------------------------------------------

def _fetch_gw_stats_for_players(player_ids: list[int], gw: int) -> dict[int, dict] | None:
    """Real per-gameweek points/minutes for exactly the players who need
    it (typically 15), via FPL's per-player history endpoint -- not the
    bootstrap snapshot's `event_points` field, which only ever reflects
    the SINGLE most-recently-finished gameweek, not arbitrary past ones.
    Returns None (meaning "can't score yet, try again later") on ANY
    failure -- a partial/wrong score is worse than staying pending."""
    stats: dict[int, dict] = {}
    for pid in player_ids:
        try:
            summary = fpl_client.fetch_player_summary(int(pid))
        except Exception:  # noqa: BLE001 -- no network, rate limit, etc.
            return None
        row = next((h for h in summary.get("history", []) if h.get("round") == gw), None)
        if row is None:
            return None
        stats[pid] = {
            "points": int(row.get("total_points", 0)),
            "minutes": int(row.get("minutes", 0)),
        }
    return stats


def _formation_legal(pos_counts: dict[str, int]) -> bool:
    d, m, f = pos_counts.get("DEF", 0), pos_counts.get("MID", 0), pos_counts.get("FWD", 0)
    g = pos_counts.get("GKP", 0)
    return g == 1 and 3 <= d <= 5 and 2 <= m <= 5 and 1 <= f <= 3 and (d + m + f) == 10


def _apply_auto_subs(starting_ids: list[int], bench_order_ids: list[int],
                     gw_stats: dict[int, dict], positions_by_id: dict[int, str]) -> dict:
    starting = list(starting_ids)
    bench = list(bench_order_ids)
    subs: list[dict] = []

    def played(pid: int) -> bool:
        return gw_stats.get(pid, {}).get("minutes", 0) > 0

    starting_gk = next((pid for pid in starting if positions_by_id.get(pid) == "GKP"), None)
    if starting_gk is not None and not played(starting_gk):
        bench_gk = next((pid for pid in bench if positions_by_id.get(pid) == "GKP"), None)
        if bench_gk is not None and played(bench_gk):
            starting[starting.index(starting_gk)] = bench_gk
            bench.remove(bench_gk)
            subs.append({"out": starting_gk, "in": bench_gk})

    def pos_counts(ids: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pid in ids:
            p = positions_by_id.get(pid)
            counts[p] = counts.get(p, 0) + 1
        return counts

    blanks = [pid for pid in starting if positions_by_id.get(pid) != "GKP" and not played(pid)]
    for blank in blanks:
        if blank not in starting:
            continue  # already handled indirectly (shouldn't happen, defensive)
        for cand in list(bench):
            if positions_by_id.get(cand) == "GKP" or not played(cand):
                continue
            trial = [cand if pid == blank else pid for pid in starting]
            if _formation_legal(pos_counts(trial)):
                subs.append({"out": blank, "in": cand})
                starting = trial
                bench.remove(cand)
                break

    return {"final_xi_ids": starting, "subs_applied": subs}


def _score_gameweek(entry: dict, gw_stats: dict[int, dict],
                    positions_by_id: dict[int, str]) -> dict:
    auto = _apply_auto_subs(entry["starting_xi_ids"], entry["bench_order_ids"],
                            gw_stats, positions_by_id)
    final_xi = auto["final_xi_ids"]
    chip = entry.get("chip_played")
    scoring_ids = entry["squad_ids"] if chip == "bench_boost" else final_xi

    def pts(pid: int) -> int:
        return gw_stats.get(pid, {}).get("points", 0)

    total = sum(pts(pid) for pid in scoring_ids)

    captain_id, vice_id = entry.get("captain_id"), entry.get("vice_captain_id")
    captain_used = None
    if captain_id in final_xi and gw_stats.get(captain_id, {}).get("minutes", 0) > 0:
        captain_used = captain_id
    elif vice_id in final_xi and gw_stats.get(vice_id, {}).get("minutes", 0) > 0:
        captain_used = vice_id

    if captain_used is not None:
        total += pts(captain_used) * (2 if chip == "triple_captain" else 1)

    return {
        "points": total,
        "final_xi_ids": final_xi,
        "subs_applied": auto["subs_applied"],
        "captain_used_id": captain_used,
    }


# ---------------------------------------------------------------------------
# Deciding the next gameweek
# ---------------------------------------------------------------------------

def _decide_chip(buyable_pool_df: pd.DataFrame, squad_df: pd.DataFrame,
                 gw: int, chips_used_this_half: set[str]) -> dict:
    """Evaluated in Wildcard > Free Hit > Bench Boost > Triple Captain
    order -- only one chip can be played per gameweek, and a full rebuild
    (Wildcard) or a one-week rescue (Free Hit) are the highest-impact,
    least-reversible calls, so they get first refusal."""
    status = chip_strategy.chip_status(gw, chips_used_this_half)

    def urgent(name: str) -> bool:
        return name not in chips_used_this_half and status["chips"][name]["urgency"] == "urgent"

    def available(name: str) -> bool:
        return name not in chips_used_this_half

    one_week = squad_df.copy()
    one_week["xp"] = pd.to_numeric(one_week["xp_gw1"], errors="coerce").fillna(0.0)
    xi_result = squad_optimizer.pick_starting_xi(one_week)

    if available("wildcard"):
        fresh = squad_optimizer.pick_squad(buyable_pool_df)
        gain = fresh["total_xp"] - float(squad_df["xp"].sum())
        if gain >= WILDCARD_GAIN_THRESHOLD or urgent("wildcard"):
            return {"chip": "wildcard", "fresh_squad": fresh["squad"],
                   "reason": f"a full rebuild projects {gain:+.1f} xP over the next 4 "
                             "gameweeks versus holding the current squad"}

    if available("free_hit"):
        fh_pool = buyable_pool_df.copy()
        fh_pool["xp"] = pd.to_numeric(fh_pool["xp_gw1"], errors="coerce").fillna(0.0)
        fresh_one_week = squad_optimizer.pick_squad(fh_pool)
        deficit = fresh_one_week["total_xp"] - xi_result["xi_xp"]
        if deficit >= FREE_HIT_DEFICIT_THRESHOLD or urgent("free_hit"):
            return {"chip": "free_hit", "fresh_squad": fresh_one_week["squad"],
                   "reason": f"this squad projects {deficit:.1f} xP behind a fresh "
                             "one-week-optimal XI (a blank-heavy week for these players)"}

    if available("bench_boost"):
        bench_xp = float(xi_result["bench_xp"])
        if bench_xp >= BENCH_BOOST_XP_THRESHOLD or urgent("bench_boost"):
            return {"chip": "bench_boost", "fresh_squad": None,
                   "reason": f"the bench alone projects {bench_xp:.1f} xP this gameweek"}

    if available("triple_captain"):
        captain_xp = float(xi_result["captain"]["xp"])
        if captain_xp >= TRIPLE_CAPTAIN_XP_THRESHOLD or urgent("triple_captain"):
            return {"chip": "triple_captain", "fresh_squad": None,
                   "reason": f"the captain projects {captain_xp:.1f} xP this gameweek alone"}

    return {"chip": None, "fresh_squad": None, "reason": None}


def _decide_next_gameweek(full_pool_df: pd.DataFrame, buyable_pool_df: pd.DataFrame,
                          prev_entry: dict, next_gw: int,
                          chips_used_this_half: set[str]) -> dict:
    if prev_entry.get("chip_played") == "free_hit":
        # Free Hit is a one-week-only swap -- the squad, bank, and free
        # transfers all revert to what they were immediately before it.
        base_ids = prev_entry["pre_free_hit_squad_ids"]
        bank = prev_entry["pre_free_hit_bank"]
        free_transfers = prev_entry["pre_free_hit_free_transfers"]
    else:
        base_ids = prev_entry["squad_ids"]
        bank = prev_entry["bank_after"]
        free_transfers = prev_entry["free_transfers_after"]

    squad_df = _squad_df_from_ids(full_pool_df, base_ids)

    chip_decision = (
        _decide_chip(buyable_pool_df, squad_df, next_gw, chips_used_this_half)
        if next_gw >= CHIP_ELIGIBLE_FROM_GW else {"chip": None, "fresh_squad": None, "reason": None}
    )
    chip = chip_decision["chip"]
    pre_free_hit_fields: dict[str, Any] = {}
    transfers: list[dict] = []

    if chip == "wildcard":
        new_squad_df = chip_decision["fresh_squad"]
        bank_after = round(squad_optimizer.DEFAULT_BUDGET - float(new_squad_df["price_m"].sum()), 1)
        free_transfers_next = 1
        note = f"WILDCARD played -- {chip_decision['reason']}. Full squad rebuild."
    elif chip == "free_hit":
        new_squad_df = chip_decision["fresh_squad"]
        bank_after = bank
        free_transfers_next = free_transfers
        pre_free_hit_fields = {
            "pre_free_hit_squad_ids": base_ids,
            "pre_free_hit_bank": bank,
            "pre_free_hit_free_transfers": free_transfers,
        }
        note = f"FREE HIT played -- {chip_decision['reason']}. Squad reverts next gameweek."
    else:
        result = squad_optimizer.optimize_transfers(
            squad_df, buyable_pool_df, free_transfers=free_transfers, bank=bank)
        new_squad_df = result["new_squad"]
        bank_after = result["bank_after"]
        if result["recommendation"] == "no_transfer":
            free_transfers_next = min(free_transfers + 1, 2)
            transfer_note = "Held -- no transfer had a positive net expected gain this week."
        else:
            transfers = result["transfers"]
            free_transfers_next = 1
            move_desc = ", ".join(
                f"{t['out']['web_name']} → {t['in']['web_name']}" for t in transfers)
            transfer_note = f"Transfer: {move_desc} (net {result['net_gain']:+.1f} xP over 4 GWs)."
        if chip == "bench_boost":
            note = f"BENCH BOOST played -- {chip_decision['reason']}. {transfer_note}"
        elif chip == "triple_captain":
            note = f"TRIPLE CAPTAIN played -- {chip_decision['reason']}. {transfer_note}"
        else:
            note = transfer_note

    # Starting XI / captain for the SINGLE upcoming gameweek, not the
    # 4-week cumulative outlook -- a real manager captains off this
    # week's projection, not next month's.
    one_week_squad = new_squad_df.copy()
    one_week_squad["xp"] = pd.to_numeric(one_week_squad["xp_gw1"], errors="coerce").fillna(0.0)
    xi_result = squad_optimizer.pick_starting_xi(one_week_squad)

    starting_xi_ids = [_to_native(i) for i in xi_result["starting_xi"]["id"].tolist()]
    bench_order_ids = [_to_native(i) for i in xi_result["bench"]["id"].tolist()]
    captain_id = _to_native(xi_result["captain"]["id"])
    projected_points = _projected_points_for_selection(
        new_squad_df, starting_xi_ids, bench_order_ids, captain_id, chip)

    return {
        "gameweek": next_gw,
        "squad_ids": [_to_native(i) for i in new_squad_df["id"].tolist()],
        "starting_xi_ids": starting_xi_ids,
        "bench_order_ids": bench_order_ids,
        "captain_id": captain_id,
        "vice_captain_id": _to_native(xi_result["vice_captain"]["id"]),
        "chip_played": chip,
        "transfers": transfers,
        "bank_after": bank_after,
        "free_transfers_after": free_transfers_next,
        "points": None,
        "projected_points": projected_points,
        "note": note,
        **pre_free_hit_fields,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def advance(full_pool_df: pd.DataFrame, buyable_pool_df: pd.DataFrame,
           bootstrap: dict) -> dict:
    """Load (or seed) the season log, then catch up any real gameweek(s)
    that have finished since it was last updated: score them for real,
    decide + lock in the following gameweek's move, and persist. Safe to
    call on every page load -- it's a no-op once there's nothing new to
    process (including, right now, before GW1 has even been played)."""
    log = _load_log()
    if log is None:
        log = seed_log(buyable_pool_df)

    events_by_id = {e["id"]: e for e in bootstrap.get("events", [])}
    changed = False

    while True:
        latest = log["history"][-1]
        gw = latest["gameweek"]
        if latest["points"] is not None:
            break

        event = events_by_id.get(gw)
        if not event or not event.get("finished"):
            break  # this gameweek hasn't actually finished yet -- stay pending

        gw_stats = _fetch_gw_stats_for_players(latest["squad_ids"], gw)
        if gw_stats is None:
            break  # couldn't fetch real results (no network, not published yet)

        positions_by_id = _positions_by_id(full_pool_df, latest["squad_ids"])
        scored = _score_gameweek(latest, gw_stats, positions_by_id)
        latest["points"] = scored["points"]
        latest["final_xi_ids"] = scored["final_xi_ids"]
        latest["subs_applied"] = scored["subs_applied"]
        latest["captain_used_id"] = scored["captain_used_id"]
        changed = True

        if gw < SEASON_END_GW:
            next_gw = gw + 1
            half = chip_strategy.current_half(next_gw)
            chips_used_this_half = set(log["chips_used"].get(half, []))
            next_entry = _decide_next_gameweek(
                full_pool_df, buyable_pool_df, latest, next_gw, chips_used_this_half)
            if next_entry["chip_played"]:
                log["chips_used"].setdefault(half, [])
                if next_entry["chip_played"] not in log["chips_used"][half]:
                    log["chips_used"][half].append(next_entry["chip_played"])
            log["history"].append(next_entry)
        # loop again in case a second gameweek has ALSO already finished
        # (the app wasn't opened for a while) -- each catch-up decision
        # still only ever uses live-at-the-time projections, never
        # reconstructs what they "would have been" on the actual date.

    if changed:
        _save_log(log)
    return log


def load_log() -> dict | None:
    """Public read-only accessor -- returns whatever's currently persisted
    (or None if the season hasn't been seeded yet), without trying to
    advance it. Used by the web layer to show *something* if `advance()`
    itself raised (e.g. a mid-catch-up failure)."""
    return _load_log()


__all__ = [
    "LOG_PATH", "SEASON_END_GW", "CHIP_ELIGIBLE_FROM_GW",
    "seed_log", "advance", "load_log",
]
