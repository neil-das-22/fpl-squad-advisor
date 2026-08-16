"""
Orchestrator (manager agent) -- consolidates research-agent findings and
Neil's own player judgments into the model's expected points, re-runs the
optimizer, and produces the final gameweek recommendation.

This is currently run manually per gameweek: the position/fixtures/chip
research agents are dispatched live (via Claude subagents in this Cowork
session), their findings are hand-transcribed into
data/processed/research_overrides_gw<N>.csv (an auditable, versioned record
of what was adjusted and why), and this script does the deterministic part:
apply overrides -> exclude non-starters -> re-optimize -> report.

If this project later moves to a standalone Anthropic-API agent runner
(see PROJECT_PLAN.md), that runner's job is just to produce the same
overrides CSV format automatically -- this script doesn't need to change.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import xp_model  # noqa: E402
import squad_optimizer  # noqa: E402
import fpl_client  # noqa: E402

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Columns carried through from the player table onto the xP table so the
# reporting layer can show availability/news next to a recommendation.
_PLAYER_CONTEXT_COLUMNS = ["id", "status_meaning", "news",
                           "chance_of_playing_next_round"]


def _finished_gameweeks(bootstrap: dict) -> int:
    """How many gameweeks of the CURRENT season have actually been played.

    This has to come from `events`, not from the player pool: pre-rollover,
    bootstrap-static still reports last season's minutes, so
    `xp_model.infer_matches_played()` would conclude the season is 37 matches
    old when it is 0 matches old, and every player would be scored off
    "current-season" evidence that is really last season's. See the PRE-SEASON
    CARRYOVER note in models/xp_model.py.
    """
    return sum(1 for e in bootstrap.get("events", []) if e.get("finished"))


def build_xp_table(gameweek: int, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Build data/processed/xp_gw{N}.csv from the raw payloads on disk.

    Deterministic and offline: reads data/raw/bootstrap_static.json and
    data/raw/fixtures.json (whatever `fpl_client.run_pipeline()` last pulled),
    joins the prior-season reference table on `code`, and runs the xP model.

    The prior-season join is what closes xp_model's KNOWN DATA GAP #1. If the
    archive CSV isn't on disk this prints a note and carries on with all-NaN
    prior-season columns -- the model then behaves exactly as it did before the
    fix (flat start-probability prior), which is degraded but not wrong.
    """
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    with open(os.path.join(raw_dir, "bootstrap_static.json")) as f:
        bootstrap = json.load(f)
    with open(os.path.join(raw_dir, "fixtures.json")) as f:
        fixtures = json.load(f)

    players_df = fpl_client.load_players_with_prior_season(bootstrap, verbose=verbose)
    teams_df = fpl_client.load_teams_df(bootstrap)
    fixtures_df = fpl_client.load_fixtures_df(fixtures, bootstrap)

    matches_played = _finished_gameweeks(bootstrap)
    if verbose:
        print(f"[XP BUILD] gameweek {gameweek}, "
              f"{matches_played} completed gameweek(s) of the current season")

    xp_df = xp_model.calculate_xp_for_gameweek(
        players_df, teams_df, fixtures_df, gameweek=gameweek,
        matches_played=float(matches_played),
    )

    context = players_df[[c for c in _PLAYER_CONTEXT_COLUMNS
                          if c in players_df.columns]]
    xp_df = xp_df.merge(context, on="id", how="left", suffixes=("", "_player"))

    if write:
        processed = os.path.join(PROJECT_ROOT, "data", "processed")
        out_path = os.path.join(processed, f"xp_gw{gameweek}.csv")
        xp_df.to_csv(out_path, index=False)
        # Refresh the shared player table too, so the CSV on disk carries the
        # same `code` + prior-season columns the xP table was built from rather
        # than a stale schema from an older run.
        players_df.to_csv(os.path.join(processed, "players.csv"), index=False)
        if verbose:
            print(f"[XP BUILD] wrote {len(xp_df)} rows to {out_path}")
    return xp_df


def run_gameweek(gameweek: int, avoid_web_name_team: list[tuple[str, str]],
                  must_include_web_name_team: list[tuple[str, str]] | None = None,
                  rebuild_xp: bool = True) -> dict:
    xp_path = os.path.join(PROJECT_ROOT, "data", "processed", f"xp_gw{gameweek}.csv")
    overrides_path = os.path.join(
        PROJECT_ROOT, "data", "processed", f"research_overrides_gw{gameweek}.csv"
    )

    if rebuild_xp or not os.path.exists(xp_path):
        xp_df = build_xp_table(gameweek)
    else:
        xp_df = pd.read_csv(xp_path)
    overrides_df = pd.read_csv(overrides_path) if os.path.exists(overrides_path) else None

    adjusted = xp_model.apply_manual_adjustments(xp_df, overrides_df)
    unmatched = adjusted.attrs.get("unmatched_overrides", [])

    # Match on (web_name, team_name), not name alone -- multiple players can
    # share a surname (e.g. two different "Davies") and a name-only match
    # would silently exclude the wrong one.
    avoid_set = set(avoid_web_name_team)
    is_avoided = adjusted.apply(
        lambda r: (r["web_name"], r["team_name"]) in avoid_set, axis=1
    )
    exclude_ids = adjusted.loc[is_avoided, "id"].tolist()

    must_set = set(must_include_web_name_team or [])
    is_must = adjusted.apply(
        lambda r: (r["web_name"], r["team_name"]) in must_set, axis=1
    )
    must_include_ids = adjusted.loc[is_must, "id"].tolist()

    unconstrained = squad_optimizer.pick_squad(adjusted, exclude_ids=exclude_ids)
    squad_result = squad_optimizer.pick_squad(
        adjusted, exclude_ids=exclude_ids, must_include_ids=must_include_ids
    )
    xi_result = squad_optimizer.pick_starting_xi(squad_result["squad"])

    return {
        "adjusted_xp": adjusted,
        "unmatched_overrides": unmatched,
        "excluded": adjusted[is_avoided][["web_name", "team_name", "position"]],
        "unconstrained_squad_result": unconstrained,
        "squad_result": squad_result,
        "xi_result": xi_result,
        "judgment_cost_xp": unconstrained["total_xp"] - squad_result["total_xp"],
    }


AVOID_GW1 = [
    ("Meslier", "Arsenal"), ("Heaton", "Man Utd"), ("Davies", "Liverpool"),
    ("Pecsi", "Liverpool"), ("Penders", "Chelsea"),               # GK backups
    ("Chalobah", "Chelsea"), ("Amass", "Man Utd"), ("Ramsay", "Liverpool"),  # DEF exiting/bench
    ("N.Jackson", "Chelsea"), ("Obi", "Man Utd"),                 # FWD unlikely starters
    # Welbeck was originally logged as a soft "downgrade" in the overrides
    # CSV, but the FWD research agent's own language ("rotation/backup only
    # ... expected to compete with Emegha for backup minutes, not a start")
    # is really an avoid-level call. A flat -0.5 xP nudge isn't enough to
    # keep a backup out once other picks shift the budget math -- this
    # surfaced for real after the DefCon/start-probability fixes changed the
    # squad enough to pull him in. Moving him here instead of relying on the
    # soft adjustment.
    ("Welbeck", "Chelsea"),
]


if __name__ == "__main__":
    # From docs/player_judgments.md: Haaland tagged "must-have" (Neil's own call).
    MUST_INCLUDE_GW1 = [("Haaland", "Man City")]
    result = run_gameweek(1, AVOID_GW1, MUST_INCLUDE_GW1)
    print(f"Judgment cost of forcing in Haaland: {result['judgment_cost_xp']:.2f} xP "
          f"vs the unconstrained optimum ({result['unconstrained_squad_result']['total_xp']:.2f} xP)")
    print()

    sq = result["squad_result"]
    xi = result["xi_result"]

    print(f"Solver: {sq['solver']}  |  optimal: {sq['optimal']}")
    print(f"Squad cost: £{sq['total_cost']}m of £{sq['budget']}m "
          f"(bank £{sq['remaining_budget']}m)  |  predicted xP: {sq['total_xp']:.2f}")
    print()
    print("Excluded from consideration (research-flagged non-starters):")
    print(result["excluded"].to_string(index=False))
    print()
    print("15-man squad:")
    print(sq["squad"][["web_name", "team_name", "position", "price_m", "xp_model", "xp"]]
          .to_string(index=False))
    print()
    print(f"Starting XI ({xi['formation']}):")
    print(xi["starting_xi"][["web_name", "team_name", "position", "xp"]].to_string(index=False))
    print(f"Captain: {xi['captain']}  |  Vice-captain: {xi['vice_captain']}")
    print()
    if result["unmatched_overrides"]:
        print("Unmatched override rows (informational):", len(result["unmatched_overrides"]))
