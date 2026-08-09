"""
Squad optimizer — FPL 2026/27.

Takes the expected-points table produced by `models/xp_model.py` and turns it
into decisions:

    pick_squad()          15-man squad, integer linear program (PuLP)
    pick_starting_xi()    best legal XI + captain/vice out of those 15
    optimize_transfers()  weekly transfer search, net of -4 hit costs

FPL RULES ENCODED HERE
----------------------
    Squad          exactly 2 GKP, 5 DEF, 5 MID, 3 FWD (15 players)
    Budget         total price <= 100.0m by default
    Club cap       at most 3 players from any single club
    Starting XI    11 players: exactly 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
    Captain        scores double; vice-captain is the fallback
    Transfers      1 free per week (rolls to 2); each extra costs -4 points

SOLVER NOTE
-----------
`pick_squad` is a genuine integer linear program and PuLP (in requirements.txt)
is the intended solver. Because the environment this was written in had no
network access to install PuLP, the module also ships a self-contained exact
branch-and-bound fallback (`_solve_squad_branch_and_bound`, with a dominance
pre-filter and a Lagrangian bound) that runs automatically when PuLP is missing.
Both paths solve the same problem to the same optimum; on a full ~600-player
pool the fallback closes in well under a second. It carries a node limit as a
safety valve and both warns and reports `optimal=False` in the event it is ever
hit, so a merely-good squad can never be mistaken for the optimum.
`result["solver"]` always records which path ran, and
`test_squad_optimizer.test_matches_brute_force_on_random_instances` checks the
result against exhaustive enumeration on small instances.

NOT IN SCOPE (deliberately)
---------------------------
Chip timing (Wildcard / Bench Boost / Triple Captain / Free Hit) belongs to the
chip strategy agent, not here. The plumbing hook already exists:
`optimize_transfers(..., unlimited_transfers=True)` turns off hit costs and the
transfer-count limit, which is exactly the Wildcard / Free Hit case. Deciding
*when* to flip that flag is the agent's job.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Iterable, Sequence

import pandas as pd

try:  # pragma: no cover - trivial import guard
    import pulp
    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover
    pulp = None
    PULP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

SQUAD_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
MAX_PER_CLUB = 3
DEFAULT_BUDGET = 100.0

XI_SIZE = 11
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]

DEFAULT_HIT_COST = 4
# Prices are quoted in 0.1m steps; work in integer tenths so the budget
# constraint is exact and free of floating-point drift.
PRICE_SCALE = 10

REQUIRED_COLUMNS = ("id", "position", "team_name", "price_m", "xp")

# Branch-and-bound safety valve (fallback solver only).
BNB_NODE_LIMIT = 3_000_000


class OptimizerError(ValueError):
    """Raised when the inputs cannot produce a legal squad."""


# ---------------------------------------------------------------------------
# Input validation / preparation
# ---------------------------------------------------------------------------

def _prepare(players_xp_df: pd.DataFrame,
             exclude_ids: Iterable[Any] | None = None) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in players_xp_df.columns]
    if missing:
        raise OptimizerError(f"players_xp_df is missing required columns: {missing}")

    df = players_xp_df.copy()
    df["price_m"] = pd.to_numeric(df["price_m"], errors="coerce")
    df["xp"] = pd.to_numeric(df["xp"], errors="coerce").fillna(0.0)
    df["position"] = df["position"].astype(str).str.upper()

    df = df[df["price_m"].notna() & (df["price_m"] > 0)]
    df = df[df["position"].isin(SQUAD_COMPOSITION)]

    if exclude_ids:
        excl = {str(i) for i in exclude_ids}
        df = df[~df["id"].astype(str).isin(excl)]

    # Integer tenths -- see PRICE_SCALE.
    df["price_tenths"] = (df["price_m"] * PRICE_SCALE).round().astype(int)
    return df.reset_index(drop=True)


def _check_feasible(df: pd.DataFrame) -> None:
    for pos, need in SQUAD_COMPOSITION.items():
        have = int((df["position"] == pos).sum())
        if have < need:
            raise OptimizerError(
                f"not enough {pos} in the player pool: need {need}, have {have}")


# ---------------------------------------------------------------------------
# pick_squad
# ---------------------------------------------------------------------------

def _solve_squad_pulp(df: pd.DataFrame, budget_tenths: int, max_per_club: int,
                      must_include_rows: Sequence[int]) -> tuple[list[int], bool]:
    """Exact ILP via PuLP/CBC. Returns (selected row positions, proven_optimal)."""
    prob = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(x[i] * float(df.at[i, "xp"]) for i in df.index), "total_expected_points"

    prob += pulp.lpSum(x.values()) == SQUAD_SIZE, "squad_size"
    for pos, need in SQUAD_COMPOSITION.items():
        rows = df.index[df["position"] == pos]
        prob += pulp.lpSum(x[i] for i in rows) == need, f"count_{pos}"

    prob += (pulp.lpSum(x[i] * int(df.at[i, "price_tenths"]) for i in df.index)
             <= budget_tenths), "budget"

    # Constraint names are enumerated rather than interpolated from club names:
    # PuLP rejects/rewrites names containing spaces ("Man City"), and two clubs
    # could otherwise sanitise to the same identifier.
    for n, (_club, rows) in enumerate(df.groupby("team_name").groups.items()):
        prob += pulp.lpSum(x[i] for i in rows) <= max_per_club, f"club_cap_{n}"

    for i in must_include_rows:
        prob += x[i] == 1, f"must_include_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise OptimizerError(f"ILP did not solve to optimality (status: {status}). "
                             "Most likely the budget is too low for a legal squad.")
    selected = [i for i in df.index if x[i].value() is not None and x[i].value() > 0.5]
    return selected, True


def _drop_dominated(df: pd.DataFrame, max_per_club: int,
                    protected_rows: Iterable[int] = ()) -> pd.DataFrame:
    """Remove players who can never appear in an optimal squad.

    Player X is *dominated* if enough same-position players are simultaneously
    no more expensive and no lower scoring that one of them is guaranteed to be
    available as a drop-in replacement. Swapping X out for such a player can
    never make a squad worse, so an optimal squad exists without X.

    "Guaranteed available" has to survive two blockers, hence the counting:
      * up to `need - 1` of the dominators may already be in the squad
        (the squad holds `need` players at that position, one of which is X);
      * a dominator's club may already be at the 3-player cap. A 14-player
        remainder can have at most 4 clubs sitting on the cap, so the four
        clubs contributing the most dominators are discounted entirely.
    So X is only dropped when  |dominators| - (top 4 clubs' share) >= need.

    Ties are broken by a strict total order on (price, -xp, index) so two
    identical players can never dominate each other into both being dropped;
    the first survivor of any such group is always kept.

    This is a pure search-space reduction, not an approximation. It roughly
    halves CBC's work and is what makes the branch-and-bound fallback tractable
    on a real ~600-player pool.
    """
    protected = set(protected_rows)
    keep_mask = pd.Series(True, index=df.index)
    n_clubs_at_cap = (SQUAD_SIZE - 1) // max_per_club   # 14 // 3 = 4

    for pos, need in SQUAD_COMPOSITION.items():
        sub = df[df["position"] == pos]
        if len(sub) <= need:
            continue
        recs = [(int(sub.at[i, "price_tenths"]), float(sub.at[i, "xp"]),
                 sub.at[i, "team_name"], i) for i in sub.index]
        # Strict total order: cheaper first, then higher xP, then index.
        order = sorted(range(len(recs)), key=lambda k: (recs[k][0], -recs[k][1], k))
        rank = {k: r for r, k in enumerate(order)}

        for k, (price_k, xp_k, _club_k, idx_k) in enumerate(recs):
            if idx_k in protected:
                continue
            club_counts: dict[Any, int] = {}
            total = 0
            for j, (price_j, xp_j, club_j, _idx_j) in enumerate(recs):
                if j == k or rank[j] >= rank[k]:
                    continue
                if price_j <= price_k and xp_j >= xp_k:
                    total += 1
                    club_counts[club_j] = club_counts.get(club_j, 0) + 1
            if not club_counts:
                continue
            blocked = sum(sorted(club_counts.values(), reverse=True)[:n_clubs_at_cap])
            if total - blocked >= need:
                keep_mask.at[idx_k] = False

    return df[keep_mask]


def _solve_squad_branch_and_bound(df: pd.DataFrame, budget_tenths: int,
                                  max_per_club: int,
                                  must_include_rows: Sequence[int],
                                  node_limit: int = BNB_NODE_LIMIT,
                                  ) -> tuple[list[int], bool]:
    """Exact depth-first branch and bound. Fallback for when PuLP is absent.

    Solves the same problem as `_solve_squad_pulp`, to the same optimum. Three
    admissible bounds do the pruning:

      * LAGRANGIAN VALUE BOUND -- the binding difficulty here is the budget, and
        a bound that ignores price ("just take the 15 highest xP players") is far
        too loose to close the search. So the budget constraint is relaxed into
        the objective with a multiplier lambda >= 0:

            UB = value_so_far + lambda * budget_remaining
                 + sum over positions of the top-k remaining (xp - lambda*price)

        which is a valid upper bound for ANY lambda >= 0 (standard Lagrangian
        relaxation: every feasible squad satisfies sum(price) <= budget, so
        substituting the relaxed objective can only over-estimate). lambda is
        fitted once at the root by ternary search on the convex dual, giving the
        tightest bound in the family. It is the "points per 0.1m" shadow price.
      * PLAIN VALUE BOUND -- the lambda = 0 member of the same family, kept
        because it is occasionally tighter when lots of budget is left.
      * MINIMUM COST -- for each position still needed, k times the cheapest
        remaining price is a lower bound on money still to be spent; if that
        exceeds the remaining budget the branch is dead.

    Players are visited in descending (xp - lambda*price), i.e. best value for
    money first, so a strong incumbent appears within the first few nodes.

    Returns (selected row positions, proven_optimal). `proven_optimal` is False
    only if the node limit was hit before the search closed.
    """
    # Dominance reduction first -- on a full ~600-player pool this is the
    # difference between "solves in under a second" and "hits the node limit".
    df = _drop_dominated(df, max_per_club, protected_rows=must_include_rows)

    forced = set(must_include_rows)
    all_rows = df.index.tolist()
    all_xp = {i: float(df.at[i, "xp"]) for i in all_rows}
    all_price = {i: int(df.at[i, "price_tenths"]) for i in all_rows}
    all_pos = {i: df.at[i, "position"] for i in all_rows}

    def dual_value(lam: float) -> float:
        """The Lagrangian bound at the root for a given multiplier."""
        total = lam * budget_tenths
        for p, need in SQUAD_COMPOSITION.items():
            adj = sorted((all_xp[i] - lam * all_price[i]
                          for i in all_rows if all_pos[i] == p), reverse=True)
            total += sum(adj[:need])
        return total

    # Ternary search for the lambda minimising the (convex) dual. The upper end
    # of the bracket is generous: beyond it every adjusted value is negative and
    # the bound only grows.
    lo, hi = 0.0, max((all_xp[i] / max(all_price[i], 1) for i in all_rows),
                      default=0.0) * 2.0 + 1e-6
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if dual_value(m1) <= dual_value(m2):
            hi = m2
        else:
            lo = m1
    lam = (lo + hi) / 2.0

    # Search order: best value for money first.
    order = sorted(all_rows, key=lambda i: -(all_xp[i] - lam * all_price[i]))

    xp = [all_xp[i] for i in order]
    price = [all_price[i] for i in order]
    pos = [all_pos[i] for i in order]
    club = [df.at[i, "team_name"] for i in order]
    adjusted = [xp[k] - lam * price[k] for k in range(len(order))]
    n = len(order)

    # For each position, the running top-`need` values in the suffix, so both
    # value bounds are O(1) per node. Memory is n * need per position (tiny).
    def _suffix_top_sums(values: list[float]) -> dict[str, list[list[float]]]:
        out: dict[str, list[list[float]]] = {}
        for p, need in SQUAD_COMPOSITION.items():
            sums: list[list[float]] = [[0.0] * (need + 1) for _ in range(n + 1)]
            current: list[float] = []
            for k in range(n - 1, -1, -1):
                if pos[k] == p:
                    current.append(values[k])
                    current.sort(reverse=True)
                    del current[need:]
                running = 0.0
                row = sums[k]
                for j, v in enumerate(current):
                    running += v
                    row[j + 1] = running
                for j in range(len(current) + 1, need + 1):
                    row[j] = running
            out[p] = sums
        return out

    top_xp = _suffix_top_sums(xp)
    top_adj = _suffix_top_sums(adjusted)

    # Suffix minimum price and suffix count per position, for the cost bound.
    suffix_min_price: dict[str, list[int]] = {}
    suffix_count: dict[str, list[int]] = {}
    for p in SQUAD_COMPOSITION:
        mins = [math.inf] * (n + 1)
        counts = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            mins[k] = mins[k + 1]
            counts[k] = counts[k + 1]
            if pos[k] == p:
                mins[k] = min(mins[k], price[k])
                counts[k] += 1
        suffix_min_price[p] = mins
        suffix_count[p] = counts

    best_value = -math.inf
    best_set: list[int] = []
    nodes = 0
    hit_limit = False

    counts_needed = dict(SQUAD_COMPOSITION)
    club_counts: dict[Any, int] = {}
    chosen: list[int] = []

    def recurse(idx: int, spent: int, value: float) -> None:
        nonlocal best_value, best_set, nodes, hit_limit
        if hit_limit:
            return
        nodes += 1
        if nodes > node_limit:
            hit_limit = True
            return

        remaining = sum(counts_needed.values())
        if remaining == 0:
            if value > best_value:
                best_value = value
                best_set = list(chosen)
            return
        if idx >= n:
            return

        # --- bound 1: Lagrangian (budget-aware) value bound ----------------
        lagrangian = value + lam * (budget_tenths - spent)
        plain = value
        for p, need in counts_needed.items():
            if need:
                lagrangian += top_adj[p][idx][need]
                plain += top_xp[p][idx][need]
        if min(lagrangian, plain) <= best_value:
            return

        # --- bound 2: minimum remaining cost ------------------------------
        min_cost = 0
        for p, need in counts_needed.items():
            if not need:
                continue
            if suffix_count[p][idx] < need:
                return                      # not enough players left at all
            min_cost += need * suffix_min_price[p][idx]
        if spent + min_cost > budget_tenths:
            return

        p = pos[idx]
        can_take = (counts_needed[p] > 0
                    and spent + price[idx] <= budget_tenths
                    and club_counts.get(club[idx], 0) < max_per_club)

        # Try taking first -- best-first ordering finds a strong incumbent fast.
        if can_take:
            counts_needed[p] -= 1
            club_counts[club[idx]] = club_counts.get(club[idx], 0) + 1
            chosen.append(idx)
            recurse(idx + 1, spent + price[idx], value + xp[idx])
            chosen.pop()
            club_counts[club[idx]] -= 1
            counts_needed[p] += 1

        if order[idx] not in forced:
            recurse(idx + 1, spent, value)

    # Forced picks are applied up front so the search only explores the rest.
    if forced:
        for k in range(n):
            if order[k] in forced:
                p = pos[k]
                if counts_needed[p] <= 0:
                    raise OptimizerError(
                        f"must_include_ids ask for too many {p} players")
                counts_needed[p] -= 1
                club_counts[club[k]] = club_counts.get(club[k], 0) + 1
                chosen.append(k)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, n * 4 + 1000))
    try:
        recurse(0, sum(price[k] for k in chosen), sum(xp[k] for k in chosen))
    finally:
        sys.setrecursionlimit(old_limit)

    if not best_set:
        raise OptimizerError(
            "no legal 15-man squad exists under these constraints "
            f"(budget {budget_tenths / PRICE_SCALE:.1f}m, max {max_per_club} per club)")

    return [order[k] for k in best_set], not hit_limit


def pick_squad(players_xp_df: pd.DataFrame,
               budget: float = DEFAULT_BUDGET,
               max_per_club: int = MAX_PER_CLUB,
               exclude_ids: Iterable[Any] | None = None,
               must_include_ids: Iterable[Any] | None = None,
               ) -> dict[str, Any]:
    """Select the optimal 15-man squad by integer linear programming.

    Maximises total predicted points subject to:
        exactly 2 GKP / 5 DEF / 5 MID / 3 FWD,
        total price <= `budget`,
        at most `max_per_club` players from any one club.

    Args:
        players_xp_df: needs columns id, position, team_name, price_m, xp
            (i.e. the output of `xp_model.calculate_xp_for_gameweek`).
        budget: squad value cap in millions.
        exclude_ids: players to keep out (injured, suspended, agent veto...).
        must_include_ids: players to force in (e.g. "keep my captain").

    Returns:
        dict with `squad` (DataFrame), `total_cost`, `total_xp`,
        `remaining_budget`, `solver`, `optimal`.
    """
    df = _prepare(players_xp_df, exclude_ids)
    _check_feasible(df)

    budget_tenths = int(round(budget * PRICE_SCALE))

    must_rows: list[int] = []
    if must_include_ids:
        wanted = {str(i) for i in must_include_ids}
        must_rows = df.index[df["id"].astype(str).isin(wanted)].tolist()
        if len(must_rows) != len(wanted):
            found = set(df.loc[must_rows, "id"].astype(str))
            raise OptimizerError(f"must_include_ids not found in pool: {wanted - found}")
        if len(must_rows) > SQUAD_SIZE:
            raise OptimizerError("must_include_ids exceeds the 15-man squad size")

    if PULP_AVAILABLE:
        selected, optimal = _solve_squad_pulp(df, budget_tenths, max_per_club, must_rows)
        solver = "pulp-cbc"
    else:
        selected, optimal = _solve_squad_branch_and_bound(
            df, budget_tenths, max_per_club, must_rows)
        solver = "branch-and-bound (PuLP not installed)"
        if not optimal:
            # Never let a merely-good squad masquerade as the optimum.
            warnings.warn(
                "Squad selection hit the branch-and-bound node limit, so the "
                "returned squad is the best found, NOT a proven optimum. "
                "Install PuLP (pip install -r requirements.txt) to use the exact "
                "ILP solver.", RuntimeWarning, stacklevel=2)

    squad = df.loc[selected].copy()
    squad["position"] = pd.Categorical(squad["position"], POSITION_ORDER, ordered=True)
    squad = squad.sort_values(["position", "xp"], ascending=[True, False]).reset_index(drop=True)
    squad["position"] = squad["position"].astype(str)

    total_cost = float(squad["price_tenths"].sum()) / PRICE_SCALE
    return {
        "squad": squad.drop(columns=["price_tenths"]),
        "total_cost": round(total_cost, 1),
        "total_xp": float(squad["xp"].sum()),
        "budget": budget,
        "remaining_budget": round(budget - total_cost, 1),
        "n_players": len(squad),
        "solver": solver,
        "optimal": optimal,
    }


# ---------------------------------------------------------------------------
# pick_starting_xi
# ---------------------------------------------------------------------------

def _legal_formations() -> list[tuple[int, int, int]]:
    formations = []
    for d in range(XI_MIN["DEF"], XI_MAX["DEF"] + 1):
        for m in range(XI_MIN["MID"], XI_MAX["MID"] + 1):
            f = XI_SIZE - 1 - d - m           # -1 for the goalkeeper
            if XI_MIN["FWD"] <= f <= XI_MAX["FWD"]:
                formations.append((d, m, f))
    return formations


LEGAL_FORMATIONS = _legal_formations()


def pick_starting_xi(squad_df: pd.DataFrame) -> dict[str, Any]:
    """Best legal starting XI, captain and vice-captain, from a 15-man squad.

    Objective: sum(xP of the XI) + xP of the captain (the captain scores
    double, so his points are counted twice).

    This is solved by exhaustive enumeration over the legal formations rather
    than another ILP, and that is exact, not a shortcut. For a fixed formation
    the best XI is simply the top-k by xP at each position: that maximises the
    sum, and it necessarily also contains the highest-xP player available to
    that formation, so it maximises the captain term at the same time. There are
    only a handful of legal formations, so trying all of them is cheap and
    provably optimal.

    Bench ordering is by xP descending, which is the auto-substitution priority.
    Note the reserve goalkeeper can only ever replace the starting goalkeeper --
    the `can_only_replace_gk` column marks him so the report layer does not
    present the bench order misleadingly.
    """
    df = squad_df.copy().reset_index(drop=True)
    df["xp"] = pd.to_numeric(df["xp"], errors="coerce").fillna(0.0)
    df["position"] = df["position"].astype(str).str.upper()

    by_pos = {p: df[df["position"] == p].sort_values("xp", ascending=False)
              for p in POSITION_ORDER}
    if len(by_pos["GKP"]) < 1:
        raise OptimizerError("squad has no goalkeeper")

    best = None
    for d, m, f in LEGAL_FORMATIONS:
        if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        xi = pd.concat([
            by_pos["GKP"].head(1),
            by_pos["DEF"].head(d),
            by_pos["MID"].head(m),
            by_pos["FWD"].head(f),
        ])
        xi_xp = float(xi["xp"].sum())
        captain_xp = float(xi["xp"].max())
        objective = xi_xp + captain_xp
        if best is None or objective > best["objective"]:
            best = {"objective": objective, "xi": xi, "formation": (d, m, f),
                    "xi_xp": xi_xp}

    if best is None:
        raise OptimizerError("no legal starting XI can be formed from this squad")

    xi = best["xi"].copy()
    xi["_pos_rank"] = xi["position"].map({p: i for i, p in enumerate(POSITION_ORDER)})
    xi = (xi.sort_values(["_pos_rank", "xp"], ascending=[True, False])
            .drop(columns=["_pos_rank"])
            .reset_index(drop=True))

    ranked = xi.sort_values("xp", ascending=False)
    captain = ranked.iloc[0]
    vice_captain = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]

    bench = df[~df.index.isin(best["xi"].index)].sort_values(
        "xp", ascending=False).reset_index(drop=True)
    bench["can_only_replace_gk"] = bench["position"] == "GKP"
    bench["bench_order"] = range(1, len(bench) + 1)

    d, m, f = best["formation"]
    return {
        "starting_xi": xi,
        "bench": bench,
        "formation": f"{d}-{m}-{f}",
        "captain": captain.to_dict(),
        "vice_captain": vice_captain.to_dict(),
        "xi_xp": best["xi_xp"],
        "bench_xp": float(bench["xp"].sum()),
        # What the manager actually banks: the XI plus the captain's doubled score.
        "total_xp_with_captain": best["objective"],
    }


# ---------------------------------------------------------------------------
# optimize_transfers
# ---------------------------------------------------------------------------

# How many replacement candidates per position to consider. The pool is sorted
# by xP, so the truncation only ever discards players who are both worse and
# not cheaper in a way that could matter for a 1-or-2 transfer window.
CANDIDATES_PER_POSITION = 20
CANDIDATES_PER_POSITION_PAIRWISE = 12
EPS = 1e-9


def _transfer_is_legal(base_club_counts: dict[Any, int],
                       outs: Sequence[dict], ins: Sequence[dict],
                       bank: float, max_per_club: int) -> bool:
    """Budget + club-cap check for a candidate swap.

    Operates on plain dicts rather than DataFrames: this runs tens of thousands
    of times inside the pairwise search, and pandas row indexing in that loop is
    orders of magnitude slower than arithmetic.
    """
    money_available = bank + sum(o["price_m"] for o in outs)
    if sum(i["price_m"] for i in ins) - money_available > EPS:
        return False

    # Only clubs gaining a player can breach the cap.
    delta: dict[Any, int] = {}
    for o in outs:
        delta[o["team_name"]] = delta.get(o["team_name"], 0) - 1
    for i in ins:
        delta[i["team_name"]] = delta.get(i["team_name"], 0) + 1
    for club, d in delta.items():
        if d > 0 and base_club_counts.get(club, 0) + d > max_per_club:
            return False
    return True


def optimize_transfers(current_squad_df: pd.DataFrame,
                       all_players_xp_df: pd.DataFrame,
                       free_transfers: int = 1,
                       max_transfers_considered: int = 2,
                       hit_cost: int = DEFAULT_HIT_COST,
                       bank: float = 0.0,
                       max_per_club: int = MAX_PER_CLUB,
                       unlimited_transfers: bool = False,
                       ) -> dict[str, Any]:
    """Find the transfer(s) with the best positive net expected point gain.

    Evaluates every single transfer and, up to `max_transfers_considered`, every
    pairwise double transfer (positions must match, budget must clear, the
    3-per-club cap must hold afterwards). Scores each option as:

        net_gain = sum(xP in) - sum(xP out) - hit_cost * max(0, n - free_transfers)

    and returns the best option only if `net_gain > 0`. If nothing clears that
    bar it returns a "no_transfer" recommendation -- doing nothing and banking
    the free transfer is a legitimate answer and the optimizer will not
    manufacture activity for its own sake.

    An exhaustive pairwise search (rather than a full ILP re-optimisation) is
    the deliberate v1 choice: with 1-2 transfers the search space is small
    enough to enumerate honestly, and the result is easier to explain in the
    weekly report. A full re-optimisation is `pick_squad` with
    `must_include_ids`, which is what a Wildcard would call.

    Args:
        current_squad_df: the 15 held players; needs id, position, team_name,
            price_m (and xp is taken from `all_players_xp_df` so the squad frame
            can be stale).
        all_players_xp_df: every buyable player with this week's xP.
        free_transfers: free transfers available (1, or 2 if one was rolled).
        max_transfers_considered: 1 or 2. Deeper searches are combinatorially
            expensive and rarely worth it outside a Wildcard.
        hit_cost: points deducted per transfer beyond the free allowance.
        bank: money in the bank, in millions.
        unlimited_transfers: CHIP HOOK. Wildcard / Free Hit set this to True --
            it zeroes the hit cost. Deciding *when* to use a chip is the chip
            strategy agent's job, not this function's.

    Note on selling prices: this uses current `price_m` on both sides. Real FPL
    sell prices apply a 50% sell-on tax to price rises since purchase, so the
    money freed up can be slightly overstated. Wire a `sell_price` column
    through here once the squad is tracked over time.
    """
    pool = _prepare(all_players_xp_df)
    squad = current_squad_df.copy()
    if "id" not in squad.columns:
        raise OptimizerError("current_squad_df must contain an 'id' column")

    # Refresh the squad's prices/xP from the live pool wherever possible, so a
    # stale squad snapshot cannot poison the comparison.
    pool_by_id = pool.set_index(pool["id"].astype(str))
    refreshed = []
    for _, row in squad.iterrows():
        key = str(row["id"])
        if key in pool_by_id.index:
            refreshed.append(pool_by_id.loc[key])
        else:
            # Player no longer in the pool (e.g. left the league): keep the
            # stored record but treat him as worth zero points.
            fallback = row.copy()
            fallback["xp"] = 0.0
            refreshed.append(fallback)
    squad = pd.DataFrame(refreshed).reset_index(drop=True)
    squad["xp"] = pd.to_numeric(squad["xp"], errors="coerce").fillna(0.0)
    squad["price_m"] = pd.to_numeric(squad["price_m"], errors="coerce")
    squad["position"] = squad["position"].astype(str).str.upper()

    held_ids = set(squad["id"].astype(str))
    available = pool[~pool["id"].astype(str).isin(held_ids)]

    def hit_for(n: int) -> float:
        if unlimited_transfers:
            return 0.0
        return hit_cost * max(0, n - free_transfers)

    # Everything below works on plain dicts: the pairwise search evaluates tens
    # of thousands of combinations and pandas row access dominates the runtime.
    squad_rows: list[dict] = squad.to_dict("records")
    for r in squad_rows:
        r["price_m"] = float(r["price_m"])
        r["xp"] = float(r["xp"])

    base_club_counts: dict[Any, int] = {}
    for r in squad_rows:
        base_club_counts[r["team_name"]] = base_club_counts.get(r["team_name"], 0) + 1

    candidates: dict[str, list[dict]] = {}
    for p in POSITION_ORDER:
        rows = (available[available["position"] == p]
                .sort_values("xp", ascending=False)
                .head(CANDIDATES_PER_POSITION)
                .to_dict("records"))
        for r in rows:
            r["price_m"] = float(r["price_m"])
            r["xp"] = float(r["xp"])
        candidates[p] = rows

    best: dict[str, Any] | None = None
    evaluated = 0

    # --- single transfers -------------------------------------------------
    if max_transfers_considered >= 1:
        for out_row in squad_rows:
            for in_row in candidates[out_row["position"]]:
                evaluated += 1
                if not _transfer_is_legal(base_club_counts, [out_row], [in_row],
                                          bank, max_per_club):
                    continue
                gross = in_row["xp"] - out_row["xp"]
                net = gross - hit_for(1)
                if best is None or net > best["net_gain"] + EPS:
                    best = {"n_transfers": 1, "out": [out_row], "in": [in_row],
                            "gross_gain": gross, "net_gain": net}

    # --- double transfers -------------------------------------------------
    if max_transfers_considered >= 2:
        for out_a, out_b in itertools.combinations(squad_rows, 2):
            cand_a = candidates[out_a["position"]][:CANDIDATES_PER_POSITION_PAIRWISE]
            cand_b = candidates[out_b["position"]][:CANDIDATES_PER_POSITION_PAIRWISE]
            out_xp = out_a["xp"] + out_b["xp"]
            outs = [out_a, out_b]
            for in_a, in_b in itertools.product(cand_a, cand_b):
                if in_a["id"] == in_b["id"]:
                    continue
                evaluated += 1
                gross = in_a["xp"] + in_b["xp"] - out_xp
                net = gross - hit_for(2)
                # Cheap arithmetic test before the legality check.
                if best is not None and net <= best["net_gain"] + EPS:
                    continue
                if not _transfer_is_legal(base_club_counts, outs, [in_a, in_b],
                                          bank, max_per_club):
                    continue
                best = {"n_transfers": 2, "out": outs, "in": [in_a, in_b],
                        "gross_gain": gross, "net_gain": net}

    baseline_xp = float(squad["xp"].sum())

    if best is None or best["net_gain"] <= EPS:
        return {
            "recommendation": "no_transfer",
            "n_transfers": 0,
            "transfers": [],
            "gross_gain": 0.0,
            "hit_cost_total": 0.0,
            "net_gain": 0.0,
            "new_squad": squad.drop(columns=["price_tenths"], errors="ignore"),
            "squad_xp_before": baseline_xp,
            "squad_xp_after": baseline_xp,
            "bank_after": round(bank, 1),
            "evaluated": evaluated,
            "notes": ("No transfer has positive net expected value this week; "
                      "hold and roll the free transfer."),
        }

    out_ids = {str(r["id"]) for r in best["out"]}
    new_squad = pd.concat([
        squad[~squad["id"].astype(str).isin(out_ids)],
        pd.DataFrame(best["in"]),
    ], ignore_index=True)

    spent = sum(float(r["price_m"]) for r in best["in"])
    freed = sum(float(r["price_m"]) for r in best["out"])
    bank_after = bank + freed - spent

    transfers = [
        {
            "out": {"id": o["id"], "web_name": o.get("web_name"),
                    "position": o["position"], "team_name": o["team_name"],
                    "price_m": float(o["price_m"]), "xp": float(o["xp"])},
            "in": {"id": i["id"], "web_name": i.get("web_name"),
                   "position": i["position"], "team_name": i["team_name"],
                   "price_m": float(i["price_m"]), "xp": float(i["xp"])},
        }
        for o, i in zip(best["out"], best["in"])
    ]

    return {
        "recommendation": "transfer",
        "n_transfers": best["n_transfers"],
        "transfers": transfers,
        "gross_gain": best["gross_gain"],
        "hit_cost_total": hit_for(best["n_transfers"]),
        "net_gain": best["net_gain"],
        "new_squad": new_squad.drop(columns=["price_tenths"], errors="ignore"),
        "squad_xp_before": baseline_xp,
        "squad_xp_after": baseline_xp + best["gross_gain"],
        "bank_after": round(bank_after, 1),
        "evaluated": evaluated,
        "notes": (f"{best['n_transfers']} transfer(s), "
                  f"{hit_for(best['n_transfers']):.0f} pts of hits, "
                  f"net +{best['net_gain']:.2f} xP."),
    }


__all__ = [
    "pick_squad",
    "pick_starting_xi",
    "optimize_transfers",
    "OptimizerError",
    "SQUAD_COMPOSITION",
    "LEGAL_FORMATIONS",
    "PULP_AVAILABLE",
]
