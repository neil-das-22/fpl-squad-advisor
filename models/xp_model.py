"""
Expected points (xP) model — FPL 2026/27.

WHAT THIS IS
------------
A transparent, fully-explainable statistical model that estimates the expected
FPL points a player will score in ONE upcoming gameweek against ONE fixture.
Every scoring rule is modelled as a probability-weighted expected value and
returned as a separate component, so any number the system produces can be
traced back to "6 pts per goal x 0.42 expected goals" rather than falling out
of a black box.

Deliberately NOT machine learning. A gradient-boosted model trained on
historical gameweeks is an explicit later stretch goal (see PROJECT_PLAN.md
build step 3/11); it must be validated by backtest before it replaces this.
This module is the baseline that backtest has to beat.

SCORING RULES IMPLEMENTED (2026/27)
-----------------------------------
  Appearance          1 pt for 1-59 mins, 2 pts for 60+ mins
  Goal                GKP/DEF 6, MID 5, FWD 4
  Assist              3 (all positions)
  Clean sheet         GKP/DEF 4, MID 1, FWD 0 -- only if 60+ mins played
  Goals conceded      GKP/DEF only, -1 per 2 conceded (floor)
  Saves               GKP only, +1 per 3 saves (floor)
  Penalty save        +5     Penalty miss  -2
  Yellow card         -1     Red card      -3     Own goal  -2
  Defensive contrib.  DEF: +2 flat at 10+ CBIT (clearances/blocks/
                      interceptions/tackles).
                      MID/FWD: +2 flat at 12+ CBIRT (as above + recoveries).
                      Capped at 2 -- no reward for exceeding the threshold, so
                      it is modelled as P(threshold) * 2.
  Bonus               3/2/1 to the top-3 BPS scorers in each match. The BPS
                      formula is not fully public and was recalibrated for
                      26/27, so this is a documented heuristic proxy -- see
                      `expected_bonus_points()`.

ALL NUMERIC CONSTANTS BELOW ARE v1 HEURISTICS. They are hand-set from
league-average football priors, not fitted. They are collected at the top of
the file specifically so backtesting (build step 11) can tune them without
touching the logic.

KNOWN DATA GAPS (things data/fpl_client.py does not fetch yet)
--------------------------------------------------------------
  1. `minutes_prev_season` / prior-season starts. At GW1 of a new season every
     player's `minutes` is 0, so start probability has nothing to stand on.
     `estimate_start_probability()` therefore falls back to a flat prior and
     raises the flag "start_prob_default". FIX: extend fpl_client with the
     element-summary `history_past` endpoint, or vaastav's archive, to populate
     a `minutes_prev_season` column. Until then, the position/rotation agents
     are expected to supply `start_probability` overrides.
  2. Defensive-contribution counting stats (CBIT / CBIRT). The FPL API exposes
     these but `load_players_df()` does not keep them. Until it does, DefCon
     uses positional priors -- pass `defcon_per90` to override.
  3. Cards / own goals. `yellow_cards`, `red_cards`, `own_goals` are not kept
     either, so card risk uses positional league-average rates.
  4. Penalty duty. Not in the schema at all. Penalty-save and penalty-miss
     terms default to 0 and only activate if `penalty_share` /
     `penalty_save_share` are passed in (the FWD/MID research agents' job).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

# ---------------------------------------------------------------------------
# SCORING TABLES (hard rules -- do not "tune" these, they are the game's rules)
# ---------------------------------------------------------------------------

GOAL_POINTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
SAVE_POINTS_PER_N_SAVES = (1, 3)          # +1 point per 3 saves
CONCEDED_PENALTY_PER_N = (-1, 2)          # -1 point per 2 goals conceded
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
DEFCON_POINTS = 2

# DefCon thresholds. NOTE: the official rules group goalkeepers with defenders
# on the 10-CBIT threshold, but a GK realistically never reaches it, and the
# GK's defensive value is already captured by the saves + clean-sheet terms.
# Set DEFCON_THRESHOLD["GKP"] = 10 and give GKP a prior below if that changes.
DEFCON_THRESHOLD = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# ---------------------------------------------------------------------------
# v1 HEURISTIC CONSTANTS (tunable by backtest -- build step 11)
# ---------------------------------------------------------------------------

# --- Minutes / appearance -------------------------------------------------
DEFAULT_START_PROBABILITY = 0.65   # used when we have no minutes history at all
SUB_APPEARANCE_RATE = 0.30         # P(appears off the bench | did not start)
P_60_GIVEN_START = 0.86            # P(reaches 60 mins | started)
MEAN_MINUTES_IF_START = 80.0
MEAN_MINUTES_IF_SUB = 18.0
MIN_START_PROBABILITY = 0.02
MAX_START_PROBABILITY = 0.98

# Availability multipliers applied to start probability when
# `chance_of_playing_next_round` is null. Keyed on the FPL `status` code.
STATUS_AVAILABILITY = {
    "a": 1.00,   # available
    "d": 0.50,   # doubtful
    "i": 0.00,   # injured
    "s": 0.00,   # suspended
    "u": 0.00,   # unavailable
    "n": 0.00,   # not in squad
}

# --- Per-90 rate shrinkage (empirical-Bayes style) -------------------------
# A player's observed per-90 rate is shrunk toward a positional prior with a
# weight equivalent to PRIOR_WEIGHT_90S full matches of "fake" prior data.
# Thin samples therefore regress hard toward the prior instead of producing
# 3.0 xG/90 off 40 minutes of football.
PRIOR_WEIGHT_90S = 6.0
PROMOTED_PRIOR_WEIGHT_MULTIPLIER = 2.0   # promoted clubs: shrink twice as hard

# League-average per-90 priors by position (non-penalty-inclusive xG/xA).
XG90_PRIOR = {"GKP": 0.00, "DEF": 0.055, "MID": 0.115, "FWD": 0.300}
XA90_PRIOR = {"GKP": 0.005, "DEF": 0.060, "MID": 0.130, "FWD": 0.120}

# Minimum minutes before we trust a player's own rate at all (below this the
# shrinkage above does nearly all the work anyway; this is belt-and-braces).
MIN_MINUTES_FOR_OWN_RATE = 180.0

# --- Fixture / team-strength model ----------------------------------------
LEAGUE_AVG_GOALS_CONCEDED = 1.40   # goals per team per match, PL long-run avg
ATTACK_DEFENCE_EXPONENT = 1.40     # sensitivity of scoreline to strength ratio
HOME_CONCEDE_MULTIPLIER = 0.88     # you concede less at home
AWAY_CONCEDE_MULTIPLIER = 1.12
HOME_ATTACK_MULTIPLIER = 1.10
AWAY_ATTACK_MULTIPLIER = 0.92

# FPL's own 1-5 fixture-difficulty rating. Kept deliberately mild because it is
# strongly correlated with the team-strength ratings we already use -- this is
# a nudge on top, not a second full model.
DIFFICULTY_CONCEDE_MULTIPLIER = {1: 0.82, 2: 0.91, 3: 1.00, 4: 1.10, 5: 1.20}
DIFFICULTY_ATTACK_MULTIPLIER = {1: 1.22, 2: 1.10, 3: 1.00, 4: 0.90, 5: 0.80}

LAMBDA_CONCEDED_BOUNDS = (0.25, 3.20)
CLEAN_SHEET_PROB_BOUNDS = (0.05, 0.65)
ATTACK_MULTIPLIER_BOUNDS = (0.40, 2.20)

# Fallback team strength if a team row is missing a rating (keeps the model
# running on partial data rather than exploding).
DEFAULT_TEAM_STRENGTH = 1150.0

# --- Saves ----------------------------------------------------------------
# Implied by a ~68% league save rate: shots on target faced ~= goals / (1-0.68),
# so saves ~= goals * 0.68/0.32 ~= 2.1x expected goals conceded.
SAVES_PER_GOAL_CONCEDED = 2.10

# --- Defensive contribution -----------------------------------------------
# Per-90 CBIT (DEF) / CBIRT (MID, FWD) priors. Used until fpl_client fetches
# the real counting stats -- see KNOWN DATA GAPS #2.
DEFCON_PER90_PRIOR = {"GKP": 0.0, "DEF": 6.5, "MID": 8.5, "FWD": 5.0}
# Counting stats like tackles/recoveries are over-dispersed relative to Poisson
# (role matters enormously: a ball-winning CDM vs an inverted full-back). We
# model the count as negative-binomial with variance = mean * this factor, which
# fattens the tail and stops us badly under-rating threshold-hitters.
DEFCON_OVERDISPERSION = 1.60

# --- Cards / own goals (positional league-average per-90 rates) ------------
YELLOW_PER90_PRIOR = {"GKP": 0.040, "DEF": 0.200, "MID": 0.185, "FWD": 0.130}
RED_PER90_PRIOR = {"GKP": 0.006, "DEF": 0.014, "MID": 0.010, "FWD": 0.008}
OWN_GOAL_PER90_PRIOR = {"GKP": 0.004, "DEF": 0.011, "MID": 0.004, "FWD": 0.002}

# --- Penalties (only active when a penalty share is supplied) --------------
TEAM_PENALTIES_PER_MATCH = 0.14    # ~5.3 pens per team per 38-game season
PENALTY_CONVERSION_RATE = 0.78     # so miss rate = 0.22
PENALTY_FACED_PER_MATCH = 0.14     # pens faced by a GK's team, same base rate
PENALTY_SAVE_RATE = 0.22           # P(GK saves a penalty he faces)

# --- Bonus proxy ----------------------------------------------------------
# See expected_bonus_points() for the full rationale. These weights convert
# "expected standout involvement" into expected bonus points.
BONUS_WEIGHT_GOAL = 4.00
BONUS_WEIGHT_ASSIST = 2.60
BONUS_WEIGHT_CLEAN_SHEET = 1.20    # multiplied by P(CS), GKP/DEF only
BONUS_WEIGHT_SAVE_TRIPLE = 0.45    # per expected 3-save block, GKP only
BONUS_WEIGHT_DEFCON = 0.55         # multiplied by P(DefCon threshold)
BONUS_WEIGHT_APPEARANCE = 0.06     # everyone who plays has some BPS floor
BONUS_SCALE = 0.30
BONUS_POINTS_CAP = 1.80            # nobody averages more than this per match

# --- Promoted-team handling -----------------------------------------------
# Promoted clubs (see fpl_client.PROMOTED_TEAMS_2026_27) have no top-flight
# underlying data. Rather than trusting a thin/absent sample or silently
# treating them as average, we (a) shrink their per-90 rates twice as hard
# (PROMOTED_PRIOR_WEIGHT_MULTIPLIER), (b) discount attacking output, and
# (c) cap clean-sheet probability well below the league ceiling. Every player
# affected carries the "promoted_team_fallback" flag so the output is auditable
# and a human/agent can override it.
PROMOTED_ATTACK_DISCOUNT = 0.85
PROMOTED_CLEAN_SHEET_CAP = 0.32
PROMOTED_XG90_PRIOR_DISCOUNT = 0.80   # their positional prior is below-average


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float.

    The FPL API returns several numeric fields as strings ("0.42", "12.3") and
    a CSV round-trip through data/processed/ turns *everything* into strings,
    so every read of a numeric column goes through here.
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Field access that works for dicts, pandas Series and namedtuple-ish rows."""
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return getattr(row, key, default)
    if value is None:
        return default
    # pandas gives NaN for missing values
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def expected_floor_divide(lam: float, divisor: int, k_max: int = 30) -> float:
    """E[floor(K / divisor)] where K ~ Poisson(lam).

    Used for the two "per N events" scoring rules (saves per 3, conceded per 2).
    Those are step functions, so E[floor(K/n)] != floor(E[K]/n) -- computing the
    expectation properly matters (a GK facing 1.4 expected goals is not simply
    "0 penalty points").
    """
    if lam <= 0:
        return 0.0
    total = 0.0
    for k in range(k_max + 1):
        total += _poisson_pmf(k, lam) * (k // divisor)
    return total


def _negbinom_pmf(k: int, mean: float, overdispersion: float) -> float:
    """Negative-binomial pmf parameterised by mean and var = mean * overdispersion."""
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    p = 1.0 / overdispersion
    r = mean * p / (1.0 - p)
    return math.exp(
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log1p(-p)
    )


def prob_at_least(threshold: int, mean: float, overdispersion: float = 1.0,
                  k_max: int = 60) -> float:
    """P(count >= threshold) for a Poisson (overdispersion=1) or NB count."""
    if threshold is None:
        return 0.0
    if mean <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    pmf = (_poisson_pmf if overdispersion <= 1.0
           else lambda k, m: _negbinom_pmf(k, m, overdispersion))
    below = sum(pmf(k, mean) for k in range(int(threshold)))
    return _clip(1.0 - below, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Minutes model
# ---------------------------------------------------------------------------

def availability_multiplier(player_row: Any) -> tuple[float, list[str]]:
    """Injury/suspension haircut on start probability, with an audit flag.

    `chance_of_playing_next_round` is the more specific signal when FPL
    publishes it (it is null for fully-fit players), so it wins over `status`.
    """
    flags: list[str] = []
    chance = _get(player_row, "chance_of_playing_next_round", None)
    status = str(_get(player_row, "status", "a") or "a").lower()

    if chance is not None and str(chance).strip() != "":
        mult = _clip(_to_float(chance, 100.0) / 100.0, 0.0, 1.0)
        if mult < 1.0:
            flags.append(f"availability_{mult:.2f}")
        return mult, flags

    mult = STATUS_AVAILABILITY.get(status, 1.0)
    if mult < 1.0:
        flags.append(f"status_{status}")
    return mult, flags


def estimate_start_probability(player_row: Any,
                               matches_played: float | None = None,
                               default_start_probability: float = DEFAULT_START_PROBABILITY,
                               ) -> tuple[float, list[str]]:
    """Estimate P(player starts this match).

    PLACEHOLDER WARNING (KNOWN DATA GAP #1): the only minutes signal available
    from `load_players_df()` is *current-season* `minutes`/`starts`. In GW1 of a
    new season both are 0 for everyone, so this returns a flat prior and flags
    "start_prob_default". Once fpl_client exposes prior-season minutes
    (`minutes_prev_season`), feed it in here -- or pass `start_probability`
    explicitly, which is how the position research agents are meant to inject
    their judgement.

    Args:
        player_row: a row from load_players_df().
        matches_played: how many league matches the season is into. If None it
            is inferred by the batch caller from the max minutes in the player
            pool (the busiest outfielder plays ~every minute).
    Returns:
        (probability, flags)
    """
    avail_mult, flags = availability_multiplier(player_row)

    minutes = _to_float(_get(player_row, "minutes", 0.0))
    starts = _to_float(_get(player_row, "starts", 0.0))

    if matches_played is None or matches_played < 1 or minutes <= 0:
        base = default_start_probability
        flags.append("start_prob_default")
    elif starts > 0:
        base = starts / matches_played
    else:
        # No `starts` column (older payloads) -- back out an implied start rate
        # from minutes, assuming a start is worth ~MEAN_MINUTES_IF_START mins.
        base = minutes / (MEAN_MINUTES_IF_START * matches_played)
        flags.append("start_prob_from_minutes")

    p_start = _clip(base * avail_mult, 0.0, MAX_START_PROBABILITY)
    if p_start > 0:
        p_start = max(p_start, MIN_START_PROBABILITY)
    return p_start, flags


def minutes_distribution(p_start: float, availability: float = 1.0) -> dict[str, float]:
    """Turn a start probability into the appearance quantities the rules need.

    `availability` gates the substitute-appearance term as well as the start
    term. Without it, an injured player with p_start = 0 would still be credited
    with a 30% chance of coming off the bench -- and therefore non-zero xP,
    which is exactly wrong for someone who is not in the squad at all.
    """
    p_start = _clip(p_start, 0.0, 1.0)
    availability = _clip(availability, 0.0, 1.0)
    p_appear = _clip(p_start + (1.0 - p_start) * SUB_APPEARANCE_RATE * availability, 0.0, 1.0)
    p_60 = _clip(p_start * P_60_GIVEN_START, 0.0, p_appear)
    p_short = max(0.0, p_appear - p_60)
    expected_minutes = (p_start * MEAN_MINUTES_IF_START
                        + max(0.0, p_appear - p_start) * MEAN_MINUTES_IF_SUB)
    return {
        "p_start": p_start,
        "p_appear": p_appear,
        "p_60": p_60,
        "p_short_appearance": p_short,
        "expected_minutes": expected_minutes,
    }


# ---------------------------------------------------------------------------
# Per-90 rates
# ---------------------------------------------------------------------------

def shrunk_per90_rate(total: float, minutes: float, prior_rate: float,
                      prior_weight_90s: float = PRIOR_WEIGHT_90S) -> float:
    """Empirical-Bayes style shrinkage of an observed per-90 rate to a prior.

        rate = (observed_total + prior_rate * prior_weight)
               / (n_90s_played  + prior_weight)

    With zero minutes this returns the prior exactly, which is the behaviour we
    want at the start of a season; with a large sample the prior washes out.
    """
    n90 = max(0.0, minutes) / 90.0
    if minutes < MIN_MINUTES_FOR_OWN_RATE:
        # Tiny samples: still blend, but the prior dominates by construction.
        total = max(0.0, total)
    return (total + prior_rate * prior_weight_90s) / (n90 + prior_weight_90s)


# ---------------------------------------------------------------------------
# Fixture model: expected goals conceded, clean sheet probability, attack boost
# ---------------------------------------------------------------------------

def _strength(team_row: Any, key: str) -> float:
    value = _to_float(_get(team_row, key, None), 0.0)
    if value <= 0:
        value = _to_float(_get(team_row, "strength_overall_home", None), 0.0)
    return value if value > 0 else DEFAULT_TEAM_STRENGTH


def fixture_context(own_team_row: Any, opponent_team_row: Any, is_home: bool,
                    difficulty: int | None) -> dict[str, float]:
    """Derive the fixture-level quantities every scoring component depends on.

    Model (documented v1 heuristic):

      lambda_conceded = LEAGUE_AVG
                        * (opp_attack / own_defence) ** ATTACK_DEFENCE_EXPONENT
                        * home/away multiplier
                        * difficulty multiplier

      P(clean sheet)  = exp(-lambda_conceded)          [Poisson, k = 0]

    Deriving the clean-sheet probability from the same lambda that drives the
    goals-conceded penalty keeps the two internally consistent -- a defender
    cannot simultaneously be likely to keep a clean sheet and likely to ship
    three. exp(-1.4) = 0.247, which lines up with the real league-wide
    clean-sheet rate of roughly a quarter of team-matches.

    The attack multiplier is the mirror image, scaling a player's own xG/xA per
    90 by how favourable this specific fixture is.
    """
    if difficulty is None:
        difficulty = 3
    difficulty = int(_clip(_to_float(difficulty, 3.0), 1, 5))

    own_def = _strength(own_team_row,
                        "strength_defence_home" if is_home else "strength_defence_away")
    own_att = _strength(own_team_row,
                        "strength_attack_home" if is_home else "strength_attack_away")
    opp_def = _strength(opponent_team_row,
                        "strength_defence_away" if is_home else "strength_defence_home")
    opp_att = _strength(opponent_team_row,
                        "strength_attack_away" if is_home else "strength_attack_home")

    concede_ratio = (opp_att / own_def) ** ATTACK_DEFENCE_EXPONENT
    lam = (LEAGUE_AVG_GOALS_CONCEDED
           * concede_ratio
           * (HOME_CONCEDE_MULTIPLIER if is_home else AWAY_CONCEDE_MULTIPLIER)
           * DIFFICULTY_CONCEDE_MULTIPLIER[difficulty])
    lam = _clip(lam, *LAMBDA_CONCEDED_BOUNDS)

    attack_ratio = (own_att / opp_def) ** ATTACK_DEFENCE_EXPONENT
    attack_mult = (attack_ratio
                   * (HOME_ATTACK_MULTIPLIER if is_home else AWAY_ATTACK_MULTIPLIER)
                   * DIFFICULTY_ATTACK_MULTIPLIER[difficulty])
    attack_mult = _clip(attack_mult, *ATTACK_MULTIPLIER_BOUNDS)

    p_cs = _clip(math.exp(-lam), *CLEAN_SHEET_PROB_BOUNDS)

    return {
        "lambda_conceded": lam,
        "clean_sheet_prob": p_cs,
        "attack_multiplier": attack_mult,
        "difficulty": float(difficulty),
        "is_home": 1.0 if is_home else 0.0,
    }


# ---------------------------------------------------------------------------
# Bonus points proxy
# ---------------------------------------------------------------------------

def expected_bonus_points(position: str, expected_goals: float,
                          expected_assists: float, clean_sheet_prob: float,
                          expected_saves: float, defcon_prob: float,
                          p_appear: float) -> float:
    """HEURISTIC PROXY for expected bonus points -- NOT the real BPS algorithm.

    Why a proxy: bonus (3/2/1) goes to the top three BPS scorers *within a
    match*, which makes it (a) a rank statistic across 28-ish players we would
    have to co-simulate, and (b) driven by a BPS formula that FPL has never
    fully published and quietly recalibrated for 26/27 (weights shifted toward
    goalkeepers, full-backs, attacking mids and forwards, and away from the
    tackle/recovery counts that now double-count with DefCon).

    What this does instead: build an "expected standout involvement" score from
    the returns that dominate BPS in practice (goals, assists, clean sheets,
    save blocks, hitting the DefCon threshold), then map it linearly to points
    and cap it. Calibration targets, chosen from observed season averages:
      - a premium forward on ~0.7 xG lands near 0.85 bonus/match
      - a first-choice defender in a good defence lands near 0.25-0.40
      - a fringe/rotation player lands near 0.05

    TODO (build step 11, backtest): once per-gameweek BPS history can be pulled
    (element-summary `history` includes `bps`), fit this properly -- ideally as
    P(finish top-3 BPS in this fixture) x point value, estimated against real
    match data. Until then treat bonus as the least trustworthy component; it is
    returned separately from the total precisely so it can be zeroed or
    re-weighted without touching anything else.
    """
    score = (
        BONUS_WEIGHT_GOAL * expected_goals
        + BONUS_WEIGHT_ASSIST * expected_assists
        + BONUS_WEIGHT_APPEARANCE * p_appear
        + BONUS_WEIGHT_DEFCON * defcon_prob
    )
    if position in ("GKP", "DEF"):
        score += BONUS_WEIGHT_CLEAN_SHEET * clean_sheet_prob
    if position == "GKP":
        score += BONUS_WEIGHT_SAVE_TRIPLE * (expected_saves / 3.0)

    return _clip(BONUS_SCALE * score, 0.0, BONUS_POINTS_CAP)


# ---------------------------------------------------------------------------
# Core single-fixture xP calculation
# ---------------------------------------------------------------------------

def calculate_xp(player_row: Any,
                 fixture_row: Any,
                 opponent_team_row: Any,
                 own_team_row: Any,
                 **overrides: Any) -> dict[str, Any]:
    """Expected FPL points for one player in one fixture.

    Args:
        player_row: row from `fpl_client.load_players_df()`.
        fixture_row: row from `fpl_client.load_fixtures_df()`.
        opponent_team_row / own_team_row: rows from `load_teams_df()`.
        **overrides: any of --
            start_probability (float 0-1)  -- bypass the minutes placeholder
            expected_minutes (float)       -- bypass the minutes model entirely
            matches_played (float)         -- season progress, for start-rate calc
            clean_sheet_probability (float)
            lambda_conceded (float)
            attack_multiplier (float)
            defcon_per90 (float)           -- player's CBIT/CBIRT rate per 90
            penalty_share (float 0-1)      -- share of the team's penalties taken
            penalty_save_share (float 0-1) -- share of penalties faced by this GK
            is_home (bool)                 -- normally inferred from the fixture

    Returns:
        dict with one key per scoring component (`xp_*`), a `xp_total`, the
        intermediate probabilities used, and a `flags` list recording every
        fallback that fired.
    """
    flags: list[str] = []

    position = str(_get(player_row, "position", "MID") or "MID").upper()
    if position not in GOAL_POINTS:
        flags.append(f"unknown_position_{position}")
        position = "MID"

    team_name = _get(player_row, "team_name", None)
    own_team_name = _get(own_team_row, "name", team_name)
    home_team = _get(fixture_row, "home_team", None)

    if "is_home" in overrides:
        is_home = bool(overrides["is_home"])
    else:
        is_home = (home_team is not None
                   and str(home_team) == str(own_team_name or team_name))

    difficulty = _get(fixture_row, "team_h_difficulty" if is_home else "team_a_difficulty", 3)

    is_promoted = bool(_get(player_row, "is_promoted", False)) or bool(
        _get(own_team_row, "is_promoted", False))
    if is_promoted:
        flags.append("promoted_team_fallback")

    # --- minutes ----------------------------------------------------------
    availability, _ = availability_multiplier(player_row)
    if "start_probability" in overrides and overrides["start_probability"] is not None:
        p_start = _clip(_to_float(overrides["start_probability"], DEFAULT_START_PROBABILITY), 0.0, 1.0)
        flags.append("start_prob_override")
        # An explicit override is a stronger signal than the availability flag
        # (that is the whole point of the hook), so do not double-discount it.
        availability = 1.0
    else:
        p_start, minute_flags = estimate_start_probability(
            player_row, matches_played=overrides.get("matches_played"))
        flags.extend(minute_flags)

    mins = minutes_distribution(p_start, availability)
    if "expected_minutes" in overrides and overrides["expected_minutes"] is not None:
        mins["expected_minutes"] = max(0.0, _to_float(overrides["expected_minutes"], 0.0))
        flags.append("expected_minutes_override")
    minutes_share = mins["expected_minutes"] / 90.0

    # --- fixture context --------------------------------------------------
    ctx = fixture_context(own_team_row, opponent_team_row, is_home, difficulty)
    lam = _to_float(overrides.get("lambda_conceded"), ctx["lambda_conceded"])
    attack_mult = _to_float(overrides.get("attack_multiplier"), ctx["attack_multiplier"])
    p_cs = _to_float(overrides.get("clean_sheet_probability"), ctx["clean_sheet_prob"])

    if is_promoted:
        # Conservative fallback: cap the upside rather than trusting thin data.
        attack_mult *= PROMOTED_ATTACK_DISCOUNT
        p_cs = min(p_cs, PROMOTED_CLEAN_SHEET_CAP)

    # --- per-90 attacking rates ------------------------------------------
    minutes_played = _to_float(_get(player_row, "minutes", 0.0))
    xg_total = _to_float(_get(player_row, "expected_goals", 0.0))
    xa_total = _to_float(_get(player_row, "expected_assists", 0.0))

    prior_weight = PRIOR_WEIGHT_90S
    xg_prior = XG90_PRIOR[position]
    xa_prior = XA90_PRIOR[position]
    if is_promoted:
        prior_weight *= PROMOTED_PRIOR_WEIGHT_MULTIPLIER
        xg_prior *= PROMOTED_XG90_PRIOR_DISCOUNT
        xa_prior *= PROMOTED_XG90_PRIOR_DISCOUNT

    if minutes_played <= 0:
        flags.append("no_minutes_history")

    xg90 = shrunk_per90_rate(xg_total, minutes_played, xg_prior, prior_weight)
    xa90 = shrunk_per90_rate(xa_total, minutes_played, xa_prior, prior_weight)

    exp_goals = xg90 * minutes_share * attack_mult
    exp_assists = xa90 * minutes_share * attack_mult

    # --- components -------------------------------------------------------
    xp_appearance = 1.0 * mins["p_short_appearance"] + 2.0 * mins["p_60"]
    xp_goals = exp_goals * GOAL_POINTS[position]
    xp_assists = exp_assists * ASSIST_POINTS
    # Clean sheet only counts with 60+ minutes played.
    xp_clean_sheet = CLEAN_SHEET_POINTS[position] * p_cs * mins["p_60"]

    # Goals conceded: -1 per 2 conceded *while the player is on the pitch*, so
    # the lambda is scaled by his share of the match, and the step function is
    # integrated properly rather than applied to the mean.
    xp_conceded = 0.0
    if position in ("GKP", "DEF"):
        lam_on_pitch = lam * minutes_share
        pts, per = CONCEDED_PENALTY_PER_N
        xp_conceded = pts * expected_floor_divide(lam_on_pitch, per)

    # Saves: +1 per 3. Save volume scales with how much shooting the opponent
    # does, which lambda already encodes.
    xp_saves = 0.0
    expected_saves = 0.0
    if position == "GKP":
        expected_saves = SAVES_PER_GOAL_CONCEDED * lam * minutes_share
        pts, per = SAVE_POINTS_PER_N_SAVES
        xp_saves = pts * expected_floor_divide(expected_saves, per)

    # DefCon: flat +2 at the threshold, so model P(threshold) * 2 and nothing
    # more -- exceeding the threshold is worth zero extra points.
    threshold = DEFCON_THRESHOLD[position]
    defcon_per90 = overrides.get("defcon_per90")
    if defcon_per90 is None:
        defcon_per90 = DEFCON_PER90_PRIOR[position]
        if threshold is not None:
            flags.append("defcon_prior")
    defcon_per90 = _to_float(defcon_per90, 0.0)

    defcon_prob = 0.0
    if threshold is not None and defcon_per90 > 0:
        # Conditional on appearing, then weighted by P(appear): a player who
        # does not play cannot rack up clearances.
        mean_if_playing = defcon_per90 * (MEAN_MINUTES_IF_START / 90.0)
        defcon_prob = prob_at_least(threshold, mean_if_playing,
                                    DEFCON_OVERDISPERSION) * mins["p_60"]
    xp_defcon = DEFCON_POINTS * defcon_prob

    # Cards / own goals: small negative drag, positional priors (DATA GAP #3).
    exp_yellow = YELLOW_PER90_PRIOR[position] * minutes_share
    exp_red = RED_PER90_PRIOR[position] * minutes_share
    exp_og = OWN_GOAL_PER90_PRIOR[position] * minutes_share
    xp_cards = (YELLOW_CARD_POINTS * exp_yellow
                + RED_CARD_POINTS * exp_red
                + OWN_GOAL_POINTS * exp_og)

    # Penalties: inert unless a share is supplied (DATA GAP #4). Note the goal
    # points from a converted penalty are already inside xG, so only the *miss*
    # penalty is added here -- adding the goal again would double count.
    xp_penalties = 0.0
    pen_share = _to_float(overrides.get("penalty_share"), 0.0)
    if pen_share > 0:
        exp_pens_taken = TEAM_PENALTIES_PER_MATCH * pen_share * attack_mult * mins["p_start"]
        xp_penalties += PENALTY_MISS_POINTS * exp_pens_taken * (1.0 - PENALTY_CONVERSION_RATE)
        flags.append("penalty_taker")
    pen_save_share = _to_float(overrides.get("penalty_save_share"), 0.0)
    if pen_save_share > 0 and position == "GKP":
        exp_pens_faced = PENALTY_FACED_PER_MATCH * pen_save_share * (lam / LEAGUE_AVG_GOALS_CONCEDED)
        xp_penalties += PENALTY_SAVE_POINTS * exp_pens_faced * PENALTY_SAVE_RATE * mins["p_start"]

    xp_bonus = expected_bonus_points(
        position=position,
        expected_goals=exp_goals,
        expected_assists=exp_assists,
        clean_sheet_prob=p_cs,
        expected_saves=expected_saves,
        defcon_prob=defcon_prob,
        p_appear=mins["p_appear"],
    )

    total = (xp_appearance + xp_goals + xp_assists + xp_clean_sheet
             + xp_conceded + xp_saves + xp_defcon + xp_cards
             + xp_penalties + xp_bonus)

    opponent_name = _get(opponent_team_row, "name", None)
    if opponent_name is None:
        opponent_name = _get(fixture_row, "away_team" if is_home else "home_team", "UNK")

    return {
        "player_id": _get(player_row, "id", None),
        "web_name": _get(player_row, "web_name", None),
        "position": position,
        "team_name": team_name,
        "opponent": opponent_name,
        "is_home": bool(is_home),
        "fixture_id": _get(fixture_row, "id", None),
        "gameweek": _get(fixture_row, "gameweek", None),
        "difficulty": ctx["difficulty"],
        # intermediate quantities (kept for explainability / debugging)
        "p_start": mins["p_start"],
        "p_appear": mins["p_appear"],
        "p_60": mins["p_60"],
        "expected_minutes": mins["expected_minutes"],
        "clean_sheet_prob": p_cs,
        "lambda_conceded": lam,
        "attack_multiplier": attack_mult,
        "expected_goals": exp_goals,
        "expected_assists": exp_assists,
        "expected_saves": expected_saves,
        "defcon_prob": defcon_prob,
        # scoring components
        "xp_appearance": xp_appearance,
        "xp_goals": xp_goals,
        "xp_assists": xp_assists,
        "xp_clean_sheet": xp_clean_sheet,
        "xp_goals_conceded": xp_conceded,
        "xp_saves": xp_saves,
        "xp_defcon": xp_defcon,
        "xp_cards": xp_cards,
        "xp_penalties": xp_penalties,
        "xp_bonus": xp_bonus,
        "xp_total": total,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Batch: one gameweek, all players
# ---------------------------------------------------------------------------

COMPONENT_COLUMNS = [
    "xp_appearance", "xp_goals", "xp_assists", "xp_clean_sheet",
    "xp_goals_conceded", "xp_saves", "xp_defcon", "xp_cards",
    "xp_penalties", "xp_bonus",
]


def infer_matches_played(players_df: pd.DataFrame) -> float:
    """How many league matches the season is into, inferred from the pool.

    The busiest outfielder plays close to every minute, so max(minutes)/90 is a
    good proxy. Returns 0 pre-season (every player on 0 minutes), which is the
    signal `estimate_start_probability` uses to fall back to its flat prior.
    """
    if "minutes" not in players_df.columns or len(players_df) == 0:
        return 0.0
    max_minutes = pd.to_numeric(players_df["minutes"], errors="coerce").max()
    if pd.isna(max_minutes) or max_minutes <= 0:
        return 0.0
    return max(1.0, round(float(max_minutes) / 90.0))


def calculate_xp_for_gameweek(players_df: pd.DataFrame,
                              teams_df: pd.DataFrame,
                              fixtures_df: pd.DataFrame,
                              gameweek: int,
                              player_overrides: Mapping[Any, Mapping[str, Any]] | None = None,
                              **global_overrides: Any) -> pd.DataFrame:
    """Expected points for every player for one gameweek.

    Handles the three fixture cases explicitly:
      * 1 fixture  -- normal.
      * 0 fixtures -- BLANK gameweek for that club. xP = 0 (the player cannot
        score points), `n_fixtures` = 0, flag "blank_gameweek". They are kept in
        the frame rather than dropped so the optimizer still sees them (a
        blanking player is a valid, just worthless, squad member).
      * 2+ fixtures -- DOUBLE gameweek. Components are summed across fixtures,
        which is correct: appearance/goal/clean-sheet points all accrue per
        match independently.

    Args:
        player_overrides: optional {player_id: {override_key: value}} mapping,
            applied per player (this is the channel the research agents use to
            inject e.g. a known start probability).
        **global_overrides: applied to every player (e.g. matches_played).

    Returns:
        DataFrame, one row per player, with `xp` as the headline column.
    """
    player_overrides = player_overrides or {}

    if "matches_played" not in global_overrides:
        global_overrides["matches_played"] = infer_matches_played(players_df)

    teams_by_name: dict[str, Any] = {}
    for _, trow in teams_df.iterrows():
        teams_by_name[str(trow["name"])] = trow

    gw_fixtures = fixtures_df[pd.to_numeric(fixtures_df["gameweek"], errors="coerce") == gameweek]

    # team name -> list of (fixture_row, is_home)
    fixtures_by_team: dict[str, list[tuple[Any, bool]]] = {}
    for _, frow in gw_fixtures.iterrows():
        fixtures_by_team.setdefault(str(frow["home_team"]), []).append((frow, True))
        fixtures_by_team.setdefault(str(frow["away_team"]), []).append((frow, False))

    rows: list[dict[str, Any]] = []
    for _, prow in players_df.iterrows():
        team_name = str(prow.get("team_name", ""))
        own_team_row = teams_by_name.get(team_name)
        player_fixtures = fixtures_by_team.get(team_name, [])

        base = {
            "id": prow.get("id"),
            "web_name": prow.get("web_name"),
            "full_name": prow.get("full_name"),
            "team_name": prow.get("team_name"),
            "team_short": prow.get("team_short"),
            "position": prow.get("position"),
            "price_m": _to_float(prow.get("price_m"), 0.0),
            "is_promoted": bool(prow.get("is_promoted", False)),
            "status": prow.get("status"),
            "gameweek": gameweek,
        }

        overrides = dict(global_overrides)
        overrides.update(player_overrides.get(prow.get("id"), {}))

        if not player_fixtures:
            rows.append({
                **base,
                "n_fixtures": 0,
                "opponents": "",
                **{c: 0.0 for c in COMPONENT_COLUMNS},
                "xp": 0.0,
                "p_start": 0.0,
                "expected_minutes": 0.0,
                "clean_sheet_prob": 0.0,
                "flags": "blank_gameweek",
            })
            continue

        totals = {c: 0.0 for c in COMPONENT_COLUMNS}
        total_xp = 0.0
        opponents: list[str] = []
        all_flags: list[str] = []
        p_start_first = 0.0
        exp_minutes_sum = 0.0
        cs_first = 0.0

        for i, (frow, is_home) in enumerate(player_fixtures):
            opp_name = str(frow["away_team"] if is_home else frow["home_team"])
            opp_row = teams_by_name.get(opp_name)
            result = calculate_xp(prow, frow, opp_row, own_team_row,
                                  is_home=is_home, **overrides)
            for c in COMPONENT_COLUMNS:
                totals[c] += result[c]
            total_xp += result["xp_total"]
            opponents.append(f"{opp_name} ({'H' if is_home else 'A'})")
            all_flags.extend(result["flags"])
            exp_minutes_sum += result["expected_minutes"]
            if i == 0:
                p_start_first = result["p_start"]
                cs_first = result["clean_sheet_prob"]

        if len(player_fixtures) > 1:
            all_flags.append("double_gameweek")

        rows.append({
            **base,
            "n_fixtures": len(player_fixtures),
            "opponents": ", ".join(opponents),
            **totals,
            "xp": total_xp,
            "p_start": p_start_first,
            "expected_minutes": exp_minutes_sum,
            "clean_sheet_prob": cs_first,
            "flags": ";".join(dict.fromkeys(all_flags)),  # de-dupe, keep order
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("xp", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Manual adjustment hook (fan-judgment / research-agent integration point)
# ---------------------------------------------------------------------------

VALID_ADJUSTMENT_TYPES = {"upgrade", "downgrade", "multiplier", "flat_override"}

# Default step size for an upgrade/downgrade with no explicit value, in points.
DEFAULT_ADJUSTMENT_STEP = 0.5


def apply_manual_adjustments(xp_df: pd.DataFrame,
                             overrides_df: pd.DataFrame | None) -> pd.DataFrame:
    """Apply human/agent judgement on top of model xP.

    This is the integration point for `docs/player_judgments.md` (Neil's own
    football calls) and, later, for the position research agents. Parsing the
    markdown is deliberately NOT this function's job -- an orchestrator agent
    turns the doc into `overrides_df` and hands it over, so the model layer
    stays a pure numeric transform and every adjustment is auditable.

    Args:
        xp_df: output of `calculate_xp_for_gameweek()`.
        overrides_df: columns --
            player_name | web_name | id   (any one is enough to match)
            adjustment_type: upgrade | downgrade | multiplier | flat_override
            value: float. For upgrade/downgrade the magnitude is used (sign is
                   implied by the type). Optional -- defaults to
                   DEFAULT_ADJUSTMENT_STEP for upgrade/downgrade.
            reason: optional free text, carried through to `adjustment_notes`.

    Returns:
        A copy of xp_df with `xp_model` (the untouched model number), an updated
        `xp`, and `adjustment_notes`. Rows that match nothing are reported in
        the returned frame's `.attrs["unmatched_overrides"]`.

    Ordering: adjustments are applied in row order, so a `flat_override` placed
    last wins. That is intentional -- a hard override is the strongest signal.
    """
    out = xp_df.copy()
    if "xp_model" not in out.columns:
        out["xp_model"] = out["xp"]
    if "adjustment_notes" not in out.columns:
        out["adjustment_notes"] = ""

    unmatched: list[dict[str, Any]] = []
    out.attrs["unmatched_overrides"] = unmatched

    if overrides_df is None or len(overrides_df) == 0:
        return out

    for _, orow in overrides_df.iterrows():
        adj_type = str(_get(orow, "adjustment_type", "") or "").strip().lower()
        if adj_type not in VALID_ADJUSTMENT_TYPES:
            unmatched.append({"row": dict(orow), "reason": f"unknown adjustment_type '{adj_type}'"})
            continue

        mask = pd.Series(False, index=out.index)
        pid = _get(orow, "id", None)
        if pid is not None and "id" in out.columns:
            mask = mask | (out["id"].astype(str) == str(pid))

        for name_col in ("web_name", "player_name", "full_name"):
            name = _get(orow, name_col, None)
            if name is None or str(name).strip() == "":
                continue
            name = str(name).strip().lower()
            for target_col in ("web_name", "full_name"):
                if target_col in out.columns:
                    mask = mask | (out[target_col].astype(str).str.strip().str.lower() == name)

        if not mask.any():
            unmatched.append({"row": dict(orow), "reason": "no matching player"})
            continue

        raw_value = _get(orow, "value", None)
        if raw_value is None or str(raw_value).strip() == "":
            value = DEFAULT_ADJUSTMENT_STEP if adj_type in ("upgrade", "downgrade") else None
            if value is None:
                unmatched.append({"row": dict(orow), "reason": f"'{adj_type}' requires a value"})
                continue
        else:
            value = _to_float(raw_value, 0.0)

        if adj_type == "upgrade":
            out.loc[mask, "xp"] = out.loc[mask, "xp"] + abs(value)
        elif adj_type == "downgrade":
            out.loc[mask, "xp"] = out.loc[mask, "xp"] - abs(value)
        elif adj_type == "multiplier":
            out.loc[mask, "xp"] = out.loc[mask, "xp"] * value
        elif adj_type == "flat_override":
            out.loc[mask, "xp"] = value

        reason = _get(orow, "reason", "") or ""
        note = f"{adj_type}={value}" + (f" ({reason})" if reason else "")
        existing = out.loc[mask, "adjustment_notes"].astype(str)
        out.loc[mask, "adjustment_notes"] = [
            note if e in ("", "nan") else f"{e}; {note}" for e in existing
        ]

    # A negative predicted score over a whole gameweek is not a meaningful
    # recommendation signal, so downgrades floor at zero. Model output itself is
    # left unfloored in `xp_model` for transparency.
    out["xp"] = out["xp"].astype(float).clip(lower=0.0)
    out.attrs["unmatched_overrides"] = unmatched
    return out


__all__ = [
    "calculate_xp",
    "calculate_xp_for_gameweek",
    "apply_manual_adjustments",
    "estimate_start_probability",
    "minutes_distribution",
    "fixture_context",
    "expected_bonus_points",
    "expected_floor_divide",
    "prob_at_least",
    "shrunk_per90_rate",
    "infer_matches_played",
]
