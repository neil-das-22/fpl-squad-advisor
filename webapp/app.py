"""
FPL Squad Advisor -- public website backend.

Reuses the existing, tested Python model (xp_model, squad_optimizer,
fixture_run, orchestrator, fpl_client) directly rather than reimplementing
any of it in JavaScript. This is a Tornado app (not FastAPI/Flask -- pip
installs are blocked in this dev sandbox for anything not already
present, and tornado happened to already be installed; functionally it's
an equivalent choice, a real production-grade async Python web framework
that deploys the same way on any standard Python host).

WHAT'S REAL vs. WHAT'S CACHED RIGHT NOW
The player pool and xP table are built at startup from whatever is on
disk in data/raw/. As of the AI Performance page, startup now ALSO makes
a best-effort attempt to refresh those raw files from the live FPL API
first (see `_refresh_raw_data()` below) -- this matters beyond freshness:
it's the only way the app can ever see a newly-finished real gameweek, which
the AI manager's season log needs in order to advance. If that live call
fails (no network -- e.g. this dev sandbox, see fpl_client.py's module
docstring), startup falls back to whatever was already cached on disk, same
graceful-degradation pattern used everywhere else in this project.

THE ONE LIVE NETWORK CALL THIS APP MAKES PER REQUEST
Looking up a manager's own team (`/team/<id>`) calls FPL's public
`/entry/{id}/` and `/entry/{id}/event/{n}/picks/` endpoints directly,
every time -- that data is REAL, per-request, not cached, since it's
specific to whoever's looking up their team. This sandbox's own outbound
network can't reach fantasy.premierleague.com (documented throughout this
project), so that specific call will fail here in dev -- the handler
below falls back to a labelled demo squad so the PAGE can still be built,
tested, and reviewed end-to-end; the live lookup itself only needs a real
network to reach FPL, which any real hosting provider has.
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys

import pandas as pd
import requests
import tornado.ioloop
import tornado.web

# OCR is an OPTIONAL dependency (the `pytesseract` package plus the
# separate `tesseract` system binary -- see webapp/README.md for the
# one-line install command). If either is missing on a given machine, the
# photo-upload team import degrades to a clear error message instead of
# crashing the whole app; team-ID lookup and everything else works either way.
try:
    import pytesseract
    from PIL import Image as PILImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

WEBAPP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBAPP_ROOT)

for sub in ("data", "models", "optimization", "agents"):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, sub))

import fpl_client  # noqa: E402
import fixture_run  # noqa: E402
import orchestrator  # noqa: E402
import squad_optimizer  # noqa: E402
import xp_model  # noqa: E402
import ai_manager  # noqa: E402
import chip_strategy  # noqa: E402

MULTI_GW_START = 1
MULTI_GW_LENGTH = 4
DEFAULT_GAMEWEEK = 1  # which gameweek "my current squad" is looked up for

# Formation display order, attacking end first -- matches how FPL's own app
# lays a squad out on the pitch (keeper nearest the bottom of the screen).
PITCH_POSITION_ORDER = ["FWD", "MID", "DEF", "GKP"]


def _photo_url(code) -> str:
    """FPL's public headshot CDN, keyed by a player's stable `code` (not
    `id` -- see the comment on `code` in data/fpl_client.py). Returns ""
    for a missing/unparseable code so the template can fall back to a
    placeholder instead of requesting a broken image."""
    if code is None or pd.isna(code):
        return ""
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return ""
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code_int}.png"


def _badge_url(team_code) -> str:
    """FPL's team-crest CDN, keyed by the club's stable `team_code` (not
    `team_name`). This exists as a SEPARATE image from the player headshot
    because the two lag independently: FPL updates a club's crest the
    moment its badge changes, but a transferred player's headshot can keep
    showing their old club's kit for weeks until FPL re-shoots it. The
    crest is the more reliable "which club is this really" signal, so it's
    rendered as a small badge overlay rather than relied on via the photo."""
    if team_code is None or pd.isna(team_code):
        return ""
    try:
        code_int = int(team_code)
    except (TypeError, ValueError):
        return ""
    return f"https://resources.premierleague.com/premierleague/badges/50/t{code_int}.png"


# ---------------------------------------------------------------------------
# Photo-upload team import: OCR + fuzzy name matching
# ---------------------------------------------------------------------------
#
# Deliberately NOT trying to recover formation/bench order/captain from the
# image layout (that would need real spatial/vision analysis of where text
# sits on the pitch graphic, not just OCR text extraction) -- this reads
# whatever names tesseract can find anywhere in the image, fuzzy-matches
# them against the real player pool, and hands the recognized 15 to the
# SAME pick_starting_xi() used everywhere else on the site to pick the
# strongest legal XI from them. Clearly disclosed to the user as a best
# guess, not a perfect read of their actual picks.

_OCR_IGNORE_WORDS = {
    "pick", "team", "points", "bench", "transfers", "fixtures", "gameweek",
    "gw", "captain", "vice", "total", "bank", "value", "live", "fpl",
    "my", "squad", "sub", "subs", "starting", "formation", "pts", "avg",
    "average", "rank", "deadline", "chip", "wildcard", "menu", "help",
}


def _clean_ocr_line(line: str) -> str:
    line = re.sub(r"[^A-Za-z'.\- ]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _match_players_from_text(raw_text: str, pool_df: pd.DataFrame,
                             min_score: float = 0.74, max_players: int = 15):
    """Fuzzy-matches OCR'd text against `pool_df`'s `web_name` column.
    Returns (matched_ids, {id: best_score}), best matches first, capped at
    `max_players`. A candidate line only counts once, matched to whichever
    player it looks most like -- this is deliberately conservative (a
    fairly high similarity threshold) since a false match is worse than a
    missed one here: pick_starting_xi() will just report "not enough
    players" rather than silently building the wrong squad."""
    candidates: set[str] = set()
    for line in raw_text.splitlines():
        cleaned = _clean_ocr_line(line)
        if len(cleaned) < 3 or cleaned.lower() in _OCR_IGNORE_WORDS:
            continue
        candidates.add(cleaned)

    names = pool_df["web_name"].astype(str).tolist()
    ids = pool_df["id"].tolist()

    best_match: dict = {}
    for cand in candidates:
        cand_lower = cand.lower()
        best_id, best_score = None, 0.0
        for name, pid in zip(names, ids):
            score = difflib.SequenceMatcher(None, cand_lower, name.lower()).ratio()
            if score > best_score:
                best_score, best_id = score, pid
        if best_id is not None and best_score >= min_score:
            if best_id not in best_match or best_score > best_match[best_id]:
                best_match[best_id] = best_score

    ranked = sorted(best_match.items(), key=lambda kv: kv[1], reverse=True)[:max_players]
    return [pid for pid, _ in ranked], dict(ranked)


# Columns pulled onto a player-detail dict, split by which pool table they
# live on: current-season counting stats + model output are wide (raw_players
# has the counting stats, `players`/adjusted has the model's xp + photo/badge
# urls); prior-season columns only exist on raw_players.
_DETAIL_CURRENT_COLS = [
    "id", "web_name", "full_name", "team_name", "team_short", "position",
    "price_m", "photo_url", "badge_url", "status", "status_meaning", "news",
    "selected_by_percent", "form", "total_points", "points_per_game",
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "expected_goals", "expected_assists",
    "expected_goal_involvements", "bonus", "bps", "ict_index",
    "creativity", "threat", "yellow_cards", "red_cards",
    "clearances_blocks_interceptions", "tackles", "recoveries",
    "penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order",
]
_DETAIL_PRIOR_COLS = [
    "minutes_prev_season", "starts_prev_season", "total_points_prev_season",
    "price_prev_season_m", "points_per_million_prev_season",
    "clean_sheets_prev_season", "clearances_blocks_interceptions_prev_season",
    "tackles_prev_season", "recoveries_prev_season",
    "expected_goals_prev_season", "expected_assists_prev_season",
    "goals_scored_prev_season", "assists_prev_season",
    "creativity_prev_season", "threat_prev_season",
]
_DETAIL_XP_COLS = ["xp", "xp_gw1", "xp_gw2", "xp_gw3", "xp_gw4",
                   "n_fixtures_total", "n_blank_gameweeks", "n_double_gameweeks"]


def _safe_float(v):
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


def _per90(total, minutes) -> float:
    m = _safe_float(minutes)
    t = _safe_float(total)
    if not m or m <= 0 or t is None:
        return 0.0
    return round(t / m * 90, 2)


# Stats a player is ranked against, within their own position, for the
# percentile-profile radar + "where they rank" list. `def_mid_only` stats
# (pure defensive output) are skipped for GKP/FWD -- ranking a striker's
# tackle count against strikers isn't a meaningful signal either way.
_RANK_STATS = [
    ("total_points", "FPL points", False),
    ("form_numeric", "Form", False),
    ("ict_index_numeric", "ICT Index", False),
    ("bonus", "Bonus points", False),
    ("attacking_p90", "Attacking output (G+A per90)", False),
    ("defensive_p90", "Defensive actions per90", True),
]


def _position_pool_stats(position: str) -> pd.DataFrame:
    """The same-position slice of the buyable pool, with the raw counting
    stats needed for ranking joined on and coerced to numeric -- FPL's API
    returns `form`/`ict_index` as strings, which `.rank()` can't touch
    directly."""
    pool = POOL.buyable()
    pool = pool[pool["position"] == position][["id"]]

    raw_cols = ["id", "total_points", "form", "ict_index", "bonus",
               "goals_scored", "assists", "minutes", "tackles",
               "clearances_blocks_interceptions", "recoveries"]
    raw_cols = [c for c in raw_cols if c in POOL.raw_players.columns]
    raw = POOL.raw_players[raw_cols]

    merged = pool.merge(raw, on="id", how="left")
    merged["form_numeric"] = pd.to_numeric(merged.get("form"), errors="coerce")
    merged["ict_index_numeric"] = pd.to_numeric(merged.get("ict_index"), errors="coerce")

    minutes = pd.to_numeric(merged.get("minutes"), errors="coerce")
    safe_minutes = minutes.where(minutes > 0)  # avoid divide-by-zero; NaN minutes -> NaN rate, not 0
    goals = pd.to_numeric(merged.get("goals_scored"), errors="coerce").fillna(0)
    assists = pd.to_numeric(merged.get("assists"), errors="coerce").fillna(0)
    tackles = pd.to_numeric(merged.get("tackles"), errors="coerce").fillna(0)
    cbi = pd.to_numeric(merged.get("clearances_blocks_interceptions"), errors="coerce").fillna(0)
    recoveries = pd.to_numeric(merged.get("recoveries"), errors="coerce").fillna(0)

    merged["attacking_p90"] = ((goals + assists) / safe_minutes * 90).fillna(0)
    merged["defensive_p90"] = ((tackles + cbi + recoveries) / safe_minutes * 90).fillna(0)
    return merged


def _rank_and_percentile(frame: pd.DataFrame, col: str, player_id: int):
    """Where `player_id` sits on `col` within `frame`, among same-position
    players. Returns (rank_from_top, pool_size, percentile_0_100) where
    rank 1 = best and percentile is "beats X% of the position" (so higher
    is always better, regardless of the underlying stat). All None if the
    player or the stat isn't found."""
    if col not in frame.columns or player_id not in frame["id"].values:
        return (None, int(len(frame)), None)
    series = frame[col]
    n = int(series.notna().sum())
    player_rows = frame.loc[frame["id"] == player_id, col]
    if player_rows.empty or n == 0:
        return (None, n, None)
    player_val = player_rows.iloc[0]
    if pd.isna(player_val):
        return (None, n, None)
    rank_from_top = int((series > player_val).sum()) + 1
    percentile = float(series.rank(pct=True)[frame["id"] == player_id].iloc[0]) * 100
    return (rank_from_top, n, round(percentile, 1))


def _player_ranks(position: str, player_id: int) -> dict:
    frame = _position_pool_stats(position)
    ranks = {}
    for col, label, def_mid_only in _RANK_STATS:
        if def_mid_only and position not in ("DEF", "MID"):
            continue
        rank, n, pct = _rank_and_percentile(frame, col, player_id)
        ranks[col] = {"label": label, "rank": rank, "n": n, "percentile": pct}
    return ranks


def player_detail(player_id: int) -> dict | None:
    """Everything the player-detail page and the compare page need for one
    player, flattened into a single JSON/template-safe dict. Pulls from
    TWO pool tables (see column-list comment above) since the counting
    stats and the model's xP projection live on different frames -- the
    caller shouldn't have to know that split exists.

    Returns None if the id isn't in the pool (unknown/bad id -- caller
    should 404).
    """
    raw_match = POOL.raw_players[POOL.raw_players["id"] == player_id]
    xp_match = POOL.players[POOL.players["id"] == player_id]
    if raw_match.empty or xp_match.empty:
        return None

    raw_dict = raw_match.iloc[0].to_dict()
    xp_dict = xp_match.iloc[0].to_dict()

    out: dict = {}
    for col in _DETAIL_CURRENT_COLS:
        out[col] = raw_dict.get(col, xp_dict.get(col))
    for col in _DETAIL_PRIOR_COLS:
        out[col] = raw_dict.get(col)
    for col in _DETAIL_XP_COLS:
        out[col] = xp_dict.get(col)

    minutes = raw_dict.get("minutes") or 0
    out["goals_p90"] = _per90(raw_dict.get("goals_scored"), minutes)
    out["assists_p90"] = _per90(raw_dict.get("assists"), minutes)
    out["xg_p90"] = _per90(raw_dict.get("expected_goals"), minutes)
    out["xa_p90"] = _per90(raw_dict.get("expected_assists"), minutes)
    out["tackles_p90"] = _per90(raw_dict.get("tackles"), minutes)
    out["cbi_p90"] = _per90(raw_dict.get("clearances_blocks_interceptions"), minutes)
    out["recoveries_p90"] = _per90(raw_dict.get("recoveries"), minutes)
    out["has_prior_season"] = bool(pd.notna(raw_dict.get("total_points_prev_season")))

    position = out.get("position")
    if position:
        ranks = _player_ranks(position, player_id)
        out["ranks"] = ranks
        out["position_pool_size"] = next(
            (r["n"] for r in ranks.values() if r["n"] is not None), 0)
    else:
        out["ranks"] = {}
        out["position_pool_size"] = 0

    out["upcoming_fixtures"] = _upcoming_fixtures_with_history(
        out.get("team_name"), raw_dict.get("code"), n=MULTI_GW_LENGTH)

    # NaN/pd.NA -> None everywhere, so the template and json.dumps both get
    # a clean, JSON-safe value instead of choking on a pandas missing-value
    # sentinel. Skip dict/list values (ranks, upcoming_fixtures) -- pd.isna()
    # on a list-like returns an array, not a bool, which blows up the `if`
    # below; those structures are already built clean by their own code.
    for k, v in list(out.items()):
        if isinstance(v, (dict, list)):
            continue
        try:
            is_na = bool(pd.isna(v))
        except (TypeError, ValueError):
            is_na = False
        if is_na:
            out[k] = None

    return out


# Cache-busting token for static assets, derived from style.css's own mtime
# so editing the file automatically invalidates any browser cache on the
# next server restart -- no manual version bump, no leftover stale CSS.
try:
    _STYLE_PATH = os.path.join(WEBAPP_ROOT, "static", "css", "style.css")
    ASSET_VERSION = str(int(os.path.getmtime(_STYLE_PATH)))
except OSError:
    ASSET_VERSION = "1"


# ---------------------------------------------------------------------------
# In-memory data pool, built once at startup (see module docstring)
# ---------------------------------------------------------------------------

def _load_last_season_opponent_history() -> pd.DataFrame | None:
    """Per-fixture 2025/26 results (`merged_gw.csv`, the vaastav-style
    archive already used elsewhere in this project), reshaped to
    (player_code, opponent_team_code) -> what happened. Both codes are
    STABLE identifiers (see the `code` comment in data/fpl_client.py) --
    `merged_gw.csv`'s own `element`/`opponent_team` columns are that
    SEASON's ids, which get reshuffled every year, so both have to be
    translated through that season's own players_raw.csv / teams.csv
    before they mean anything against this season's pool. Returns None
    (not an exception) if the archive isn't on disk -- the "history vs
    this opponent" section just won't have anything to show, same
    graceful-degradation pattern as the rest of this file's optional data.
    """
    hist_dir = os.path.join(PROJECT_ROOT, "data", "raw", "historical_2025_26")
    gw_path = os.path.join(hist_dir, "merged_gw.csv")
    players_path = os.path.join(hist_dir, "players_raw.csv")
    teams_path = os.path.join(hist_dir, "teams.csv")
    if not (os.path.exists(gw_path) and os.path.exists(players_path) and os.path.exists(teams_path)):
        return None

    gw = pd.read_csv(gw_path, usecols=["element", "opponent_team", "total_points",
                                        "goals_scored", "assists", "minutes"])
    season_players = pd.read_csv(players_path, usecols=["id", "code"])
    season_teams = pd.read_csv(teams_path, usecols=["id", "code"])

    gw = gw.merge(season_players.rename(columns={"id": "element", "code": "player_code"}),
                  on="element", how="left")
    gw = gw.merge(season_teams.rename(columns={"id": "opponent_team", "code": "opponent_team_code"}),
                  on="opponent_team", how="left")
    gw = gw.dropna(subset=["player_code", "opponent_team_code"])
    gw["player_code"] = gw["player_code"].astype("int64")
    gw["opponent_team_code"] = gw["opponent_team_code"].astype("int64")
    return gw[["player_code", "opponent_team_code", "total_points",
              "goals_scored", "assists", "minutes"]]


def _upcoming_fixtures_with_history(team_name: str, player_code, n: int = 4) -> list[dict]:
    """This player's next `n` fixtures, each with how they did against
    that SAME opponent last season (if they and the opponent were both
    in the league then). All return values are plain Python types --
    this list bypasses the generic NaN-sanitize pass in player_detail()
    (see the isinstance guard there), so it has to come out clean itself."""
    fixtures = POOL.fixtures
    if fixtures is None or fixtures.empty or not team_name:
        return []

    mask = (fixtures["home_team"] == team_name) | (fixtures["away_team"] == team_name)
    if "finished" in fixtures.columns:
        mask &= ~fixtures["finished"].fillna(False).astype(bool)
    upcoming = fixtures[mask].sort_values("gameweek").head(n)

    team_code_by_name = {}
    if POOL.teams is not None and "code" in POOL.teams.columns:
        team_code_by_name = dict(zip(POOL.teams["name"], POOL.teams["code"]))

    player_code_int = None
    if player_code is not None:
        try:
            if not pd.isna(player_code):
                player_code_int = int(player_code)
        except (TypeError, ValueError):
            player_code_int = None

    hist = POOL.last_season_history

    out = []
    for _, row in upcoming.iterrows():
        is_home = bool(row["home_team"] == team_name)
        opponent = row["away_team"] if is_home else row["home_team"]
        opponent_short = row.get("away_short") if is_home else row.get("home_short")
        difficulty = row.get("team_h_difficulty") if is_home else row.get("team_a_difficulty")

        history = None
        opp_code = team_code_by_name.get(opponent)
        if hist is not None and player_code_int is not None and opp_code is not None and pd.notna(opp_code):
            sub = hist[(hist["player_code"] == player_code_int) &
                      (hist["opponent_team_code"] == int(opp_code))]
            if len(sub):
                history = {
                    "matches": int(len(sub)),
                    "total_points": int(sub["total_points"].sum()),
                    "goals": int(sub["goals_scored"].sum()),
                    "assists": int(sub["assists"].sum()),
                    "minutes": int(sub["minutes"].sum()),
                }

        out.append({
            "gameweek": int(row["gameweek"]) if pd.notna(row.get("gameweek")) else None,
            "opponent": opponent,
            "opponent_short": opponent_short if pd.notna(opponent_short) else "",
            "opponent_badge_url": _badge_url(opp_code) if opp_code is not None else "",
            "is_home": is_home,
            "difficulty": int(difficulty) if pd.notna(difficulty) else None,
            "history_vs_opponent": history,
        })
    return out


_DEFAULT_ALT_PRICE_DELTA = 1.0  # default "similar price range" ceiling, in £m over the player's own price


def _price_alternatives(player_id: int, max_delta: float = _DEFAULT_ALT_PRICE_DELTA, n: int = 8) -> list[dict]:
    """Same-position players (excluding this one) priced at or below
    `player_id`'s own price plus `max_delta` -- covers both "slightly
    pricier upgrade" and "cheaper-but-comparable" alternatives in one
    list, ranked by projected points. `max_delta` is a UI-adjustable
    threshold (see player.html), not a fixed rule."""
    pool = POOL.buyable()
    row = pool[pool["id"] == player_id]
    if row.empty:
        return []
    position = row.iloc[0]["position"]
    price = float(row.iloc[0]["price_m"])
    candidates = pool[(pool["position"] == position) & (pool["id"] != player_id)
                      & (pool["price_m"] <= price + max_delta)]
    candidates = candidates.sort_values("xp", ascending=False).head(n)
    cols = [c for c in ["id", "web_name", "team_name", "position", "price_m",
                        "photo_url", "badge_url", "xp"]
           if c in candidates.columns]
    return candidates[cols].to_dict("records")


def _avg_fixture_difficulty(team_name: str, n: int = MULTI_GW_LENGTH) -> float | None:
    """Mean FPL fixture-difficulty rating (1 easiest - 5 hardest) across a
    team's next `n` unplayed fixtures, from that team's own side of each
    matchup (their opponent's difficulty, not their own). None if the
    team has no unplayed fixtures on the board (shouldn't happen mid
    season, but the fixtures table could in principle be stale/partial)."""
    fixtures = POOL.fixtures
    if fixtures is None or fixtures.empty or not team_name:
        return None
    mask = (fixtures["home_team"] == team_name) | (fixtures["away_team"] == team_name)
    if "finished" in fixtures.columns:
        mask &= ~fixtures["finished"].fillna(False).astype(bool)
    upcoming = fixtures[mask].sort_values("gameweek").head(n)
    if upcoming.empty:
        return None
    diffs = []
    for _, row in upcoming.iterrows():
        is_home = row["home_team"] == team_name
        d = row.get("team_h_difficulty") if is_home else row.get("team_a_difficulty")
        if pd.notna(d):
            diffs.append(float(d))
    return round(sum(diffs) / len(diffs), 1) if diffs else None


# Minimum minutes for a player to count in a RATE stat (points-per-million,
# differentials, correlations, etc.) -- roughly 10 full matches. Without
# this, a player who played 20 minutes and scored a fluke goal looks like
# the best-value pick in the league; total-count stats (most goals, most
# bonus) don't need it since a low-minutes player can't lead those anyway.
_ANALYTICS_MIN_MINUTES = 900

# Human-readable labels for the underlying stats used in the correlation
# panels -- keeps the template from having to know column-name conventions.
_STAT_LABELS = {
    "ict_index": "ICT Index",
    "influence": "Influence",
    "creativity": "Creativity",
    "threat": "Threat",
    "bps": "Bonus Points System score",
    "expected_goals": "Expected goals (xG)",
    "expected_assists": "Expected assists (xA)",
    "expected_goal_involvements": "Expected goal involvements",
    "defensive_contribution": "Defensive contribution",
    "tackles": "Tackles",
    "clearances_blocks_interceptions": "Clearances/blocks/interceptions",
    "recoveries": "Recoveries",
    "saves": "Saves",
    "selected_by_percent": "Ownership %",
    "price_m": "Price",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "expected_goals_conceded": "Expected goals conceded",
}

# Deliberately EXCLUDES minutes/starts (more game time mechanically means
# more chances to accumulate ANY counting stat -- not an interesting
# finding) and goals_scored/assists/clean_sheets/bonus/total_points
# themselves (the "obvious" stats the correlation panel exists to look
# past). What's left is "quality/style of play" stats -- the underlying
# ICT sub-indices, the model-style expected-output numbers, defensive
# actions, discipline, price, and ownership.
_POINTS_CANDIDATES = ["ict_index", "influence", "creativity", "threat", "bps",
                      "expected_goals", "expected_assists", "expected_goal_involvements",
                      "defensive_contribution", "tackles", "clearances_blocks_interceptions",
                      "recoveries", "saves", "selected_by_percent", "price_m",
                      "yellow_cards", "red_cards", "expected_goals_conceded"]

_GA_CANDIDATES = ["ict_index", "influence", "creativity", "threat", "expected_goals",
                  "expected_assists", "expected_goal_involvements", "bps",
                  "selected_by_percent", "price_m"]

# goals_conceded is deliberately excluded here -- a clean sheet is LITERALLY
# defined as goals_conceded == 0 for that match, so correlating the two is
# circular, not informative. expected_goals_conceded (the model's pre-match
# estimate) is the genuinely interesting version of that question.
_CS_CANDIDATES = ["tackles", "clearances_blocks_interceptions", "recoveries",
                  "defensive_contribution", "saves", "expected_goals_conceded",
                  "ict_index", "influence", "threat", "bps"]

_CORRELATION_MIN_SAMPLE = 15  # skip a position/target combo below this qualified-player count


def _position_correlations(qualified_df: pd.DataFrame, position: str, target_col: str,
                           candidate_cols: list[str], top_n: int = 6) -> list[dict]:
    """Pearson correlation between `target_col` and each candidate stat,
    within one position's qualified (>= _ANALYTICS_MIN_MINUTES) players.
    Sorted by strength (|r|) regardless of direction, so a strong negative
    correlate (e.g. yellow cards) surfaces just as readily as a strong
    positive one. Returns [] if the sample's too small or the target has
    no variance (shouldn't happen with real season data, but a defensive
    guard against a degenerate/near-empty position slice)."""
    sub = qualified_df[qualified_df["position"] == position]
    if len(sub) < _CORRELATION_MIN_SAMPLE or sub[target_col].std() == 0:
        return []
    rows = []
    for col in candidate_cols:
        if col not in sub.columns or sub[col].std() == 0:
            continue
        r = sub[target_col].corr(sub[col])
        if pd.isna(r):
            continue
        rows.append({"stat": col, "label": _STAT_LABELS.get(col, col), "r": round(float(r), 3)})
    rows.sort(key=lambda d: abs(d["r"]), reverse=True)
    return rows[:top_n]


def _stat_top_players(qualified_df: pd.DataFrame, position: str, stat_col: str,
                      n: int = 10) -> list[dict]:
    """Top-n qualified players at `position` by raw `stat_col` value --
    backs the "click a correlation bar to see who's actually leading it"
    interaction on the analytics page."""
    sub = qualified_df[qualified_df["position"] == position]
    if stat_col not in sub.columns or sub.empty:
        return []
    top = sub.sort_values(stat_col, ascending=False).head(n)
    cols = [c for c in ["id", "web_name", "team_name", "team_short", "photo_url", "badge_url", stat_col]
           if c in top.columns]
    records = top[cols].to_dict("records")
    for r in records:
        val = r.pop(stat_col, None)
        r["value"] = round(float(val), 2) if pd.notna(val) else None
    return records


def _scatter_points(frame: pd.DataFrame, x_col: str, y_col: str,
                    extra_cols: list[str] | None = None) -> list[dict]:
    pts = []
    for _, row in frame.iterrows():
        x, y = row.get(x_col), row.get(y_col)
        if pd.notna(x) and pd.notna(y):
            pt = {"x": round(float(x), 2), "y": round(float(y), 2),
                 "name": row.get("web_name", "")}
            for col in (extra_cols or []):
                val = row.get(col)
                pt[col] = val.item() if hasattr(val, "item") else val
            pts.append(pt)
    return pts


def _build_last_season_analytics() -> dict | None:
    """Creative last-season (2025/26) analytics, built from the same
    archive already used for prior-season priors elsewhere in this
    project (players_raw.csv, teams.csv, team_possession.csv). Purely a
    reporting page -- nothing computed here feeds the xP model. Returns
    None if the archive isn't on disk, same graceful-degradation pattern
    as everywhere else in this file.
    """
    hist_dir = os.path.join(PROJECT_ROOT, "data", "raw", "historical_2025_26")
    players_path = os.path.join(hist_dir, "players_raw.csv")
    teams_path = os.path.join(hist_dir, "teams.csv")
    possession_path = os.path.join(hist_dir, "team_possession.csv")
    if not (os.path.exists(players_path) and os.path.exists(teams_path)):
        return None

    df = pd.read_csv(players_path)
    # `players_raw.csv` already carries each player's own `code` AND
    # `team_code` natively (same as the live bootstrap payload) -- pull
    # `id` from `teams.csv` for the join, but do NOT pull `code` from it
    # too, or pandas silently suffixes the collision (`code_x`/`code_y`,
    # `team_code_x`/`team_code_y`) and bare `df["code"]`/`df["team_code"]`
    # stop existing. This exact collision was already hit and fixed twice
    # elsewhere in this project (fpl_client.py, and earlier in this same
    # function) -- not pulling the redundant column is the simplest fix.
    teams = pd.read_csv(teams_path, usecols=["id", "name", "short_name"])
    position_names = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    df = df.merge(
        teams.rename(columns={"id": "team", "name": "team_name", "short_name": "team_short"}),
        on="team", how="left")
    df["position"] = df["element_type"].map(position_names)
    df["price_m"] = pd.to_numeric(df["now_cost"], errors="coerce") / 10.0

    numeric_cols = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets",
                    "goals_conceded", "saves", "bonus", "bps", "expected_goals",
                    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
                    "ict_index", "influence", "creativity", "threat",
                    "defensive_contribution", "tackles", "clearances_blocks_interceptions",
                    "recoveries", "yellow_cards", "red_cards", "selected_by_percent"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    price_safe = df["price_m"].where(df["price_m"] > 0)
    df["points_per_million"] = (df["total_points"] / price_safe).fillna(0.0).round(2)
    df["goal_involvements"] = df["goals_scored"] + df["assists"]
    df["goals_minus_xg"] = (df["goals_scored"] - df["expected_goals"]).round(2)
    df["photo_url"] = df["code"].apply(_photo_url)
    df["badge_url"] = df["team_code"].apply(_badge_url)

    qualified = df[df["minutes"] >= _ANALYTICS_MIN_MINUTES].copy()

    # "What predicts FPL points (beyond the obvious)?" -- per-position
    # correlations against the underlying quality-of-play stats, plus the
    # same question asked of goal involvements and clean sheets
    # specifically. See the candidate-list comments above for what's
    # deliberately excluded and why.
    positions = ["GKP", "DEF", "MID", "FWD"]
    correlations: dict[str, dict[str, list[dict]]] = {}
    for pos in positions:
        entry = {"points": _position_correlations(qualified, pos, "total_points", _POINTS_CANDIDATES)}
        if pos != "GKP":
            entry["goal_involvements"] = _position_correlations(
                qualified, pos, "goal_involvements", _GA_CANDIDATES)
        entry["clean_sheets"] = _position_correlations(qualified, pos, "clean_sheets", _CS_CANDIDATES)
        correlations[pos] = entry

    # Top-10 players for each stat actually shown in the "points" bar
    # charts, keyed by position then stat column -- backs the "click a bar
    # to see who's actually leading it" interaction on the analytics page.
    stat_leaders: dict[str, dict[str, list[dict]]] = {}
    for pos in positions:
        stat_leaders[pos] = {
            row["stat"]: _stat_top_players(qualified, pos, row["stat"])
            for row in correlations[pos]["points"]
        }

    # Headline finding: the single strongest non-obvious points correlate,
    # whichever position it comes from -- paired with a scatter of that
    # exact relationship so the claim is visually checkable, not just a
    # number in a table.
    top_points_correlation = None
    for pos, entry in correlations.items():
        if entry["points"]:
            candidate = dict(entry["points"][0])
            candidate["position"] = pos
            if top_points_correlation is None or abs(candidate["r"]) > abs(top_points_correlation["r"]):
                top_points_correlation = candidate

    hero_scatter = []
    if top_points_correlation:
        hero_scatter = _scatter_points(
            qualified[qualified["position"] == top_points_correlation["position"]],
            top_points_correlation["stat"], "total_points")

    price_vs_points = _scatter_points(qualified, "price_m", "total_points", ["position"])
    ownership_vs_points = _scatter_points(qualified, "selected_by_percent", "total_points", ["position"])

    lead_cols = ["id", "web_name", "team_name", "team_short", "position",
                "price_m", "photo_url", "badge_url"]

    def _lead(frame: pd.DataFrame, extra_cols: list[str]) -> list[dict]:
        return frame[lead_cols + extra_cols].to_dict("records")

    best_value = _lead(
        qualified.sort_values("points_per_million", ascending=False).head(15),
        ["total_points", "points_per_million", "minutes"])

    best_value_by_position = {
        pos: _lead(
            qualified[qualified["position"] == pos]
            .sort_values("points_per_million", ascending=False).head(5),
            ["total_points", "points_per_million"])
        for pos in ["GKP", "DEF", "MID", "FWD"]
    }

    goal_contributions = _lead(
        df.sort_values("goal_involvements", ascending=False).head(10),
        ["goals_scored", "assists", "goal_involvements"])

    finishing_pool = qualified[qualified["expected_goals"] >= 2].copy()
    overperformers = _lead(
        finishing_pool.sort_values("goals_minus_xg", ascending=False).head(6),
        ["goals_scored", "expected_goals", "goals_minus_xg"])
    underperformers = _lead(
        finishing_pool.sort_values("goals_minus_xg", ascending=True).head(6),
        ["goals_scored", "expected_goals", "goals_minus_xg"])

    iron_men = _lead(
        df.sort_values("minutes", ascending=False).head(8),
        ["minutes", "total_points"])

    bonus_leaders = _lead(
        df.sort_values("bonus", ascending=False).head(8),
        ["bonus", "total_points"])

    differentials = _lead(
        qualified[(qualified["selected_by_percent"] < 10) & (qualified["total_points"] >= 100)]
        .sort_values("total_points", ascending=False).head(8),
        ["total_points", "selected_by_percent", "points_per_million"])

    busts = _lead(
        qualified[qualified["selected_by_percent"] >= 15]
        .sort_values("points_per_million", ascending=True).head(6),
        ["total_points", "points_per_million", "selected_by_percent"])

    # Team defense: each team's own #1 goalkeeper (most minutes) stands in
    # for the team's season defensive record -- summing clean_sheets/
    # goals_conceded across ALL players on a team would wildly overcount
    # since those columns are per-player ("conceded while I was on the
    # pitch"), not already a team total.
    gk = df[df["position"] == "GKP"].sort_values("minutes", ascending=False)
    team_defense = (gk.groupby(["team_name", "team_short"], as_index=False).first()
                    [["team_name", "team_short", "clean_sheets", "goals_conceded", "minutes"]]
                    .sort_values("clean_sheets", ascending=False)
                    .to_dict("records"))

    team_attack = (df.groupby(["team_name", "team_short"], as_index=False)
                  .agg(goals_scored=("goals_scored", "sum"),
                       assists=("assists", "sum"),
                       squad_points=("total_points", "sum"))
                  .sort_values("goals_scored", ascending=False)
                  .to_dict("records"))

    team_possession = []
    if os.path.exists(possession_path):
        poss = pd.read_csv(possession_path).sort_values(
            "possession_pct_prev_season", ascending=False)
        team_possession = poss.to_dict("records")

    league_summary = {
        "total_players": int(len(df)),
        "total_goals": int(df["goals_scored"].sum()),
        "total_assists": int(df["assists"].sum()),
        "total_points_awarded": int(df["total_points"].sum()),
        "top_scorer": df.loc[df["goals_scored"].idxmax(), "web_name"] if len(df) else None,
        "top_scorer_goals": int(df["goals_scored"].max()) if len(df) else 0,
        "top_points": df.loc[df["total_points"].idxmax(), "web_name"] if len(df) else None,
        "top_points_value": int(df["total_points"].max()) if len(df) else 0,
        "most_owned": df.loc[df["selected_by_percent"].idxmax(), "web_name"] if len(df) else None,
        "most_owned_pct": float(df["selected_by_percent"].max()) if len(df) else 0.0,
        "avg_price": round(float(df["price_m"].mean()), 1) if len(df) else 0.0,
    }

    return {
        "best_value": best_value,
        "best_value_by_position": best_value_by_position,
        "goal_contributions": goal_contributions,
        "overperformers": overperformers,
        "underperformers": underperformers,
        "iron_men": iron_men,
        "bonus_leaders": bonus_leaders,
        "differentials": differentials,
        "busts": busts,
        "team_defense": team_defense,
        "team_attack": team_attack,
        "team_possession": team_possession,
        "league_summary": league_summary,
        "min_minutes": _ANALYTICS_MIN_MINUTES,
        "correlations": correlations,
        "stat_leaders": stat_leaders,
        "top_points_correlation": top_points_correlation,
        "hero_scatter": hero_scatter,
        "price_vs_points": price_vs_points,
        "ownership_vs_points": ownership_vs_points,
    }


def _refresh_raw_data() -> bool:
    """Best-effort live refresh of data/raw/bootstrap_static.json and
    fixtures.json from the real FPL API, run once at startup before the
    Pool is built. Returns True if it succeeded, False if it fell back to
    the existing cached files (no network, FPL unreachable, etc.) -- never
    raises, since a failed refresh should degrade to "use what's on disk",
    not take the whole app down.

    This matters beyond just data freshness: the AI Performance page's
    season log can only ever detect a newly-finished real gameweek by
    seeing it in this file's `events` list, so without this refresh the
    log would never advance no matter how many real gameweeks pass.
    """
    try:
        bootstrap = fpl_client.fetch_bootstrap_static()
        fixtures = fpl_client.fetch_fixtures()
    except Exception as exc:  # noqa: BLE001 -- refresh is optional, never fatal
        print(f"[STARTUP] live data refresh unavailable ({exc}) -- using cached data/raw/ files")
        return False

    warnings = fpl_client.validate_schema(bootstrap, fixtures)
    for w in warnings:
        print(f"[STARTUP] live data refresh schema warning: {w}")

    fpl_client.save_raw(bootstrap, "bootstrap_static.json")
    fpl_client.save_raw(fixtures, "fixtures.json")
    print("[STARTUP] refreshed data/raw/ from the live FPL API")
    return True


class Pool:
    """Holds everything the site's pages read from -- one shared object,
    rebuilt by `refresh()`, read by every request handler."""

    def __init__(self) -> None:
        self.players: pd.DataFrame | None = None   # adjusted, multi-GW xP table
        self.teams: pd.DataFrame | None = None
        self.fixtures: pd.DataFrame | None = None
        self.raw_players: pd.DataFrame | None = None  # pre-xP, for shortlist tools
        self.last_season_history: pd.DataFrame | None = None  # per-fixture 2025/26 results
        self.analytics: dict | None = None  # last-season leaderboards, see _build_last_season_analytics()
        self.built_at: str | None = None

    def refresh(self) -> None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        with open(os.path.join(raw_dir, "bootstrap_static.json")) as f:
            bootstrap = json.load(f)
        with open(os.path.join(raw_dir, "fixtures.json")) as f:
            fixtures_raw = json.load(f)

        players_df = fpl_client.load_players_with_prior_season(bootstrap, verbose=False)
        teams_df = fpl_client.load_teams_df(bootstrap)
        team_prior = None
        try:
            team_prior = fpl_client.load_team_prior_season_reference()
        except FileNotFoundError:
            pass
        teams_df = fpl_client.attach_team_prior_season_stats(teams_df, team_prior)
        fixtures_df = fpl_client.load_fixtures_df(fixtures_raw, bootstrap)

        xp_df = orchestrator.build_multi_gw_xp_table(
            MULTI_GW_START, MULTI_GW_LENGTH, write=False, verbose=False)

        overrides_path = os.path.join(PROJECT_ROOT, "data", "processed",
                                      f"research_overrides_gw{MULTI_GW_START}.csv")
        overrides_df = pd.read_csv(overrides_path) if os.path.exists(overrides_path) else None
        adjusted = xp_model.apply_manual_adjustments(xp_df, overrides_df)

        avoid_set = set(orchestrator.AVOID_GW1)
        is_avoided = adjusted.apply(
            lambda r: (r["web_name"], r["team_name"]) in avoid_set, axis=1)
        adjusted = adjusted.copy()
        adjusted["excluded_by_research"] = is_avoided
        adjusted["photo_url"] = (
            adjusted["code"].apply(_photo_url) if "code" in adjusted.columns
            else ""
        )
        adjusted["badge_url"] = (
            adjusted["team_code"].apply(_badge_url) if "team_code" in adjusted.columns
            else ""
        )

        self.raw_players = players_df
        self.teams = teams_df
        self.fixtures = fixtures_df
        self.players = adjusted
        self.last_season_history = _load_last_season_opponent_history()
        self.analytics = _build_last_season_analytics()
        self.built_at = pd.Timestamp.utcnow().isoformat()

    def buyable(self) -> pd.DataFrame:
        """The pool the optimizer should choose from: not research-flagged
        as a confirmed non-starter. Everyone still SHOWS on the browse
        page (excluding a player from view entirely would hide real
        information); this filter is specifically for "who should the
        optimizer suggest," where a confirmed backup should never surface."""
        return self.players[~self.players["excluded_by_research"]].copy()


POOL = Pool()


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def render(handler: tornado.web.RequestHandler, template: str, **kwargs) -> None:
    handler.render(template, built_at=POOL.built_at, asset_version=ASSET_VERSION, **kwargs)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

class HomeHandler(tornado.web.RequestHandler):
    """Team-ID lookup + top picks -- reached at /home (linked from the nav
    and from the Overview page), one step in from the actual "/" landing
    page now that Overview owns that slot."""

    def get(self) -> None:
        top_picks = (POOL.buyable()
                    .sort_values("xp", ascending=False)
                    .head(6)[["web_name", "team_name", "position", "price_m", "xp"]]
                    .to_dict("records"))
        render(self, "home.html", top_picks=top_picks, error=None)


class OverviewHandler(tornado.web.RequestHandler):
    """The site's actual landing page ("/") -- a static guide to what
    every other page does, in a few bullets each. No data dependency at
    all (doesn't touch POOL), so it always renders even if the player
    pool failed to build for some reason."""

    def get(self) -> None:
        render(self, "overview.html")


def _render_squad_page(handler, squad_df: pd.DataFrame, manager_name: str, team_name: str,
                       *, demo_mode: bool = False, entry_id: int | None = None,
                       note: str | None = None, bank: float = 0.0) -> None:
    """Shared by TeamHandler (team-ID lookup) and TeamPhotoHandler (photo
    import) -- both end up with the same shape of input (a 15-row squad_df
    with squad_position/is_captain/is_vice_captain/multiplier set) and want
    the same team.html output, so the actual page-building logic only
    needs to exist once."""
    squad_df = squad_df.sort_values("squad_position")
    # FPL's own convention (see fpl_client.parse_manager_squad): squad
    # slots 1-11 are the starting XI in formation order, 12-15 the
    # bench. Only starters' points count toward the total (bench
    # points are worthless unless Bench Boost is active, which this
    # v1 doesn't model yet).
    starters = squad_df[squad_df["squad_position"] <= 11]
    bench = squad_df[squad_df["squad_position"] > 11]
    base_total = float(starters["xp"].sum())
    captain_row = starters[starters["is_captain"]]
    # multiplier is 2 for a normal captain, 3 under Triple Captain --
    # the "extra" on top of his own already-counted xp is (multiplier - 1).
    captain_extra = float((captain_row["xp"] * (captain_row["multiplier"] - 1)).sum())
    total_xp = base_total + captain_extra

    squad_ids = set(squad_df["id"].tolist())

    def _alternatives_for(position: str, n: int = 3) -> list[dict]:
        """Same-position players NOT already in this squad, ranked by
        projected points -- realistic upgrade candidates for a weak link,
        not a budget-aware transfer plan (that's what the Transfer
        recommendation section below already does with real hit-cost math)."""
        pool = POOL.buyable()
        pool = pool[(pool["position"] == position) & (~pool["id"].isin(squad_ids))]
        return pool.sort_values("xp", ascending=False).head(n)[
            ["id", "web_name", "team_name", "price_m", "xp"]].to_dict("records")

    weakest = []
    for _, row in starters.sort_values("xp").head(3).iterrows():
        weakest.append({
            "id": row["id"], "web_name": row["web_name"], "team_name": row["team_name"],
            "position": row["position"], "xp": row["xp"],
            "alternatives": _alternatives_for(row["position"]),
        })

    # Group the starting XI into pitch rows (attacking end first) for
    # the formation view -- whatever shape the XI actually is (3-4-3,
    # 4-4-2, etc.) falls out naturally since we're just filtering by
    # position, not assuming a fixed count per row.
    pitch_rows = [
        {"position": pos, "players": starters[starters["position"] == pos]
                                      .sort_values("xp", ascending=False)
                                      .to_dict("records")}
        for pos in PITCH_POSITION_ORDER
    ]
    pitch_rows = [row for row in pitch_rows if row["players"]]

    transfer_rec = None
    try:
        transfer_rec = squad_optimizer.optimize_transfers(
            squad_df, POOL.buyable(), free_transfers=1, bank=bank)
    except Exception:  # noqa: BLE001
        transfer_rec = None

    render(handler, "team.html", error=None, demo_mode=demo_mode, note=note,
          manager_name=manager_name, team_name=team_name, entry_id=entry_id,
          pitch_rows=pitch_rows, bench=bench.to_dict("records"),
          total_xp=total_xp, captain_extra=captain_extra, base_total=base_total,
          weakest=weakest, transfer_rec=transfer_rec,
          gw_start=MULTI_GW_START, gw_end=MULTI_GW_START + MULTI_GW_LENGTH - 1)


def _friendly_lookup_error(exc: Exception, entry_id: int) -> str:
    """Turn whatever exception the live FPL lookup raised into a message
    that actually helps a real (non-sandboxed) user fix it, instead of the
    old one-size-fits-all "unavailable in this environment" line -- that
    line was accurate for the dev sandbox this was originally built in
    (see fpl_client.py's module docstring) but unhelpful/misleading if a
    real live lookup on a real machine failed for a different reason."""
    text = str(exc)

    if isinstance(exc, requests.exceptions.SSLError) or "CERTIFICATE_VERIFY_FAILED" in text:
        return (
            "SSL certificate verification failed while contacting the FPL API. This is a "
            "known macOS + python.org-installer issue, not a bug in this app -- open your "
            "Applications/Python 3.x folder and double-click “Install Certificates.command”, "
            "then try the lookup again."
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return "The FPL API took too long to respond (30s timeout). Check your connection and try again."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "Couldn't establish a connection to fantasy.premierleague.com -- check your internet "
            "connection, or whether a firewall/VPN/network filter on this machine is blocking it."
        )
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            return f"No FPL team found with ID {entry_id} -- double-check the number from your team's URL."
        if status == 429:
            return "The FPL API is rate-limiting requests right now -- wait a moment and try again."
        return f"The FPL API returned an error (HTTP {status}) for team {entry_id}."
    return f"Unexpected error reaching the FPL API: {text}"


class TeamHandler(tornado.web.RequestHandler):
    def get(self, entry_id_str: str) -> None:
        try:
            entry_id = int(entry_id_str)
        except ValueError:
            render(self, "team.html", error="That doesn't look like a valid team ID -- "
                                             "it should be a number.", squad=None)
            return

        demo_mode = False
        bank = 0.0
        lookup_error = None
        try:
            entry = fpl_client.fetch_manager_entry(entry_id)
            picks_payload = fpl_client.fetch_manager_picks(entry_id, DEFAULT_GAMEWEEK)
            squad_df = fpl_client.parse_manager_squad(picks_payload, POOL.players)
            manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
            team_name = entry.get("name", "Your team")
            bank = float(picks_payload.get("entry_history", {}).get("bank", 0)) / 10.0
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            # The live lookup failed -- fall back to a labelled demo squad
            # (our own GW1 recommendation, built via the same tested
            # pick_squad()/pick_starting_xi() pipeline as
            # reports/gw1_recommendation.md, not an ad hoc top-N list) so
            # the page itself is still reviewable end to end. Unlike the
            # old version of this handler, the *actual* reason is captured
            # and shown (see _friendly_lookup_error) instead of silently
            # swallowed behind one generic message -- that was hiding real,
            # fixable causes (bad cert store, firewall, wrong team ID, FPL
            # rate limiting) behind a message that only made sense for the
            # network-restricted dev sandbox this was first built in.
            demo_mode = True
            manager_name, team_name = "Demo Manager", "Preview Squad (demo data -- live FPL lookup unavailable)"
            demo_squad_result = squad_optimizer.pick_squad(
                POOL.buyable(),
                must_include_ids=POOL.buyable()[
                    POOL.buyable()["web_name"] == "Haaland"]["id"].tolist(),
            )
            demo_xi = squad_optimizer.pick_starting_xi(demo_squad_result["squad"])
            squad_df = pd.concat([demo_xi["starting_xi"], demo_xi["bench"]], ignore_index=True)
            squad_df["is_captain"] = squad_df["xp"] == demo_xi["starting_xi"]["xp"].max()
            squad_df["is_vice_captain"] = False
            squad_df["multiplier"] = squad_df["is_captain"].map({True: 2, False: 1})
            squad_df["squad_position"] = range(1, len(squad_df) + 1)
            lookup_error = _friendly_lookup_error(exc, entry_id)

        _render_squad_page(self, squad_df, manager_name, team_name,
                           demo_mode=demo_mode, entry_id=entry_id, bank=bank,
                           note=lookup_error)


class TeamPhotoHandler(tornado.web.RequestHandler):
    """Upload-a-screenshot alternative to team-ID lookup. OCRs the image,
    fuzzy-matches recognized text against the player pool, and hands
    whichever players it's confident about to the same pick_starting_xi()
    used everywhere else on the site. This can't recover a manager's REAL
    bench order or captain choice from a static image -- it picks the
    strongest legal XI from the players it found instead, and says so.
    """

    def post(self) -> None:
        files = self.request.files.get("photo")
        if not files:
            render(self, "team.html",
                  error="No photo was uploaded -- choose a screenshot of your squad and try again.")
            return

        if not _OCR_AVAILABLE:
            render(self, "team.html",
                  error="Photo import needs OCR support (the `pytesseract` Python package plus the "
                        "separate `tesseract` program) which isn't installed in this environment. "
                        "See the README for the one-line install command, or use your team ID instead.")
            return

        try:
            image = PILImage.open(io.BytesIO(files[0]["body"]))
        except Exception:  # noqa: BLE001
            render(self, "team.html",
                  error="Couldn't read that file as an image -- try a PNG or JPG screenshot.")
            return

        try:
            raw_text = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001 -- e.g. tesseract binary missing at runtime
            render(self, "team.html",
                  error=f"OCR couldn't run on this image ({exc}). Use your team ID instead.")
            return

        matched_ids, _scores = _match_players_from_text(raw_text, POOL.players)
        if len(matched_ids) < 11:
            render(self, "team.html",
                  error=(f"Only recognized {len(matched_ids)} player name(s) in that photo -- "
                         "not enough to build a squad. Try a clearer, full screenshot of your "
                         "\"My Team\" pitch view showing all 15 names, or use your team ID instead."))
            return

        squad_df = POOL.players[POOL.players["id"].isin(matched_ids)].copy()
        try:
            xi_result = squad_optimizer.pick_starting_xi(squad_df)
        except squad_optimizer.OptimizerError as exc:
            recognized = ", ".join(sorted(squad_df["web_name"].tolist()))
            render(self, "team.html",
                  error=(f"Recognized {len(matched_ids)} players ({recognized}) but couldn't form "
                         f"a legal starting XI from them ({exc}). OCR likely misread a name or two "
                         "-- try a clearer photo, or use your team ID instead."))
            return

        squad_df = pd.concat([xi_result["starting_xi"], xi_result["bench"]], ignore_index=True)
        squad_df["is_captain"] = squad_df["id"] == xi_result["captain"]["id"]
        squad_df["is_vice_captain"] = squad_df["id"] == xi_result["vice_captain"]["id"]
        squad_df["multiplier"] = squad_df["is_captain"].map({True: 2, False: 1})
        squad_df["squad_position"] = range(1, len(squad_df) + 1)

        note = (
            f"Built from your photo -- recognized {len(matched_ids)} of 15 players. A static image "
            "can't tell us your real bench order or who you actually captained, so this is the "
            "strongest legal XI and captain from the players we found. Double-check it matches "
            "your real picks."
        )
        _render_squad_page(self, squad_df, "Your team", "Imported from photo",
                           demo_mode=False, entry_id=None, note=note)


class PlayersHandler(tornado.web.RequestHandler):
    """Search-driven player page -- replaces the old full-table browse view
    and the separate /compare page. The whole buyable pool's lightweight
    fields ship as one embedded JSON blob (small enough at ~600 players)
    so the search box, single-player preview, and up-to-4 comparison all
    run client-side with no extra round trips."""

    def get(self) -> None:
        cols = ["id", "web_name", "team_name", "team_short", "position",
                "price_m", "photo_url", "badge_url"]
        pool = POOL.buyable().sort_values(["position", "web_name"])
        cols = [c for c in cols if c in pool.columns]
        render(self, "players.html", players=pool[cols].to_dict("records"))


class PlayerHandler(tornado.web.RequestHandler):
    """One player's own page: current-season output, last season for
    comparison, and this model's own GW1-4 projection -- the "why is this
    a good pick" view, reached by clicking any player card."""

    def get(self, player_id_str: str) -> None:
        try:
            player_id = int(player_id_str)
        except ValueError:
            self.set_status(404)
            render(self, "player.html", error="That doesn't look like a valid player ID.", p=None)
            return

        detail = player_detail(player_id)
        if detail is None:
            self.set_status(404)
            render(self, "player.html", error="Couldn't find that player.", p=None)
            return

        render(self, "player.html", error=None, p=detail,
              gw_start=MULTI_GW_START, gw_end=MULTI_GW_START + MULTI_GW_LENGTH - 1)


class PlayerJSONHandler(tornado.web.RequestHandler):
    """Backs the /compare page's fetch-on-select flow -- one player's full
    detail dict as JSON, same shape `player_detail()` returns."""

    def get(self, player_id_str: str) -> None:
        try:
            player_id = int(player_id_str)
        except ValueError:
            self.set_status(400)
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"error": "invalid id"}))
            return

        detail = player_detail(player_id)
        if detail is None:
            self.set_status(404)
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"error": "not found"}))
            return

        self.set_header("Content-Type", "application/json")
        # default=str as a defensive net -- player_detail() already
        # sanitizes NaN/pd.NA, but this guarantees the endpoint never
        # 500s on some future column with an unexpected pandas dtype.
        self.write(json.dumps(detail, default=str))


class AlternativesJSONHandler(tornado.web.RequestHandler):
    """Backs the "Alternatives in a similar price range" panel on the
    player-detail page -- same-position players priced up to a
    (user-adjustable) delta above this one, ranked by projected points."""

    def get(self, player_id_str: str) -> None:
        try:
            player_id = int(player_id_str)
        except ValueError:
            self.set_status(400)
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"error": "invalid id"}))
            return

        try:
            max_delta = float(self.get_argument("max_delta", str(_DEFAULT_ALT_PRICE_DELTA)))
        except ValueError:
            max_delta = _DEFAULT_ALT_PRICE_DELTA
        max_delta = max(0.0, min(max_delta, 10.0))

        alternatives = _price_alternatives(player_id, max_delta=max_delta)
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({"max_delta": max_delta, "alternatives": alternatives}, default=str))


def _build_ai_performance_context(log: dict, full_pool_df: pd.DataFrame, bootstrap: dict) -> dict:
    """Turns the AI manager's raw season log (see agents/ai_manager.py)
    into everything ai_performance.html needs: the current squad in the
    same pitch-view shape team.html already uses, a gameweek-by-gameweek
    log (with the real FPL average score alongside for comparison, from
    bootstrap's own `events[].average_entry_score`), and chip status."""
    events_by_id = {e["id"]: e for e in bootstrap.get("events", [])}
    history = log["history"]
    current_entry = history[-1]
    current_gw = current_entry["gameweek"]

    squad_df = full_pool_df[full_pool_df["id"].isin(current_entry["squad_ids"])].copy()
    squad_df["is_captain"] = squad_df["id"] == current_entry["captain_id"]
    squad_df["is_vice_captain"] = squad_df["id"] == current_entry["vice_captain_id"]

    starters = squad_df[squad_df["id"].isin(current_entry["starting_xi_ids"])]
    bench_order = current_entry["bench_order_ids"]
    bench = squad_df[squad_df["id"].isin(bench_order)].copy()
    bench["_order"] = bench["id"].map({pid: i for i, pid in enumerate(bench_order)})
    bench = bench.sort_values("_order").drop(columns=["_order"])

    pitch_rows = [
        {"position": pos, "players": starters[starters["position"] == pos]
                                      .sort_values("xp", ascending=False)
                                      .to_dict("records")}
        for pos in PITCH_POSITION_ORDER
    ]
    pitch_rows = [row for row in pitch_rows if row["players"]]

    finished = [h for h in history if h["points"] is not None]
    total_points = sum(h["points"] for h in finished)
    running = 0
    gw_log = []
    for h in finished:
        running += h["points"]
        event = events_by_id.get(h["gameweek"], {})
        predicted = h.get("projected_points")  # absent on log entries seeded before this field existed
        gw_log.append({
            "gameweek": h["gameweek"],
            "points": h["points"],
            "projected_points": predicted,
            "delta": round(h["points"] - predicted, 1) if predicted is not None else None,
            "cumulative": running,
            "average_entry_score": event.get("average_entry_score"),
            "chip_played": h["chip_played"],
            "note": h["note"],
        })

    # Prediction accuracy for the most recently finished gameweek -- the new
    # box only appears once there's a finished gameweek with a stored
    # pre-match projection to compare against (older log entries may not
    # have one; that's handled by the None check, not a crash).
    accuracy = None
    if gw_log and gw_log[-1]["projected_points"] is not None:
        last = gw_log[-1]
        accuracy = {
            "gameweek": last["gameweek"],
            "predicted": last["projected_points"],
            "actual": last["points"],
            "delta": last["delta"],
        }

    half = chip_strategy.current_half(current_gw)
    chips_used_this_half = set(log["chips_used"].get(half, []))
    status = chip_strategy.chip_status(current_gw, chips_used_this_half)
    current_event = events_by_id.get(current_gw, {})

    return {
        "pitch_rows": pitch_rows,
        "bench": bench.to_dict("records"),
        "captain_id": current_entry["captain_id"],
        "vice_captain_id": current_entry["vice_captain_id"],
        "current_gw": current_gw,
        "current_note": current_entry["note"],
        "current_chip": current_entry.get("chip_played"),
        "deadline": current_event.get("deadline_time"),
        "gw_finished": bool(current_event.get("finished")),
        "total_points": total_points,
        "squad_value": round(float(squad_df["price_m"].sum()), 1),
        "bank": current_entry.get("bank_after", 0.0),
        "free_transfers": current_entry.get("free_transfers_after"),
        "upcoming_projected_points": current_entry.get("projected_points"),
        "accuracy": accuracy,
        "gw_log": list(reversed(gw_log)),
        "chip_status": status,
        "season_start_note": history[0]["note"],
        "season_complete": current_gw > ai_manager.SEASON_END_GW,
        "chip_labels": {"wildcard": "Wildcard", "free_hit": "Free Hit",
                        "bench_boost": "Bench Boost", "triple_captain": "Triple Captain"},
    }


class AIPerformanceHandler(tornado.web.RequestHandler):
    """The AI Manager's own season -- a fully autonomous manager built on
    this project's own model (squad_optimizer + a handful of new chip
    heuristics, see agents/ai_manager.py), making real, committed
    transfer and chip decisions gameweek by gameweek as the season
    actually progresses. Not a live recalculated snapshot -- every past
    decision was locked in using whatever the model projected at the
    time and is never revised in hindsight."""

    def get(self) -> None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        try:
            with open(os.path.join(raw_dir, "bootstrap_static.json")) as f:
                bootstrap = json.load(f)
        except Exception:  # noqa: BLE001
            bootstrap = {"events": []}

        error = None
        try:
            log = ai_manager.advance(POOL.players, POOL.buyable(), bootstrap)
        except Exception as exc:  # noqa: BLE001 -- show whatever's saved, don't 500
            error = f"Couldn't update the AI manager's season log right now: {exc}"
            log = ai_manager.load_log()

        if log is None:
            render(self, "ai_performance.html",
                  error=error or "The AI manager's season log hasn't been built yet.", data=None)
            return

        data = _build_ai_performance_context(log, POOL.players, bootstrap)
        render(self, "ai_performance.html", error=error, data=data,
              gw_start=MULTI_GW_START, gw_end=MULTI_GW_START + MULTI_GW_LENGTH - 1)


class AnalyticsHandler(tornado.web.RequestHandler):
    """Standalone last-season (2025/26) reporting page -- value leaders,
    goal contribution, finishing over/underperformance, team-level
    attack/defense/possession, and a few fun league-wide facts. See
    _build_last_season_analytics() for how each leaderboard is built."""

    def get(self) -> None:
        if POOL.analytics is None:
            render(self, "analytics.html", data=None)
            return
        render(self, "analytics.html", data=POOL.analytics)


class FinderHandler(tornado.web.RequestHandler):
    """Pick a position and a price ceiling, get the best next-4-gameweek
    picks that fit. Optionally aware of an existing squad (via team ID):
    if that squad already has 3 players from a club (FPL's own per-club
    cap is 3), this won't suggest a 4th from the same club -- a
    recommendation you couldn't legally act on anyway."""

    def get(self) -> None:
        position = self.get_argument("position", "ALL")
        max_price_raw = self.get_argument("max_price", "15.0")
        team_id_raw = self.get_argument("team_id", "").strip()

        try:
            max_price = float(max_price_raw)
        except ValueError:
            max_price = 15.0
        max_price = max(3.5, min(max_price, 16.0))

        excluded_clubs: list[str] = []
        team_lookup_error = None
        if team_id_raw:
            try:
                team_id = int(team_id_raw)
                picks_payload = fpl_client.fetch_manager_picks(team_id, DEFAULT_GAMEWEEK)
                squad_df = fpl_client.parse_manager_squad(picks_payload, POOL.players)
                counts = squad_df["team_name"].value_counts()
                excluded_clubs = [club for club, n in counts.items() if n >= 3]
            except ValueError:
                team_lookup_error = "That doesn't look like a valid team ID -- showing all clubs."
            except Exception:  # noqa: BLE001
                team_lookup_error = (
                    "Couldn't look up that team right now (network lookup unavailable in this "
                    "environment, or the ID doesn't exist) -- showing recommendations without "
                    "the club-limit check.")

        pool = POOL.buyable()
        if position != "ALL":
            pool = pool[pool["position"] == position]
        pool = pool[pool["price_m"] <= max_price]
        if excluded_clubs:
            pool = pool[~pool["team_name"].isin(excluded_clubs)]
        pool = pool.sort_values("xp", ascending=False).head(30)

        results = pool.to_dict("records")
        fixture_cache: dict[str, float | None] = {}
        for r in results:
            team = r["team_name"]
            if team not in fixture_cache:
                fixture_cache[team] = _avg_fixture_difficulty(team)
            r["fixture_difficulty"] = fixture_cache[team]

        render(self, "finder.html", results=results, position=position,
              max_price=max_price, team_id=team_id_raw, excluded_clubs=excluded_clubs,
              team_lookup_error=team_lookup_error,
              gw_start=MULTI_GW_START, gw_end=MULTI_GW_START + MULTI_GW_LENGTH - 1)


class TeamLookupRedirectHandler(tornado.web.RequestHandler):
    """No-JS fallback for the home page's lookup form -- JS intercepts the
    submit and redirects client-side in the normal case, but a form should
    still work if JS is somehow unavailable."""

    def get(self) -> None:
        team_id = self.get_argument("id", "").strip()
        if team_id.isdigit():
            self.redirect(f"/team/{team_id}")
        else:
            self.redirect("/home")


class PlayersJSONHandler(tornado.web.RequestHandler):
    """Backs the client-side comparison chart on /players -- every buyable
    player's headline numbers, small enough to ship as one JSON blob
    rather than a per-player round trip."""

    def get(self) -> None:
        cols = ["id", "web_name", "team_name", "position", "price_m", "xp",
                "xp_gw1", "xp_gw2", "xp_gw3", "xp_gw4"]
        pool = POOL.buyable()
        cols = [c for c in cols if c in pool.columns]
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(pool[cols].to_dict("records")))


def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", OverviewHandler),
            (r"/overview", tornado.web.RedirectHandler, {"url": "/"}),
            (r"/home", HomeHandler),
            (r"/team/([0-9]+)", TeamHandler),
            (r"/team-photo", TeamPhotoHandler),
            (r"/team-lookup", TeamLookupRedirectHandler),
            (r"/players", PlayersHandler),
            (r"/player/([0-9]+)", PlayerHandler),
            (r"/compare", tornado.web.RedirectHandler, {"url": "/players"}),
            (r"/analytics", AnalyticsHandler),
            (r"/finder", FinderHandler),
            (r"/ai-performance", AIPerformanceHandler),
            (r"/api/players.json", PlayersJSONHandler),
            (r"/api/player/([0-9]+)\.json", PlayerJSONHandler),
            (r"/api/player/([0-9]+)/alternatives\.json", AlternativesJSONHandler),
        ],
        template_path=os.path.join(WEBAPP_ROOT, "templates"),
        static_path=os.path.join(WEBAPP_ROOT, "static"),
        debug=True,
    )


if __name__ == "__main__":
    print("[STARTUP] checking for live FPL data ...")
    _refresh_raw_data()

    print("[STARTUP] building player pool from data/raw/ ...")
    POOL.refresh()
    print(f"[STARTUP] pool ready: {len(POOL.players)} players, built {POOL.built_at}")

    port = int(os.environ.get("PORT", 8888))
    app = make_app()
    app.listen(port)
    print(f"[STARTUP] listening on http://0.0.0.0:{port}")
    tornado.ioloop.IOLoop.current().start()
