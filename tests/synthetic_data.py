"""
Synthetic FPL data, shaped exactly like `data/fpl_client.py` output.

WHY THIS EXISTS
---------------
The live FPL API is not reachable from the dev sandbox this was built in (see
the fpl_client docstring), so `data/processed/*.csv` does not exist with real
numbers yet. Everything downstream -- the xP model and the squad optimizer --
is therefore developed and tested against this module. The moment real CSVs
land in data/processed/, the same code runs unmodified: these frames match
`load_players_df()`, `load_teams_df()` and `load_fixtures_df()` column for
column.

FIDELITY NOTES
--------------
* Several numeric fields are deliberately emitted as STRINGS ("0.42", "12.3"),
  because that is what the real FPL API returns for `expected_goals`,
  `expected_assists`, `form`, `points_per_game`, `selected_by_percent` and
  `ict_index` -- and a CSV round-trip stringifies everything anyway. This forces
  the model layer to coerce types instead of assuming floats.
* Player names are invented. Club names are real (they have to be, for the
  promoted-team logic), stats are plausible fiction.
* The fixture set is built to exercise the awkward cases on purpose:
    - Brighton have NO gameweek-1 fixture   -> BLANK gameweek
    - Everton and Fulham each play TWICE in gameweek 1 -> DOUBLE gameweek
"""

from __future__ import annotations

import pandas as pd

# Promoted for 2026/27 -- mirrors fpl_client.PROMOTED_TEAMS_2026_27.
PROMOTED_TEAMS = {"Coventry", "Ipswich", "Hull"}


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

_TEAMS = [
    # id, name, short, strength, ovr_h, ovr_a, att_h, att_a, def_h, def_a
    (1, "Man City",  "MCI", 5, 1350, 1370, 1360, 1380, 1340, 1350),
    (2, "Arsenal",   "ARS", 5, 1330, 1340, 1330, 1340, 1350, 1330),
    (3, "Liverpool", "LIV", 5, 1320, 1330, 1340, 1330, 1300, 1290),
    (4, "Newcastle", "NEW", 4, 1230, 1210, 1220, 1200, 1240, 1220),
    (5, "Brighton",  "BHA", 3, 1150, 1130, 1160, 1140, 1140, 1120),
    (6, "Everton",   "EVE", 3, 1120, 1100, 1090, 1070, 1150, 1130),
    (7, "Brentford", "BRE", 3, 1130, 1110, 1140, 1120, 1110, 1090),
    (8, "Fulham",    "FUL", 3, 1140, 1120, 1130, 1110, 1130, 1110),
    (9, "Coventry",  "COV", 2, 1040, 1010, 1030, 1000, 1050, 1020),
]


def make_sample_teams_df() -> pd.DataFrame:
    """9 clubs with varied strength ratings, one of them promoted (Coventry)."""
    df = pd.DataFrame(_TEAMS, columns=[
        "id", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ])
    df["is_promoted"] = df["name"].isin(PROMOTED_TEAMS)
    return df


_SHORT_BY_TEAM = {name: short for _, name, short, *_ in _TEAMS}


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
# Columns follow load_players_df() exactly.
#
# The pool is built so squad selection is genuinely constrained:
#   * 8 GKP / 12 DEF / 13 MID / 8 FWD = 41 players
#   * prices span 3.9m - 14.5m, so a 15-man squad under 100.0m forces real
#     trade-offs rather than "buy the top 15"
#   * the strongest players are concentrated in 3 clubs, so the max-3-per-club
#     rule actually binds
#   * a handful of injured/doubtful players exercise the availability logic

_PLAYER_SPECS = [
    # (first, last, web, team, pos, price, minutes, starts, goals, assists,
    #  xg, xa, clean_sheets, goals_conceded, xgc, bonus, bps, ict, status, chance, sel)
    # ---- GOALKEEPERS -----------------------------------------------------
    ("Tomas",   "Ekdahl",    "Ekdahl",   "Man City",  "GKP", 5.5, 3330, 37, 0, 1, 0.00, 0.05, 17, 30, 31.2, 18, 720, 68.0,  "a", 100, 18.4),
    ("Rui",     "Barbosa",   "Barbosa",  "Arsenal",   "GKP", 5.4, 3240, 36, 0, 0, 0.00, 0.02, 16, 33, 34.0, 15, 690, 62.1,  "a", 100, 14.2),
    ("Declan",  "Marsh",     "Marsh",    "Liverpool", "GKP", 5.2, 3060, 34, 0, 0, 0.00, 0.01, 13, 41, 42.8, 12, 640, 58.3,  "a", 100,  9.8),
    ("Kasper",  "Lund",      "Lund",     "Newcastle", "GKP", 4.9, 3420, 38, 0, 0, 0.00, 0.00, 11, 49, 50.1, 14, 655, 60.4,  "a", 100,  7.1),
    ("Adrien",  "Vasseur",   "Vasseur",  "Brighton",  "GKP", 4.6, 2790, 31, 0, 0, 0.00, 0.00,  8, 44, 45.6,  9, 520, 47.9,  "a", 100,  3.3),
    ("Milos",   "Petrovic",  "Petrovic", "Everton",   "GKP", 4.4, 3420, 38, 0, 0, 0.00, 0.00,  9, 55, 56.7, 11, 580, 52.2,  "a", 100,  2.6),
    ("Iwan",    "Prosser",   "Prosser",  "Brentford", "GKP", 4.3, 1800, 20, 0, 0, 0.00, 0.00,  4, 29, 30.2,  4, 300, 27.5,  "a", 100,  1.1),
    ("Femi",    "Adeyemo",   "Adeyemo",  "Coventry",  "GKP", 3.9,    0,  0, 0, 0, 0.00, 0.00,  0,  0,  0.0,  0,   0,  0.0,  "a", 100,  0.4),

    # ---- DEFENDERS -------------------------------------------------------
    ("Nils",    "Karlsson",  "Karlsson", "Man City",  "DEF", 6.4, 2880, 32, 4, 7, 2.90, 5.60, 15, 24, 25.9, 22, 730, 152.0, "a", 100, 32.7),
    ("Andre",   "Fonseca",   "Fonseca",  "Man City",  "DEF", 5.6, 2340, 26, 2, 2, 1.70, 1.90, 12, 20, 21.4, 11, 540, 96.4,  "a", 100,  8.9),
    ("Jorrit",  "van Dael",  "van Dael", "Arsenal",   "DEF", 6.1, 3060, 34, 3, 5, 2.40, 4.10, 15, 30, 30.9, 19, 700, 138.2, "a", 100, 27.5),
    ("Callum",  "Ridley",    "Ridley",   "Arsenal",   "DEF", 5.3, 2700, 30, 1, 1, 0.90, 1.30, 13, 27, 28.1, 10, 590, 88.7,  "a", 100,  6.2),
    ("Marcelo", "Ibarra",    "Ibarra",   "Liverpool", "DEF", 6.6, 2970, 33, 5, 6, 3.80, 5.20, 12, 38, 39.4, 24, 760, 168.5, "a", 100, 38.1),
    ("Stefan",  "Novak",     "Novak",    "Liverpool", "DEF", 4.9, 1620, 18, 0, 1, 0.40, 0.80,  7, 22, 23.0,  5, 330, 51.3,  "d",  50,  2.0),
    ("Ola",     "Bergstrom", "Bergstrom","Newcastle", "DEF", 5.1, 3150, 35, 2, 3, 1.60, 2.70, 11, 42, 43.6, 14, 640, 104.9, "a", 100, 11.4),
    ("Josh",    "Trentham",  "Trentham", "Newcastle", "DEF", 4.5, 2250, 25, 1, 0, 0.70, 0.60,  8, 31, 32.5,  6, 430, 62.8,  "a", 100,  3.7),
    ("Ayo",     "Balogun",   "Balogun",  "Brighton",  "DEF", 4.8, 2880, 32, 2, 2, 1.30, 1.80,  8, 45, 46.3,  9, 520, 88.1,  "a", 100,  5.5),
    ("Pierre",  "Coste",     "Coste",    "Everton",   "DEF", 4.4, 3060, 34, 1, 1, 0.80, 0.90,  9, 48, 49.7,  8, 500, 71.6,  "a", 100,  4.1),
    ("Sam",     "Whittle",   "Whittle",  "Brentford", "DEF", 4.3, 2610, 29, 1, 2, 0.90, 1.40,  7, 43, 44.2,  7, 470, 76.9,  "a", 100,  2.8),
    ("Dele",    "Amadi",     "Amadi",    "Coventry",  "DEF", 4.0,    0,  0, 0, 0, 0.00, 0.00,  0,  0,  0.0,  0,   0,   0.0, "a", 100,  0.6),

    # ---- MIDFIELDERS -----------------------------------------------------
    ("Rafael",  "Duarte",    "Duarte",   "Man City",  "MID", 12.8, 2790, 31, 17, 12, 15.40, 10.20, 13, 22, 23.1, 34, 890, 342.6, "a", 100, 51.2),
    ("Kai",     "Brenner",   "Brenner",  "Man City",  "MID",  7.2, 2340, 26,  6,  8,  5.90,  7.10, 11, 19, 20.3, 15, 610, 178.4, "a", 100, 12.9),
    ("Bruno",   "Sertao",    "Sertao",   "Arsenal",   "MID", 10.4, 2880, 32, 11, 13, 9.80, 11.40, 12, 26, 27.2, 27, 810, 288.1, "a", 100, 40.6),
    ("Owen",    "Fairclough","Fairclough","Arsenal",  "MID",  5.8, 2520, 28,  3,  4, 2.60,  3.30, 12, 24, 25.0, 11, 560, 118.7, "a", 100,  7.8),
    ("Mateo",   "Aguirre",   "Aguirre",  "Liverpool", "MID", 11.6, 2700, 30, 13, 11, 12.10, 9.60, 10, 33, 34.5, 29, 840, 305.9, "a", 100, 44.3),
    ("Ibrahim", "Toure",     "Toure",    "Liverpool", "MID",  6.3, 2160, 24,  4,  5, 3.70,  4.40,  9, 28, 29.1, 12, 570, 141.2, "a", 100,  9.1),
    ("Danny",   "Wexford",   "Wexford",  "Newcastle", "MID",  7.8, 2970, 33,  8,  7, 7.20,  6.30, 10, 39, 40.2, 18, 690, 210.5, "a", 100, 21.7),
    ("Ryo",     "Nakashima", "Nakashima","Newcastle", "MID",  5.5, 2430, 27,  3,  3, 2.40,  2.90,  8, 33, 34.4,  9, 500, 112.0, "a", 100,  4.9),
    ("Lucas",   "Mendel",    "Mendel",   "Brighton",  "MID",  6.9, 2790, 31,  7,  6, 6.10,  5.50,  7, 42, 43.1, 16, 650, 189.3, "a", 100, 15.3),
    ("Karim",   "Belhadj",   "Belhadj",  "Brighton",  "MID",  5.2, 2250, 25,  2,  4, 2.10,  3.60,  6, 36, 37.0,  8, 470, 104.6, "a", 100,  3.4),
    ("Nathan",  "Okoro",     "Okoro",    "Everton",   "MID",  5.9, 2880, 32,  5,  4, 4.30,  3.80,  8, 45, 46.5, 12, 560, 133.8, "a", 100,  6.7),
    ("Finn",    "Halvorsen", "Halvorsen","Brentford", "MID",  6.1, 2700, 30,  6,  5, 5.20,  4.60,  6, 44, 45.3, 13, 580, 152.7, "a", 100,  8.2),
    ("Emeka",   "Nwosu",     "Nwosu",    "Coventry",  "MID",  4.5,    0,  0,  0,  0, 0.00,  0.00,  0,  0,  0.0,  0,   0,   0.0, "a", 100,  1.2),

    # ---- FORWARDS --------------------------------------------------------
    ("Viktor",  "Ahlberg",   "Ahlberg",  "Man City",  "FWD", 14.5, 2880, 32, 27, 6, 25.80, 5.10, 12, 21, 22.4, 39, 940, 388.2, "a", 100, 62.8),
    ("Diego",   "Palermo",   "Palermo",  "Arsenal",   "FWD",  9.1, 2610, 29, 15, 5, 14.20, 4.30, 11, 25, 26.0, 24, 770, 251.6, "a", 100, 29.4),
    ("Samuel",  "Aleixo",    "Aleixo",   "Liverpool", "FWD",  8.4, 2340, 26, 12, 7, 11.60, 6.20,  9, 30, 31.3, 21, 720, 236.9, "a", 100, 24.1),
    ("Jonas",   "Ritter",    "Ritter",   "Newcastle", "FWD",  7.1, 2520, 28, 10, 4,  9.40, 3.50,  8, 36, 37.2, 17, 640, 194.7, "a", 100, 13.6),
    ("Tariq",   "Mensah",    "Mensah",   "Brighton",  "FWD",  6.4, 2160, 24,  8, 3,  7.80, 2.80,  5, 34, 35.1, 13, 540, 158.3, "a", 100,  7.9),
    ("Gio",     "Lanzini",   "Lanzini",  "Everton",   "FWD",  5.6, 2430, 27,  6, 3,  6.30, 2.60,  6, 41, 42.5, 10, 480, 129.4, "i",   0,  2.2),
    ("Marc",    "Delacroix", "Delacroix","Fulham",    "FWD",  6.0, 2610, 29,  7, 4,  6.90, 3.40,  7, 39, 40.1, 12, 520, 146.2, "a", 100,  5.8),
    ("Kofi",    "Asante",    "Asante",   "Coventry",  "FWD",  4.7,    0,  0,  0, 0,  0.00, 0.00,  0,  0,  0.0,  0,   0,   0.0, "a", 100,  1.9),

    # ---- extra Fulham cover so Fulham (a double-gameweek club) is pickable -
    ("Louis",   "Grandin",   "Grandin",  "Fulham",    "GKP",  4.5, 3060, 34, 0, 0, 0.00, 0.00,  8, 47, 48.3, 10, 540, 49.1,  "a", 100,  3.0),
    ("Theo",    "Marchand",  "Marchand", "Fulham",    "DEF",  4.6, 2880, 32, 2, 2, 1.10, 1.60,  8, 44, 45.0,  9, 510, 84.3,  "a", 100,  4.4),
    ("Aiden",   "Rourke",    "Rourke",   "Fulham",    "MID",  5.7, 2700, 30, 4, 6, 3.90, 5.10,  7, 41, 42.2, 12, 560, 147.5, "a", 100,  6.1),
    ("Bart",    "Hulsman",   "Hulsman",  "Everton",   "DEF",  4.2, 2340, 26, 0, 1, 0.50, 0.90,  7, 38, 39.4,  5, 400, 58.2,  "a", 100,  1.8),
]


def make_sample_players_df() -> pd.DataFrame:
    """~45 players across all four positions, spread over the 9 sample clubs.

    Mirrors `fpl_client.load_players_df()`: same columns, same order, and the
    same string-typed numeric fields the live API returns.
    """
    rows = []
    for pid, spec in enumerate(_PLAYER_SPECS, start=1):
        (first, last, web, team, pos, price, minutes, starts, goals, assists,
         xg, xa, clean_sheets, goals_conceded, xgc, bonus, bps, ict,
         status, chance, sel) = spec

        # Rough points reconstruction so total_points/form/ppg are self-consistent.
        appearance_pts = starts * 2
        goal_pts = goals * {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}[pos]
        cs_pts = clean_sheets * {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}[pos]
        conceded_pts = -(goals_conceded // 2) if pos in ("GKP", "DEF") else 0
        total_points = appearance_pts + goal_pts + assists * 3 + cs_pts + conceded_pts + bonus
        games = max(starts, 1)

        rows.append({
            "id": pid,
            # FPL's stable cross-season identifier (see fpl_client.load_players_df).
            # Deliberately NOT equal to `id`, because in the real payload they
            # are unrelated and any code that conflates them is broken.
            "code": 100000 + pid * 7,
            "full_name": f"{first} {last}",
            "web_name": web,
            "team_name": team,
            "team_short": _SHORT_BY_TEAM[team],
            "is_promoted": team in PROMOTED_TEAMS,
            "position": pos,
            "price_m": float(price),
            "total_points": int(total_points),
            # API returns these as strings -- keep it that way on purpose.
            "points_per_game": f"{total_points / games:.1f}",
            "form": f"{min(9.9, total_points / games * 0.9):.1f}",
            "selected_by_percent": f"{sel:.1f}",
            "minutes": int(minutes),
            "starts": int(starts),
            "goals_scored": int(goals),
            "assists": int(assists),
            "clean_sheets": int(clean_sheets),
            "goals_conceded": int(goals_conceded),
            "expected_goals": f"{xg:.2f}",
            "expected_assists": f"{xa:.2f}",
            "expected_goal_involvements": f"{xg + xa:.2f}",
            "expected_goals_conceded": f"{xgc:.2f}",
            "bonus": int(bonus),
            "bps": int(bps),
            "ict_index": f"{ict:.1f}",
            "status": status,
            "status_meaning": {"a": "available", "d": "doubtful", "i": "injured",
                               "s": "suspended", "u": "unavailable",
                               "n": "not in squad"}[status],
            "chance_of_playing_next_round": chance,
            "news": "" if status == "a" else "Knock - assessed ahead of the weekend",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# GW1 is built to hit every edge case at once:
#   Brighton        -> 0 fixtures  (BLANK)
#   Everton, Fulham -> 2 fixtures  (DOUBLE)
#   everyone else   -> 1 fixture
# GW2 is a plain single round (Coventry blank) so multi-gameweek code has
# something ordinary to run against.

_FIXTURES = [
    # id, gw, kickoff, home, away, h_diff, a_diff, finished, h_score, a_score
    (1, 1, "2026-08-21T19:00:00Z", "Man City",  "Coventry",  2, 5, False, None, None),
    (2, 1, "2026-08-22T14:00:00Z", "Arsenal",   "Everton",   2, 5, False, None, None),
    (3, 1, "2026-08-22T14:00:00Z", "Liverpool", "Brentford", 2, 4, False, None, None),
    (4, 1, "2026-08-22T16:30:00Z", "Newcastle", "Fulham",    3, 4, False, None, None),
    # second GW1 fixture for Everton and Fulham -> double gameweek for both
    (5, 1, "2026-08-24T19:45:00Z", "Everton",   "Fulham",    3, 3, False, None, None),

    (6, 2, "2026-08-29T14:00:00Z", "Brighton",  "Man City",  5, 2, False, None, None),
    (7, 2, "2026-08-29T14:00:00Z", "Arsenal",   "Liverpool", 4, 4, False, None, None),
    (8, 2, "2026-08-29T16:30:00Z", "Newcastle", "Everton",   2, 4, False, None, None),
    (9, 2, "2026-08-30T14:00:00Z", "Brentford", "Fulham",    3, 3, False, None, None),
]


def make_sample_fixtures_df() -> pd.DataFrame:
    """Fixtures for gameweeks 1 and 2, including a blank and a double in GW1."""
    df = pd.DataFrame(_FIXTURES, columns=[
        "id", "gameweek", "kickoff_time", "home_team", "away_team",
        "team_h_difficulty", "team_a_difficulty", "finished",
        "team_h_score", "team_a_score",
    ])
    df["home_short"] = df["home_team"].map(_SHORT_BY_TEAM)
    df["away_short"] = df["away_team"].map(_SHORT_BY_TEAM)
    return df[[
        "id", "gameweek", "kickoff_time", "home_team", "home_short",
        "away_team", "away_short", "team_h_difficulty", "team_a_difficulty",
        "finished", "team_h_score", "team_a_score",
    ]]


def make_sample_overrides_df() -> pd.DataFrame:
    """A small `player_judgments`-style override table for testing the hook."""
    return pd.DataFrame([
        {"web_name": "Ahlberg",  "adjustment_type": "upgrade",       "value": 1.0,
         "reason": "penalties + always captained at home"},
        {"web_name": "Lanzini",  "adjustment_type": "flat_override", "value": 0.0,
         "reason": "injured, will not feature"},
        {"web_name": "Nwosu",    "adjustment_type": "multiplier",    "value": 0.5,
         "reason": "promoted side, unproven at this level"},
        {"web_name": "NotARealPlayer", "adjustment_type": "upgrade", "value": 2.0,
         "reason": "should be reported as unmatched"},
    ])


if __name__ == "__main__":
    players = make_sample_players_df()
    teams = make_sample_teams_df()
    fixtures = make_sample_fixtures_df()
    print(f"players : {len(players)} rows")
    print(players["position"].value_counts().to_string())
    print(f"\nteams   : {len(teams)} rows "
          f"({int(teams['is_promoted'].sum())} promoted)")
    print(f"fixtures: {len(fixtures)} rows across "
          f"gameweeks {sorted(fixtures['gameweek'].unique())}")
