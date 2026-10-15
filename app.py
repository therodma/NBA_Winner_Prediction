"""
app.py — Flask web server for NBA predictions
"""

import os
import time
import json
import warnings
import threading
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import CORS

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.live.nba.endpoints import boxscore as live_boxscore
from nba_api.stats.endpoints import LeagueDashPlayerStats, ScoreboardV3, BoxScoreTraditionalV3

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

MODELS_DIR   = Path("models")
RAW          = Path("data/raw")
SLEEP        = 0.3
PST          = timezone(timedelta(hours=-7))

GIST_ID      = os.environ.get("GIST_ID", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GIST_FILE    = "history.json"
GIST_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

HISTORY_FILE = Path("data/history.json")

FEATURE_COLS = [
    "DIFF_OFF_RATING", "DIFF_DEF_RATING", "DIFF_NET_RATING", "DIFF_PACE",
    "DIFF_W_PCT", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_AST", "DIFF_TOV",
    "HOME_ROLL10_WIN_PCT", "AWAY_ROLL10_WIN_PCT", "DIFF_ROLL10_WIN_PCT",
    "HOME_ROLL10_PM",      "AWAY_ROLL10_PM",      "DIFF_ROLL10_PM",
    "HOME_REST",   "AWAY_REST",   "DIFF_REST",
    "HOME_B2B",    "AWAY_B2B",
    "H2H_HOME_WIN_PCT",
    "HOME_STAR_AVAIL", "AWAY_STAR_AVAIL", "DIFF_STAR_AVAIL",
    "HOME_TRAVEL",     "AWAY_TRAVEL",     "DIFF_TRAVEL",
    "HOME_SOS",        "AWAY_SOS",        "DIFF_SOS",
]

# ── load once at startup ──────────────────────────────────────────────────────

print("Loading models and data...")
MODELS = {
    "LogisticRegression": joblib.load(MODELS_DIR / "logisticregression_v2.pkl"),
    "RandomForest":       joblib.load(MODELS_DIR / "randomforest_v2.pkl"),
    "GradientBoosting":   joblib.load(MODELS_DIR / "gradientboosting_v2.pkl"),
    "Ensemble":           joblib.load(MODELS_DIR / "ensemble_v2.pkl"),
}
MARGIN_MODEL = joblib.load(MODELS_DIR / "margin_regressor_v2.pkl")
LOGS         = pd.read_csv(RAW / "game_logs.csv",  parse_dates=["GAME_DATE"])
MATCHUPS     = pd.read_csv(RAW / "matchups.csv",   parse_dates=["GAME_DATE"])
STATS        = pd.read_csv(RAW / "team_stats.csv")
HOME_COURT   = pd.read_csv(RAW / "home_court_strength.csv") if (RAW / "home_court_strength.csv").exists() else pd.DataFrame()
print("Ready.")

ARENA_COORDS = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621), "BKN": (40.6826, -73.9754),
    "CHA": (35.2251, -80.8392), "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077), "DET": (42.3410, -83.0553),
    "GSW": (37.7680, -122.3877), "HOU": (29.7508, -95.3621), "IND": (39.7640, -86.1555),
    "LAC": (34.0430, -118.2673), "LAL": (34.0430, -118.2673), "MEM": (35.1382, -90.0505),
    "MIA": (25.7814, -80.1870), "MIL": (43.0450, -87.9170), "MIN": (44.9795, -93.2762),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934), "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839), "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4997), "SAS": (29.4270, -98.4375),
    "TOR": (43.6435, -79.3791), "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}

def _haversine(c1, c2):
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.8
    lat1, lon1 = map(radians, c1); lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_travel_miles(team_tri):
    """Miles traveled to reach current city from last game's city."""
    t = LOGS[LOGS["TEAM_ABBREVIATION"] == team_tri].sort_values("GAME_DATE") if "TEAM_ABBREVIATION" in LOGS.columns else pd.DataFrame()
    if len(t) < 2:
        return 0.0
    last = t.iloc[-1]
    prev = t.iloc[-2]
    def arena_of(row):
        if "vs." in str(row.get("MATCHUP", "")):
            return row.get("TEAM_ABBREVIATION", "")
        return str(row.get("MATCHUP", "")).split("@ ")[-1].strip()
    curr_arena = arena_of(last)
    prev_arena = arena_of(prev)
    if curr_arena == prev_arena:
        return 0.0
    c1 = ARENA_COORDS.get(prev_arena)
    c2 = ARENA_COORDS.get(curr_arena)
    return _haversine(c1, c2) if c1 and c2 else 0.0

def get_sos(team_id):
    """Average opponent NET_RATING over last 10 games."""
    t = LOGS[LOGS["TEAM_ID"] == team_id].sort_values("GAME_DATE").tail(10)
    if len(t) == 0:
        return 0.0
    opp_nets = []
    for _, row in t.iterrows():
        opp_id = row.get("OPP_TEAM_ID") if "OPP_TEAM_ID" in LOGS.columns else None
        if opp_id:
            opp_stats = get_stats(opp_id)
            opp_nets.append(float(opp_stats["NET_RATING"]) if "NET_RATING" in opp_stats.index else 0.0)
    return float(np.mean(opp_nets)) if opp_nets else 0.0


TEAM_TIMEZONE = {
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6,
    "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5, "GSW": -8,
    "HOU": -6, "IND": -5, "LAC": -8, "LAL": -8, "MEM": -6,
    "MIA": -5, "MIL": -6, "MIN": -6, "NOP": -6, "NYK": -5,
    "OKC": -6, "ORL": -5, "PHI": -5, "PHX": -7, "POR": -8,
    "SAC": -8, "SAS": -6, "TOR": -5, "UTA": -7, "WAS": -5,
}

# ── in-memory cache ───────────────────────────────────────────────────────────

_cache        = {"data": None, "updated": None, "date": None}
_schedule_cache = {"data": None, "date": None}
_cache_lock   = threading.Lock()
_prob_history = {}  # {game_id: [{t, home_prob}]}

# ── feature helpers ───────────────────────────────────────────────────────────

def get_rolling_form(team_id):
    t = LOGS[LOGS["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    last10, last20 = t.tail(10), t.tail(20)
    if len(last10) == 0:
        return 0.5, 0.0, 0.5, 0.0, 0
    win10 = (last10["WL"] == "W").mean()
    pm10  = last10["PLUS_MINUS"].astype(float).mean()
    win20 = (last20["WL"] == "W").mean() if len(last20) > 0 else win10
    pm20  = last20["PLUS_MINUS"].astype(float).mean() if len(last20) > 0 else pm10
    streak = 0
    for _, row in t.iloc[::-1].iterrows():
        if streak == 0:
            streak = 1 if row["WL"] == "W" else -1
        elif (streak > 0 and row["WL"] == "W") or (streak < 0 and row["WL"] == "L"):
            streak += 1 if streak > 0 else -1
        else:
            break
    return win10, pm10, win20, pm20, streak

def get_rest(team_id):
    t = LOGS[LOGS["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    if len(t) == 0:
        return 7, 0
    rest = (pd.Timestamp(date.today()) - t["GAME_DATE"].iloc[-1]).days
    return int(rest), int(rest == 1)

def get_h2h(home_id, away_id):
    pair = MATCHUPS[
        ((MATCHUPS["HOME_TEAM_ID"] == home_id) & (MATCHUPS["AWAY_TEAM_ID"] == away_id)) |
        ((MATCHUPS["HOME_TEAM_ID"] == away_id) & (MATCHUPS["AWAY_TEAM_ID"] == home_id))
    ].tail(20)
    if len(pair) == 0:
        return 0.5
    wins = (
        ((pair["HOME_TEAM_ID"] == home_id) & (pair["HOME_TEAM_WIN"] == 1)) |
        ((pair["AWAY_TEAM_ID"] == home_id) & (pair["HOME_TEAM_WIN"] == 0))
    ).sum()
    return wins / len(pair)

def get_stats(team_id):
    s = STATS[STATS["TEAM_ID"] == team_id].sort_values("SEASON")
    return s.iloc[-1] if len(s) else pd.Series(dtype=float)

def get_home_court_strength(team_id):
    if HOME_COURT.empty:
        return 0.0
    row = HOME_COURT[HOME_COURT["TEAM_ID"] == team_id].sort_values("SEASON")
    return float(row["HOME_COURT_STRENGTH"].iloc[-1]) if len(row) else 0.0

def build_features(home_id, away_id, home_tri="", away_tri=""):
    hs, as_ = get_stats(home_id), get_stats(away_id)
    hwp, hpm, hwp20, hpm20, hstreak = get_rolling_form(home_id)
    awp, apm, awp20, apm20, astreak = get_rolling_form(away_id)
    hr, hb2b = get_rest(home_id)
    ar, ab2b = get_rest(away_id)
    h2h  = get_h2h(home_id, away_id)
    hcs  = get_home_court_strength(home_id)
    acs  = get_home_court_strength(away_id)

    # v2 features
    h_travel = get_travel_miles(home_tri)
    a_travel = get_travel_miles(away_tri)
    h_sos    = get_sos(home_id)
    a_sos    = get_sos(away_id)
    # star availability: use injury report proxy (1.0 = full strength)
    h_star = 1.0
    a_star = 1.0

    def diff(col):
        hv = float(hs[col]) if col in hs.index else 0.0
        av = float(as_[col]) if col in as_.index else 0.0
        return hv - av

    return {
        "DIFF_OFF_RATING":  diff("OFF_RATING"),  "DIFF_DEF_RATING": diff("DEF_RATING"),
        "DIFF_NET_RATING":  diff("NET_RATING"),  "DIFF_PACE":       diff("PACE"),
        "DIFF_W_PCT":       diff("W_PCT"),        "DIFF_FG_PCT":     diff("FG_PCT"),
        "DIFF_FG3_PCT":     diff("FG3_PCT"),      "DIFF_FT_PCT":     diff("FT_PCT"),
        "DIFF_REB":         diff("REB"),           "DIFF_AST":        diff("AST"),
        "DIFF_TOV":         diff("TOV"),
        "HOME_ROLL10_WIN_PCT": hwp,  "AWAY_ROLL10_WIN_PCT": awp,  "DIFF_ROLL10_WIN_PCT": hwp - awp,
        "HOME_ROLL10_PM":      hpm,  "AWAY_ROLL10_PM":      apm,  "DIFF_ROLL10_PM":      hpm - apm,
        "HOME_REST": hr, "AWAY_REST": ar, "DIFF_REST": hr - ar,
        "HOME_B2B": hb2b, "AWAY_B2B": ab2b,
        "H2H_HOME_WIN_PCT": h2h,
        "HOME_STAR_AVAIL": h_star, "AWAY_STAR_AVAIL": a_star, "DIFF_STAR_AVAIL": h_star - a_star,
        "HOME_TRAVEL": h_travel,  "AWAY_TRAVEL": a_travel,  "DIFF_TRAVEL": h_travel - a_travel,
        "HOME_SOS":    h_sos,     "AWAY_SOS":    a_sos,     "DIFF_SOS":    h_sos - a_sos,
    }

# ── feature importance / explainability ──────────────────────────────────────

# Pre-compute RF feature importances once at startup
_RF_IMPORTANCES = None
try:
    _rf = MODELS["RandomForest"]
    # Pipeline wraps the classifier; get the actual estimator
    _clf = _rf.named_steps["clf"] if hasattr(_rf, "named_steps") else _rf
    if hasattr(_clf, "feature_importances_"):
        _RF_IMPORTANCES = dict(zip(FEATURE_COLS, _clf.feature_importances_))
except Exception as _e:
    print(f"[importance] could not load RF importances: {_e}")

# Human-readable labels for features
_FEAT_LABELS = {
    "DIFF_NET_RATING":      "Net Rating edge",
    "DIFF_OFF_RATING":      "Offensive Rating edge",
    "DIFF_DEF_RATING":      "Defensive Rating edge",
    "DIFF_W_PCT":           "Win % edge",
    "DIFF_ROLL10_WIN_PCT":  "Last-10 form edge",
    "DIFF_ROLL10_PM":       "Last-10 point diff edge",
    "DIFF_REST":            "Rest days edge",
    "HOME_B2B":             "Home team on B2B",
    "AWAY_B2B":             "Away team on B2B",
    "H2H_HOME_WIN_PCT":     "Head-to-head history",
    "DIFF_TRAVEL":          "Travel distance edge",
    "DIFF_SOS":             "Strength of schedule edge",
    "DIFF_STAR_AVAIL":      "Star player availability",
}

def build_top_factors(feats: dict, home_tri: str, away_tri: str, home_prob: float) -> list:
    """Return top 3 most influential factors for this prediction."""
    if not _RF_IMPORTANCES:
        return []
    home_favored = home_prob >= 0.5
    fav = home_tri if home_favored else away_tri
    dog = away_tri if home_favored else home_tri

    scored = []
    for feat, importance in _RF_IMPORTANCES.items():
        val = feats.get(feat, 0.0)
        if abs(val) < 1e-6 and feat not in ("HOME_B2B", "AWAY_B2B"):
            continue
        label = _FEAT_LABELS.get(feat, feat)
        # Determine direction: positive diff = home advantage
        if feat == "HOME_B2B":
            if val:
                desc = f"{home_tri} on back-to-back (fatigue risk)"
            else:
                continue
        elif feat == "AWAY_B2B":
            if val:
                desc = f"{away_tri} on back-to-back (fatigue risk)"
            else:
                continue
        elif feat == "AWAY_TRAVEL_TZ_DIFF":
            if val >= 2:
                desc = f"{away_tri} traveling {int(val)} time zones"
            else:
                continue
        elif feat == "H2H_HOME_WIN_PCT":
            if val >= 0.6:
                desc = f"{home_tri} leads H2H ({round(val*100)}%)"
            elif val <= 0.4:
                desc = f"{away_tri} leads H2H ({round((1-val)*100)}%)"
            else:
                continue
        else:
            # diff features: positive = home advantage
            if val > 0:
                desc = f"{home_tri} {label} (+{val:.1f})"
            else:
                desc = f"{away_tri} {label} ({val:.1f})"
        scored.append((importance * abs(val if feat not in ("HOME_B2B","AWAY_B2B","H2H_HOME_WIN_PCT") else 1), desc))

    scored.sort(reverse=True)
    return [desc for _, desc in scored[:3]]


# ── history / accuracy tracking ───────────────────────────────────────────────

def _load_history_file() -> dict:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        print(f"[history] load error: {e}")
    return {}

def _save_history_file(data: dict):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[history] save error: {e}")

def record_final_games(games: list):
    """Called after each cache refresh — persist any newly-final games."""
    final = [g for g in games if g.get("game_status") == 3 and "pred_winner" in g]
    if not final:
        return
    today_str = date.today().isoformat()
    history = _load_history_file()
    existing = {g["game_id"] for day in history.values() for g in day}
    new_games = [g for g in final if g["game_id"] not in existing]
    if not new_games:
        return
    if today_str not in history:
        history[today_str] = []
    for g in new_games:
        history[today_str].append({
            "game_id":    g["game_id"],
            "home_tri":   g["home_tri"],
            "away_tri":   g["away_tri"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "pred_winner":g["pred_winner"],
            "home_prob":  g["home_prob"],
        })
    _save_history_file(history)
    print(f"[history] saved {len(new_games)} final game(s) for {today_str}")

def compute_accuracy() -> dict:
    """Return season-to-date correct/total/pct from history file."""
    history = _load_history_file()
    correct = total = 0
    recent = []
    for day_str in sorted(history.keys(), reverse=True):
        for g in history[day_str]:
            total += 1
            home_won = g["home_score"] > g["away_score"]
            pred_correct = (home_won and g["pred_winner"] == g["home_tri"]) or \
                           (not home_won and g["pred_winner"] == g["away_tri"])
            if pred_correct:
                correct += 1
            if len(recent) < 20:
                recent.append({
                    "date":       day_str,
                    "home_tri":   g["home_tri"],
                    "away_tri":   g["away_tri"],
                    "home_score": g["home_score"],
                    "away_score": g["away_score"],
                    "pred_winner":g["pred_winner"],
                    "correct":    pred_correct,
                })
    pct = round(correct / total * 100, 1) if total else None
    return {"correct": correct, "total": total, "pct": pct, "recent": recent}


# ── box score helpers ─────────────────────────────────────────────────────────

def parse_minutes(raw):
    if not raw or raw == "PT" or raw == "":
        return "0:00"
    m = raw.replace("PT", "").replace("M", ":").split(":")[0:2]
    return ":".join(m) if len(m) == 2 else raw

def parse_players_v3(team_data):
    players = []
    for p in team_data.get("players", []):
        s = p.get("statistics", {})
        pts = s.get("points", 0)
        mins = s.get("minutes", "")
        if not mins and pts == 0:
            continue  # didn't play
        players.append({
            "name":    f"{p.get('firstName','')} {p.get('familyName','')}".strip(),
            "starter": p.get("position", "") != "",
            "min":     parse_minutes(mins),
            "pts":     pts,
            "reb":     s.get("reboundsTotal", 0),
            "ast":     s.get("assists", 0),
            "stl":     s.get("steals", 0),
            "blk":     s.get("blocks", 0),
            "tov":     s.get("turnovers", 0),
            "fg":      f"{s.get('fieldGoalsMade',0)}/{s.get('fieldGoalsAttempted',0)}",
            "3p":      f"{s.get('threePointersMade',0)}/{s.get('threePointersAttempted',0)}",
            "ft":      f"{s.get('freeThrowsMade',0)}/{s.get('freeThrowsAttempted',0)}",
            "pm":      int(s.get("plusMinusPoints", 0)),
        })
    players.sort(key=lambda x: (-x["pts"], not x["starter"]))
    return players

# ── history helpers (GitHub Gist) ─────────────────────────────────────────────

def load_history():
    if not GIST_ID:
        return {}
    try:
        r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=GIST_HEADERS, timeout=10)
        raw_url = r.json()["files"][GIST_FILE]["raw_url"]
        return requests.get(raw_url, timeout=10).json()
    except Exception as e:
        print(f"[gist] load error: {e}")
        return {}

def save_results(date_str, games):
    final_games = [g for g in games if g.get("game_status") == 3 and "pred_winner" in g]
    if not final_games or not GIST_ID:
        return
    history = load_history()
    history[date_str] = final_games
    try:
        requests.patch(
            f"https://api.github.com/gists/{GIST_ID}",
            headers=GIST_HEADERS,
            json={"files": {GIST_FILE: {"content": json.dumps(history)}}},
            timeout=10
        )
    except Exception as e:
        print(f"[gist] save error: {e}")

# ── injury / player helpers ───────────────────────────────────────────────────

_injury_cache      = {"data": {}, "updated": None}
_player_stats_cache = {"data": {}, "updated": None}

def fetch_player_stats():
    try:
        time.sleep(SLEEP)
        df = LeagueDashPlayerStats(
            season="2024-25",
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        df["IMPACT"] = (
            df["PTS"] * 1.0 + df["AST"] * 0.7 + df["REB"] * 0.4 +
            df["STL"] * 1.0 + df["BLK"] * 0.7 - df["TOV"] * 0.5
        )
        result = {}
        for _, row in df.iterrows():
            result[row["PLAYER_NAME"]] = {"impact": float(row["IMPACT"]), "pts": float(row["PTS"])}
        _player_stats_cache["data"]    = result
        _player_stats_cache["updated"] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        print(f"[player_stats] fetch error: {e}")
        return _player_stats_cache["data"]

def get_player_stats():
    updated = _player_stats_cache["updated"]
    if updated is None or (datetime.now(timezone.utc) - updated).seconds > 3600:
        return fetch_player_stats()
    return _player_stats_cache["data"]

def player_impact_adj(player_name, player_stats):
    """Win prob shift if this player is out. Star=5%, solid=3.5%, role=2%, bench=1%."""
    p = player_stats.get(player_name)
    if not p:
        return 0.02
    impact = p["impact"]
    if impact >= 30:   return 0.05
    elif impact >= 20: return 0.035
    elif impact >= 10: return 0.02
    else:              return 0.01

def fetch_injury_report():
    try:
        time.sleep(SLEEP)
        url  = "https://stats.nba.com/js/data/leaders/00_player_status.json"
        hdrs = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/", "Accept": "application/json"}
        r    = requests.get(url, headers=hdrs, timeout=10)
        report = {}
        for player in r.json().get("playerStatuses", []):
            tid    = player.get("teamId")
            name   = player.get("playerName", "")
            status = player.get("playerStatus", "")
            reason = player.get("injuryDescription", "")
            if tid not in report:
                report[tid] = []
            report[tid].append({"name": name, "status": status, "reason": reason})
        _injury_cache["data"]    = report
        _injury_cache["updated"] = datetime.now(timezone.utc)
        return report
    except Exception as e:
        print(f"[injury] fetch error: {e}")
        return _injury_cache["data"]

def get_injury_report():
    updated = _injury_cache["updated"]
    if updated is None or (datetime.now(timezone.utc) - updated).seconds > 600:
        return fetch_injury_report()
    return _injury_cache["data"]

def injury_adjustment(team_id, injury_report):
    """Flat adjustment for confirmed out/doubtful players."""
    adj = 0.0
    for p in injury_report.get(team_id, []):
        s = p["status"].lower()
        if s == "out":            adj -= 0.025
        elif s == "doubtful":     adj -= 0.015
        elif s == "questionable": adj -= 0.007
    return max(-0.15, adj)

def build_scenarios(base_prob, home_id, away_id, injury_report, player_stats):
    """
    For each questionable/doubtful player, return two win prob scenarios:
    - if they play vs if they don't.
    """
    scenarios = []
    for team_id, is_home in [(home_id, True), (away_id, False)]:
        for p in injury_report.get(team_id, []):
            if p["status"].lower() not in ("questionable", "doubtful"):
                continue
            adj = player_impact_adj(p["name"], player_stats)
            if is_home:
                plays_prob = round(min(0.99, max(0.01, base_prob + adj)), 3)
                sits_prob  = round(min(0.99, max(0.01, base_prob - adj)), 3)
            else:
                plays_prob = round(min(0.99, max(0.01, base_prob - adj)), 3)
                sits_prob  = round(min(0.99, max(0.01, base_prob + adj)), 3)
            scenarios.append({
                "player":          p["name"],
                "status":          p["status"],
                "reason":          p.get("reason", ""),
                "team":            "home" if is_home else "away",
                "plays_home_prob": plays_prob,
                "sits_home_prob":  sits_prob,
            })
    return scenarios

def build_analysis(home_tri, away_tri, feats, home_prob, home_injuries, away_injuries, player_stats):
    """Generate a plain-English pre-game analysis summary."""
    lines = []
    fav  = home_tri if home_prob >= 0.5 else away_tri
    dog  = away_tri if home_prob >= 0.5 else home_tri
    fav_prob = round(max(home_prob, 1-home_prob) * 100)

    lines.append(f"{fav} are favored at {fav_prob}% to win.")

    # net rating edge
    net = feats.get("DIFF_NET_RATING", 0)
    if abs(net) >= 3:
        better = home_tri if net > 0 else away_tri
        lines.append(f"{better} have a significant net rating edge ({abs(net):.1f} pts/100).")

    # rolling form
    h10 = feats.get("HOME_ROLL10_WIN_PCT", 0.5)
    a10 = feats.get("AWAY_ROLL10_WIN_PCT", 0.5)
    if abs(h10 - a10) >= 0.2:
        hotter = home_tri if h10 > a10 else away_tri
        lines.append(f"{hotter} are in better recent form (last 10 games).")

    # streak
    hs = feats.get("HOME_STREAK", 0)
    as_ = feats.get("AWAY_STREAK", 0)
    if abs(hs) >= 3:
        lines.append(f"{home_tri} are on a {'win' if hs > 0 else 'loss'} streak of {abs(hs)}.")
    if abs(as_) >= 3:
        lines.append(f"{away_tri} are on a {'win' if as_ > 0 else 'loss'} streak of {abs(as_)}.")

    # rest
    hr = feats.get("HOME_REST", 2)
    ar = feats.get("AWAY_REST", 2)
    if feats.get("HOME_B2B"):
        lines.append(f"{home_tri} are on a back-to-back, which may affect performance.")
    if feats.get("AWAY_B2B"):
        lines.append(f"{away_tri} are on a back-to-back, which may affect performance.")
    if hr > ar + 1:
        lines.append(f"{home_tri} have more rest ({hr} days vs {ar}).")
    elif ar > hr + 1:
        lines.append(f"{away_tri} have more rest ({ar} days vs {hr}).")

    # home court
    hcs = feats.get("HOME_COURT_STRENGTH", 0)
    if hcs >= 0.1:
        lines.append(f"{home_tri} have a strong home court advantage.")

    # travel
    tz = feats.get("AWAY_TRAVEL_TZ_DIFF", 0)
    if tz >= 2:
        lines.append(f"{away_tri} are traveling across {int(tz)} time zones.")

    # h2h
    h2h = feats.get("H2H_HOME_WIN_PCT", 0.5)
    if h2h >= 0.65:
        lines.append(f"{home_tri} have dominated this matchup historically ({round(h2h*100)}% H2H win rate).")
    elif h2h <= 0.35:
        lines.append(f"{away_tri} have dominated this matchup historically ({round((1-h2h)*100)}% H2H win rate).")

    # injuries
    for inj, tri in [(home_injuries, home_tri), (away_injuries, away_tri)]:
        out = [p["name"] for p in inj if p["status"].lower() == "out"]
        q   = [p["name"] for p in inj if p["status"].lower() in ("questionable", "doubtful")]
        if out:
            lines.append(f"{tri} are missing {', '.join(out[:3])} (Out).")
        if q:
            lines.append(f"{', '.join(q[:2])} ({tri}) are questionable.")

    return " ".join(lines)



# ── news helpers ─────────────────────────────────────────────────────────────

TEAM_NAMES = {
    "ATL": ["Hawks", "Atlanta"], "BOS": ["Celtics", "Boston"], "BKN": ["Nets", "Brooklyn"],
    "CHA": ["Hornets", "Charlotte"], "CHI": ["Bulls", "Chicago"], "CLE": ["Cavaliers", "Cleveland"],
    "DAL": ["Mavericks", "Dallas"], "DEN": ["Nuggets", "Denver"], "DET": ["Pistons", "Detroit"],
    "GSW": ["Warriors", "Golden State"], "HOU": ["Rockets", "Houston"], "IND": ["Pacers", "Indiana"],
    "LAC": ["Clippers", "LA Clippers"], "LAL": ["Lakers", "LA Lakers", "Los Angeles Lakers"],
    "MEM": ["Grizzlies", "Memphis"], "MIA": ["Heat", "Miami"], "MIL": ["Bucks", "Milwaukee"],
    "MIN": ["Timberwolves", "Minnesota"], "NOP": ["Pelicans", "New Orleans"],
    "NYK": ["Knicks", "New York"], "OKC": ["Thunder", "Oklahoma City"],
    "ORL": ["Magic", "Orlando"], "PHI": ["76ers", "Sixers", "Philadelphia"],
    "PHX": ["Suns", "Phoenix"], "POR": ["Trail Blazers", "Portland"],
    "SAC": ["Kings", "Sacramento"], "SAS": ["Spurs", "San Antonio"],
    "TOR": ["Raptors", "Toronto"], "UTA": ["Jazz", "Utah"], "WAS": ["Wizards", "Washington"],
}

_news_cache = {"items": [], "updated": None}

def fetch_news():
    try:
        import xml.etree.ElementTree as ET
        r = requests.get(
            "https://www.espn.com/espn/rss/nba/news",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            if title and link:
                items.append({"title": title, "link": link})
        _news_cache["items"]   = items
        _news_cache["updated"] = datetime.now(timezone.utc)
        return items
    except Exception as e:
        print(f"[news] fetch error: {e}")
        return _news_cache["items"]

def get_news():
    updated = _news_cache["updated"]
    if updated is None or (datetime.now(timezone.utc) - updated).seconds > 1800:
        return fetch_news()
    return _news_cache["items"]

def news_for_team(tri, all_news, max_items=2):
    keywords = TEAM_NAMES.get(tri, [])
    results = []
    for item in all_news:
        if any(kw.lower() in item["title"].lower() for kw in keywords):
            results.append(item)
        if len(results) >= max_items:
            break
    return results


def fetch_predictions():
    time.sleep(SLEEP)

    # ScoreboardV3 is more reliable than the live scoreboard endpoint
    games = []
    try:
        sb    = ScoreboardV3(game_date=date.today().strftime("%m/%d/%Y"), timeout=30)
        games = sb.get_dict()["scoreboard"]["games"]
    except Exception as e:
        print(f"[scoreboard] ScoreboardV3 failed: {e}, trying live API")
        try:
            time.sleep(SLEEP)
            sb2   = live_scoreboard.ScoreBoard()
            games = sb2.get_dict()["scoreboard"]["games"]
        except Exception as e2:
            print(f"[scoreboard] live API also failed: {e2}")

    injury_report = get_injury_report()
    player_stats  = get_player_stats()
    all_news      = get_news()
    results = []
    for g in games:
        home_id     = g["homeTeam"]["teamId"]
        away_id     = g["awayTeam"]["teamId"]
        home_tri    = g["homeTeam"]["teamTricode"]
        away_tri    = g["awayTeam"]["teamTricode"]
        home_name   = g["homeTeam"]["teamCity"] + " " + g["homeTeam"]["teamName"]
        away_name   = g["awayTeam"]["teamCity"] + " " + g["awayTeam"]["teamName"]
        home_rec    = f"{g['homeTeam']['wins']}-{g['homeTeam']['losses']}"
        away_rec    = f"{g['awayTeam']['wins']}-{g['awayTeam']['losses']}"
        status      = g["gameStatusText"]
        game_status = g["gameStatus"]
        home_score  = g["homeTeam"]["score"]
        away_score  = g["awayTeam"]["score"]
        game_id     = g["gameId"]
        period      = g.get("period", 0)
        game_clock  = g.get("gameClock", "")
        try:
            feats = build_features(home_id, away_id, home_tri, away_tri)
            X = pd.DataFrame([feats], columns=FEATURE_COLS).values
            probs, preds = {}, {}
            for name, model in MODELS.items():
                p = model.predict_proba(X)[0, 1]
                probs[name] = round(float(p), 3)
                preds[name] = int(p >= 0.5)
            margin   = float(MARGIN_MODEL.predict(X)[0])
            raw_prob = probs["Ensemble"]

            # apply injury adjustments pre-game
            if game_status == 1:
                home_adj  = injury_adjustment(home_id, injury_report)
                away_adj  = injury_adjustment(away_id, injury_report)
                home_prob = round(min(0.99, max(0.01, raw_prob + home_adj - away_adj)), 3)
            else:
                home_prob = raw_prob

            away_prob   = round(1 - home_prob, 3)
            pred_winner = home_tri if home_prob >= 0.5 else away_tri
            agreement   = sum(
                preds[m] == preds["Ensemble"]
                for m in ["LogisticRegression", "RandomForest", "GradientBoosting"]
            )

            # track prob history for live games — always add point each refresh
            if game_status == 2:
                ts = datetime.now(PST).strftime("%I:%M")
                if game_id not in _prob_history:
                    _prob_history[game_id] = []
                _prob_history[game_id].append({"t": ts, "home_prob": home_prob})
                if len(_prob_history[game_id]) > 100:
                    _prob_history[game_id].pop(0)

            # build scenarios, analysis, and top factors for pre-game
            scenarios    = []
            analysis     = ""
            top_factors  = []
            if game_status == 1:
                scenarios   = build_scenarios(home_prob, home_id, away_id, injury_report, player_stats)
                analysis    = build_analysis(home_tri, away_tri, feats, home_prob,
                                             injury_report.get(home_id, []),
                                             injury_report.get(away_id, []), player_stats)
                top_factors = build_top_factors(feats, home_tri, away_tri, home_prob)

            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": game_id,
                "home_id": home_id, "away_id": away_id,
                "home_score": home_score, "away_score": away_score, "period": period, "game_clock": game_clock,
                "home_prob": home_prob, "away_prob": away_prob,
                "pred_winner": pred_winner,
                "margin": round(margin, 1),
                "agreement": f"{agreement}/3",
                "prob_lr": probs["LogisticRegression"],
                "prob_rf": probs["RandomForest"],
                "prob_gb": probs["GradientBoosting"],
                "home_b2b": bool(feats["HOME_B2B"]),
                "away_b2b": bool(feats["AWAY_B2B"]),
                "prob_history":   _prob_history.get(game_id, []),
                "home_injuries":  injury_report.get(home_id, []),
                "away_injuries":  injury_report.get(away_id, []),
                "scenarios":      scenarios,
                "analysis":       analysis,
                "top_factors":    top_factors,
                "home_news":      news_for_team(home_tri, all_news) if game_status == 1 else [],
                "away_news":      news_for_team(away_tri, all_news) if game_status == 1 else [],
            })
        except Exception as e:
            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": game_id,
                "home_id": home_id, "away_id": away_id,
                "home_score": home_score, "away_score": away_score, "period": period, "game_clock": game_clock,
                "error": str(e),
            })
    return results

def refresh_cache():
    try:
        today_str = date.today().isoformat()
        with _cache_lock:
            if _cache["date"] != today_str:
                _cache["data"] = None
                _cache["date"] = today_str
                _prob_history.clear()
                print(f"[cache] new day {today_str} — cleared stale games")
        results = fetch_predictions()
        updated = datetime.now(timezone(timedelta(hours=-7))).strftime("%I:%M %p PST")
        with _cache_lock:
            _cache["data"]    = results
            _cache["updated"] = updated
            _cache["date"]    = today_str
        record_final_games(results)
        print(f"[cache] refreshed at {updated}")
    except Exception as e:
        print(f"[cache] refresh error: {e}")

# ── start scheduler (runs on Render + locally) ───────────────────────────────

_scheduler = BackgroundScheduler()
_scheduler.add_job(refresh_cache, "interval", seconds=60)
_scheduler.add_job(refresh_cache, "cron", hour=0, minute=5)
_scheduler.add_job(
    lambda: requests.get("https://web-production-438de.up.railway.app/api/predictions", timeout=10),
    "interval", minutes=10
)
_scheduler.start()
threading.Thread(target=refresh_cache, daemon=True).start()
print("Scheduler started — cache warming up in background.")

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", today=date.today().strftime("%B %d, %Y"))

@app.route("/api/boxscore/<game_id>")
def get_boxscore(game_id):
    try:
        time.sleep(SLEEP)
        bs        = live_boxscore.BoxScore(game_id)
        data      = bs.get_dict()["game"]
        home_data = data["homeTeam"]
        away_data = data["awayTeam"]
        def parse_periods(td):
            return [{"period": p["period"], "type": p["periodType"], "score": p["score"]} for p in td["periods"]]
        return jsonify({
            "home_tri":     home_data["teamTricode"],
            "away_tri":     away_data["teamTricode"],
            "home_score":   home_data["statistics"]["points"],
            "away_score":   away_data["statistics"]["points"],
            "home_periods": parse_periods(home_data),
            "away_periods": parse_periods(away_data),
            "home_players": parse_players_v3(home_data),
            "away_players": parse_players_v3(away_data),
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/predictions")
def predictions():
    with _cache_lock:
        data    = _cache["data"]
        updated = _cache["updated"]
    if data is None:
        try:
            data    = fetch_predictions()
            updated = datetime.now(timezone(timedelta(hours=-7))).strftime("%I:%M %p PST")
            with _cache_lock:
                _cache["data"]    = data
                _cache["updated"] = updated
        except Exception as e:
            return jsonify({"error": str(e), "games": []})
    return jsonify({"games": data, "date": date.today().isoformat(), "updated": updated})

@app.route("/api/accuracy")
def accuracy():
    return jsonify(compute_accuracy())

@app.route("/api/schedule")
def schedule():
    """Return games for the next 7 days (excluding today), cached per day."""
    today_str = date.today().isoformat()
    with _cache_lock:
        if _schedule_cache["date"] == today_str and _schedule_cache["data"] is not None:
            return jsonify({"games": _schedule_cache["data"]})
    results = []
    for delta in range(1, 8):
        d = date.today() + timedelta(days=delta)
        try:
            time.sleep(SLEEP)
            sb = ScoreboardV3(game_date=d.strftime("%m/%d/%Y"), timeout=20)
            games = sb.get_dict()["scoreboard"]["games"]
            date_disp = d.strftime("%a %b ") + str(d.day)  # avoids %-d on Windows
            for g in games:
                results.append({
                    "date":      d.isoformat(),
                    "date_disp": date_disp,
                    "home_tri":  g["homeTeam"]["teamTricode"],
                    "away_tri":  g["awayTeam"]["teamTricode"],
                    "home_name": g["homeTeam"]["teamCity"] + " " + g["homeTeam"]["teamName"],
                    "away_name": g["awayTeam"]["teamCity"] + " " + g["awayTeam"]["teamName"],
                    "status":    g["gameStatusText"],
                })
        except Exception as e:
            print(f"[schedule] {d}: {e}")
    with _cache_lock:
        _schedule_cache["data"] = results
        _schedule_cache["date"] = today_str
    return jsonify({"games": results})

@app.route("/api/yesterday")
def yesterday():
    history       = load_history()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    return jsonify({"games": history.get(yesterday_str, []), "date": yesterday_str})

if __name__ == "__main__":
    import webbrowser
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    print("NBA Predictions running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
