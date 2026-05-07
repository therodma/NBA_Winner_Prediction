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

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

MODELS_DIR   = Path("models")
RAW          = Path("data/raw")
SLEEP        = 0.3

GIST_ID      = os.environ.get("GIST_ID", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GIST_FILE    = "history.json"
GIST_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

FEATURE_COLS = [
    "DIFF_OFF_RATING", "DIFF_DEF_RATING", "DIFF_NET_RATING", "DIFF_PACE",
    "DIFF_W_PCT", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_AST", "DIFF_TOV",
    "HOME_ROLL10_WIN_PCT", "AWAY_ROLL10_WIN_PCT", "DIFF_ROLL10_WIN_PCT",
    "HOME_ROLL10_PM", "AWAY_ROLL10_PM", "DIFF_ROLL10_PM",
    "HOME_ROLL20_WIN_PCT", "AWAY_ROLL20_WIN_PCT", "DIFF_ROLL20_WIN_PCT",
    "HOME_ROLL20_PM", "AWAY_ROLL20_PM", "DIFF_ROLL20_PM",
    "HOME_STREAK", "AWAY_STREAK", "DIFF_STREAK",
    "HOME_REST", "AWAY_REST", "DIFF_REST",
    "HOME_B2B", "AWAY_B2B",
    "H2H_HOME_WIN_PCT",
]

# ── load once at startup ──────────────────────────────────────────────────────

print("Loading models and data...")
MODELS = {
    "LogisticRegression": joblib.load(MODELS_DIR / "logisticregression.pkl"),
    "RandomForest":       joblib.load(MODELS_DIR / "randomforest.pkl"),
    "GradientBoosting":   joblib.load(MODELS_DIR / "gradientboosting.pkl"),
    "Ensemble":           joblib.load(MODELS_DIR / "ensemble.pkl"),
}
MARGIN_MODEL = joblib.load(MODELS_DIR / "margin_regressor.pkl")
LOGS         = pd.read_csv(RAW / "game_logs.csv",  parse_dates=["GAME_DATE"])
MATCHUPS     = pd.read_csv(RAW / "matchups.csv",   parse_dates=["GAME_DATE"])
STATS        = pd.read_csv(RAW / "team_stats.csv")
print("Ready.")

# ── in-memory cache ───────────────────────────────────────────────────────────

_cache = {"data": None, "updated": None}
_cache_lock = threading.Lock()

# prob history: {game_id: [{time, home_prob}]}
_prob_history = {}

# ── feature helpers ───────────────────────────────────────────────────────────

def get_rolling_form(team_id):
    t = LOGS[LOGS["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    last10 = t.tail(10)
    last20 = t.tail(20)
    if len(last10) == 0:
        return 0.5, 0.0, 0.5, 0.0, 0
    win10 = (last10["WL"] == "W").mean()
    pm10  = last10["PLUS_MINUS"].astype(float).mean()
    win20 = (last20["WL"] == "W").mean() if len(last20) > 0 else win10
    pm20  = last20["PLUS_MINUS"].astype(float).mean() if len(last20) > 0 else pm10
    # streak: positive = win streak, negative = loss streak
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

def build_features(home_id, away_id):
    hs, as_ = get_stats(home_id), get_stats(away_id)
    hwp, hpm, hwp20, hpm20, hstreak = get_rolling_form(home_id)
    awp, apm, awp20, apm20, astreak = get_rolling_form(away_id)
    hr, hb2b = get_rest(home_id)
    ar, ab2b = get_rest(away_id)
    h2h = get_h2h(home_id, away_id)
    return {
        "DIFF_OFF_RATING":     hs["OFF_RATING"]  - as_["OFF_RATING"],
        "DIFF_DEF_RATING":     hs["DEF_RATING"]  - as_["DEF_RATING"],
        "DIFF_NET_RATING":     hs["NET_RATING"]  - as_["NET_RATING"],
        "DIFF_PACE":           hs["PACE"]        - as_["PACE"],
        "DIFF_W_PCT":          hs["W_PCT"]       - as_["W_PCT"],
        "DIFF_FG_PCT":         hs["FG_PCT"]      - as_["FG_PCT"],
        "DIFF_FG3_PCT":        hs["FG3_PCT"]     - as_["FG3_PCT"],
        "DIFF_FT_PCT":         hs["FT_PCT"]      - as_["FT_PCT"],
        "DIFF_REB":            hs["REB"]         - as_["REB"],
        "DIFF_AST":            hs["AST"]         - as_["AST"],
        "DIFF_TOV":            hs["TOV"]         - as_["TOV"],
        "HOME_ROLL10_WIN_PCT": hwp,  "AWAY_ROLL10_WIN_PCT": awp,
        "DIFF_ROLL10_WIN_PCT": hwp - awp,
        "HOME_ROLL10_PM":      hpm,  "AWAY_ROLL10_PM": apm,
        "DIFF_ROLL10_PM":      hpm - apm,
        "HOME_ROLL20_WIN_PCT": hwp20, "AWAY_ROLL20_WIN_PCT": awp20,
        "DIFF_ROLL20_WIN_PCT": hwp20 - awp20,
        "HOME_ROLL20_PM":      hpm20, "AWAY_ROLL20_PM": apm20,
        "DIFF_ROLL20_PM":      hpm20 - apm20,
        "HOME_STREAK": hstreak, "AWAY_STREAK": astreak,
        "DIFF_STREAK": hstreak - astreak,
        "HOME_REST": hr, "AWAY_REST": ar, "DIFF_REST": hr - ar,
        "HOME_B2B": hb2b, "AWAY_B2B": ab2b,
        "H2H_HOME_WIN_PCT": h2h,
    }

# ── box score helpers ─────────────────────────────────────────────────────────

def parse_minutes(raw):
    m = raw.replace("PT", "").replace("M", ":").split(":")[0:2]
    return ":".join(m) if len(m) == 2 else raw

def parse_players(team_data):
    players = []
    for p in team_data["players"]:
        if p["played"] != "1":
            continue
        s = p["statistics"]
        players.append({
            "name":    p["name"],
            "starter": p["starter"] == "1",
            "min":     parse_minutes(s.get("minutesCalculated", "PT0M")),
            "pts":     s["points"],
            "reb":     s["reboundsTotal"],
            "ast":     s["assists"],
            "stl":     s["steals"],
            "blk":     s["blocks"],
            "tov":     s["turnovers"],
            "fg":      f"{s['fieldGoalsMade']}/{s['fieldGoalsAttempted']}",
            "3p":      f"{s['threePointersMade']}/{s['threePointersAttempted']}",
            "ft":      f"{s['freeThrowsMade']}/{s['freeThrowsAttempted']}",
            "pm":      int(s["plusMinusPoints"]),
        })
    players.sort(key=lambda x: (-x["pts"], not x["starter"]))
    return players

def parse_team_stats(team_data):
    s = team_data["statistics"]
    return {
        "pts":     s["points"],
        "reb":     s["reboundsTotal"],
        "ast":     s["assists"],
        "tov":     s["turnoversTotal"],
        "fg_pct":  round(s["fieldGoalsPercentage"] * 100, 1),
        "3p_pct":  round(s["threePointersPercentage"] * 100, 1),
        "ft_pct":  round(s["freeThrowsPercentage"] * 100, 1),
        "stl":     s["steals"],
        "blk":     s["blocks"],
        "paint":   s["pointsInThePaint"],
        "fb_pts":  s["pointsFastBreak"],
        "biggest_lead": s["biggestLead"],
    }

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

# ── core prediction fetch (used by cache + scheduler) ─────────────────────────

def fetch_predictions():
    time.sleep(SLEEP)
    sb    = live_scoreboard.ScoreBoard()
    games = sb.get_dict()["scoreboard"]["games"]
    results = []
    for g in games:
        home_id   = g["homeTeam"]["teamId"]
        away_id   = g["awayTeam"]["teamId"]
        home_tri  = g["homeTeam"]["teamTricode"]
        away_tri  = g["awayTeam"]["teamTricode"]
        home_name = g["homeTeam"]["teamCity"] + " " + g["homeTeam"]["teamName"]
        away_name = g["awayTeam"]["teamCity"] + " " + g["awayTeam"]["teamName"]
        home_rec  = f"{g['homeTeam']['wins']}-{g['homeTeam']['losses']}"
        away_rec  = f"{g['awayTeam']['wins']}-{g['awayTeam']['losses']}"
        status      = g["gameStatusText"]
        game_status = g["gameStatus"]
        home_score  = g["homeTeam"]["score"]
        away_score  = g["awayTeam"]["score"]
        game_id     = g["gameId"]
        try:
            feats = build_features(home_id, away_id)
            X = pd.DataFrame([feats], columns=FEATURE_COLS).values
            probs, preds = {}, {}
            for name, model in MODELS.items():
                p = model.predict_proba(X)[0, 1]
                probs[name] = round(float(p), 3)
                preds[name] = int(p >= 0.5)
            margin      = float(MARGIN_MODEL.predict(X)[0])
            home_prob   = probs["Ensemble"]
            away_prob   = round(1 - home_prob, 3)
            pred_winner = home_tri if home_prob >= 0.5 else away_tri
            agreement   = sum(
                preds[m] == preds["Ensemble"]
                for m in ["LogisticRegression", "RandomForest", "GradientBoosting"]
            )
            # track prob history for live games
            if game_status == 2:
                ts = datetime.now(timezone.utc).strftime("%H:%M")
                if game_id not in _prob_history:
                    _prob_history[game_id] = []
                hist = _prob_history[game_id]
                if not hist or hist[-1]["home_prob"] != home_prob:
                    hist.append({"t": ts, "home_prob": home_prob})
                    if len(hist) > 50:
                        hist.pop(0)
            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": game_id,
                "home_id": home_id, "away_id": away_id,
                "home_score": home_score, "away_score": away_score,
                "home_prob": home_prob, "away_prob": away_prob,
                "pred_winner": pred_winner,
                "margin": round(margin, 1),
                "agreement": f"{agreement}/3",
                "prob_lr": probs["LogisticRegression"],
                "prob_rf": probs["RandomForest"],
                "prob_gb": probs["GradientBoosting"],
                "home_b2b": bool(feats["HOME_B2B"]),
                "away_b2b": bool(feats["AWAY_B2B"]),
                "prob_history": _prob_history.get(game_id, []),
            })
        except Exception as e:
            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": game_id,
                "home_id": home_id, "away_id": away_id,
                "home_score": home_score, "away_score": away_score,
                "error": str(e),
            })
    return results

def refresh_cache():
    try:
        results = fetch_predictions()
        updated = datetime.now(timezone.utc).strftime("%I:%M %p UTC")
        with _cache_lock:
            _cache["data"] = results
            _cache["updated"] = updated
        save_results(date.today().isoformat(), results)
        print(f"[cache] refreshed at {updated}")
    except Exception as e:
        print(f"[cache] refresh error: {e}")

def scheduled_save():
    refresh_cache()

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
        def parse_periods(team_data):
            return [
                {"period": p["period"], "type": p["periodType"], "score": p["score"]}
                for p in team_data["periods"]
            ]
        return jsonify({
            "home_tri":     home_data["teamTricode"],
            "away_tri":     away_data["teamTricode"],
            "home_score":   home_data["statistics"]["points"],
            "away_score":   away_data["statistics"]["points"],
            "home_periods": parse_periods(home_data),
            "away_periods": parse_periods(away_data),
            "home_players": parse_players(home_data),
            "away_players": parse_players(away_data),
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
            data = fetch_predictions()
            updated = datetime.now(timezone.utc).strftime("%I:%M %p UTC")
            with _cache_lock:
                _cache["data"] = data
                _cache["updated"] = updated
        except Exception as e:
            return jsonify({"error": str(e), "games": []})
    return jsonify({
        "games":   data,
        "date":    date.today().isoformat(),
        "updated": updated,
    })

@app.route("/api/yesterday")
def yesterday():
    history = load_history()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    games = history.get(yesterday_str, [])
    return jsonify({"games": games, "date": yesterday_str})

if __name__ == "__main__":
    import webbrowser

    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_cache, "interval", seconds=60)
    scheduler.add_job(scheduled_save, "cron", hour=0, minute=5)
    scheduler.add_job(
        lambda: requests.get("https://nba-winner-prediction.onrender.com/", timeout=10),
        "interval", minutes=10
    )
    scheduler.start()
    print("Scheduler started — refreshing every 60s, saving nightly at 12:05 AM.")

    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    print("NBA Predictions running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
