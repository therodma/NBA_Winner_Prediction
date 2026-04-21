"""
app.py — Flask web server for NBA predictions
"""

import time
import json
import warnings
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
HISTORY_FILE = Path("data/history.json")
SLEEP        = 0.6

FEATURE_COLS = [
    "DIFF_OFF_RATING", "DIFF_DEF_RATING", "DIFF_NET_RATING", "DIFF_PACE",
    "DIFF_W_PCT", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_AST", "DIFF_TOV",
    "HOME_ROLL10_WIN_PCT", "AWAY_ROLL10_WIN_PCT", "DIFF_ROLL10_WIN_PCT",
    "HOME_ROLL10_PM", "AWAY_ROLL10_PM", "DIFF_ROLL10_PM",
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

# ── feature helpers ───────────────────────────────────────────────────────────

def get_rolling_form(team_id):
    t = LOGS[LOGS["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    last10 = t.tail(10)
    if len(last10) == 0:
        return 0.5, 0.0
    return (last10["WL"] == "W").mean(), last10["PLUS_MINUS"].astype(float).mean()

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
    hwp, hpm = get_rolling_form(home_id)
    awp, apm = get_rolling_form(away_id)
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
        "HOME_ROLL10_WIN_PCT": hwp, "AWAY_ROLL10_WIN_PCT": awp,
        "DIFF_ROLL10_WIN_PCT": hwp - awp,
        "HOME_ROLL10_PM":      hpm, "AWAY_ROLL10_PM": apm,
        "DIFF_ROLL10_PM":      hpm - apm,
        "HOME_REST": hr, "AWAY_REST": ar, "DIFF_REST": hr - ar,
        "HOME_B2B": hb2b, "AWAY_B2B": ab2b,
        "H2H_HOME_WIN_PCT": h2h,
    }

# ── box score helpers ─────────────────────────────────────────────────────────

def parse_minutes(raw):
    """Convert PT24M41.00S -> '24:41'"""
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
    """Extract actual team-level stats from box score for comparison."""
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

def build_comparison(home_tri, away_tri, home_id, away_id,
                     actual_home, actual_away,
                     pred_margin, pred_winner,
                     actual_home_score, actual_away_score):
    """
    Build predicted vs actual comparison for a final game.
    'Predicted' team stats come from season averages (what the model expected).
    'Actual' stats come from the box score.
    """
    hs = get_stats(home_id)
    as_ = get_stats(away_id)

    actual_margin  = actual_home_score - actual_away_score
    actual_winner  = home_tri if actual_margin > 0 else away_tri
    winner_correct = pred_winner == actual_winner
    margin_error   = round(abs(pred_margin - actual_margin), 1)

    def stat_row(label, pred_home, pred_away, act_home, act_away, fmt="{:.1f}", pct=False):
        suffix = "%" if pct else ""
        return {
            "label":      label,
            "pred_home":  fmt.format(pred_home) + suffix,
            "pred_away":  fmt.format(pred_away) + suffix,
            "act_home":   fmt.format(act_home)  + suffix,
            "act_away":   fmt.format(act_away)  + suffix,
            "home_delta": round(act_home - pred_home, 1),
            "away_delta": round(act_away - pred_away, 1),
        }

    rows = [
        stat_row("Points",   float(hs["PTS"]),     float(as_["PTS"]),
                              actual_home["pts"],   actual_away["pts"],    fmt="{:.0f}"),
        stat_row("Rebounds",  float(hs["REB"]),     float(as_["REB"]),
                              actual_home["reb"],   actual_away["reb"],    fmt="{:.0f}"),
        stat_row("Assists",   float(hs["AST"]),     float(as_["AST"]),
                              actual_home["ast"],   actual_away["ast"],    fmt="{:.0f}"),
        stat_row("Turnovers", float(hs["TOV"]),     float(as_["TOV"]),
                              actual_home["tov"],   actual_away["tov"],    fmt="{:.0f}"),
        stat_row("FG%",       float(hs["FG_PCT"])*100,  float(as_["FG_PCT"])*100,
                              actual_home["fg_pct"], actual_away["fg_pct"], fmt="{:.1f}", pct=True),
        stat_row("3P%",       float(hs["FG3_PCT"])*100, float(as_["FG3_PCT"])*100,
                              actual_home["3p_pct"], actual_away["3p_pct"], fmt="{:.1f}", pct=True),
        stat_row("FT%",       float(hs["FT_PCT"])*100,  float(as_["FT_PCT"])*100,
                              actual_home["ft_pct"], actual_away["ft_pct"], fmt="{:.1f}", pct=True),
    ]

    return {
        "pred_winner":    pred_winner,
        "actual_winner":  actual_winner,
        "winner_correct": winner_correct,
        "pred_margin":    round(pred_margin, 1),
        "actual_margin":  actual_margin,
        "margin_error":   margin_error,
        "home_tri":       home_tri,
        "away_tri":       away_tri,
        "rows":           rows,
    }

# ── history helpers ──────────────────────────────────────────────────────────

def load_history():
    if not HISTORY_FILE.exists():
        return {}
    with open(HISTORY_FILE) as f:
        return json.load(f)

def save_results(date_str, games):
    """Save final games for a given date to history.json."""
    final_games = [g for g in games if g.get("game_status") == 3 and "pred_winner" in g]
    if not final_games:
        return
    history = load_history()
    history[date_str] = final_games
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def scheduled_save():
    """Background job: fetch today's final games and save to history."""
    try:
        time.sleep(SLEEP)
        sb    = live_scoreboard.ScoreBoard()
        games = sb.get_dict()["scoreboard"]["games"]
    except Exception as e:
        print(f"[scheduler] scoreboard error: {e}")
        return

    results = []
    for g in games:
        home_id   = g["homeTeam"]["teamId"]
        away_id   = g["awayTeam"]["teamId"]
        home_tri  = g["homeTeam"]["teamTricode"]
        away_tri  = g["awayTeam"]["teamTricode"]
        game_status = g["gameStatus"]
        home_score  = g["homeTeam"]["score"]
        away_score  = g["awayTeam"]["score"]
        if game_status != 3:
            continue
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
            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": g["homeTeam"]["teamCity"] + " " + g["homeTeam"]["teamName"],
                "away_name": g["awayTeam"]["teamCity"] + " " + g["awayTeam"]["teamName"],
                "home_rec": f"{g['homeTeam']['wins']}-{g['homeTeam']['losses']}",
                "away_rec": f"{g['awayTeam']['wins']}-{g['awayTeam']['losses']}",
                "status": g["gameStatusText"], "game_status": game_status,
                "game_id": g["gameId"],
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
            })
        except Exception as e:
            print(f"[scheduler] prediction error for {home_tri} vs {away_tri}: {e}")

    if results:
        save_results(date.today().isoformat(), results)
        print(f"[scheduler] saved {len(results)} final games for {date.today().isoformat()}")

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


@app.route("/api/comparison/<game_id>/<int:home_id>/<int:away_id>/<int:home_score>/<int:away_score>/<pred_winner>/<float:pred_margin>")
def get_comparison(game_id, home_id, away_id, home_score, away_score, pred_winner, pred_margin):
    try:
        time.sleep(SLEEP)
        bs        = live_boxscore.BoxScore(game_id)
        data      = bs.get_dict()["game"]
        home_data = data["homeTeam"]
        away_data = data["awayTeam"]

        actual_home = parse_team_stats(home_data)
        actual_away = parse_team_stats(away_data)

        comparison = build_comparison(
            home_tri=home_data["teamTricode"],
            away_tri=away_data["teamTricode"],
            home_id=home_id,
            away_id=away_id,
            actual_home=actual_home,
            actual_away=actual_away,
            pred_margin=pred_margin,
            pred_winner=pred_winner,
            actual_home_score=home_score,
            actual_away_score=away_score,
        )
        return jsonify(comparison)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predictions")
def predictions():
    try:
        time.sleep(SLEEP)
        sb    = live_scoreboard.ScoreBoard()
        games = sb.get_dict()["scoreboard"]["games"]
    except Exception as e:
        return jsonify({"error": str(e), "games": []})

    if not games:
        return jsonify({"games": [], "date": date.today().isoformat()})

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
        game_status = g["gameStatus"]  # 1=pre, 2=live, 3=final
        home_score  = g["homeTeam"]["score"]
        away_score  = g["awayTeam"]["score"]

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

            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": g["gameId"],
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
            })
        except Exception as e:
            results.append({
                "home_tri": home_tri, "away_tri": away_tri,
                "home_name": home_name, "away_name": away_name,
                "home_rec": home_rec, "away_rec": away_rec,
                "status": status, "game_status": game_status,
                "game_id": g["gameId"],
                "home_id": home_id, "away_id": away_id,
                "home_score": home_score, "away_score": away_score,
                "error": str(e),
            })

    save_results(date.today().isoformat(), results)

    return jsonify({
        "games": results,
        "date": date.today().isoformat(),
        "updated": datetime.now(timezone.utc).strftime("%I:%M %p UTC"),
    })


@app.route("/api/yesterday")
def yesterday():
    history = load_history()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    games = history.get(yesterday_str, [])
    return jsonify({"games": games, "date": yesterday_str})


if __name__ == "__main__":
    import threading, webbrowser

    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_save, "cron", hour=0, minute=5)
    scheduler.add_job(
        lambda: __import__('requests').get("https://nba-winner-prediction.onrender.com/", timeout=10),
        "interval", minutes=10
    )
    scheduler.start()
    print("Scheduler started — saving results nightly at 12:05 AM, pinging every 10 min.")

    def open_browser():
        webbrowser.open("http://localhost:5000")
    threading.Timer(1.2, open_browser).start()
    print("NBA Predictions running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
