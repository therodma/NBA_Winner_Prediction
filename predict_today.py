"""
predict_today.py — Step 4: Daily Prediction Pipeline
Fetches today's NBA schedule, builds features for each matchup,
runs all trained models, and prints a prediction table.
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
RAW        = Path("data/raw")
PROCESSED  = Path("data/processed")
SLEEP      = 0.6

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


# ── load saved assets ─────────────────────────────────────────────────────────

def load_models():
    return {
        "LogisticRegression": joblib.load(MODELS_DIR / "logisticregression.pkl"),
        "RandomForest":       joblib.load(MODELS_DIR / "randomforest.pkl"),
        "GradientBoosting":   joblib.load(MODELS_DIR / "gradientboosting.pkl"),
        "Ensemble":           joblib.load(MODELS_DIR / "ensemble.pkl"),
    }


def load_history():
    """Load game logs and matchup history for feature computation."""
    logs     = pd.read_csv(RAW / "game_logs.csv", parse_dates=["GAME_DATE"])
    matchups = pd.read_csv(RAW / "matchups.csv",  parse_dates=["GAME_DATE"])
    stats    = pd.read_csv(RAW / "team_stats.csv")
    return logs, matchups, stats


# ── fetch today's schedule ────────────────────────────────────────────────────

def get_todays_games() -> pd.DataFrame:
    """Return today's games as a DataFrame with home/away team IDs and tricodes."""
    sb = live_scoreboard.ScoreBoard()
    games = sb.get_dict()["scoreboard"]["games"]
    if not games:
        return pd.DataFrame()

    rows = []
    for g in games:
        rows.append({
            "GAME_ID":       g["gameId"],
            "HOME_TEAM_ID":  g["homeTeam"]["teamId"],
            "HOME_TEAM":     g["homeTeam"]["teamTricode"],
            "AWAY_TEAM_ID":  g["awayTeam"]["teamId"],
            "AWAY_TEAM":     g["awayTeam"]["teamTricode"],
            "GAME_TIME":     g["gameStatusText"],
        })
    return pd.DataFrame(rows)


# ── feature computation for today's games ────────────────────────────────────

def get_rolling_form(logs: pd.DataFrame, team_id: int) -> dict:
    """Last 10 games rolling win% and +/- for a team."""
    team_logs = logs[logs["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    last10 = team_logs.tail(10)
    if len(last10) == 0:
        return {"roll10_win_pct": 0.5, "roll10_pm": 0.0}
    win_pct = (last10["WL"] == "W").mean()
    pm      = last10["PLUS_MINUS"].astype(float).mean()
    return {"roll10_win_pct": win_pct, "roll10_pm": pm}


def get_rest_days(logs: pd.DataFrame, team_id: int) -> dict:
    """Days since last game and B2B flag for a team."""
    team_logs = logs[logs["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    if len(team_logs) == 0:
        return {"rest_days": 7, "is_b2b": 0}
    last_game = team_logs["GAME_DATE"].iloc[-1]
    rest = (pd.Timestamp(date.today()) - last_game).days
    return {"rest_days": int(rest), "is_b2b": int(rest == 1)}


def get_h2h(matchups: pd.DataFrame, home_id: int, away_id: int) -> float:
    """Historical H2H win% for the home team vs this away team."""
    pair = matchups[
        ((matchups["HOME_TEAM_ID"] == home_id) & (matchups["AWAY_TEAM_ID"] == away_id)) |
        ((matchups["HOME_TEAM_ID"] == away_id) & (matchups["AWAY_TEAM_ID"] == home_id))
    ].tail(20)
    if len(pair) == 0:
        return 0.5
    home_wins = (
        ((pair["HOME_TEAM_ID"] == home_id) & (pair["HOME_TEAM_WIN"] == 1)) |
        ((pair["AWAY_TEAM_ID"] == home_id) & (pair["HOME_TEAM_WIN"] == 0))
    ).sum()
    return home_wins / len(pair)


def get_current_season_stats(stats: pd.DataFrame, team_id: int) -> pd.Series:
    """Most recent season stats for a team."""
    team_stats = stats[stats["TEAM_ID"] == team_id].sort_values("SEASON")
    if len(team_stats) == 0:
        return pd.Series(dtype=float)
    return team_stats.iloc[-1]


def build_game_features(game: pd.Series, logs: pd.DataFrame,
                        matchups: pd.DataFrame, stats: pd.DataFrame) -> dict:
    home_id = game["HOME_TEAM_ID"]
    away_id = game["AWAY_TEAM_ID"]

    home_stats = get_current_season_stats(stats, home_id)
    away_stats = get_current_season_stats(stats, away_id)
    home_form  = get_rolling_form(logs, home_id)
    away_form  = get_rolling_form(logs, away_id)
    home_rest  = get_rest_days(logs, home_id)
    away_rest  = get_rest_days(logs, away_id)
    h2h        = get_h2h(matchups, home_id, away_id)

    def s(col, home, away):
        hv = home.get(col, np.nan) if isinstance(home, dict) else home[col]
        av = away.get(col, np.nan) if isinstance(away, dict) else away[col]
        return hv - av

    return {
        "DIFF_OFF_RATING":      s("OFF_RATING", home_stats, away_stats),
        "DIFF_DEF_RATING":      s("DEF_RATING", home_stats, away_stats),
        "DIFF_NET_RATING":      s("NET_RATING", home_stats, away_stats),
        "DIFF_PACE":            s("PACE", home_stats, away_stats),
        "DIFF_W_PCT":           s("W_PCT", home_stats, away_stats),
        "DIFF_FG_PCT":          s("FG_PCT", home_stats, away_stats),
        "DIFF_FG3_PCT":         s("FG3_PCT", home_stats, away_stats),
        "DIFF_FT_PCT":          s("FT_PCT", home_stats, away_stats),
        "DIFF_REB":             s("REB", home_stats, away_stats),
        "DIFF_AST":             s("AST", home_stats, away_stats),
        "DIFF_TOV":             s("TOV", home_stats, away_stats),
        "HOME_ROLL10_WIN_PCT":  home_form["roll10_win_pct"],
        "AWAY_ROLL10_WIN_PCT":  away_form["roll10_win_pct"],
        "DIFF_ROLL10_WIN_PCT":  home_form["roll10_win_pct"] - away_form["roll10_win_pct"],
        "HOME_ROLL10_PM":       home_form["roll10_pm"],
        "AWAY_ROLL10_PM":       away_form["roll10_pm"],
        "DIFF_ROLL10_PM":       home_form["roll10_pm"] - away_form["roll10_pm"],
        "HOME_REST":            home_rest["rest_days"],
        "AWAY_REST":            away_rest["rest_days"],
        "DIFF_REST":            home_rest["rest_days"] - away_rest["rest_days"],
        "HOME_B2B":             home_rest["is_b2b"],
        "AWAY_B2B":             away_rest["is_b2b"],
        "H2H_HOME_WIN_PCT":     h2h,
    }


# ── prediction table ──────────────────────────────────────────────────────────

def predict(games: pd.DataFrame, logs: pd.DataFrame,
            matchups: pd.DataFrame, stats: pd.DataFrame,
            models: dict, margin_model) -> pd.DataFrame:

    feature_rows = []
    for _, game in games.iterrows():
        feature_rows.append(build_game_features(game, logs, matchups, stats))

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLS).values

    results = games[["HOME_TEAM", "AWAY_TEAM", "GAME_TIME"]].copy()

    probs = {}
    preds = {}
    for name, model in models.items():
        p = model.predict_proba(X)[:, 1]
        probs[name] = p
        preds[name] = (p >= 0.5).astype(int)

    # Ensemble win probability
    results["HOME_WIN_PROB"] = probs["Ensemble"].round(3)
    results["PRED_WINNER"] = np.where(
        probs["Ensemble"] >= 0.5,
        results["HOME_TEAM"],
        results["AWAY_TEAM"],
    )
    results["PRED_MARGIN"] = margin_model.predict(X).round(1)

    # Model agreement: how many of the 3 base models agree with ensemble
    agreement = (
        (preds["LogisticRegression"] == preds["Ensemble"]).astype(int) +
        (preds["RandomForest"]       == preds["Ensemble"]).astype(int) +
        (preds["GradientBoosting"]   == preds["Ensemble"]).astype(int)
    )
    results["MODEL_AGREE"] = agreement.astype(str) + "/3"

    # Individual model probs for transparency
    results["PROB_LR"] = probs["LogisticRegression"].round(3)
    results["PROB_RF"] = probs["RandomForest"].round(3)
    results["PROB_GB"] = probs["GradientBoosting"].round(3)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"=== NBA Predictions for {date.today()} ===\n")

    print("Loading models and history...")
    models       = load_models()
    margin_model = joblib.load(MODELS_DIR / "margin_regressor.pkl")
    logs, matchups, stats = load_history()

    print("Fetching today's schedule...")
    time.sleep(SLEEP)
    games = get_todays_games()

    if games.empty:
        print("No games scheduled today.")
        return

    print(f"Found {len(games)} game(s). Building features and predicting...\n")
    results = predict(games, logs, matchups, stats, models, margin_model)

    # Pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print("-" * 100)
    print(f"{'HOME':<6} {'AWAY':<6} {'PRED WINNER':<12} {'HOME WIN%':<12} {'PRED MARGIN':<14} {'AGREE':<8} {'LR%':<7} {'RF%':<7} {'GB%':<7} TIME")
    print("-" * 100)
    for _, r in results.iterrows():
        margin_str = f"+{r['PRED_MARGIN']:.1f}" if r["PRED_MARGIN"] >= 0 else f"{r['PRED_MARGIN']:.1f}"
        print(
            f"{r['HOME_TEAM']:<6} {r['AWAY_TEAM']:<6} {r['PRED_WINNER']:<12} "
            f"{r['HOME_WIN_PROB']:<12.1%} {margin_str:<14} {r['MODEL_AGREE']:<8} "
            f"{r['PROB_LR']:<7.1%} {r['PROB_RF']:<7.1%} {r['PROB_GB']:<7.1%} "
            f"{r['GAME_TIME']}"
        )
    print("-" * 100)
    print("\nNote: margin is from home team's perspective (positive = home team wins by that amount)")


if __name__ == "__main__":
    main()
