"""
improvements.py — Step 5: Accuracy Improvements
Adds three new feature groups to the feature matrix and retrains models:
  1. Star player availability (injury proxy via missing game logs)
  2. Travel distance/fatigue (miles traveled before each game)
  3. Strength of schedule (avg opponent net rating, last 10 games)

Run this AFTER collect_data.py, engineer_features.py, and train_models.py.
Outputs: data/processed/features_v2.csv and updated models in models/
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from nba_api.stats.endpoints import LeagueDashPlayerStats, TeamGameLog
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS    = Path("models")
SLEEP     = 0.6

# ── Arena coordinates (lat, lon) ─────────────────────────────────────────────
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


def haversine_miles(coord1, coord2) -> float:
    """Great-circle distance in miles between two (lat, lon) pairs."""
    R = 3958.8
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Improvement 1: Star player availability ───────────────────────────────────

def fetch_player_game_logs(season: str) -> pd.DataFrame:
    """Fetch per-game player logs to detect absences."""
    time.sleep(SLEEP)
    # We use team game logs per player via LeagueDashPlayerStats filtered by game
    # Instead, use the game log approach: players who appear in box scores
    from nba_api.stats.endpoints import LeagueGameLog
    time.sleep(SLEEP)
    df = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",  # player-level logs
    ).get_data_frames()[0]
    df["SEASON"] = season
    return df


def build_star_availability(seasons):
    """
    For each team-game, compute the fraction of top-5 (by season minutes)
    players who actually played. Returns a DataFrame keyed by (TEAM_ID, GAME_ID).
    """
    print("  Fetching player game logs (this takes ~2 min)...")
    frames = []
    for s in seasons:
        print(f"    Season {s}...")
        frames.append(fetch_player_game_logs(s))
    player_logs = pd.concat(frames, ignore_index=True)

    # Identify top-5 players by total minutes per team per season
    player_logs["MIN_NUM"] = pd.to_numeric(player_logs["MIN"], errors="coerce").fillna(0)
    season_mins = (
        player_logs.groupby(["SEASON", "TEAM_ID", "PLAYER_ID"])["MIN_NUM"]
        .sum().reset_index()
    )
    season_mins["RANK"] = season_mins.groupby(["SEASON", "TEAM_ID"])["MIN_NUM"].rank(
        ascending=False, method="first"
    )
    stars = season_mins[season_mins["RANK"] <= 5][["SEASON", "TEAM_ID", "PLAYER_ID"]]

    # For each game, check which stars played
    played = player_logs[["SEASON", "TEAM_ID", "GAME_ID", "PLAYER_ID"]].copy()
    played["PLAYED"] = 1

    star_games = stars.merge(
        played, on=["SEASON", "TEAM_ID", "PLAYER_ID"], how="left"
    )
    star_games["PLAYED"] = star_games["PLAYED"].fillna(0)

    availability = (
        star_games.groupby(["TEAM_ID", "GAME_ID"])["PLAYED"]
        .mean()
        .reset_index()
        .rename(columns={"PLAYED": "STAR_AVAILABILITY"})
    )
    availability["GAME_ID"] = availability["GAME_ID"].astype("int64")
    return availability


# ── Improvement 2: Travel distance ────────────────────────────────────────────

def build_travel_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, compute miles traveled from the previous game's arena.
    Home games after away games = long travel. Consecutive home games = 0.
    """
    logs = game_logs.copy().sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    def get_arena(matchup, team_abbr):
        if "vs." in matchup:
            return team_abbr
        else:
            return matchup.split("@ ")[-1].strip()

    logs["GAME_ARENA"] = logs.apply(
        lambda r: get_arena(r["MATCHUP"], r["TEAM_ABBREVIATION"]), axis=1
    )

    travel_miles = []
    for team_id, group in logs.groupby("TEAM_ID"):
        arenas = group["GAME_ARENA"].tolist()
        distances = [0.0]
        for i in range(1, len(arenas)):
            prev, curr = arenas[i-1], arenas[i]
            if prev == curr:
                distances.append(0.0)
            elif prev in ARENA_COORDS and curr in ARENA_COORDS:
                distances.append(haversine_miles(ARENA_COORDS[prev], ARENA_COORDS[curr]))
            else:
                distances.append(0.0)
        for idx, dist in zip(group.index, distances):
            travel_miles.append((idx, dist))

    travel_series = pd.Series(
        {idx: dist for idx, dist in travel_miles}, name="TRAVEL_MILES"
    )
    logs["TRAVEL_MILES"] = travel_series
    result = logs[["TEAM_ID", "GAME_ID", "TRAVEL_MILES"]].copy()
    result["GAME_ID"] = result["GAME_ID"].astype("int64")
    return result


# ── Improvement 3: Strength of schedule ───────────────────────────────────────

def build_sos_features(matchups: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, compute the average opponent NET_RATING over the last 10 games
    (shift by 1 to avoid leakage). Higher = tougher schedule.
    """
    # Map team_id -> net_rating per season
    net_map = team_stats.set_index(["TEAM_ID", "SEASON"])["NET_RATING"].to_dict()

    matchups = matchups.copy().sort_values("GAME_DATE").reset_index(drop=True)

    def get_net(team_id, season):
        return net_map.get((team_id, season), 0.0)

    # Build a per-team list of opponent net ratings in game order
    home_opp_net = matchups.apply(lambda r: get_net(r["AWAY_TEAM_ID"], r["SEASON"]), axis=1)
    away_opp_net = matchups.apply(lambda r: get_net(r["HOME_TEAM_ID"], r["SEASON"]), axis=1)

    # For each team, we need their last-10-game opponent net rating
    # Build a long format: (GAME_ID, TEAM_ID, OPP_NET)
    home_rows = matchups[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID"]].copy()
    home_rows["TEAM_ID"] = home_rows["HOME_TEAM_ID"]
    home_rows["OPP_NET"] = home_opp_net.values

    away_rows = matchups[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID"]].copy()
    away_rows["TEAM_ID"] = away_rows["AWAY_TEAM_ID"]
    away_rows["OPP_NET"] = away_opp_net.values

    long = pd.concat([
        home_rows[["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_NET"]],
        away_rows[["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_NET"]],
    ]).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    long["SOS_LAST10"] = (
        long.groupby("TEAM_ID")["OPP_NET"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.0)
    )

    sos = long[["GAME_ID", "TEAM_ID", "SOS_LAST10"]]
    home_sos = sos.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "SOS_LAST10": "HOME_SOS"})
    away_sos = sos.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "SOS_LAST10": "AWAY_SOS"})

    result = matchups[["GAME_ID"]].merge(
        home_sos[home_sos["HOME_TEAM_ID"].isin(matchups["HOME_TEAM_ID"])],
        on="GAME_ID", how="left"
    )
    # Simpler: just merge on GAME_ID + team
    home_sos2 = long[["GAME_ID", "TEAM_ID", "SOS_LAST10"]].merge(
        matchups[["GAME_ID", "HOME_TEAM_ID"]], on="GAME_ID"
    )
    home_sos2 = home_sos2[home_sos2["TEAM_ID"] == home_sos2["HOME_TEAM_ID"]][["GAME_ID", "SOS_LAST10"]].rename(
        columns={"SOS_LAST10": "HOME_SOS"}
    )
    away_sos2 = long[["GAME_ID", "TEAM_ID", "SOS_LAST10"]].merge(
        matchups[["GAME_ID", "AWAY_TEAM_ID"]], on="GAME_ID"
    )
    away_sos2 = away_sos2[away_sos2["TEAM_ID"] == away_sos2["AWAY_TEAM_ID"]][["GAME_ID", "SOS_LAST10"]].rename(
        columns={"SOS_LAST10": "AWAY_SOS"}
    )

    sos_df = matchups[["GAME_ID"]].merge(home_sos2, on="GAME_ID", how="left") \
                                   .merge(away_sos2, on="GAME_ID", how="left")
    sos_df["DIFF_SOS"] = sos_df["HOME_SOS"] - sos_df["AWAY_SOS"]
    return sos_df.fillna(0.0)


# ── retrain with new features ─────────────────────────────────────────────────

FEATURE_COLS_V2 = [
    "DIFF_OFF_RATING", "DIFF_DEF_RATING", "DIFF_NET_RATING", "DIFF_PACE",
    "DIFF_W_PCT", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_AST", "DIFF_TOV",
    "HOME_ROLL10_WIN_PCT", "AWAY_ROLL10_WIN_PCT", "DIFF_ROLL10_WIN_PCT",
    "HOME_ROLL10_PM", "AWAY_ROLL10_PM", "DIFF_ROLL10_PM",
    "HOME_REST", "AWAY_REST", "DIFF_REST",
    "HOME_B2B", "AWAY_B2B",
    "H2H_HOME_WIN_PCT",
    # New features
    "HOME_STAR_AVAIL", "AWAY_STAR_AVAIL", "DIFF_STAR_AVAIL",
    "HOME_TRAVEL", "AWAY_TRAVEL", "DIFF_TRAVEL",
    "HOME_SOS", "AWAY_SOS", "DIFF_SOS",
]

TEST_SEASON = "2024-25"


def retrain(features_v2: pd.DataFrame):
    train = features_v2[features_v2["SEASON"] != TEST_SEASON]
    test  = features_v2[features_v2["SEASON"] == TEST_SEASON]

    X_train = train[FEATURE_COLS_V2].values
    X_test  = test[FEATURE_COLS_V2].values
    y_train = train["HOME_TEAM_WIN"].values
    y_test  = test["HOME_TEAM_WIN"].values
    y_train_reg = train["POINT_MARGIN"].values
    y_test_reg  = test["POINT_MARGIN"].values

    print(f"\n  Train: {len(train)} | Test: {len(test)}")
    print("  Classifier results:")

    lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=0.5))])
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    ensemble = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft")

    for name, model in [("LogisticRegression", lr), ("RandomForest", rf),
                        ("GradientBoosting", gb), ("Ensemble", ensemble)]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        ll  = log_loss(y_test, probs)
        print(f"    {name:20s}  acc={acc:.4f}  log_loss={ll:.4f}")
        fname = "ensemble" if name == "Ensemble" else name.lower()
        joblib.dump(model, MODELS / f"{fname}_v2.pkl")

    # Margin regressor
    reg = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    reg.fit(X_train, y_train_reg)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_reg, reg.predict(X_test))
    print(f"    MarginRegressor (Ridge)       MAE={mae:.2f} pts")
    joblib.dump(reg, MODELS / "margin_regressor_v2.pkl")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    seasons = ["2022-23", "2023-24", "2024-25"]

    print("Loading base features and raw data...")
    features   = pd.read_csv(PROCESSED / "features.csv", parse_dates=["GAME_DATE"])
    game_logs  = pd.read_csv(RAW / "game_logs.csv", parse_dates=["GAME_DATE"])
    matchups   = pd.read_csv(RAW / "matchups.csv",  parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(RAW / "team_stats.csv")

    # Bridge team IDs back into features (needed for improvement merges)
    # Ensure GAME_ID is int64 consistently across all dataframes
    features["GAME_ID"]  = features["GAME_ID"].astype("int64")
    matchups["GAME_ID"]  = matchups["GAME_ID"].astype("int64")
    game_logs["GAME_ID"] = game_logs["GAME_ID"].astype("int64")
    id_bridge = matchups[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]]
    features = features.merge(id_bridge, on="GAME_ID", how="left")

    # ── Improvement 1: Star availability ─────────────────────────────────────
    print("\n=== Improvement 1: Star Player Availability ===")
    star_avail = build_star_availability(seasons)
    home_avail = star_avail.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "STAR_AVAILABILITY": "HOME_STAR_AVAIL"})
    away_avail = star_avail.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "STAR_AVAILABILITY": "AWAY_STAR_AVAIL"})
    features = features.merge(home_avail, on=["HOME_TEAM_ID", "GAME_ID"], how="left")
    features = features.merge(away_avail, on=["AWAY_TEAM_ID", "GAME_ID"], how="left")
    features["HOME_STAR_AVAIL"] = features["HOME_STAR_AVAIL"].fillna(1.0)
    features["AWAY_STAR_AVAIL"] = features["AWAY_STAR_AVAIL"].fillna(1.0)
    features["DIFF_STAR_AVAIL"] = features["HOME_STAR_AVAIL"] - features["AWAY_STAR_AVAIL"]
    print(f"  Star availability range: {features['HOME_STAR_AVAIL'].min():.2f} - {features['HOME_STAR_AVAIL'].max():.2f}")

    # ── Improvement 2: Travel distance ───────────────────────────────────────
    print("\n=== Improvement 2: Travel Distance ===")
    travel = build_travel_features(game_logs)
    home_travel = travel.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "TRAVEL_MILES": "HOME_TRAVEL"})
    away_travel = travel.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "TRAVEL_MILES": "AWAY_TRAVEL"})
    features = features.merge(home_travel, on=["HOME_TEAM_ID", "GAME_ID"], how="left")
    features = features.merge(away_travel, on=["AWAY_TEAM_ID", "GAME_ID"], how="left")
    features["HOME_TRAVEL"] = features["HOME_TRAVEL"].fillna(0.0)
    features["AWAY_TRAVEL"] = features["AWAY_TRAVEL"].fillna(0.0)
    features["DIFF_TRAVEL"]  = features["HOME_TRAVEL"] - features["AWAY_TRAVEL"]
    print(f"  Avg away team travel: {features['AWAY_TRAVEL'].mean():.0f} miles")
    print(f"  Avg home team travel: {features['HOME_TRAVEL'].mean():.0f} miles")

    # ── Improvement 3: Strength of schedule ──────────────────────────────────
    print("\n=== Improvement 3: Strength of Schedule ===")
    sos = build_sos_features(matchups, team_stats)
    features = features.merge(sos, on="GAME_ID", how="left")
    features[["HOME_SOS", "AWAY_SOS", "DIFF_SOS"]] = features[["HOME_SOS", "AWAY_SOS", "DIFF_SOS"]].fillna(0.0)
    print(f"  SOS range: {features['HOME_SOS'].min():.2f} to {features['HOME_SOS'].max():.2f}")

    # Drop bridge columns before saving
    features = features.drop(columns=["HOME_TEAM_ID", "AWAY_TEAM_ID"])

    # Save v2 features
    out = PROCESSED / "features_v2.csv"
    features.to_csv(out, index=False)
    print(f"\nSaved {len(features)} rows, {len(features.columns)} cols -> data/processed/features_v2.csv")

    # ── Retrain ───────────────────────────────────────────────────────────────
    print("\n=== Retraining models with new features ===")
    retrain(features)

    print("\nDone. New models saved as *_v2.pkl in models/")


if __name__ == "__main__":
    main()
