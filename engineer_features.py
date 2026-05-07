"""
engineer_features.py — Step 2: Feature Engineering
Joins matchups with season-level team stats and produces the final feature matrix.
Output: data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_data():
    matchups = pd.read_csv(RAW / "matchups.csv", parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(RAW / "team_stats.csv")
    return matchups, team_stats


def join_team_stats(matchups: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Attach season-level stats for home and away teams."""
    home_stats = team_stats.add_prefix("H_").rename(columns={"H_TEAM_ID": "HOME_TEAM_ID", "H_SEASON": "SEASON"})
    away_stats = team_stats.add_prefix("A_").rename(columns={"A_TEAM_ID": "AWAY_TEAM_ID", "A_SEASON": "SEASON"})

    df = matchups.merge(home_stats, on=["HOME_TEAM_ID", "SEASON"], how="left")
    df = df.merge(away_stats, on=["AWAY_TEAM_ID", "SEASON"], how="left")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all difference and derived features."""
    # Fill nulls in rolling columns (first games of season have no prior history)
    df["HOME_ROLL10_WIN_PCT"] = df["HOME_ROLL10_WIN_PCT"].fillna(0.5)
    df["AWAY_ROLL10_WIN_PCT"] = df["AWAY_ROLL10_WIN_PCT"].fillna(0.5)
    df["HOME_ROLL10_PM"] = df["HOME_ROLL10_PM"].fillna(0.0)
    df["AWAY_ROLL10_PM"] = df["AWAY_ROLL10_PM"].fillna(0.0)

    features = pd.DataFrame()

    # Identifiers (not used as model inputs)
    features["GAME_ID"]    = df["GAME_ID"]
    features["GAME_DATE"]  = df["GAME_DATE"]
    features["SEASON"]     = df["SEASON"]
    features["HOME_TEAM"]  = df["HOME_TEAM"]
    features["AWAY_TEAM"]  = df["AWAY_TEAM"]

    # ── Season-level difference features (home minus away) ──────────────────
    features["DIFF_OFF_RATING"]  = df["H_OFF_RATING"]  - df["A_OFF_RATING"]
    features["DIFF_DEF_RATING"]  = df["H_DEF_RATING"]  - df["A_DEF_RATING"]
    features["DIFF_NET_RATING"]  = df["H_NET_RATING"]  - df["A_NET_RATING"]
    features["DIFF_PACE"]        = df["H_PACE"]        - df["A_PACE"]
    features["DIFF_W_PCT"]       = df["H_W_PCT"]       - df["A_W_PCT"]
    features["DIFF_FG_PCT"]      = df["H_FG_PCT"]      - df["A_FG_PCT"]
    features["DIFF_FG3_PCT"]     = df["H_FG3_PCT"]     - df["A_FG3_PCT"]
    features["DIFF_FT_PCT"]      = df["H_FT_PCT"]      - df["A_FT_PCT"]
    features["DIFF_REB"]         = df["H_REB"]         - df["A_REB"]
    features["DIFF_AST"]         = df["H_AST"]         - df["A_AST"]
    features["DIFF_TOV"]         = df["H_TOV"]         - df["A_TOV"]  # positive = home turns it over more

    # ── Rolling form features ────────────────────────────────────────────────
    features["HOME_ROLL10_WIN_PCT"]  = df["HOME_ROLL10_WIN_PCT"]
    features["AWAY_ROLL10_WIN_PCT"]  = df["AWAY_ROLL10_WIN_PCT"]
    features["DIFF_ROLL10_WIN_PCT"]  = df["HOME_ROLL10_WIN_PCT"] - df["AWAY_ROLL10_WIN_PCT"]
    features["HOME_ROLL10_PM"]       = df["HOME_ROLL10_PM"]
    features["AWAY_ROLL10_PM"]       = df["AWAY_ROLL10_PM"]
    features["DIFF_ROLL10_PM"]       = df["HOME_ROLL10_PM"] - df["AWAY_ROLL10_PM"]

    features["HOME_ROLL20_WIN_PCT"]  = df["HOME_ROLL20_WIN_PCT"].fillna(0.5)
    features["AWAY_ROLL20_WIN_PCT"]  = df["AWAY_ROLL20_WIN_PCT"].fillna(0.5)
    features["DIFF_ROLL20_WIN_PCT"]  = features["HOME_ROLL20_WIN_PCT"] - features["AWAY_ROLL20_WIN_PCT"]
    features["HOME_ROLL20_PM"]       = df["HOME_ROLL20_PM"].fillna(0.0)
    features["AWAY_ROLL20_PM"]       = df["AWAY_ROLL20_PM"].fillna(0.0)
    features["DIFF_ROLL20_PM"]       = features["HOME_ROLL20_PM"] - features["AWAY_ROLL20_PM"]

    features["HOME_STREAK"]          = df["HOME_STREAK"].fillna(0)
    features["AWAY_STREAK"]          = df["AWAY_STREAK"].fillna(0)
    features["DIFF_STREAK"]          = features["HOME_STREAK"] - features["AWAY_STREAK"]

    # ── Rest / fatigue features ──────────────────────────────────────────────
    features["HOME_REST"]  = df["HOME_REST"]
    features["AWAY_REST"]  = df["AWAY_REST"]
    features["DIFF_REST"]  = df["HOME_REST"] - df["AWAY_REST"]
    features["HOME_B2B"]   = df["HOME_B2B"]
    features["AWAY_B2B"]   = df["AWAY_B2B"]

    # ── H2H ─────────────────────────────────────────────────────────────────
    features["H2H_HOME_WIN_PCT"] = df["H2H_HOME_WIN_PCT"]

    # ── Targets ─────────────────────────────────────────────────────────────
    features["HOME_TEAM_WIN"] = df["HOME_TEAM_WIN"]
    features["POINT_MARGIN"]  = df["POINT_MARGIN"]

    return features.sort_values("GAME_DATE").reset_index(drop=True)


def main():
    print("Loading raw data...")
    matchups, team_stats = load_data()
    print(f"  {len(matchups)} matchups, {len(team_stats)} team-season rows")

    print("Joining team stats...")
    df = join_team_stats(matchups, team_stats)

    missing = df[["H_OFF_RATING", "A_OFF_RATING"]].isnull().sum()
    if missing.any():
        print(f"  WARNING: {missing.to_dict()} nulls after join — check TEAM_ID/SEASON alignment")

    print("Building features...")
    features = build_features(df)

    out_path = PROCESSED / "features.csv"
    features.to_csv(out_path, index=False)

    print(f"  {len(features)} rows, {len(features.columns)} columns -> data/processed/features.csv")
    print()

    # Quick sanity check
    model_cols = [c for c in features.columns if c not in
                  ("GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM",
                   "HOME_TEAM_WIN", "POINT_MARGIN")]
    print(f"Feature columns ({len(model_cols)}):")
    for c in model_cols:
        print(f"  {c:30s}  mean={features[c].mean():.3f}  std={features[c].std():.3f}")

    print()
    null_counts = features.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print("Remaining nulls:", null_counts.to_dict())
    else:
        print("No nulls in feature matrix.")

    print("\nDone.")


if __name__ == "__main__":
    main()
