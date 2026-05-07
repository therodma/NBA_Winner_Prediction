"""
engineer_features.py — Step 2: Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_data():
    matchups   = pd.read_csv(RAW / "matchups.csv",            parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(RAW / "team_stats.csv")
    hcs        = pd.read_csv(RAW / "home_court_strength.csv")
    return matchups, team_stats, hcs


def join_team_stats(matchups, team_stats, hcs):
    home_stats = team_stats.add_prefix("H_").rename(columns={"H_TEAM_ID": "HOME_TEAM_ID", "H_SEASON": "SEASON"})
    away_stats = team_stats.add_prefix("A_").rename(columns={"A_TEAM_ID": "AWAY_TEAM_ID", "A_SEASON": "SEASON"})
    home_hcs   = hcs.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "SEASON": "SEASON",
                                      "HOME_COURT_STRENGTH": "HOME_COURT_STRENGTH"})
    away_hcs   = hcs.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "SEASON": "SEASON",
                                      "HOME_COURT_STRENGTH": "AWAY_COURT_STRENGTH"})

    df = matchups.merge(home_stats, on=["HOME_TEAM_ID", "SEASON"], how="left")
    df = df.merge(away_stats,       on=["AWAY_TEAM_ID", "SEASON"], how="left")
    df = df.merge(home_hcs[["HOME_TEAM_ID", "SEASON", "HOME_COURT_STRENGTH"]],
                  on=["HOME_TEAM_ID", "SEASON"], how="left")
    df = df.merge(away_hcs[["AWAY_TEAM_ID", "SEASON", "AWAY_COURT_STRENGTH"]],
                  on=["AWAY_TEAM_ID", "SEASON"], how="left")
    return df


def build_features(df):
    for col, fill in [
        ("HOME_ROLL10_WIN_PCT", 0.5), ("AWAY_ROLL10_WIN_PCT", 0.5),
        ("HOME_ROLL10_PM", 0.0),      ("AWAY_ROLL10_PM", 0.0),
        ("HOME_ROLL20_WIN_PCT", 0.5), ("AWAY_ROLL20_WIN_PCT", 0.5),
        ("HOME_ROLL20_PM", 0.0),      ("AWAY_ROLL20_PM", 0.0),
        ("HOME_STREAK", 0),           ("AWAY_STREAK", 0),
        ("HOME_COURT_STRENGTH", 0.0), ("AWAY_COURT_STRENGTH", 0.0),
    ]:
        df[col] = df[col].fillna(fill)

    f = pd.DataFrame()
    f["GAME_ID"]   = df["GAME_ID"]
    f["GAME_DATE"] = df["GAME_DATE"]
    f["SEASON"]    = df["SEASON"]
    f["HOME_TEAM"] = df["HOME_TEAM"]
    f["AWAY_TEAM"] = df["AWAY_TEAM"]

    # season-level diffs
    for stat in ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
                 "W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT",
                 "REB", "AST", "TOV", "STL", "BLK",
                 "OREB", "DREB", "TS_PCT", "AST_PCT", "REB_PCT", "TM_TOV_PCT"]:
        h, a = f"H_{stat}", f"A_{stat}"
        if h in df.columns and a in df.columns:
            f[f"DIFF_{stat}"] = df[h] - df[a]

    # rolling form
    f["HOME_ROLL10_WIN_PCT"]  = df["HOME_ROLL10_WIN_PCT"]
    f["AWAY_ROLL10_WIN_PCT"]  = df["AWAY_ROLL10_WIN_PCT"]
    f["DIFF_ROLL10_WIN_PCT"]  = df["HOME_ROLL10_WIN_PCT"] - df["AWAY_ROLL10_WIN_PCT"]
    f["HOME_ROLL10_PM"]       = df["HOME_ROLL10_PM"]
    f["AWAY_ROLL10_PM"]       = df["AWAY_ROLL10_PM"]
    f["DIFF_ROLL10_PM"]       = df["HOME_ROLL10_PM"] - df["AWAY_ROLL10_PM"]
    f["HOME_ROLL20_WIN_PCT"]  = df["HOME_ROLL20_WIN_PCT"]
    f["AWAY_ROLL20_WIN_PCT"]  = df["AWAY_ROLL20_WIN_PCT"]
    f["DIFF_ROLL20_WIN_PCT"]  = df["HOME_ROLL20_WIN_PCT"] - df["AWAY_ROLL20_WIN_PCT"]
    f["HOME_ROLL20_PM"]       = df["HOME_ROLL20_PM"]
    f["AWAY_ROLL20_PM"]       = df["AWAY_ROLL20_PM"]
    f["DIFF_ROLL20_PM"]       = df["HOME_ROLL20_PM"] - df["AWAY_ROLL20_PM"]
    f["HOME_STREAK"]          = df["HOME_STREAK"]
    f["AWAY_STREAK"]          = df["AWAY_STREAK"]
    f["DIFF_STREAK"]          = df["HOME_STREAK"] - df["AWAY_STREAK"]

    # rest / fatigue
    f["HOME_REST"]  = df["HOME_REST"]
    f["AWAY_REST"]  = df["AWAY_REST"]
    f["DIFF_REST"]  = df["HOME_REST"] - df["AWAY_REST"]
    f["HOME_B2B"]   = df["HOME_B2B"]
    f["AWAY_B2B"]   = df["AWAY_B2B"]

    # home court strength
    f["HOME_COURT_STRENGTH"]  = df["HOME_COURT_STRENGTH"]
    f["AWAY_COURT_STRENGTH"]  = df["AWAY_COURT_STRENGTH"]
    f["DIFF_COURT_STRENGTH"]  = df["HOME_COURT_STRENGTH"] - df["AWAY_COURT_STRENGTH"]

    # travel timezone disadvantage for away team
    f["AWAY_TRAVEL_TZ_DIFF"]  = df["AWAY_TRAVEL_TZ_DIFF"].fillna(0)

    # H2H
    f["H2H_HOME_WIN_PCT"] = df["H2H_HOME_WIN_PCT"]

    # targets
    f["HOME_TEAM_WIN"] = df["HOME_TEAM_WIN"]
    f["POINT_MARGIN"]  = df["POINT_MARGIN"]

    return f.sort_values("GAME_DATE").reset_index(drop=True)


def main():
    print("Loading raw data...")
    matchups, team_stats, hcs = load_data()
    print(f"  {len(matchups)} matchups")

    print("Joining stats...")
    df = join_team_stats(matchups, team_stats, hcs)

    print("Building features...")
    features = build_features(df)
    features.to_csv(PROCESSED / "features.csv", index=False)

    model_cols = [c for c in features.columns if c not in
                  ("GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM",
                   "HOME_TEAM_WIN", "POINT_MARGIN")]
    print(f"  {len(features)} rows, {len(model_cols)} feature columns -> features.csv")

    nulls = features[model_cols].isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("  Nulls:", nulls.to_dict())
    else:
        print("  No nulls.")
    print("Done.")


if __name__ == "__main__":
    main()
