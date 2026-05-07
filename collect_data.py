"""
collect_data.py — Step 1: Data Collection
Pulls 10 seasons of NBA data including game logs, team stats, home court strength,
and travel/timezone features.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from nba_api.stats.endpoints import LeagueGameLog, LeagueDashTeamStats

RAW   = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# 10 seasons — enough data without overfitting to old eras
SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]
SLEEP = 0.7

# ── timezone map (arena city -> UTC offset, standard time) ───────────────────
TEAM_TIMEZONE = {
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6,
    "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5, "GSW": -8,
    "HOU": -6, "IND": -5, "LAC": -8, "LAL": -8, "MEM": -6,
    "MIA": -5, "MIL": -6, "MIN": -6, "NOP": -6, "NYK": -5,
    "OKC": -6, "ORL": -5, "PHI": -5, "PHX": -7, "POR": -8,
    "SAC": -8, "SAS": -6, "TOR": -5, "UTA": -7, "WAS": -5,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_game_logs(season):
    time.sleep(SLEEP)
    df = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        league_id="00",
    ).get_data_frames()[0]
    df["SEASON"] = season
    return df

def fetch_team_stats(season, measure="Base"):
    time.sleep(SLEEP)
    df = LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
        timeout=30,
    ).get_data_frames()[0]
    df["SEASON"] = season
    return df

# ── game logs ─────────────────────────────────────────────────────────────────

def build_game_logs():
    frames = []
    for s in SEASONS:
        print(f"  Fetching game log: {s}")
        frames.append(fetch_game_logs(s))
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(["TEAM_ID", "GAME_DATE"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_rest_days(logs):
    logs = logs.copy()
    logs["REST_DAYS"] = (
        logs.groupby("TEAM_ID")["GAME_DATE"]
        .diff().dt.days.fillna(7).astype(int)
    )
    logs["IS_B2B"] = (logs["REST_DAYS"] == 1).astype(int)
    return logs

def add_rolling_form(logs):
    logs = logs.copy()
    logs["WIN"]        = (logs["WL"] == "W").astype(float)
    logs["PLUS_MINUS"] = logs["PLUS_MINUS"].astype(float)
    grp = logs.groupby("TEAM_ID")

    logs["ROLL10_WIN_PCT"]    = grp["WIN"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    logs["ROLL10_PLUS_MINUS"] = grp["PLUS_MINUS"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    logs["ROLL20_WIN_PCT"]    = grp["WIN"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    logs["ROLL20_PLUS_MINUS"] = grp["PLUS_MINUS"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())

    def calc_streak(series):
        result, vals = [0] * len(series), series.tolist()
        for i in range(1, len(vals)):
            prev = result[i-1]
            if (prev >= 0 and vals[i-1] == 1.0) or (prev > 0 and vals[i-1] == 1.0):
                result[i] = max(1, prev + 1) if vals[i-1] == 1.0 else min(-1, prev - 1)
            else:
                result[i] = 1 if vals[i-1] == 1.0 else -1
        return result

    logs["STREAK"] = grp["WIN"].transform(calc_streak)
    return logs

# ── home court strength ───────────────────────────────────────────────────────

def build_home_court_strength(logs):
    """
    Per team per season: home win% minus away win%.
    Positive = strong home court, negative = better on road.
    """
    home = logs[logs["MATCHUP"].str.contains(r"vs\.")].copy()
    away = logs[logs["MATCHUP"].str.contains("@")].copy()

    home_wl = home.groupby(["TEAM_ID", "SEASON"])["WIN"].mean().reset_index()
    home_wl.columns = ["TEAM_ID", "SEASON", "HOME_WIN_PCT"]

    away_wl = away.groupby(["TEAM_ID", "SEASON"])["WIN"].mean().reset_index()
    away_wl.columns = ["TEAM_ID", "SEASON", "AWAY_WIN_PCT"]

    hcs = home_wl.merge(away_wl, on=["TEAM_ID", "SEASON"], how="outer").fillna(0.5)
    hcs["HOME_COURT_STRENGTH"] = hcs["HOME_WIN_PCT"] - hcs["AWAY_WIN_PCT"]
    return hcs[["TEAM_ID", "SEASON", "HOME_COURT_STRENGTH"]]

# ── matchup table ─────────────────────────────────────────────────────────────

def build_matchup_table(logs):
    home = logs[logs["MATCHUP"].str.contains(r"vs\.")].copy()
    away = logs[logs["MATCHUP"].str.contains("@")].copy()

    home = home.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM",
        "PTS": "HOME_PTS", "WIN": "HOME_WIN",
        "REST_DAYS": "HOME_REST", "IS_B2B": "HOME_B2B",
        "ROLL10_WIN_PCT": "HOME_ROLL10_WIN_PCT", "ROLL10_PLUS_MINUS": "HOME_ROLL10_PM",
        "ROLL20_WIN_PCT": "HOME_ROLL20_WIN_PCT", "ROLL20_PLUS_MINUS": "HOME_ROLL20_PM",
        "STREAK": "HOME_STREAK",
    })
    away = away.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID", "TEAM_ABBREVIATION": "AWAY_TEAM",
        "PTS": "AWAY_PTS", "WIN": "AWAY_WIN",
        "REST_DAYS": "AWAY_REST", "IS_B2B": "AWAY_B2B",
        "ROLL10_WIN_PCT": "AWAY_ROLL10_WIN_PCT", "ROLL10_PLUS_MINUS": "AWAY_ROLL10_PM",
        "ROLL20_WIN_PCT": "AWAY_ROLL20_WIN_PCT", "ROLL20_PLUS_MINUS": "AWAY_ROLL20_PM",
        "STREAK": "AWAY_STREAK",
    })

    home_cols = ["GAME_ID", "GAME_DATE", "SEASON",
                 "HOME_TEAM_ID", "HOME_TEAM", "HOME_PTS", "HOME_WIN",
                 "HOME_REST", "HOME_B2B",
                 "HOME_ROLL10_WIN_PCT", "HOME_ROLL10_PM",
                 "HOME_ROLL20_WIN_PCT", "HOME_ROLL20_PM", "HOME_STREAK"]
    away_cols = ["GAME_ID",
                 "AWAY_TEAM_ID", "AWAY_TEAM", "AWAY_PTS", "AWAY_WIN",
                 "AWAY_REST", "AWAY_B2B",
                 "AWAY_ROLL10_WIN_PCT", "AWAY_ROLL10_PM",
                 "AWAY_ROLL20_WIN_PCT", "AWAY_ROLL20_PM", "AWAY_STREAK"]

    matchups = home[home_cols].merge(away[away_cols], on="GAME_ID")
    matchups["POINT_MARGIN"]  = matchups["HOME_PTS"] - matchups["AWAY_PTS"]
    matchups["HOME_TEAM_WIN"] = (matchups["HOME_WIN"] == 1.0).astype(int)

    # timezone travel disadvantage: away team timezone difference
    matchups["HOME_TZ"] = matchups["HOME_TEAM"].map(TEAM_TIMEZONE).fillna(-6)
    matchups["AWAY_TZ"] = matchups["AWAY_TEAM"].map(TEAM_TIMEZONE).fillna(-6)
    matchups["AWAY_TRAVEL_TZ_DIFF"] = (matchups["HOME_TZ"] - matchups["AWAY_TZ"]).abs()

    return matchups.sort_values("GAME_DATE").reset_index(drop=True)

# ── H2H ───────────────────────────────────────────────────────────────────────

def build_h2h(matchups):
    matchups = matchups.copy().sort_values("GAME_DATE").reset_index(drop=True)
    h2h_win_pct = []
    for i, row in matchups.iterrows():
        prior = matchups.iloc[:i]
        pair = prior[
            ((prior["HOME_TEAM_ID"] == row["HOME_TEAM_ID"]) & (prior["AWAY_TEAM_ID"] == row["AWAY_TEAM_ID"])) |
            ((prior["HOME_TEAM_ID"] == row["AWAY_TEAM_ID"]) & (prior["AWAY_TEAM_ID"] == row["HOME_TEAM_ID"]))
        ].tail(20)
        if len(pair) == 0:
            h2h_win_pct.append(0.5)
            continue
        home_wins = (
            ((pair["HOME_TEAM_ID"] == row["HOME_TEAM_ID"]) & (pair["HOME_TEAM_WIN"] == 1)) |
            ((pair["AWAY_TEAM_ID"] == row["HOME_TEAM_ID"]) & (pair["HOME_TEAM_WIN"] == 0))
        ).sum()
        h2h_win_pct.append(home_wins / len(pair))
    matchups["H2H_HOME_WIN_PCT"] = h2h_win_pct
    return matchups

# ── team season stats ─────────────────────────────────────────────────────────

def build_team_stats():
    base_frames, adv_frames = [], []
    for s in SEASONS:
        print(f"  Fetching base stats: {s}")
        base_frames.append(fetch_team_stats(s, "Base"))
        print(f"  Fetching advanced stats: {s}")
        adv_frames.append(fetch_team_stats(s, "Advanced"))

    base = pd.concat(base_frames, ignore_index=True)
    adv  = pd.concat(adv_frames,  ignore_index=True)

    keep_base = ["TEAM_ID", "SEASON", "W_PCT", "PTS", "REB", "AST", "TOV",
                 "FG_PCT", "FG3_PCT", "FT_PCT", "STL", "BLK",
                 "OREB", "DREB", "PF"]
    keep_adv  = ["TEAM_ID", "SEASON", "OFF_RATING", "DEF_RATING", "NET_RATING",
                 "PACE", "TS_PCT", "AST_PCT", "REB_PCT", "TM_TOV_PCT"]

    merged = base[keep_base].merge(adv[keep_adv], on=["TEAM_ID", "SEASON"])
    return merged

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Collecting game logs ===")
    logs = build_game_logs()
    logs = add_rest_days(logs)
    logs = add_rolling_form(logs)
    logs.to_csv(RAW / "game_logs.csv", index=False)
    print(f"  Saved {len(logs)} rows -> game_logs.csv")

    print("=== Building home court strength ===")
    hcs = build_home_court_strength(logs)
    hcs.to_csv(RAW / "home_court_strength.csv", index=False)
    print(f"  Saved {len(hcs)} rows -> home_court_strength.csv")

    print("=== Building matchup table ===")
    matchups = build_matchup_table(logs)
    print("=== Computing H2H records (may take a few minutes) ===")
    matchups = build_h2h(matchups)
    matchups.to_csv(RAW / "matchups.csv", index=False)
    print(f"  Saved {len(matchups)} matchups -> matchups.csv")

    print("=== Collecting team season stats ===")
    team_stats = build_team_stats()
    team_stats.to_csv(RAW / "team_stats.csv", index=False)
    print(f"  Saved {len(team_stats)} rows -> team_stats.csv")

    print("\n✓ Data collection complete.")

if __name__ == "__main__":
    main()
