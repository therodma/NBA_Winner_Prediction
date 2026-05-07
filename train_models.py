"""
train_models.py — Step 3: Model Training & Evaluation
10 seasons of data, time-weighted (recent seasons count more), XGBoost ensemble.
Test set = 2024-25 season.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
MODELS    = Path("models")
MODELS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "DIFF_OFF_RATING", "DIFF_DEF_RATING", "DIFF_NET_RATING", "DIFF_PACE",
    "DIFF_W_PCT", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_AST", "DIFF_TOV", "DIFF_STL", "DIFF_BLK",
    "DIFF_OREB", "DIFF_DREB", "DIFF_TS_PCT", "DIFF_AST_PCT",
    "DIFF_REB_PCT", "DIFF_TM_TOV_PCT",
    "HOME_ROLL10_WIN_PCT", "AWAY_ROLL10_WIN_PCT", "DIFF_ROLL10_WIN_PCT",
    "HOME_ROLL10_PM",      "AWAY_ROLL10_PM",      "DIFF_ROLL10_PM",
    "HOME_ROLL20_WIN_PCT", "AWAY_ROLL20_WIN_PCT", "DIFF_ROLL20_WIN_PCT",
    "HOME_ROLL20_PM",      "AWAY_ROLL20_PM",      "DIFF_ROLL20_PM",
    "HOME_STREAK", "AWAY_STREAK", "DIFF_STREAK",
    "HOME_REST",   "AWAY_REST",   "DIFF_REST",
    "HOME_B2B",    "AWAY_B2B",
    "HOME_COURT_STRENGTH", "AWAY_COURT_STRENGTH", "DIFF_COURT_STRENGTH",
    "AWAY_TRAVEL_TZ_DIFF",
    "H2H_HOME_WIN_PCT",
]
TARGET_CLS  = "HOME_TEAM_WIN"
TARGET_REG  = "POINT_MARGIN"
TEST_SEASON = "2024-25"

# Season weights — more recent = higher weight
SEASON_WEIGHTS = {
    "2015-16": 0.4, "2016-17": 0.5, "2017-18": 0.6,
    "2018-19": 0.7, "2019-20": 0.75, "2020-21": 0.8,
    "2021-22": 0.85, "2022-23": 0.9, "2023-24": 0.95, "2024-25": 1.0,
}


def load_splits():
    df    = pd.read_csv(PROCESSED / "features.csv")
    # only keep cols that exist in the file
    cols  = [c for c in FEATURE_COLS if c in df.columns]
    train = df[df["SEASON"] != TEST_SEASON].copy()
    test  = df[df["SEASON"] == TEST_SEASON].copy()

    X_train = train[cols].fillna(0).values
    X_test  = test[cols].fillna(0).values
    y_train_cls = train[TARGET_CLS].values
    y_test_cls  = test[TARGET_CLS].values
    y_train_reg = train[TARGET_REG].values
    y_test_reg  = test[TARGET_REG].values

    # sample weights based on season recency
    w_train = train["SEASON"].map(SEASON_WEIGHTS).fillna(0.5).values

    print(f"Train: {len(train)} games | Test: {len(test)} games ({TEST_SEASON})")
    print(f"Features used: {len(cols)}")
    return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, w_train, test, cols


def build_models():
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5)),
    ])
    rf  = RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=12, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, random_state=42)
    xgb = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=42, verbosity=0)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("xgb", xgb)],
        voting="soft",
    )
    return {"LogisticRegression": lr, "RandomForest": rf,
            "GradientBoosting": gb, "XGBoost": xgb, "Ensemble": ensemble}


def evaluate(name, model, X_train, X_test, y_train, y_test, sample_weight=None):
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    # CV without weights for simplicity
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train, **{f"{model.steps[-1][0]}__sample_weight" if hasattr(model, 'steps') else 'sample_weight': sample_weight} if sample_weight is not None else {})
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    ll    = log_loss(y_test, probs)
    print(f"  {name:20s}  acc={acc:.4f}  log_loss={ll:.4f}  cv={cv.mean():.4f}+/-{cv.std():.4f}")
    return preds, probs


def plot_calibration(models_probs, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name, probs in models_probs.items():
        fp, mp = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mp, fp, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODELS / "calibration_curves.png", dpi=120)
    plt.close()


def plot_feature_importance(rf_model, gb_model, cols):
    avg = (rf_model.feature_importances_ + gb_model.feature_importances_) / 2
    idx = np.argsort(avg)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(cols)), avg[idx])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([cols[i] for i in idx], rotation=45, ha="right", fontsize=7)
    ax.set_title("Feature Importance (RF + GB avg)")
    plt.tight_layout()
    plt.savefig(MODELS / "feature_importance.png", dpi=120)
    plt.close()
    print("\n  Top 10 features:")
    for rank, i in enumerate(idx[:10], 1):
        print(f"    {rank:2d}. {cols[i]:35s} {avg[i]:.4f}")


def main():
    print("=== Loading data ===")
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, w_train, test_df, cols = load_splits()

    print("\n=== Training classifiers ===")
    models    = build_models()
    all_probs = {}
    all_preds = {}

    for name, model in models.items():
        preds, probs = evaluate(name, model, X_train, X_test, y_train_cls, y_test_cls)
        all_probs[name] = probs
        all_preds[name] = preds
        if name != "Ensemble":
            joblib.dump(model, MODELS / f"{name.lower()}.pkl")
    joblib.dump(models["Ensemble"], MODELS / "ensemble.pkl")

    print("\n=== Calibration ===")
    plot_calibration(all_probs, y_test_cls)

    print("\n=== Feature importance ===")
    plot_feature_importance(models["RandomForest"], models["GradientBoosting"], cols)

    print("\n=== Margin regression ===")
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train_reg)
    mae = mean_absolute_error(y_test_reg, pipe.predict(X_test))
    print(f"  Ridge MAE = {mae:.2f} pts")
    joblib.dump(pipe, MODELS / "margin_regressor.pkl")

    print("\n=== Saved models ===")
    for f in sorted(MODELS.glob("*.pkl")):
        print(f"  {f.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
