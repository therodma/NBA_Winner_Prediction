"""
train_models.py — Step 3: Model Training & Evaluation
Trains Logistic Regression, Random Forest, Gradient Boosting, and an ensemble.
Test set = 2024-25 season (held out, never used in training).
Saves all models to models/ and prints evaluation metrics.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

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
TARGET_CLS = "HOME_TEAM_WIN"
TARGET_REG = "POINT_MARGIN"
TEST_SEASON = "2024-25"


# ── data ─────────────────────────────────────────────────────────────────────

def load_splits():
    df = pd.read_csv(PROCESSED / "features.csv")
    train = df[df["SEASON"] != TEST_SEASON].copy()
    test  = df[df["SEASON"] == TEST_SEASON].copy()
    X_train = train[FEATURE_COLS].values
    X_test  = test[FEATURE_COLS].values
    y_train_cls = train[TARGET_CLS].values
    y_test_cls  = test[TARGET_CLS].values
    y_train_reg = train[TARGET_REG].values
    y_test_reg  = test[TARGET_REG].values
    print(f"Train: {len(train)} games | Test: {len(test)} games ({TEST_SEASON})")
    return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, test


# ── models ────────────────────────────────────────────────────────────────────

def build_models():
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5)),
    ])
    rf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=15, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.04,
                                    subsample=0.8, random_state=42)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                        eval_metric="logloss", random_state=42, verbosity=0)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("xgb", xgb)],
        voting="soft",
    )
    return {"LogisticRegression": lr, "RandomForest": rf, "GradientBoosting": gb,
            "XGBoost": xgb, "Ensemble": ensemble}


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(name, model, X_train, X_test, y_train, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    probs  = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, preds)
    ll     = log_loss(y_test, probs)
    print(f"  {name:20s}  acc={acc:.4f}  log_loss={ll:.4f}  cv={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")
    return preds, probs


def plot_calibration(models_probs, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, probs in models_probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODELS / "calibration_curves.png", dpi=120)
    plt.close()
    print("  Saved -> models/calibration_curves.png")


def plot_feature_importance(rf_model, gb_model):
    rf_imp = rf_model.feature_importances_
    gb_imp = gb_model.feature_importances_
    avg_imp = (rf_imp + gb_imp) / 2
    idx = np.argsort(avg_imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(FEATURE_COLS))
    ax.bar(x, avg_imp[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_title("Average Feature Importance (RF + GB)")
    plt.tight_layout()
    plt.savefig(MODELS / "feature_importance.png", dpi=120)
    plt.close()
    print("  Saved -> models/feature_importance.png")

    print("\n  Top 10 features:")
    for rank, i in enumerate(idx[:10], 1):
        print(f"    {rank:2d}. {FEATURE_COLS[i]:30s} {avg_imp[i]:.4f}")


# ── margin regression (simple linear on top of features) ─────────────────────

def train_margin_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  MarginRegressor (Ridge)       MAE={mae:.2f} pts")
    joblib.dump(pipe, MODELS / "margin_regressor.pkl")
    return pipe, preds


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Loading data ===")
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, test_df = load_splits()

    print("\n=== Training & evaluating classifiers ===")
    models = build_models()
    all_probs = {}
    all_preds = {}

    for name, model in models.items():
        preds, probs = evaluate(name, model, X_train, X_test, y_train_cls, y_test_cls)
        all_probs[name] = probs
        all_preds[name] = preds
        if name != "Ensemble":
            joblib.dump(model, MODELS / f"{name.lower()}.pkl")

    # Save ensemble separately (contains fitted sub-models)
    joblib.dump(models["Ensemble"], MODELS / "ensemble.pkl")

    print("\n=== Calibration curves ===")
    plot_calibration(all_probs, y_test_cls)

    print("\n=== Feature importance ===")
    plot_feature_importance(models["RandomForest"], models["GradientBoosting"])

    print("\n=== Margin regression ===")
    margin_model, margin_preds = train_margin_model(X_train, X_test, y_train_reg, y_test_reg)

    # Save a sample prediction table
    print("\n=== Sample predictions (first 10 test games) ===")
    sample = test_df.head(10)[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", TARGET_CLS, TARGET_REG]].copy()
    sample["PRED_LR"]       = all_preds["LogisticRegression"][:10]
    sample["PRED_RF"]       = all_preds["RandomForest"][:10]
    sample["PRED_GB"]       = all_preds["GradientBoosting"][:10]
    sample["PRED_ENSEMBLE"] = all_preds["Ensemble"][:10]
    sample["PROB_ENSEMBLE"] = all_probs["Ensemble"][:10].round(3)
    sample["PRED_MARGIN"]   = margin_preds[:10].round(1)
    print(sample.to_string(index=False))

    print("\n=== Saved models ===")
    for f in sorted(MODELS.glob("*.pkl")):
        print(f"  {f.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
