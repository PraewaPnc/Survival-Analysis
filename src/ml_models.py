"""Phase 8.4 — weather-aware failure-risk classifiers.

Trains and compares three gradient/ensemble tree models on the time-based
train/test split: Random Forest, XGBoost, and LightGBM.

    define_features → build_preprocessor → train(RF, XGB, LightGBM)
                    → tune thresholds (val slice) → evaluate → importances → save

All three models have built-in class-imbalance handling, so no resampling
(SMOTE) is used — that would double-correct the imbalance:
  - RF       — class_weight="balanced"
  - XGBoost  — scale_pos_weight = neg/pos
  - LightGBM — class_weight="balanced"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET = "fault_to_failure_flag"

# Columns excluded from the feature matrix (4.1). event_id is added to the
# spec's list: it is a 5,342-level identifier, the same family as span_id.
EXCLUDE_COLS: list[str] = [
    TARGET, "span_id", "forecast_date", "installation_date",
    "age_at_forecast_days", "failure_probability", "line_id", "tower_id",
    "event_id",
]

# Categorical features to one-hot encode (4.2). "region" is added to the spec's
# list — a genuine 4-level categorical that would otherwise be silently dropped.
CATEGORICAL_FEATURES: list[str] = [
    "component", "event_type", "coastal_proximity", "pollution_severity",
    "encroachment_severity", "fault_type_most_common", "HI_class_last", "region",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 4.1 — Feature matrix
# ---------------------------------------------------------------------------

def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the time-based train/test CSVs.

    Returns:
        Tuple of (train_df, test_df).
    """
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def define_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Resolve the feature columns and split into categorical/numeric.

    Args:
        df: Training DataFrame (used to discover column dtypes).

    Returns:
        Tuple of (feature_cols, categorical_cols, numeric_cols).
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]
    return feature_cols, categorical, numeric


# ---------------------------------------------------------------------------
# 4.2 — Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """StandardScaler on numeric + one-hot on categorical features.

    Args:
        numeric: Numeric feature column names.
        categorical: Categorical feature column names.

    Returns:
        Unfitted ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 4.4 — Model factories
# ---------------------------------------------------------------------------

def build_rf() -> RandomForestClassifier:
    """Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_xgb(scale_pos_weight: float, n_estimators: int = 200,
              early_stopping: bool = False) -> XGBClassifier:
    """XGBoost with scale_pos_weight; optional early stopping."""
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
        early_stopping_rounds=20 if early_stopping else None,
    )


def build_lgbm(n_estimators: int = 200) -> LGBMClassifier:
    """LightGBM with balanced class weights (mirrors RF's strategy)."""
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


# ---------------------------------------------------------------------------
# 4.5 — Evaluation
# ---------------------------------------------------------------------------

def _metrics(name: str, y_true: np.ndarray, proba: np.ndarray,
             threshold: float = 0.5) -> dict[str, object]:
    """Compute the five evaluation metrics for one model at a given threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "F1": f1_score(y_true, pred),
        "ROC_AUC": roc_auc_score(y_true, proba),  # threshold-independent
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "Accuracy": accuracy_score(y_true, pred),
    }


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold in [0.1, 0.6) that maximises F1 (grid step 0.01).

    Tuned on a held-out validation slice (never on test).

    Args:
        y_true: Binary ground-truth labels.
        proba: Predicted positive-class probabilities.

    Returns:
        The F1-maximising threshold.
    """
    thresholds = np.arange(0.1, 0.6, 0.01)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def plot_importance(model: "object", feature_names: list[str], title: str, path: Path,
                    top_n: int = 20) -> None:
    """Save a horizontal bar chart of the top-N feature importances."""
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    top = imp.tail(top_n)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.index, top.values, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _final_table(test_proba: dict[str, np.ndarray], y_test: np.ndarray,
                 thresholds: dict[str, float]) -> pd.DataFrame:
    """Build the test-metrics-at-val-tuned-threshold comparison table."""
    rows: list[dict[str, object]] = []
    for name, proba in test_proba.items():
        m = _metrics(name, y_test, proba, threshold=thresholds[name])
        rows.append({
            "Model": name, "val_tuned_threshold": round(thresholds[name], 2),
            "F1": m["F1"], "Precision": m["Precision"],
            "Recall": m["Recall"], "ROC_AUC": m["ROC_AUC"], "Accuracy": m["Accuracy"],
        })
    return pd.DataFrame(rows)


def _save_importances(rf, xgb, lgbm, feature_names: list[str]) -> None:
    """Save the three feature-importance figures."""
    plot_importance(rf, feature_names, "Random Forest — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_rf.png")
    plot_importance(xgb, feature_names, "XGBoost — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_xgboost.png")
    plot_importance(lgbm, feature_names, "LightGBM — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_lightgbm.png")


def _save_models(rf, xgb, lgbm) -> None:
    """Persist the three fitted models (compressed; RF is ~160 MB uncompressed)."""
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl", compress=3)
    joblib.dump(xgb, MODELS_DIR / "xgboost_model.pkl", compress=3)
    joblib.dump(lgbm, MODELS_DIR / "lightgbm_model.pkl", compress=3)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> pd.DataFrame:
    """Train on a 90% slice, tune thresholds on the held-out 10%, evaluate on test.

    Returns:
        The final comparison DataFrame (also saved to CSV).
    """
    np.random.seed(RANDOM_STATE)
    for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test()
    feature_cols, categorical, numeric = define_features(train_df)

    if verbose:
        print(f"=== 4.1 Feature matrix: {len(feature_cols)} features ===")
        print(f"  numeric ({len(numeric)}): {numeric}")
        print(f"  categorical ({len(categorical)}): {categorical}")

    X_train, y_train = train_df[feature_cols], train_df[TARGET].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[TARGET].to_numpy()

    # Stratified 10% validation slice — models train on 90%, thresholds tuned on 10%.
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE,
    )

    # --- 4.2 preprocessing (fit on the 90% train slice only) ---------------
    pre = build_preprocessor(numeric, categorical)
    X_tr_p = pre.fit_transform(X_tr_raw)
    X_val_p = pre.transform(X_val_raw)
    X_test_p = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")
    if verbose:
        print(f"\n=== 4.2 Preprocessed matrix: {X_tr_p.shape[1]} columns "
              f"(train slice {X_tr_p.shape[0]}, val {X_val_p.shape[0]}, test {X_test_p.shape[0]}) ===")

    # --- 4.4 train models (on the 90% slice) -------------------------------
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())

    rf = build_rf()
    rf.fit(X_tr_p, y_tr)

    xgb = build_xgb(scale_pos_weight=neg / pos, early_stopping=True)
    xgb.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)

    lgbm = build_lgbm()
    lgbm.fit(
        X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    # --- Validation-slice probabilities → tune thresholds ------------------
    val_proba = {
        "RandomForest": rf.predict_proba(X_val_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_val_p)[:, 1],
        "LightGBM": lgbm.predict_proba(X_val_p)[:, 1],
    }
    thresholds = {name: tune_threshold(y_val, p) for name, p in val_proba.items()}
    joblib.dump(thresholds, MODELS_DIR / "thresholds.pkl")
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # --- 4.5 evaluate on test ----------------------------------------------
    test_proba = {
        "RandomForest": rf.predict_proba(X_test_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_test_p)[:, 1],
        "LightGBM": lgbm.predict_proba(X_test_p)[:, 1],
    }
    pd.DataFrame([_metrics(n, y_test, p) for n, p in test_proba.items()]).to_csv(
        TABLES_DIR / "ml_model_comparison.csv", index=False)

    final_cmp = _final_table(test_proba, y_test, thresholds)
    final_cmp.to_csv(TABLES_DIR / "ml_model_comparison_final.csv", index=False)

    _save_importances(rf, xgb, lgbm, feature_names)
    _save_models(rf, xgb, lgbm)

    if verbose:
        print("\n=== 4.5 Final comparison (test set @ validation-tuned threshold) ===")
        print(final_cmp.to_string(index=False))
        primary = final_cmp.loc[final_cmp["Recall"].idxmax(), "Model"]
        print(f"\nPrimary model (highest recall at val-tuned threshold): {primary}")
        print(f"Thresholds: {thresholds}")

    return final_cmp


def refit_full_train(verbose: bool = True) -> pd.DataFrame:
    """Refit all models on 100% of train, keeping the val-tuned thresholds.

    Thresholds were fixed on the held-out validation slice by `run()`; now that
    they are frozen (models/thresholds.json), the models are refit on the full
    train set. XGBoost and LightGBM reuse the tree counts from their early-stop
    runs (no further early stopping). Saved models, preprocessor, importance
    plots, and ml_model_comparison_final.csv are overwritten.

    Returns:
        Final comparison DataFrame (test metrics at the frozen thresholds).
    """
    np.random.seed(RANDOM_STATE)
    train_df, test_df = load_train_test()
    feature_cols, categorical, numeric = define_features(train_df)
    X_train, y_train = train_df[feature_cols], train_df[TARGET].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[TARGET].to_numpy()

    thresholds = json.loads((MODELS_DIR / "thresholds.json").read_text())
    xgb_n = int(joblib.load(MODELS_DIR / "xgboost_model.pkl").best_iteration) + 1
    lgbm_n = int(joblib.load(MODELS_DIR / "lightgbm_model.pkl").best_iteration_)

    pre = build_preprocessor(numeric, categorical)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    rf = build_rf()
    rf.fit(X_train_p, y_train)

    xgb = build_xgb(scale_pos_weight=neg / pos, n_estimators=xgb_n, early_stopping=False)
    xgb.fit(X_train_p, y_train)

    lgbm = build_lgbm(n_estimators=lgbm_n)
    lgbm.fit(X_train_p, y_train)

    test_proba = {
        "RandomForest": rf.predict_proba(X_test_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_test_p)[:, 1],
        "LightGBM": lgbm.predict_proba(X_test_p)[:, 1],
    }
    final_cmp = _final_table(test_proba, y_test, thresholds)
    final_cmp.to_csv(TABLES_DIR / "ml_model_comparison_final.csv", index=False)

    _save_importances(rf, xgb, lgbm, feature_names)
    _save_models(rf, xgb, lgbm)

    if verbose:
        print(f"\n=== Refit on full train ({len(train_df):,} rows) "
              f"@ frozen val-tuned thresholds ===")
        print(f"Tree counts from early-stop runs — XGB: {xgb_n}, LightGBM: {lgbm_n}")
        print(final_cmp.to_string(index=False))
        print(f"Saved refit models to {MODELS_DIR}/ (preprocessor refit on full train)")

    return final_cmp


if __name__ == "__main__":
    run()
    refit_full_train()
