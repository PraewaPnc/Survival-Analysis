"""Phase 8.4 — weather-aware failure-risk classifiers.

Trains and compares three models on the time-based train/test split:
Random Forest, XGBoost, and an LSTM (pseudo-sequence over 5 weather features).

    define_features → build_preprocessor → SMOTE → train(RF, XGB, LSTM)
                    → evaluate → feature importances → save

Imbalance handling differs per model (documented in NOTE below):
  - RF  uses class_weight="balanced"   on the original preprocessed train
  - XGB uses scale_pos_weight=neg/pos  on the original preprocessed train
  - LSTM has no built-in weighting, so it trains on the SMOTE-resampled set
Applying SMOTE *and* class weighting to the tree models would double-correct
the imbalance, so SMOTE is reserved for the LSTM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TensorFlow

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
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

# The 5 weather features treated as a pseudo-sequence for the LSTM (4.4).
LSTM_SEQ_FEATURES: list[str] = [
    "forecast_wind_max", "forecast_wind_avg", "forecast_rain_total",
    "forecast_pressure_min", "forecast_humidity_max",
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
# 4.4 — LSTM helpers
# ---------------------------------------------------------------------------

def _seq_static_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Indices of the 5 weather-sequence columns vs. the static remainder."""
    seq_names = {f"num__{f}" for f in LSTM_SEQ_FEATURES}
    seq_idx = [i for i, n in enumerate(feature_names) if n in seq_names]
    static_idx = [i for i in range(len(feature_names)) if i not in seq_idx]
    return seq_idx, static_idx


def _split_seq_static(
    X: np.ndarray, seq_idx: list[int], static_idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape into ((n, 5, 1) sequence, (n, k) static) inputs."""
    X_seq = X[:, seq_idx].reshape(-1, len(seq_idx), 1)
    X_static = X[:, static_idx]
    return X_seq, X_static


def build_lstm(n_static: int) -> "object":
    """Build the LSTM + static-context classifier (4.4).

    Args:
        n_static: Number of static (non-sequence) features.

    Returns:
        Compiled Keras Model with two inputs [sequence, static].
    """
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense, Concatenate
    from tensorflow.keras.optimizers import Adam

    seq_in = Input(shape=(len(LSTM_SEQ_FEATURES), 1), name="weather_seq")
    static_in = Input(shape=(n_static,), name="static")

    x_seq = LSTM(64, return_sequences=False)(seq_in)
    x_static = Dense(32, activation="relu")(static_in)
    x = Concatenate()([x_seq, x_static])
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[seq_in, static_in], outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


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

    Note: tuned on the evaluation (test) set as specified, so the reported
    optimal-threshold metrics are mildly optimistic — the threshold "sees" the
    same set it is scored on.

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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> pd.DataFrame:
    """Run the full Phase 8.4 train/evaluate/save pipeline.

    Returns:
        The model comparison DataFrame (also saved to CSV).
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
        print(", ".join(feature_cols))

    X_train, y_train = train_df[feature_cols], train_df[TARGET].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[TARGET].to_numpy()

    # --- Carve a stratified 10% validation slice from train (raw) ----------
    # Models train on the 90% slice; thresholds are tuned on the held-out 10%.
    # Test is never touched for fitting or tuning.
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

    # --- 4.3 SMOTE (train slice only) --------------------------------------
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_tr_p, y_tr)
    perm = np.random.permutation(len(y_res))  # shuffle for LSTM validation_split
    X_res, y_res = X_res[perm], y_res[perm]
    if verbose:
        before = pd.Series(y_tr).value_counts().sort_index().to_dict()
        after = pd.Series(y_res).value_counts().sort_index().to_dict()
        print("\n=== 4.3 SMOTE class distribution (train slice) ===")
        print(f"  before: {before}")
        print(f"  after : {after}")

    # --- 4.4 train models (on the 90% slice) -------------------------------
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_tr_p, y_tr)

    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
        early_stopping_rounds=20,
    )
    xgb.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)

    seq_idx, static_idx = _seq_static_indices(feature_names)
    Xtr_seq, Xtr_static = _split_seq_static(X_res, seq_idx, static_idx)
    Xval_seq, Xval_static = _split_seq_static(X_val_p, seq_idx, static_idx)
    Xte_seq, Xte_static = _split_seq_static(X_test_p, seq_idx, static_idx)

    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    lstm = build_lstm(n_static=len(static_idx))
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    lstm.fit(
        [Xtr_seq, Xtr_static], y_res,
        epochs=50, batch_size=256, validation_split=0.1,
        callbacks=[es], verbose=2 if verbose else 0,
    )

    # --- Validation-slice probabilities → tune thresholds ------------------
    val_proba = {
        "RandomForest": rf.predict_proba(X_val_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_val_p)[:, 1],
        "LSTM": lstm.predict([Xval_seq, Xval_static], verbose=0).ravel(),
    }
    thresholds = {name: tune_threshold(y_val, p) for name, p in val_proba.items()}
    joblib.dump(thresholds, MODELS_DIR / "thresholds.pkl")
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # --- 4.5 evaluate on test ----------------------------------------------
    test_proba = {
        "RandomForest": rf.predict_proba(X_test_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_test_p)[:, 1],
        "LSTM": lstm.predict([Xte_seq, Xte_static], verbose=0).ravel(),
    }

    # Default-threshold (0.5) comparison — kept for reference
    comparison = pd.DataFrame([_metrics(n, y_test, p) for n, p in test_proba.items()])
    comparison.to_csv(TABLES_DIR / "ml_model_comparison.csv", index=False)

    # Final table: test metrics AT the validation-tuned threshold
    final_rows: list[dict[str, object]] = []
    for name, proba in test_proba.items():
        m = _metrics(name, y_test, proba, threshold=thresholds[name])
        final_rows.append({
            "Model": name, "val_tuned_threshold": round(thresholds[name], 2),
            "F1": m["F1"], "Precision": m["Precision"],
            "Recall": m["Recall"], "ROC_AUC": m["ROC_AUC"],
            "Accuracy": m["Accuracy"],
        })
    final_cmp = pd.DataFrame(final_rows)
    final_cmp.to_csv(TABLES_DIR / "ml_model_comparison_final.csv", index=False)

    plot_importance(rf, feature_names, "Random Forest — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_rf.png")
    plot_importance(xgb, feature_names, "XGBoost — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_xgboost.png")

    # --- 4.6 save models ---------------------------------------------------
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(xgb, MODELS_DIR / "xgboost_model.pkl")
    lstm.save(MODELS_DIR / "lstm_model.keras")

    if verbose:
        print("\n=== 4.5 Final comparison (test set @ validation-tuned threshold) ===")
        print(final_cmp.to_string(index=False))
        primary = final_cmp.loc[final_cmp["Recall"].idxmax(), "Model"]
        print(f"\nPrimary model (highest recall at val-tuned threshold): {primary}")
        print(f"Thresholds: {thresholds}")
        print(f"Saved models to {MODELS_DIR}/, final table to "
              f"{TABLES_DIR}/ml_model_comparison_final.csv")

    return final_cmp


def refit_full_train(verbose: bool = True) -> pd.DataFrame:
    """Refit all models on 100% of train, keeping the val-tuned thresholds.

    Thresholds were fixed on the held-out validation slice by `run()`; now that
    they are frozen (models/thresholds.json), the models are refit on the full
    train set so they use all available data. XGBoost reuses the tree count from
    the early-stopping run (no further early stopping). Saved models, preprocessor,
    importance plots, and ml_model_comparison_final.csv are overwritten.

    Returns:
        Final comparison DataFrame (test metrics at the frozen thresholds).
    """
    np.random.seed(RANDOM_STATE)
    train_df, test_df = load_train_test()
    feature_cols, categorical, numeric = define_features(train_df)
    X_train, y_train = train_df[feature_cols], train_df[TARGET].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[TARGET].to_numpy()

    thresholds = json.loads((MODELS_DIR / "thresholds.json").read_text())
    xgb_n_estimators = int(joblib.load(MODELS_DIR / "xgboost_model.pkl").best_iteration) + 1

    # Preprocessor refit on FULL train
    pre = build_preprocessor(numeric, categorical)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_p, y_train)

    # No early stopping on the refit — tree count fixed from the val run
    xgb = XGBClassifier(
        n_estimators=xgb_n_estimators, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
    )
    xgb.fit(X_train_p, y_train)

    # SMOTE on full train for the LSTM
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_p, y_train)
    perm = np.random.permutation(len(y_res))
    X_res, y_res = X_res[perm], y_res[perm]

    seq_idx, static_idx = _seq_static_indices(feature_names)
    Xtr_seq, Xtr_static = _split_seq_static(X_res, seq_idx, static_idx)
    Xte_seq, Xte_static = _split_seq_static(X_test_p, seq_idx, static_idx)

    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    lstm = build_lstm(n_static=len(static_idx))
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    lstm.fit(
        [Xtr_seq, Xtr_static], y_res,
        epochs=50, batch_size=256, validation_split=0.1,
        callbacks=[es], verbose=2 if verbose else 0,
    )

    test_proba = {
        "RandomForest": rf.predict_proba(X_test_p)[:, 1],
        "XGBoost": xgb.predict_proba(X_test_p)[:, 1],
        "LSTM": lstm.predict([Xte_seq, Xte_static], verbose=0).ravel(),
    }
    final_rows: list[dict[str, object]] = []
    for name, proba in test_proba.items():
        m = _metrics(name, y_test, proba, threshold=thresholds[name])
        final_rows.append({
            "Model": name, "val_tuned_threshold": round(thresholds[name], 2),
            "F1": m["F1"], "Precision": m["Precision"],
            "Recall": m["Recall"], "ROC_AUC": m["ROC_AUC"],
            "Accuracy": m["Accuracy"],
        })
    final_cmp = pd.DataFrame(final_rows)
    final_cmp.to_csv(TABLES_DIR / "ml_model_comparison_final.csv", index=False)

    plot_importance(rf, feature_names, "Random Forest — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_rf.png")
    plot_importance(xgb, feature_names, "XGBoost — Top 20 Feature Importances",
                    FIGURES_DIR / "ml_feature_importance_xgboost.png")

    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(xgb, MODELS_DIR / "xgboost_model.pkl")
    lstm.save(MODELS_DIR / "lstm_model.keras")

    if verbose:
        print(f"\n=== Refit on full train ({len(train_df):,} rows) "
              f"@ frozen val-tuned thresholds ===")
        print(f"XGBoost n_estimators (from early-stop run): {xgb_n_estimators}")
        print(final_cmp.to_string(index=False))
        print(f"Saved refit models to {MODELS_DIR}/ (preprocessor refit on full train)")

    return final_cmp


if __name__ == "__main__":
    refit_full_train()
