"""Phase 8.5 — combined failure-risk score and single-span inference.

Blends the RandomForest weather-aware failure probability (Phase 8.4) with the
Weibull-AFT Remaining-Useful-Life estimate (Phase 7.4) into one risk score:

    risk_score = 0.6 * failure_prob + 0.4 * (1 - RUL_norm)

`build_combined_risk_scores` scores every row of the test set; `predict_risk`
scores a single span from a raw 72-hour forecast.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from src.ml_models import (
    CATEGORICAL_FEATURES, EXCLUDE_COLS, TARGET, define_features,
)
from src.weather_features import aggregate_forecast


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "outputs" / "tables"

TEST_PATH = DATA_DIR / "test.csv"
TRANSMISSION_PATH = DATA_DIR / "transmission_line_maintenance_data.csv"
RUL_PATH = TABLES_DIR / "rul_predictions.csv"
COMBINED_PATH = TABLES_DIR / "combined_risk_scores.csv"
TEST_PRED_PATH = TABLES_DIR / "test_predictions.csv"
WEATHER_AGG_PATH = DATA_DIR / "weather_aggregated.csv"
PRIORITY_FORECAST_PATH = TABLES_DIR / "priority_maintenance_list_forecast.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
ROC_PR_PATH = FIGURES_DIR / "ml_roc_pr_curves.png"

MODEL_COLORS = {"RandomForest": "#d62728", "XGBoost": "#1f77b4", "LightGBM": "#2ca02c"}
_KEY_TO_NAME = {"rf": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}

ID_PASSTHROUGH = ["span_id", "component", "region", "event_type", "forecast_date"]

PRIMARY_MODEL = "RandomForest"  # highest recall at val-tuned threshold (Phase 8.4)

# Combined-score weights and risk-level cut points.
W_FAILURE, W_RUL = 0.6, 0.4
RISK_BINS = [0.7, 0.5, 0.3]  # >= → Critical / High / Medium, else Low
RISK_LABELS = ["Critical", "High", "Medium", "Low"]

# Weather feature columns produced by aggregate_forecast (broadcast in inference).
WEATHER_FEATURES = [
    "forecast_wind_max", "forecast_wind_avg", "forecast_rain_total",
    "forecast_pressure_min", "forecast_humidity_max", "forecast_temp_max",
    "forecast_confidence_avg",
]


# ---------------------------------------------------------------------------
# Artifact loading (lazy, cached)
# ---------------------------------------------------------------------------

_CACHE: dict[str, object] = {}


def _load_artifacts() -> dict[str, object]:
    """Load and cache preprocessor, models, thresholds, and feature metadata."""
    if _CACHE:
        return _CACHE
    pre = joblib.load(MODELS_DIR / "preprocessor.pkl")
    _CACHE.update({
        "pre": pre,
        "rf": joblib.load(MODELS_DIR / "rf_model.pkl"),
        "xgb": joblib.load(MODELS_DIR / "xgboost_model.pkl"),
        "lightgbm": joblib.load(MODELS_DIR / "lightgbm_model.pkl"),
        "thresholds": joblib.load(MODELS_DIR / "thresholds.pkl"),
        "feature_names": list(pre.get_feature_names_out()),
    })
    return _CACHE


def _risk_level(score: float) -> str:
    """Map a risk score to its categorical level."""
    for thr, label in zip(RISK_BINS, RISK_LABELS):
        if score >= thr:
            return label
    return RISK_LABELS[-1]


def _predict_probas(X_processed: np.ndarray) -> dict[str, np.ndarray]:
    """Positive-class probabilities from all three models on a processed matrix."""
    art = _load_artifacts()
    return {
        "rf": art["rf"].predict_proba(X_processed)[:, 1],
        "xgboost": art["xgb"].predict_proba(X_processed)[:, 1],
        "lightgbm": art["lightgbm"].predict_proba(X_processed)[:, 1],
    }


# ---------------------------------------------------------------------------
# Test-set predictions (per-model probabilities + thresholded classes)
# ---------------------------------------------------------------------------

def build_test_predictions(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Score test.csv with all three models and save per-row predictions.

    Writes outputs/tables/test_predictions.csv with each model's positive-class
    probability and its class at the frozen val-tuned threshold, alongside the
    true label. Skips work if the file already exists unless ``force`` is set.

    Args:
        force: Regenerate even if the output already exists.
        verbose: Print a short report to stdout.

    Returns:
        The predictions DataFrame (loaded from disk if it already existed).
    """
    if TEST_PRED_PATH.exists() and not force:
        if verbose:
            print(f"{TEST_PRED_PATH.name} already exists — skipping (use force=True to rebuild).")
        return pd.read_csv(TEST_PRED_PATH)

    art = _load_artifacts()
    test = pd.read_csv(TEST_PATH)
    feature_cols = define_features(test)[0]
    X_p = art["pre"].transform(test[feature_cols])
    probas = _predict_probas(X_p)
    thr = art["thresholds"]

    out = test[ID_PASSTHROUGH].copy()
    out["actual"] = test[TARGET].to_numpy()
    for key, model in (("rf", "RandomForest"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
        out[f"prob_{key}"] = probas[key]
        out[f"pred_{key}"] = (probas[key] >= thr[model]).astype(int)

    out.to_csv(TEST_PRED_PATH, index=False)
    if verbose:
        n = len(out)
        print(f"Test predictions: {n} rows → {TEST_PRED_PATH.name}")
        print(f"Actual positive rate: {out['actual'].mean():.2%}")
        for key, model in (("rf", "RandomForest"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
            flagged = int(out[f"pred_{key}"].sum())
            print(f"  {model:13s} (thr {thr[model]:.2f}): {flagged} flagged "
                  f"({flagged / n:.1%})")
    return out


# ---------------------------------------------------------------------------
# Step 5 — combined risk scores over the test set
# ---------------------------------------------------------------------------

def build_combined_risk_scores(verbose: bool = True) -> pd.DataFrame:
    """Score every test row by blending RF failure prob with RUL.

    Returns:
        DataFrame written to combined_risk_scores.csv.
    """
    art = _load_artifacts()
    test = pd.read_csv(TEST_PATH)
    rul = pd.read_csv(RUL_PATH)[["span_id", "component", "RUL_days", "risk_category"]]

    # RF failure probability for every test row (primary model, val-tuned thr=0.11)
    feature_cols = define_features(test)[0]
    X_p = art["pre"].transform(test[feature_cols])
    test["failure_prob"] = art["rf"].predict_proba(X_p)[:, 1]

    # Join RUL on (span_id, component); fill misses with the component median RUL
    merged = test.merge(
        rul[["span_id", "component", "RUL_days"]], on=["span_id", "component"], how="left",
    )
    median_rul = rul.groupby("component")["RUL_days"].median()
    n_fill = int(merged["RUL_days"].isna().sum())
    merged["RUL_days"] = merged["RUL_days"].fillna(merged["component"].map(median_rul))

    # Combined score + level
    rul_norm = merged["RUL_days"] / merged["RUL_days"].max()
    merged["risk_score"] = W_FAILURE * merged["failure_prob"] + W_RUL * (1 - rul_norm)
    merged["risk_level"] = pd.cut(
        merged["risk_score"],
        bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )

    out_cols = [
        "span_id", "component", "region", "event_type", "forecast_date",
        "failure_prob", "RUL_days", "risk_score", "risk_level",
    ]
    out = merged[out_cols]
    out.to_csv(COMBINED_PATH, index=False)

    if verbose:
        n = len(out)
        print(f"Combined risk scores: {n} rows → {COMBINED_PATH.name}")
        print(f"RUL match: {n - n_fill} matched, {n_fill} filled with component-median RUL")
        dist = out["risk_level"].value_counts().reindex(RISK_LABELS)
        print("\nrisk_level distribution:")
        for level, cnt in dist.items():
            print(f"  {level:9s} {int(cnt):6d} ({cnt / n:.1%})")

    return out


# ---------------------------------------------------------------------------
# ROC and Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(verbose: bool = True) -> dict[str, dict[str, float]]:
    """Plot ROC and Precision-Recall curves for all 3 models on the test set.

    Each model's operating point (its frozen val-tuned threshold) is marked.
    Saves a 1×2 figure (ROC | PR) to outputs/figures/ml_roc_pr_curves.png.

    Returns:
        Per-model dict of {roc_auc, avg_precision}.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    art = _load_artifacts()
    test = pd.read_csv(TEST_PATH)
    y = test[TARGET].to_numpy()
    feature_cols = define_features(test)[0]
    X_p = art["pre"].transform(test[feature_cols])
    probas = _predict_probas(X_p)
    thr = art["thresholds"]
    base_rate = y.mean()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    scores: dict[str, dict[str, float]] = {}

    for key, name in _KEY_TO_NAME.items():
        proba = probas[key]
        color = MODEL_COLORS[name]
        t = thr[name]

        # ROC
        fpr, tpr, roc_t = roc_curve(y, proba)
        roc_auc = roc_auc_score(y, proba)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        i = int(np.argmin(np.abs(roc_t - t)))
        ax_roc.scatter(fpr[i], tpr[i], color=color, s=60, zorder=5,
                       edgecolor="black", linewidth=0.8)

        # Precision-Recall
        prec, rec, pr_t = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
        j = int(np.argmin(np.abs(pr_t - t)))  # pr_t aligns with prec[:-1], rec[:-1]
        ax_pr.scatter(rec[j], prec[j], color=color, s=60, zorder=5,
                      edgecolor="black", linewidth=0.8)

        scores[name] = {"roc_auc": roc_auc, "avg_precision": ap}

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves (test set)", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    ax_pr.axhline(base_rate, ls="--", color="grey", lw=1, alpha=0.7,
                  label=f"Baseline ({base_rate:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall Curves (test set)", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    fig.suptitle("Markers = each model's val-tuned operating threshold", y=0.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROC_PR_PATH, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Saved ROC + PR curves → {ROC_PR_PATH}")
        for name, s in scores.items():
            print(f"  {name:13s} ROC-AUC={s['roc_auc']:.3f}  AP={s['avg_precision']:.3f}")
    return scores


# ---------------------------------------------------------------------------
# Priority maintenance list (risk scores + weather context, ranked)
# ---------------------------------------------------------------------------

def build_priority_forecast_list(verbose: bool = True) -> pd.DataFrame:
    """Join weather features onto combined risk scores, ranked by risk_score.

    Maps `weather_aggregated.csv` (the forecast features per span×event) onto
    `combined_risk_scores.csv` so each prioritised unit carries its weather
    context, then sorts by risk_score descending.

    Returns:
        The ranked DataFrame, also saved to priority_maintenance_list_forecast.csv.
    """
    combined = pd.read_csv(COMBINED_PATH)
    weather = pd.read_csv(WEATHER_AGG_PATH)

    weather_cols = [
        "span_id", "event_type", "forecast_date", "event_id", "age_at_forecast_days",
        *WEATHER_FEATURES,
    ]
    merged = combined.merge(
        weather[weather_cols], on=["span_id", "event_type", "forecast_date"], how="left",
    )

    out_cols = [
        "span_id", "event_id", "component", "region", "event_type", "forecast_date",
        "age_at_forecast_days", *WEATHER_FEATURES,
        "RUL_days", "failure_prob", "risk_score", "risk_level",
    ]
    ranked = merged[out_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)
    ranked.to_csv(PRIORITY_FORECAST_PATH, index=False)

    if verbose:
        unmatched = ranked["forecast_wind_max"].isna().sum()
        print(f"Priority forecast list: {len(ranked)} rows → {PRIORITY_FORECAST_PATH.name}")
        print(f"Weather match: {len(ranked) - unmatched}/{len(ranked)} "
              f"({unmatched} unmatched)")
        print("\nTop 10 by risk_score:")
        show = ["span_id", "component", "region", "event_type", "risk_score",
                "risk_level", "failure_prob", "RUL_days", "forecast_wind_max",
                "forecast_rain_total"]
        print(ranked[show].head(10).to_string(index=False))

    return ranked


# ---------------------------------------------------------------------------
# Step 5.2 — single-span inference
# ---------------------------------------------------------------------------

def predict_risk(
    span_id: str,
    forecast_df: pd.DataFrame,
    rul_days: float | None = None,
) -> dict:
    """Predict failure risk for one span given a 72-hour forecast.

    Aggregates the raw forecast, attaches the span's static component covariates,
    runs all three models, and aggregates to a span-level risk by taking the
    worst (max) component probability — appropriate when missing a failure is
    costlier than a false alarm.

    Args:
        span_id: Target span identifier.
        forecast_df: Raw 72-hour forecast DataFrame (weather_forecast_72hr schema).
        rul_days: Remaining useful life (days) from the Weibull AFT model. If
            None, the combined score equals the RF failure probability alone.

    Returns:
        dict with keys: span_id, event_type, failure_prob_rf,
        failure_prob_xgboost, failure_prob_lightgbm, recommended_threshold,
        combined_risk_score, risk_level.
    """
    art = _load_artifacts()

    # Aggregate the forecast and pick this span's most severe event (max rainfall)
    agg = aggregate_forecast(forecast_df)
    agg = agg[agg["span_id"] == span_id]
    if agg.empty:
        raise ValueError(f"No forecast rows for span_id={span_id!r}")
    event = agg.loc[agg["forecast_rain_total"].idxmax()]

    # The span's static component rows from the maintenance data (6 components)
    trans = pd.read_csv(TRANSMISSION_PATH)
    rows = trans[trans["span_id"] == span_id].drop(
        columns=["fault_to_failure_flag"], errors="ignore",
    ).copy()
    if rows.empty:
        raise ValueError(f"span_id={span_id!r} not found in transmission data")

    # Broadcast the forecast-derived features onto every component row
    for col in WEATHER_FEATURES:
        rows[col] = event[col]
    rows["event_type"] = event["event_type"]
    rows["forecast_month"] = event["forecast_month"]
    rows["age_at_forecast_years"] = event["age_at_forecast_days"] / 365

    feature_cols = define_features(rows)[0]
    X_p = art["pre"].transform(rows[feature_cols])
    probas = _predict_probas(X_p)

    # Span-level risk = worst component (max probability)
    prob_rf = float(probas["rf"].max())
    prob_xgb = float(probas["xgboost"].max())
    prob_lightgbm = float(probas["lightgbm"].max())

    if rul_days is None:
        combined = prob_rf
    else:
        ref_max = pd.read_csv(RUL_PATH)["RUL_days"].max()
        rul_norm = min(max(rul_days / ref_max, 0.0), 1.0)
        combined = W_FAILURE * prob_rf + W_RUL * (1 - rul_norm)

    return {
        "span_id": span_id,
        "event_type": str(event["event_type"]),
        "failure_prob_rf": prob_rf,
        "failure_prob_xgboost": prob_xgb,
        "failure_prob_lightgbm": prob_lightgbm,
        "recommended_threshold": float(art["thresholds"][PRIMARY_MODEL]),
        "combined_risk_score": float(combined),
        "risk_level": _risk_level(combined),
    }


if __name__ == "__main__":
    build_combined_risk_scores()
