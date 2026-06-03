"""Weibull Accelerated Failure Time (AFT) model per component — Phase 7.

Fits a separate WeibullAFTFitter for each of the 6 transmission-line
component types using the same 7 standardised covariates as Phase 6.

AFT interpretation differs from Cox PH:
  - Cox: exp(coef) = Hazard Ratio (HR > 1 → more hazard → shorter survival)
  - AFT: exp(coef) = Time Ratio (TR < 1 → shorter survival time → risk)
        TR > 1 → longer survival time → protective

The lambda_ sub-model encodes the covariate effects on the (log) scale
parameter; rho_ is the shape sub-model (intercept only by default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from src.cox_model import (
    COVARIATE_LABELS,
    MODEL_COVARIATES,
    _component_prepared,
    prepare_cox_data,
    _sig_stars,
)


TABLES = Path(__file__).parent.parent / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_aft_by_component(
    df: pd.DataFrame,
    components: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    penalizer: float = 0.1,
) -> dict[str, WeibullAFTFitter]:
    """Fit a WeibullAFTFitter for each component.

    Reuses the same standardised data prepared by Phase 6's
    ``prepare_cox_data`` so that covariates are on identical scales and
    results are directly comparable with the Cox PH model.

    Args:
        df: Validated raw DataFrame.
        components: Ordered list of component names.
        duration_col: Time-to-event column.
        event_col: Binary event indicator (1=failure).
        penalizer: L2 ridge regularisation (same default as Cox model).

    Returns:
        Dict mapping component name → fitted WeibullAFTFitter.
    """
    full_prep = prepare_cox_data(df, duration_col, event_col, ref_df=df)

    fitters: dict[str, WeibullAFTFitter] = {}
    for comp in components:
        sub = _component_prepared(df, full_prep, comp)
        n_ev = int(sub[event_col].sum())
        print(f"[aft] Fitting {comp}: n={len(sub)}, events={n_ev}")
        waf = WeibullAFTFitter(penalizer=penalizer)
        try:
            waf.fit(sub, duration_col=duration_col, event_col=event_col,
                    show_progress=False)
            fitters[comp] = waf
        except Exception as exc:
            print(f"[aft] {comp} failed: {exc}")

    return fitters


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def aft_tr_table(
    fitters: dict[str, WeibullAFTFitter],
) -> pd.DataFrame:
    """Extract time ratios (TR) and 95% CI from the lambda_ sub-model.

    Only covariate rows (no Intercept) of the lambda_ sub-model are kept.
    Intercept rows and the rho_ sub-model are excluded.

    A TR < 1 indicates the covariate *shortens* expected survival (risk);
    TR > 1 indicates the covariate *extends* expected survival (protective).

    Args:
        fitters: Output of fit_aft_by_component.

    Returns:
        Long-format DataFrame with columns:
        [component, covariate, covariate_label, TR, TR_lower, TR_upper,
         coef, se, z, p, sig].
    """
    rows = []
    for comp, waf in fitters.items():
        lam = waf.summary.xs("lambda_", level="param")
        lam = lam[lam.index != "Intercept"]
        for cov in lam.index:
            p = float(lam.loc[cov, "p"])
            rows.append({
                "component":       comp,
                "covariate":       cov,
                "covariate_label": COVARIATE_LABELS.get(cov, cov),
                "TR":              round(float(lam.loc[cov, "exp(coef)"]), 4),
                "TR_lower":        round(float(lam.loc[cov, "exp(coef) lower 95%"]), 4),
                "TR_upper":        round(float(lam.loc[cov, "exp(coef) upper 95%"]), 4),
                "coef":            round(float(lam.loc[cov, "coef"]), 4),
                "se":              round(float(lam.loc[cov, "se(coef)"]), 4),
                "z":               round(float(lam.loc[cov, "z"]), 3),
                "p":               round(p, 4),
                "sig":             _sig_stars(p),
            })
    return pd.DataFrame(rows)


def aft_model_summary(
    fitters: dict[str, WeibullAFTFitter],
) -> pd.DataFrame:
    """Per-component model diagnostics: AIC, LLV, C-index, rho shape.

    Args:
        fitters: Output of fit_aft_by_component.

    Returns:
        DataFrame indexed by component with columns:
        [AIC, LLV, concordance_index, rho_intercept, median_survival_mean].
    """
    rows = []
    for comp, waf in fitters.items():
        rho = float(waf.params_.loc[("rho_", "Intercept")])
        rows.append({
            "component":          comp,
            "AIC":                round(float(waf.AIC_), 2),
            "LLV":                round(float(waf.log_likelihood_), 4),
            "concordance_index":  round(float(waf.concordance_index_), 4),
            "rho_intercept":      round(rho, 4),
        })
    return pd.DataFrame(rows).set_index("component")


def aft_covariate_wide(
    tr_table: pd.DataFrame,
    metric: str = "TR",
) -> pd.DataFrame:
    """Pivot TR table to wide format: component × covariate_label.

    Args:
        tr_table: Output of aft_tr_table.
        metric: Column to pivot — 'TR', 'p', 'sig', 'TR_lower', 'TR_upper'.

    Returns:
        Wide DataFrame (component × covariate_label).
    """
    return tr_table.pivot(index="component", columns="covariate_label", values=metric)


def predict_mean_survival(
    waf: WeibullAFTFitter,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Predict AFT S(t) for the population-mean covariate profile.

    All standardised covariates = 0 (= full-dataset mean), so this gives
    the model's prediction for an "average" span within that component.

    Args:
        waf: Fitted WeibullAFTFitter for one component.
        t_grid: Evaluation time points (days).

    Returns:
        Array of predicted survival probabilities at each t in t_grid.
    """
    # Build a single-row mean profile (all covariates = 0)
    cov_names = [c for c in waf.params_.index.get_level_values("covariate").unique()
                 if c != "Intercept"]
    profile = pd.DataFrame({c: [0.0] for c in cov_names})
    sf = waf.predict_survival_function(profile, times=t_grid)
    return sf.iloc[:, 0].values


def build_hi_profiles(n_sd: list[float] | None = None) -> pd.DataFrame:
    """Create covariate profiles varying HI_score_last at multiples of SD.

    All other covariates are held at their standardised mean (= 0).

    Args:
        n_sd: List of SD multiples for HI_score_last.
               Defaults to [-2, -1, 0, 1, 2].

    Returns:
        DataFrame with one row per profile, columns = MODEL_COVARIATES.
    """
    if n_sd is None:
        n_sd = [-2.0, -1.0, 0.0, 1.0, 2.0]

    rows = []
    for v in n_sd:
        row = {c: 0.0 for c in MODEL_COVARIATES}
        row["HI_score_last"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
# Acceleration Factors (7.2)
# ---------------------------------------------------------------------------

def extract_acceleration_factors(
    fitters: dict[str, WeibullAFTFitter],
) -> pd.DataFrame:
    """Extract Acceleration Factors (AF) per covariate × component.

    An Acceleration Factor is the exponentiated lambda_ sub-model coefficient.
    It has the same numerical value as the Time Ratio but the AFT-specific
    interpretation:

    - AF < 1: covariate **shortens** expected survival (accelerates failure)
    - AF > 1: covariate **extends**  expected survival (decelerates failure)
    - AF = 1: no effect on survival time

    This is reported on the log scale (``coef = log(AF)``), along with
    the 95% confidence interval for AF and a two-sided p-value.

    Args:
        fitters: Output of fit_aft_by_component.

    Returns:
        Long-format DataFrame with columns:
        [component, covariate, covariate_label, coef, AF, AF_lower_95,
         AF_upper_95, p, sig, effect].
        Rows are sorted by component then p-value ascending.
    """
    rows = []
    for comp, waf in fitters.items():
        lam = waf.summary.xs("lambda_", level="param")
        lam = lam[lam.index != "Intercept"]
        for cov in lam.index:
            af  = float(lam.loc[cov, "exp(coef)"])
            p   = float(lam.loc[cov, "p"])
            rows.append({
                "component":       comp,
                "covariate":       cov,
                "covariate_label": COVARIATE_LABELS.get(cov, cov),
                "coef":            round(float(lam.loc[cov, "coef"]), 4),
                "AF":              round(af, 4),
                "AF_lower_95":     round(float(lam.loc[cov, "exp(coef) lower 95%"]), 4),
                "AF_upper_95":     round(float(lam.loc[cov, "exp(coef) upper 95%"]), 4),
                "p":               round(p, 4),
                "sig":             _sig_stars(p),
                "effect":          (
                    "accelerates failure" if af < 1.0 else
                    "decelerates failure" if af > 1.0 else
                    "no effect"
                ),
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["component", "p"]).reset_index(drop=True)


def save_acceleration_factors(af_table: pd.DataFrame) -> Path:
    """Save the AF table to outputs/tables/aft_acceleration_factors.csv.

    Args:
        af_table: Output of extract_acceleration_factors.

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "aft_acceleration_factors.csv"
    af_table.to_csv(path, index=False)
    print(f"[aft] Saved → {path}")
    return path


# ---------------------------------------------------------------------------

def save_aft_results(
    tr_table: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save AFT time ratio table and model summary to outputs/tables/.

    Args:
        tr_table: Output of aft_tr_table.
        summary: Output of aft_model_summary.

    Returns:
        Tuple (tr_table_path, summary_path).
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    p1 = TABLES / "aft_time_ratios.csv"
    p2 = TABLES / "aft_model_summary.csv"
    tr_table.to_csv(p1, index=False)
    summary.to_csv(p2)
    print(f"[aft] Saved → {p1}")
    print(f"[aft] Saved → {p2}")
    return p1, p2


# ---------------------------------------------------------------------------
# Remaining Useful Life (RUL) — Phase 7.4
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS: list[tuple[float, str]] = [
    (365,  "Critical"),
    (730,  "Warning"),
    (1460, "Monitor"),
]


def _risk_category(rul_days: float) -> str:
    """Map RUL (days) to a 4-level risk category string."""
    for threshold, label in _RISK_THRESHOLDS:
        if rul_days < threshold:
            return label
    return "Healthy"


def predict_rul(
    df: pd.DataFrame,
    fitters: dict[str, WeibullAFTFitter],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    method: str = "median",
) -> pd.DataFrame:
    """Predict Remaining Useful Life (RUL) for every observation.

    For each row the component-specific WeibullAFTFitter predicts the
    total survival time under that row's covariate profile.  RUL is the
    residual life beyond the current age, clipped to zero.

    Prediction methods:

    - ``"median"`` — median survival time (50th percentile, default).
      More robust for heavily right-censored components such as Conductor
      (4.9% event rate) where the distributional tail is very uncertain.
    - ``"mean"``   — expected survival time (Weibull closed-form mean).
      Larger than the median; use when the arithmetic mean is preferred.

    Args:
        df: Validated raw DataFrame (all components, all rows).
        fitters: Output of fit_aft_by_component.
        duration_col: Column holding the observed age (current time in service).
        event_col: Binary failure indicator (not used for prediction but
            kept for context in the output).
        method: ``"median"`` or ``"mean"``.

    Returns:
        Copy of *df* with four additional columns:

        - ``predicted_lifetime_days`` — model-predicted total survival time.
        - ``RUL_days``               — max(0, predicted_lifetime − age).
        - ``RUL_years``              — RUL_days / 365, rounded to 3 d.p.
        - ``risk_category``          — "Critical" / "Warning" / "Monitor" / "Healthy".
    """
    if method not in ("median", "mean"):
        raise ValueError(f"method must be 'median' or 'mean', got {method!r}")

    full_prep = prepare_cox_data(df, duration_col, event_col, ref_df=df)

    predicted = pd.Series(np.nan, index=df.index, dtype=float)

    for comp, waf in fitters.items():
        comp_mask   = df["component"] == comp         # Boolean Series (df index)
        sub_indices = df.index[comp_mask]             # Original df indices
        sub         = full_prep[comp_mask.values].reset_index(drop=True)

        if method == "mean":
            try:
                preds = waf.predict_expectation(sub)
            except Exception:
                preds = waf.predict_median(sub)
        else:
            preds = waf.predict_median(sub)

        predicted.loc[sub_indices] = preds.values

    out = df.copy()
    out["predicted_lifetime_days"] = predicted.round(1)
    out["RUL_days"]  = (predicted - df[duration_col]).clip(lower=0).round(1)
    out["RUL_years"] = (out["RUL_days"] / 365.0).round(3)
    out["risk_category"] = out["RUL_days"].map(_risk_category)

    return out


def rul_summary_by_component(rul_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for RUL_days grouped by component.

    Args:
        rul_df: Output of predict_rul.

    Returns:
        DataFrame indexed by component with mean, median, min, max,
        and std of RUL_days.
    """
    g = rul_df.groupby("component")["RUL_days"]
    return pd.DataFrame({
        "mean_RUL_days":   g.mean().round(1),
        "median_RUL_days": g.median().round(1),
        "std_RUL_days":    g.std().round(1),
        "min_RUL_days":    g.min().round(1),
        "max_RUL_days":    g.max().round(1),
    })


def rul_risk_breakdown(rul_df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage in each risk category per component.

    Args:
        rul_df: Output of predict_rul.

    Returns:
        DataFrame indexed by component with columns:
        Critical, Warning, Monitor, Healthy (counts) and their _pct variants.
    """
    cats = ["Critical", "Warning", "Monitor", "Healthy"]
    counts = (
        rul_df.groupby(["component", "risk_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cats, fill_value=0)
    )
    total = counts.sum(axis=1)
    pct = (counts.div(total, axis=0) * 100).round(1)
    pct.columns = [f"{c}_pct" for c in cats]
    return pd.concat([counts, pct], axis=1)


def generate_priority_list(
    rul_df: pd.DataFrame,
    output_cols: list[str] | None = None,
    print_top_n: int = 20,
) -> pd.DataFrame:
    """Filter Critical units and produce a priority maintenance list.

    Selects observations where risk_category == 'Critical' (RUL < 365 days),
    sorts by RUL_days ascending (most urgent first), and saves to
    outputs/tables/priority_maintenance_list.csv.

    Args:
        rul_df: Output of predict_rul — must contain risk_category and
            RUL_days columns.
        output_cols: Columns to keep in the output. Defaults to the 11
            operationally relevant columns specified in Phase 7.7.
        print_top_n: Number of most-urgent rows to print. Set 0 to skip.

    Returns:
        Filtered, sorted DataFrame of Critical units.
    """
    if output_cols is None:
        output_cols = [
            "span_id", "line_id", "component",
            "installation_date", "maintenance_period_days",
            "HI_score_last", "HI_class_last",
            "RUL_days", "predicted_lifetime_days",
            "lightning_flash_density", "region",
        ]

    # Build deduplicated column list; RUL_days must be present for sorting
    all_needed = list(dict.fromkeys(
        [c for c in output_cols if c in rul_df.columns]
    ))
    if "RUL_days" not in all_needed:
        all_needed.append("RUL_days")

    priority = (
        rul_df[rul_df["risk_category"] == "Critical"][all_needed]
        .drop_duplicates()
        .sort_values("RUL_days")
        .reset_index(drop=True)
    )

    # Final column order = output_cols only (drop any extras added for sorting)
    final_cols = [c for c in output_cols if c in priority.columns]
    priority = priority[final_cols]

    # Save
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "priority_maintenance_list.csv"
    priority.to_csv(path, index=False)
    print(f"[aft] Saved → {path}  ({len(priority)} Critical units)")

    # Print top-N
    if print_top_n > 0:
        top = priority.head(print_top_n)
        header = (
            f"\n{'='*72}\n"
            f"  TOP {min(print_top_n, len(priority))} MOST URGENT UNITS FOR MAINTENANCE"
            f"  (of {len(priority)} Critical, RUL < 1 year)\n"
            f"{'='*72}"
        )
        print(header)
        print(top.to_string(index=False))
        print("=" * 72)

    return priority


def save_rul_predictions(
    rul_df: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    breakdown: pd.DataFrame | None = None,
) -> Path:
    """Save RUL prediction table (and optional summaries) to outputs/tables/.

    Args:
        rul_df: Output of predict_rul — full DataFrame with RUL columns.
        summary: Optional output of rul_summary_by_component.
        breakdown: Optional output of rul_risk_breakdown.

    Returns:
        Path to the main rul_predictions.csv.
    """
    TABLES.mkdir(parents=True, exist_ok=True)

    p_main = TABLES / "rul_predictions.csv"
    rul_df.to_csv(p_main, index=False)
    print(f"[aft] Saved → {p_main}")

    if summary is not None:
        p = TABLES / "rul_summary_by_component.csv"
        summary.to_csv(p)
        print(f"[aft] Saved → {p}")

    if breakdown is not None:
        p = TABLES / "rul_risk_breakdown.csv"
        breakdown.to_csv(p)
        print(f"[aft] Saved → {p}")

    return p_main
