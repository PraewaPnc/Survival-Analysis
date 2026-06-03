"""Cox Proportional Hazard model per component — Phase 6.

For each of the 6 transmission-line component types, a separate
CoxPHFitter is fitted with 7 covariates:

    lightning_flash_density, avg_wind_speed_ms, avg_humidity_pct,
    pm25_annual_avg, HI_score_last, encroachment_severity (ordinal),
    voltage_kv

Continuous covariates are z-score standardised using the *overall* dataset
statistics so that hazard ratios (HRs) are comparable across components
(1 unit = 1 SD of the full population).

L2 regularisation (penalizer=0.1) is applied throughout; this is especially
important for Conductor (82 events / 7 covariates ≈ 12 EPV).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


TABLES = Path(__file__).parent.parent / "outputs" / "tables"

# Ordinal encoding: none < minor < moderate < severe
ENCROACHMENT_ORDER: dict[str, int] = {
    "none": 0, "minor": 1, "moderate": 2, "severe": 3,
}

# Continuous covariates that will be z-score standardised
CONTINUOUS_COVARIATES: list[str] = [
    "lightning_flash_density",
    "avg_wind_speed_ms",
    "avg_humidity_pct",
    "pm25_annual_avg",
    "HI_score_last",
    "voltage_kv",
]

# All covariates entering the model (after encoding)
MODEL_COVARIATES: list[str] = CONTINUOUS_COVARIATES + ["encroachment_severity_enc"]

# Human-readable labels for plots
COVARIATE_LABELS: dict[str, str] = {
    "lightning_flash_density":  "Lightning Flash Density",
    "avg_wind_speed_ms":        "Avg Wind Speed (m/s)",
    "avg_humidity_pct":         "Avg Humidity (%)",
    "pm25_annual_avg":          "PM2.5 Annual Avg",
    "HI_score_last":            "Health Index Score",
    "voltage_kv":               "Voltage (kV)",
    "encroachment_severity_enc":"Encroachment Severity",
}

# Significance thresholds for annotation
_SIG_STARS: list[tuple[float, str]] = [
    (0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns"),
]


def _sig_stars(p: float) -> str:
    for threshold, label in _SIG_STARS:
        if p < threshold:
            return label
    return "ns"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_cox_data(
    df: pd.DataFrame,
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    ref_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Encode and standardise covariates for CoxPHFitter.

    Steps:
    1. Ordinal-encode ``encroachment_severity`` (none=0 … severe=3).
    2. Z-score standardise all continuous covariates using *ref_df* statistics
       (defaults to the passed DataFrame itself; pass the full dataset when
       fitting per-component subsets so the scale is globally comparable).
    3. Return a DataFrame containing only the model columns plus duration and
       event columns.

    Args:
        df: Validated DataFrame (one row per span × component).
        duration_col: Time-to-event column.
        event_col: Binary event indicator (1=failure).
        ref_df: Reference DataFrame whose mean/std are used for standardisation.
            Defaults to ``df`` when None.

    Returns:
        Clean DataFrame ready for CoxPHFitter.fit().
    """
    if ref_df is None:
        ref_df = df

    out = df[[duration_col, event_col] + CONTINUOUS_COVARIATES + ["encroachment_severity"]].copy()

    # Ordinal encoding
    out["encroachment_severity_enc"] = (
        out["encroachment_severity"]
        .map(ENCROACHMENT_ORDER)
        .astype(float)
    )
    out = out.drop(columns=["encroachment_severity"])

    # Z-score standardisation (using reference population statistics)
    for col in CONTINUOUS_COVARIATES:
        mu  = ref_df[col].mean()
        std = ref_df[col].std()
        out[col] = (out[col] - mu) / (std if std > 0 else 1.0)

    return out.dropna(subset=MODEL_COVARIATES)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_cox_by_component(
    df: pd.DataFrame,
    components: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    penalizer: float = 0.1,
) -> dict[str, CoxPHFitter]:
    """Fit a CoxPHFitter for each component.

    Args:
        df: Validated (raw, not yet prepared) DataFrame.
        components: Ordered list of component names.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.
        penalizer: L2 ridge regularisation strength. Larger values shrink
            coefficients toward zero; 0.1 stabilises low-EPV fits.

    Returns:
        Dict mapping component name → fitted CoxPHFitter.
    """
    # Prepare the full dataset once for globally-consistent standardisation
    full_prepared = prepare_cox_data(df, duration_col, event_col, ref_df=df)

    fitters: dict[str, CoxPHFitter] = {}
    for comp in components:
        sub = full_prepared[df["component"] == comp].reset_index(drop=True)
        n_ev = int(sub[event_col].sum())
        print(f"[cox] Fitting {comp}: n={len(sub)}, events={n_ev}, EPV={n_ev/len(MODEL_COVARIATES):.1f}")

        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(
                sub,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
            fitters[comp] = cph
        except Exception as exc:
            print(f"[cox] {comp} failed: {exc}")

    return fitters


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def cox_hr_table(
    fitters: dict[str, CoxPHFitter],
) -> pd.DataFrame:
    """Extract HR, 95% CI, z-statistic, and p-value per covariate × component.

    Args:
        fitters: Output of fit_cox_by_component.

    Returns:
        Long-format DataFrame with columns:
        [component, covariate, covariate_label, HR, HR_lower, HR_upper,
         coef, se, z, p, sig].
    """
    rows = []
    for comp, cph in fitters.items():
        s = cph.summary
        for cov in s.index:
            rows.append({
                "component":      comp,
                "covariate":      cov,
                "covariate_label": COVARIATE_LABELS.get(cov, cov),
                "HR":             round(float(s.loc[cov, "exp(coef)"]), 4),
                "HR_lower":       round(float(s.loc[cov, "exp(coef) lower 95%"]), 4),
                "HR_upper":       round(float(s.loc[cov, "exp(coef) upper 95%"]), 4),
                "coef":           round(float(s.loc[cov, "coef"]), 4),
                "se":             round(float(s.loc[cov, "se(coef)"]), 4),
                "z":              round(float(s.loc[cov, "z"]), 3),
                "p":              float(s.loc[cov, "p"]),
                "sig":            _sig_stars(float(s.loc[cov, "p"])),
            })
    return pd.DataFrame(rows)


def concordance_table(
    fitters: dict[str, CoxPHFitter],
) -> pd.DataFrame:
    """Return concordance index (C-index) per component.

    Args:
        fitters: Output of fit_cox_by_component.

    Returns:
        DataFrame with columns [component, concordance_index].
    """
    rows = [
        {"component": comp, "concordance_index": round(cph.concordance_index_, 4)}
        for comp, cph in fitters.items()
    ]
    return pd.DataFrame(rows).set_index("component")


def cox_wide_table(
    hr_table: pd.DataFrame,
    metric: str = "HR",
) -> pd.DataFrame:
    """Pivot HR table to wide format: rows=component, cols=covariate.

    Args:
        hr_table: Output of cox_hr_table.
        metric: One of 'HR', 'p', 'sig', 'HR_lower', 'HR_upper'.

    Returns:
        DataFrame (component × covariate_label) for the chosen metric.
    """
    return hr_table.pivot(index="component", columns="covariate_label", values=metric)


# ---------------------------------------------------------------------------
# Proportional hazards assumption check
# ---------------------------------------------------------------------------

def _component_prepared(
    df: pd.DataFrame,
    full_prepared: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """Return prepared rows for one component, reset to 0-based index.

    Uses the original df's boolean mask to select rows from full_prepared,
    ensuring correct alignment even if row order differs.
    """
    mask = (df["component"] == component).values
    return full_prepared[mask].reset_index(drop=True)


def check_ph_assumption(
    df: pd.DataFrame,
    fitters: dict[str, CoxPHFitter],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Schoenfeld residuals test for the proportional hazards assumption.

    For each component, runs `proportional_hazard_test` (rank-transformed
    event times) on the fitted CoxPHFitter.  A p-value < p_threshold
    indicates that the log-hazard ratio for that covariate may vary over
    time, violating the PH assumption.

    Args:
        df: Validated raw DataFrame (used to subset by component).
        fitters: Output of fit_cox_by_component.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.
        p_threshold: Significance threshold flagged as 'violated'.

    Returns:
        Long-format DataFrame with columns:
        [component, covariate, covariate_label, test_statistic, p, violated].
        Rows are sorted by component then p-value ascending.
    """
    from lifelines.statistics import proportional_hazard_test

    full_prep = prepare_cox_data(df, duration_col, event_col, ref_df=df)

    rows = []
    for comp, cph in fitters.items():
        sub = _component_prepared(df, full_prep, comp)
        try:
            result = proportional_hazard_test(cph, sub, time_transform="rank")
            for cov in result.summary.index:
                p = float(result.summary.loc[cov, "p"])
                rows.append({
                    "component":       comp,
                    "covariate":       cov,
                    "covariate_label": COVARIATE_LABELS.get(cov, cov),
                    "test_statistic":  round(float(result.summary.loc[cov, "test_statistic"]), 4),
                    "p":               round(p, 4),
                    "violated":        p < p_threshold,
                })
        except Exception as exc:
            print(f"[cox] PH test failed for {comp}: {exc}")

    out = pd.DataFrame(rows)
    return out.sort_values(["component", "p"]).reset_index(drop=True)


def compute_schoenfeld_residuals(
    df: pd.DataFrame,
    fitters: dict[str, CoxPHFitter],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> dict[str, pd.DataFrame]:
    """Compute scaled Schoenfeld residuals for each component.

    Schoenfeld residuals are defined only at observed failure times.
    Under the PH assumption they should have zero expected value and show
    no time trend.  A trend in the smoothed residuals vs. time indicates
    a time-varying effect (PH violation).

    Args:
        df: Validated raw DataFrame.
        fitters: Output of fit_cox_by_component.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.

    Returns:
        Dict: component → DataFrame indexed 0…n_events−1 with columns
        [event_time] + one column per covariate.
    """
    full_prep = prepare_cox_data(df, duration_col, event_col, ref_df=df)

    out: dict[str, pd.DataFrame] = {}
    for comp, cph in fitters.items():
        sub = _component_prepared(df, full_prep, comp)
        try:
            resid = cph.compute_residuals(sub, kind="schoenfeld")
            # resid.index = row positions in sub where event occurred
            event_times = sub.loc[resid.index, duration_col].values
            result = resid.copy().reset_index(drop=True)
            result.insert(0, "event_time", event_times)
            out[comp] = result
        except Exception as exc:
            print(f"[cox] Schoenfeld residuals failed for {comp}: {exc}")

    return out


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_cox_results(
    hr_table: pd.DataFrame,
    concordance: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save HR table and concordance index to outputs/tables/.

    Args:
        hr_table: Output of cox_hr_table.
        concordance: Output of concordance_table.

    Returns:
        Tuple (hr_table_path, concordance_path).
    """
    TABLES.mkdir(parents=True, exist_ok=True)

    p1 = TABLES / "cox_hazard_ratios.csv"
    p2 = TABLES / "cox_concordance.csv"

    hr_table.to_csv(p1, index=False)
    concordance.to_csv(p2)

    print(f"[cox] Saved → {p1}")
    print(f"[cox] Saved → {p2}")
    return p1, p2
