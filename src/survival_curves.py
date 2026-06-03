"""Kaplan-Meier, Nelson-Aalen, and log-rank tests for survival analysis.

Phase 3 of the pipeline. Fitting functions live here; all plots live in
visualization.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test


TABLES = Path(__file__).parent.parent / "outputs" / "tables"

# Canonical time points (days) used for KM summary tables and at-risk ticks
AT_RISK_TIMES: list[int] = [365, 730, 1095, 1460, 1825, 2190]


# ---------------------------------------------------------------------------
# Kaplan-Meier fitting
# ---------------------------------------------------------------------------

def fit_km_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> dict[str, KaplanMeierFitter]:
    """Fit a KaplanMeierFitter for each group.

    Args:
        df: Validated DataFrame.
        group_col: Column name to group by (e.g. 'component').
        groups: Ordered list of group values to iterate over.
        duration_col: Time-to-event column.
        event_col: Event indicator column (1=failure).

    Returns:
        Dict mapping group label → fitted KaplanMeierFitter. Order matches
        the input ``groups`` list.
    """
    fitters: dict[str, KaplanMeierFitter] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        kmf = KaplanMeierFitter(label=g)
        kmf.fit(sub[duration_col], sub[event_col])
        fitters[g] = kmf
    return fitters


# ---------------------------------------------------------------------------
# Nelson-Aalen fitting
# ---------------------------------------------------------------------------

def fit_na_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> dict[str, NelsonAalenFitter]:
    """Fit a NelsonAalenFitter for each group.

    Args:
        df: Validated DataFrame.
        group_col: Column name to group by.
        groups: Ordered list of group values.
        duration_col: Time-to-event column.
        event_col: Event indicator column.

    Returns:
        Dict mapping group label → fitted NelsonAalenFitter.
    """
    fitters: dict[str, NelsonAalenFitter] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        naf = NelsonAalenFitter(label=g)
        naf.fit(sub[duration_col], sub[event_col])
        fitters[g] = naf
    return fitters


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def km_summary_table(
    fitters: dict[str, KaplanMeierFitter],
    time_points: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Extract S(t) and its 95% CI at specified time points for each group.

    Args:
        fitters: Dict from fit_km_by_group.
        time_points: Days at which to evaluate. Defaults to AT_RISK_TIMES.

    Returns:
        DataFrame with groups as rows. Columns: S(t=Xd), S_lower, S_upper,
        and median_survival_days.
    """
    if time_points is None:
        time_points = AT_RISK_TIMES

    rows: dict[str, dict] = {}
    for label, kmf in fitters.items():
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        row: dict = {}
        for t in time_points:
            idx = sf.index[sf.index <= t]
            if len(idx):
                t_last = idx[-1]
                row[f"S(t={t}d)"] = round(sf.loc[t_last].iloc[0], 4)
                row[f"S_lower(t={t}d)"] = round(ci.loc[t_last].iloc[0], 4)
                row[f"S_upper(t={t}d)"] = round(ci.loc[t_last].iloc[1], 4)
            else:
                row[f"S(t={t}d)"] = np.nan
                row[f"S_lower(t={t}d)"] = np.nan
                row[f"S_upper(t={t}d)"] = np.nan
        row["median_survival_days"] = kmf.median_survival_time_
        rows[label] = row
    return pd.DataFrame(rows).T


def save_km_table(table: pd.DataFrame, name: str) -> Path:
    """Save KM summary table to outputs/tables/.

    Args:
        table: DataFrame from km_summary_table.
        name: Filename stem (without extension).

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    table.to_csv(path)
    print(f"[survival_curves] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Nelson-Aalen summary table
# ---------------------------------------------------------------------------

def na_summary_table(
    fitters: dict[str, NelsonAalenFitter],
    time_points: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Extract cumulative hazard H(t) and 95% CI at key time points.

    Args:
        fitters: Dict from fit_na_by_group.
        time_points: Days at which to evaluate. Defaults to AT_RISK_TIMES.

    Returns:
        DataFrame indexed by group with columns H(t=Xd), H_lower(t=Xd),
        H_upper(t=Xd) for each time point.
    """
    if time_points is None:
        time_points = AT_RISK_TIMES

    rows: dict[str, dict] = {}
    for label, naf in fitters.items():
        ch = naf.cumulative_hazard_
        ci = naf.confidence_interval_cumulative_hazard_
        row: dict = {}
        for t in time_points:
            idx = ch.index[ch.index <= t]
            if len(idx):
                t_last = idx[-1]
                row[f"H(t={t}d)"]       = round(float(ch.loc[t_last].iloc[0]), 6)
                row[f"H_lower(t={t}d)"] = round(float(ci.loc[t_last].iloc[0]), 6)
                row[f"H_upper(t={t}d)"] = round(float(ci.loc[t_last].iloc[1]), 6)
            else:
                row[f"H(t={t}d)"]       = np.nan
                row[f"H_lower(t={t}d)"] = np.nan
                row[f"H_upper(t={t}d)"] = np.nan
        rows[label] = row

    return pd.DataFrame(rows).T


def save_na_table(table: pd.DataFrame, name: str = "na_cumulative_hazard") -> Path:
    """Save Nelson-Aalen summary table to outputs/tables/.

    Args:
        table: Output of na_summary_table.
        name: Filename stem.

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    table.to_csv(path)
    print(f"[survival_curves] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Log-rank tests
# ---------------------------------------------------------------------------

def log_rank_test(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> tuple[float, float]:
    """Multivariate log-rank test: H₀ = all survival curves are identical.

    Args:
        df: Validated DataFrame.
        group_col: Column whose unique values define the groups.
        duration_col: Time-to-event column.
        event_col: Binary event indicator (1=failure).

    Returns:
        Tuple (test_statistic, p_value).
    """
    result = multivariate_logrank_test(
        df[duration_col], df[group_col], df[event_col]
    )
    return float(result.test_statistic), float(result.p_value)


def pairwise_log_rank(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> pd.DataFrame:
    """Pairwise log-rank tests between every pair of groups.

    Args:
        df: Validated DataFrame.
        group_col: Column whose unique values define the groups.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.

    Returns:
        DataFrame from lifelines pairwise_logrank_test with columns
        including 'test_statistic' and 'p'.
    """
    result = pairwise_logrank_test(
        df[duration_col], df[group_col], df[event_col]
    )
    return result.summary


def save_logrank_results(
    df: pd.DataFrame,
    group_col: str,
    name: str,
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> Path:
    """Run pairwise log-rank and save results to outputs/tables/.

    Args:
        df: Validated DataFrame.
        group_col: Column to stratify by.
        name: Output filename stem.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.

    Returns:
        Path to saved CSV.
    """
    summary = pairwise_log_rank(df, group_col, duration_col, event_col)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    summary.to_csv(path)
    print(f"[survival_curves] Saved → {path}")
    return path
