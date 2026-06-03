"""Data loading, validation, and exploratory statistics.

Phase 2 of the survival analysis pipeline:
  load_data → validate_data → censoring_report → summary_stats → save
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "transmission_line_maintenance_data.csv"
TABLES_DIR = Path(__file__).parent.parent / "outputs" / "tables"

COMPONENTS: list[str] = ["Conductor", "Damper", "Spacer", "Insulator", "Fittings", "Arrester"]
REGIONS: list[str] = ["Northeast", "North", "Central", "South"]

# Column groups used downstream
SURVIVAL_COLS: list[str] = [
    "maintenance_period_days", "event_occurred", "component", "failure_mode"
]
ENV_COLS: list[str] = [
    "lightning_flash_density", "avg_wind_speed_ms", "avg_humidity_pct",
    "pm25_annual_avg", "coastal_proximity", "pollution_severity",
]
HI_COLS: list[str] = ["HI_score_last", "HI_class_last", "HI_trend_per_year"]
ENCROACHMENT_COLS: list[str] = ["encroachment_severity", "vegetation_encroachment"]
OUTAGE_COLS: list[str] = ["total_outage_events", "fault_type_most_common", "permanent_faults"]

# Numeric covariates for correlation / covariate summary
NUMERIC_COV_COLS: list[str] = [
    "lightning_flash_density", "avg_wind_speed_ms", "avg_humidity_pct",
    "pm25_annual_avg", "HI_score_last", "HI_trend_per_year",
    "total_outage_events", "permanent_faults",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and minimally parse the maintenance CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with installation_date and maintenance_date parsed as
        datetime, all other columns left as-is.
    """
    df = pd.read_csv(
        path,
        parse_dates=["installation_date", "maintenance_date"],
    )
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate critical columns and return a cleaned copy.

    Checks performed:
    - Required columns present.
    - `event_occurred` is strictly 0/1.
    - `maintenance_period_days` is positive (rows with <= 0 are dropped).
    - `failure_mode` is null only for censored rows (warning if violated).

    Args:
        df: Raw DataFrame from load_data.

    Returns:
        Cleaned DataFrame. Rows with non-positive durations are removed.

    Raises:
        ValueError: If required columns are missing or event column is not binary.
    """
    required = ["maintenance_period_days", "event_occurred", "component", "region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    event_vals = set(df["event_occurred"].dropna().unique())
    if not event_vals.issubset({0, 1}):
        raise ValueError(f"event_occurred has unexpected values: {event_vals - {0, 1}}")

    df = df.copy()

    # Drop non-positive durations
    bad = df["maintenance_period_days"] <= 0
    if bad.any():
        print(f"[validate] Dropping {bad.sum()} row(s) with duration <= 0.")
        df = df[~bad].reset_index(drop=True)

    # Warn if censored rows have a non-null failure_mode
    if "failure_mode" in df.columns:
        inconsistent = (df["event_occurred"] == 0) & df["failure_mode"].notna()
        if inconsistent.any():
            print(
                f"[validate] Warning: {inconsistent.sum()} censored rows "
                "have a non-null failure_mode — these are kept as-is."
            )

    return df


# ---------------------------------------------------------------------------
# Censoring report
# ---------------------------------------------------------------------------

def censoring_report(df: pd.DataFrame) -> pd.DataFrame:
    """Censoring rate per component, per region, and overall.

    Args:
        df: Validated DataFrame.

    Returns:
        DataFrame indexed by (group_type, group_label) with columns:
        [n, n_events, n_censored, censoring_rate_pct, min_days, max_days,
         median_days, mean_days].
    """
    def _stats(sub: pd.DataFrame, group_type: str, label: str) -> dict:
        n = len(sub)
        n_ev = int(sub["event_occurred"].sum())
        n_cx = n - n_ev
        dur = sub["maintenance_period_days"]
        return {
            "group_type": group_type,
            "group_label": label,
            "n": n,
            "n_events": n_ev,
            "n_censored": n_cx,
            "censoring_rate_pct": round(100 * n_cx / n, 1) if n else 0.0,
            "min_days": int(dur.min()),
            "max_days": int(dur.max()),
            "median_days": dur.median(),
            "mean_days": round(dur.mean(), 1),
        }

    rows = [_stats(df, "Overall", "Overall")]
    for c in COMPONENTS:
        rows.append(_stats(df[df["component"] == c], "Component", c))
    for r in REGIONS:
        rows.append(_stats(df[df["region"] == r], "Region", r))

    report = pd.DataFrame(rows).set_index(["group_type", "group_label"])
    return report


# ---------------------------------------------------------------------------
# Summary stats (to be saved as data_summary.csv)
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Primary summary table matching Yang et al. Table 1 style.

    Columns: N, n_events, n_censored, censoring_rate, event_rate,
    min_days, median_days, mean_days, max_days, std_days.
    Rows: Overall + one per component.

    Args:
        df: Validated DataFrame.

    Returns:
        DataFrame indexed by component label (Overall + 6 components).
    """
    rows = []
    groups = [("Overall", df)] + [(c, df[df["component"] == c]) for c in COMPONENTS]

    for label, grp in groups:
        n = len(grp)
        n_ev = int(grp["event_occurred"].sum())
        n_cx = n - n_ev
        dur = grp["maintenance_period_days"]
        rows.append({
            "group": label,
            "N": n,
            "n_events": n_ev,
            "n_censored": n_cx,
            "censoring_rate": round(n_cx / n, 4) if n else 0.0,
            "event_rate": round(n_ev / n, 4) if n else 0.0,
            "min_days": int(dur.min()),
            "median_days": dur.median(),
            "mean_days": round(dur.mean(), 1),
            "max_days": int(dur.max()),
            "std_days": round(dur.std(), 1),
        })

    return pd.DataFrame(rows).set_index("group")


def save_summary(table: pd.DataFrame, name: str = "data_summary") -> Path:
    """Save summary table to outputs/tables/data_summary.csv.

    Args:
        table: DataFrame from summary_stats.
        name: Filename stem.

    Returns:
        Path to saved CSV.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    table.to_csv(path)
    print(f"[preprocessing] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Failure mode breakdown
# ---------------------------------------------------------------------------

def failure_mode_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count failures by component × failure_mode.

    Args:
        df: Validated DataFrame.

    Returns:
        Pivot table (component × failure_mode) with counts,
        plus a 'Total' column.
    """
    events = df[df["event_occurred"] == 1].copy()
    events["failure_mode"] = events["failure_mode"].fillna("Unknown")
    pivot = (
        events.groupby(["component", "failure_mode"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    return pivot


# ---------------------------------------------------------------------------
# Covariate summary (events vs. censored)
# ---------------------------------------------------------------------------

def covariate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric covariates split by event status.

    Args:
        df: Validated DataFrame.

    Returns:
        DataFrame with columns [event_mean, event_std, censored_mean,
        censored_std, diff] indexed by covariate name.
    """
    cols = [c for c in NUMERIC_COV_COLS if c in df.columns]
    rows = []
    for col in cols:
        ev = df.loc[df["event_occurred"] == 1, col].dropna()
        cx = df.loc[df["event_occurred"] == 0, col].dropna()
        ev_mean = round(ev.mean(), 3)
        cx_mean = round(cx.mean(), 3)
        rows.append({
            "covariate": col,
            "event_mean": ev_mean,
            "event_std": round(ev.std(), 3),
            "censored_mean": cx_mean,
            "censored_std": round(cx.std(), 3),
            "diff (event−censored)": round(ev_mean - cx_mean, 3),
        })
    return pd.DataFrame(rows).set_index("covariate")


# ---------------------------------------------------------------------------
# Convenience: run full Phase 2
# ---------------------------------------------------------------------------

def run_phase2(
    path: Path = DATA_PATH,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load → validate → report → save summary CSV.

    Args:
        path: Path to the raw CSV.
        verbose: Print intermediate reports to stdout.

    Returns:
        Tuple of (validated_df, summary_table).
    """
    raw = load_data(path)
    df = validate_data(raw)

    cr = censoring_report(df)
    if verbose:
        print("\n=== Censoring Report ===")
        print(cr.to_string())

    stats = summary_stats(df)
    if verbose:
        print("\n=== Summary Stats ===")
        print(stats.to_string())

    save_summary(stats, "data_summary")

    fm = failure_mode_breakdown(df)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fm.to_csv(TABLES_DIR / "failure_mode_breakdown.csv")
    if verbose:
        print("\n=== Failure Mode Breakdown ===")
        print(fm.to_string())

    return df, stats
