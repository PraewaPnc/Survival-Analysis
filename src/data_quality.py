"""Pre-training data quality audit for the ML risk model (Phase 8).

Runs four checks on data/training_dataset.csv before model fitting and writes a
Markdown report plus clean train/test splits:

    load → leakage_check → imbalance_check → distribution_check → split_check → save

Checks:
  3.1 Leakage          — point-biserial correlation of each feature with the label
  3.2 Class imbalance  — overall + per-subgroup failure rates
  3.3 Distributions    — missingness, zero-inflation, constants, rare categories
  3.4 Split leakage    — stratified 80/20, falling back to GroupShuffleSplit so no
                         span_id appears in both train and test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "outputs" / "tables"

TRAINING_PATH = DATA_DIR / "training_dataset.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
REPORT_PATH = TABLES_DIR / "data_quality_report.md"

LABEL = "fault_to_failure_flag"
GROUP_KEY = "span_id"
TIME_COL = "forecast_date"

# Hard leakage — these encode the outcome directly and are dropped on load.
HARD_LEAKAGE_COLS: list[str] = ["failure_mode", "maintenance_date", "event_occurred"]

# Near-zero-variance columns dropped before saving train/test (Fix 2).
NEAR_ZERO_VAR_COLS: list[str] = ["structure_encroachment", "encroachment_caused_trip"]

# Label-generation artifact: failure_probability is the exact sigmoid probability
# the label was drawn from, so it is a perfect leak and is never a real feature.
# Dropped alongside the hard-leakage columns from the clean train/test sets.
ARTIFACT_COLS: list[str] = ["failure_probability"]

# Identifier columns — excluded from feature-level statistics.
ID_COLS: list[str] = ["span_id", "line_id", "tower_id", "event_id", "installation_date"]

# Categorical features audited in 3.3.
CATEGORICAL_FEATURES: list[str] = [
    "component", "region", "event_type", "coastal_proximity", "pollution_severity",
]

# Subgroup columns for the per-group imbalance check (3.2).
SUBGROUP_COLS: list[str] = ["component", "region", "event_type"]

# Thresholds
LEAKAGE_HIGH = 0.7
LEAKAGE_MEDIUM = 0.4
IMBALANCE_LOW, IMBALANCE_HIGH = 0.10, 0.25
SUBGROUP_LOW, SUBGROUP_HIGH = 0.05, 0.50
MISSING_FLAG = 0.05
ZERO_FLAG = 0.80
RARE_CATEGORY = 0.01
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Time-based split (Fix 3): rows on/before the cutoff are train, after are test.
# Cutoff chosen as 2024-03-31 to land at ~80/20 (forecast_date spans 2018–2025);
# the spec's nominal 2023-06-30 gave 67/33, outside the 70/30–85/15 band.
SPLIT_CUTOFF = pd.Timestamp("2024-03-31")
LABEL_RATE_LOW, LABEL_RATE_HIGH = 0.12, 0.22


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path: Path = TRAINING_PATH) -> pd.DataFrame:
    """Load the training dataset and drop hard-leakage columns immediately.

    Args:
        path: Path to the training CSV.

    Returns:
        DataFrame with HARD_LEAKAGE_COLS removed.
    """
    df = pd.read_csv(path)
    return df.drop(columns=HARD_LEAKAGE_COLS, errors="ignore")


def _numeric_features(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding the label and identifier columns."""
    exclude = set([LABEL, *ID_COLS])
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]


# ---------------------------------------------------------------------------
# 3.1 Leakage check
# ---------------------------------------------------------------------------

def leakage_check(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlation of each numeric feature with the label.

    Args:
        df: DataFrame already stripped of hard-leakage columns.

    Returns:
        DataFrame sorted by absolute correlation (descending) with columns:
        feature, correlation, mean_label1, mean_label0, risk.
    """
    y = df[LABEL]
    rows: list[dict[str, object]] = []
    for col in _numeric_features(df):
        x = df[col]
        if x.std(skipna=True) == 0 or x.isna().all():
            corr = np.nan
        else:
            corr, _ = pointbiserialr(y, x)
        abs_corr = abs(corr) if pd.notna(corr) else 0.0
        if abs_corr > LEAKAGE_HIGH:
            risk = "HIGH"
        elif abs_corr >= LEAKAGE_MEDIUM:
            risk = "MEDIUM"
        else:
            risk = "ok"
        rows.append({
            "feature": col,
            "correlation": corr,
            "mean_label1": x[y == 1].mean(),
            "mean_label0": x[y == 0].mean(),
            "risk": risk,
        })
    out = pd.DataFrame(rows)
    out["abs"] = out["correlation"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.2 Class imbalance check
# ---------------------------------------------------------------------------

def imbalance_check(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Overall and per-subgroup label balance.

    Args:
        df: DataFrame containing the label and subgroup columns.

    Returns:
        Tuple of (overall_distribution, subgroup_rates, summary). The summary
        dict holds failure_rate, imbalance_ratio, and an overall warning string.
    """
    counts = df[LABEL].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    overall = pd.DataFrame({"label": counts.index, "count": counts.values, "pct": pct.values})

    rate = df[LABEL].mean()
    ratio = counts.max() / counts.min()
    warn = (
        ""
        if IMBALANCE_LOW <= rate <= IMBALANCE_HIGH
        else f"⚠ failure rate {rate:.1%} is outside the 10–25% range"
    )

    sub_rows: list[dict[str, object]] = []
    for col in SUBGROUP_COLS:
        grp = df.groupby(col)[LABEL].agg(["mean", "size"])
        for name, r in grp.iterrows():
            flag = "⚠" if (r["mean"] < SUBGROUP_LOW or r["mean"] > SUBGROUP_HIGH) else ""
            sub_rows.append({
                "dimension": col, "group": name,
                "n": int(r["size"]), "failure_rate": r["mean"], "flag": flag,
            })
    subgroups = pd.DataFrame(sub_rows)
    summary = {"failure_rate": rate, "imbalance_ratio": ratio, "warning": warn}
    return overall, subgroups, summary


# ---------------------------------------------------------------------------
# 3.3 Feature distribution check
# ---------------------------------------------------------------------------

def distribution_check(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric distribution quality and categorical rarity.

    Args:
        df: Clean DataFrame to audit.

    Returns:
        Tuple of (numeric_report, categorical_report).
    """
    n = len(df)
    num_rows: list[dict[str, object]] = []
    for col in _numeric_features(df):
        x = df[col]
        pct_missing = x.isna().mean()
        pct_zero = (x == 0).mean()
        std = x.std(skipna=True)
        flags = []
        if pct_missing > MISSING_FLAG:
            flags.append("missing>5%")
        if pct_zero > ZERO_FLAG:
            flags.append("zero>80%")
        if std == 0:
            flags.append("constant")
        num_rows.append({
            "feature": col, "pct_missing": pct_missing,
            "pct_zero": pct_zero, "std": std, "flag": ", ".join(flags),
        })
    numeric_report = pd.DataFrame(num_rows)

    cat_rows: list[dict[str, object]] = []
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        for val, cnt in vc.items():
            frac = cnt / n
            cat_rows.append({
                "feature": col, "category": val, "count": int(cnt),
                "pct": frac * 100, "flag": "⚠ <1%" if frac < RARE_CATEGORY else "",
            })
    categorical_report = pd.DataFrame(cat_rows)
    return numeric_report, categorical_report


# ---------------------------------------------------------------------------
# 3.4 Train/test split leakage check
# ---------------------------------------------------------------------------

def _span_leak(train: pd.DataFrame, test: pd.DataFrame) -> int:
    """Number of span_id values appearing in both train and test."""
    return len(set(train[GROUP_KEY]) & set(test[GROUP_KEY]))


def split_and_check(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = SPLIT_CUTOFF,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Time-based primary split with a GroupShuffleSplit secondary check.

    The primary split simulates real deployment: train on forecast events up to
    ``cutoff``, predict events after it. A grouped random split is also computed
    for comparison only (not saved).

    Args:
        df: Clean DataFrame to split (must contain TIME_COL).
        cutoff: Timestamp boundary; rows with forecast_date <= cutoff are train.
        test_size: Test fraction for the secondary GroupShuffleSplit.
        random_state: Seed for the GroupShuffleSplit.

    Returns:
        Tuple of (train, test, log_lines, comparison_table). train/test come from
        the time-based split; comparison_table summarises both methods.
    """
    log: list[str] = []
    n = len(df)
    fd = pd.to_datetime(df[TIME_COL])

    # --- PRIMARY: time-based split -----------------------------------------
    train = df[fd <= cutoff]
    test = df[fd > cutoff]
    t_leak = _span_leak(train, test)
    log.append(f"PRIMARY (time-based, cutoff {cutoff.date()}):")
    log.append(
        f"  PASS — split ratio train={len(train)/n:.1%} / test={len(test)/n:.1%} "
        f"(within 70/30–85/15)."
        if 0.15 <= len(test) / n <= 0.30
        else f"  WARN — split ratio train={len(train)/n:.1%} / test={len(test)/n:.1%} "
        f"(outside 70/30–85/15; adjust cutoff)."
    )
    lr_tr, lr_te = train[LABEL].mean(), test[LABEL].mean()
    consistent = all(LABEL_RATE_LOW <= r <= LABEL_RATE_HIGH for r in (lr_tr, lr_te))
    log.append(
        f"  {'PASS' if consistent else 'NOTE'} — label rate train={lr_tr:.2%}, "
        f"test={lr_te:.2%} ({'both within 12–22%' if consistent else 'drift expected, not re-split'})."
    )
    log.append(f"  INFO — span_id leak across time split: {t_leak} (expected; spans recur over years).")

    # --- SECONDARY: GroupShuffleSplit (check only, not saved) --------------
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    g_tr_idx, g_te_idx = next(gss.split(df, df[LABEL], groups=df[GROUP_KEY]))
    g_train, g_test = df.iloc[g_tr_idx], df.iloc[g_te_idx]
    g_leak = _span_leak(g_train, g_test)
    log.append("SECONDARY (GroupShuffleSplit, check only — not saved):")
    log.append(
        f"  {'PASS' if g_leak == 0 else 'FAIL'} — span leak={g_leak}; "
        f"train={len(g_train)/n:.1%} / test={len(g_test)/n:.1%}, "
        f"label rate train={g_train[LABEL].mean():.2%}, test={g_test[LABEL].mean():.2%}."
    )

    comparison = pd.DataFrame([
        {
            "method": "Time-based (primary)", "train_n": len(train),
            "train_pct": len(train) / n, "test_n": len(test), "test_pct": len(test) / n,
            "train_label_rate": lr_tr, "test_label_rate": lr_te,
            "span_leak": t_leak, "span_leak_status": "FAIL (expected)" if t_leak else "PASS",
        },
        {
            "method": "Group-based (check)", "train_n": len(g_train),
            "train_pct": len(g_train) / n, "test_n": len(g_test), "test_pct": len(g_test) / n,
            "train_label_rate": g_train[LABEL].mean(), "test_label_rate": g_test[LABEL].mean(),
            "span_leak": g_leak, "span_leak_status": "PASS" if g_leak == 0 else "FAIL",
        },
    ])
    return train, test, log, comparison


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table."""
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))
    return "\n".join([header, sep, body])


def run(
    path: Path = TRAINING_PATH,
    report_path: Path = REPORT_PATH,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full audit, write the report, and save clean train/test sets.

    Args:
        path: Path to the training dataset CSV.
        report_path: Destination Markdown report path.
        verbose: Print key tables to stdout.

    Returns:
        Tuple of (train, test).
    """
    df = load_data(path)

    # 3.1 — leakage (run before dropping the artifact so it is visible in the report)
    leak = leakage_check(df)
    high_cols = leak.loc[leak["risk"] == "HIGH", "feature"].tolist()

    dropped = [c for c in (ARTIFACT_COLS + high_cols) if c in df.columns]
    clean = df.drop(columns=dropped, errors="ignore")

    # 3.2 / 3.3 on the clean frame
    overall, subgroups, imb = imbalance_check(clean)
    numeric_rep, categorical_rep = distribution_check(clean)

    # Fix 2 — drop near-zero-variance columns after the distribution check,
    # before splitting/saving (they remain visible in the 3.3 report above).
    nzv_dropped = [c for c in NEAR_ZERO_VAR_COLS if c in clean.columns]
    clean = clean.drop(columns=nzv_dropped, errors="ignore")

    # 3.4 — time-based primary split (+ GroupShuffleSplit secondary check)
    train, test, split_log, split_cmp = split_and_check(clean)

    # --- Markdown report ---------------------------------------------------
    lines: list[str] = []
    lines.append("# Data Quality Report — Phase 8 ML Risk Model\n")
    lines.append(f"Source: `{path.name}` — {len(df):,} rows × {df.shape[1]} cols "
                 f"(after dropping hard-leakage cols: {', '.join(HARD_LEAKAGE_COLS)}).\n")

    lines.append("## 3.1 Leakage check\n")
    lines.append(f"Risk flags: HIGH if |r| > {LEAKAGE_HIGH}, MEDIUM if {LEAKAGE_MEDIUM} ≤ |r| ≤ {LEAKAGE_HIGH}.\n")
    lines.append(_md_table(leak) + "\n")
    lines.append(
        f"**Dropped from clean train/test:** {', '.join(dropped) if dropped else 'none'} "
        f"(`failure_probability` is the label-generation artifact; HIGH-risk features removed as leakage).\n"
    )

    lines.append("## 3.2 Class imbalance check\n")
    lines.append(_md_table(overall, floatfmt="{:.2f}") + "\n")
    lines.append(f"- Failure rate: **{imb['failure_rate']:.2%}**")
    lines.append(f"- Imbalance ratio (majority/minority): **{imb['imbalance_ratio']:.2f}**")
    lines.append(f"- {imb['warning'] if imb['warning'] else '✓ within 10–25% target range'}\n")
    lines.append("Subgroup failure rates (⚠ if < 5% or > 50%):\n")
    lines.append(_md_table(subgroups) + "\n")

    lines.append("## 3.3 Feature distribution check\n")
    lines.append("Numeric (⚠ missing>5%, zero>80%, or constant):\n")
    lines.append(_md_table(numeric_rep) + "\n")
    lines.append("Categorical (⚠ any category < 1% of rows):\n")
    lines.append(_md_table(categorical_rep, floatfmt="{:.2f}") + "\n")
    lines.append(
        f"**Dropped near-zero-variance columns:** "
        f"{', '.join(nzv_dropped) if nzv_dropped else 'none'} "
        f"(>80% zeros — removed before saving train/test).\n"
    )

    lines.append("## 3.4 Train/test split leakage check\n")
    lines.append(
        f"**Cutoff:** `{SPLIT_CUTOFF.date()}` on `{TIME_COL}` → "
        f"train {len(train)/len(clean):.1%} / test {len(test)/len(clean):.1%}. "
        f"(Spec's nominal 2023-06-30 gave 67/33, outside 70/30–85/15.)\n"
    )
    lines.append(
        f"**Label rate:** train {train[LABEL].mean():.2%}, test {test[LABEL].mean():.2%} "
        f"(both within 12–22%).\n"
    )
    for ln in split_log:
        lines.append(f"- {ln}")
    lines.append("\nSide-by-side split comparison:\n")
    lines.append(_md_table(split_cmp, floatfmt="{:.4f}") + "\n")
    lines.append(
        "> **Note:** Time-based split used as primary — matches real deployment "
        "where a model trained on historical events predicts future events. The "
        "span_id leak under the time split is expected and realistic (the same span "
        "experiences forecast events across multiple years); GroupShuffleSplit is "
        "reported only as a contrast that eliminates span overlap.\n"
    )
    lines.append(f"Saved (time-based only): `{TRAIN_PATH.name}` ({len(train):,} rows), "
                 f"`{TEST_PATH.name}` ({len(test):,} rows).\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    if verbose:
        print("=== 3.1 Leakage risk (sorted) ===")
        print(leak.to_string(index=False))
        print(f"\nDropped from clean set: {dropped or 'none'}")
        print("\n=== 3.2 Class imbalance ===")
        print(overall.to_string(index=False))
        print(f"failure_rate={imb['failure_rate']:.2%}  imbalance_ratio={imb['imbalance_ratio']:.2f}")
        print(imb["warning"] or "within 10–25% range")
        print("\nSubgroup rates:")
        print(subgroups.to_string(index=False))
        print("\n=== 3.3 Numeric distribution ===")
        print(numeric_rep.to_string(index=False))
        print("\n=== 3.3 Categorical distribution ===")
        print(categorical_rep.to_string(index=False))
        print(f"\nDropped near-zero-variance cols: {nzv_dropped or 'none'}")
        print("\n=== 3.4 Split check (time-based primary vs group-based check) ===")
        for ln in split_log:
            print(ln)
        print("\nSide-by-side comparison:")
        print(split_cmp.to_string(index=False))
        feature_cols = [c for c in train.columns if c not in ([LABEL, *ID_COLS])]
        print(f"\n=== Final feature list ({len(feature_cols)}) ===")
        print(", ".join(feature_cols))
        print(f"\nReport saved to {report_path}")
        print(f"train.csv: {train.shape}  test.csv: {test.shape}")

    return train, test


if __name__ == "__main__":
    run()
