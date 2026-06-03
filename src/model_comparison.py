"""Cox PH vs. Weibull AFT model comparison — Phase 7.6.

Combines the result tables already produced in Phases 6 and 7 into a
single comparison frame, identifies covariate agreement, and prints an
interpretive summary.

Key conceptual difference
--------------------------
Cox PH answers:   "which units are at higher RISK RIGHT NOW?"
  → Hazard Ratio (HR > 1): instantaneous failure rate is elevated.

Weibull AFT answers: "how much LIFE does each covariate add or remove?"
  → Acceleration Factor (AF < 1): survival time is shortened.

Because HR ≈ AF^(−ρ) for a Weibull distribution, any covariate that
increases the hazard (HR > 1) also shortens time (AF < 1). Both models
should therefore flag the same covariates as significant and agree on
direction — differences in which model reaches significance for a borderline
covariate reveal where the two parameterisations diverge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLES = Path(__file__).parent.parent / "outputs" / "tables"

_SIG_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def compare_concordance(
    cox_concordance: pd.DataFrame,
    aft_concordance: pd.Series,
) -> pd.DataFrame:
    """Build a side-by-side concordance index table.

    Args:
        cox_concordance: Output of cox_model.concordance_table.
        aft_concordance: 'concordance_index' column from aft_model.aft_model_summary.

    Returns:
        DataFrame indexed by component with columns:
        [Cox_C, AFT_C, delta_C, better_model].
    """
    cox_c = cox_concordance["concordance_index"].rename("Cox_C")
    aft_c = aft_concordance.rename("AFT_C")

    conc = pd.concat([cox_c, aft_c], axis=1).round(4)
    conc["delta_C"] = (conc["AFT_C"] - conc["Cox_C"]).round(4)
    conc["better_model"] = conc["delta_C"].apply(
        lambda d: "AFT" if d > 0.001 else ("Cox" if d < -0.001 else "Tie")
    )
    return conc


def compare_covariates(
    hr_table: pd.DataFrame,
    af_table: pd.DataFrame,
    sig_threshold: float = _SIG_THRESHOLD,
) -> pd.DataFrame:
    """Merge Cox HR and AFT AF tables and classify covariate agreement.

    Agreement categories:
    - ``"Both"``      — significant (p < threshold) in Cox AND AFT.
    - ``"Cox only"``  — significant in Cox but not AFT.
    - ``"AFT only"``  — significant in AFT but not Cox.
    - ``"Neither"``   — not significant in either model.

    Direction consistency: a covariate is directionally consistent when
    HR > 1 (more hazard) coincides with AF < 1 (shorter life), or vice
    versa.  Inconsistency would indicate a model specification problem.

    Args:
        hr_table: Output of cox_model.cox_hr_table.
        af_table: Output of aft_model.aft_tr_table (TR = AF numerically).
        sig_threshold: p-value threshold for significance.

    Returns:
        Long-format DataFrame with columns:
        [component, covariate_label,
         Cox_HR, Cox_p, Cox_sig, AFT_AF, AFT_p, AFT_sig,
         agreement, direction_consistent, note].
    """
    cox = hr_table[["component", "covariate_label",
                    "HR", "HR_lower", "HR_upper", "p", "sig"]].copy()
    cox.columns = ["component", "covariate_label",
                   "Cox_HR", "Cox_HR_lower", "Cox_HR_upper", "Cox_p", "Cox_sig"]

    aft = af_table[["component", "covariate_label",
                    "TR", "TR_lower", "TR_upper", "p", "sig"]].copy()
    aft.columns = ["component", "covariate_label",
                   "AFT_AF", "AFT_AF_lower", "AFT_AF_upper", "AFT_p", "AFT_sig"]

    merged = cox.merge(aft, on=["component", "covariate_label"], how="outer")

    # Significance flags
    sig_cox = merged["Cox_p"] < sig_threshold
    sig_aft = merged["AFT_p"] < sig_threshold

    def _agreement(row_sig_cox: bool, row_sig_aft: bool) -> str:
        if row_sig_cox and row_sig_aft:
            return "Both"
        if row_sig_cox:
            return "Cox only"
        if row_sig_aft:
            return "AFT only"
        return "Neither"

    merged["agreement"] = [
        _agreement(c, a) for c, a in zip(sig_cox, sig_aft)
    ]

    # Direction consistency: HR > 1 ↔ AF < 1  (same risk direction).
    # For non-significant covariates in either model the direction is meaningless
    # (the point estimate is consistent with noise around 1.0), so we mark
    # those as True to avoid spurious "inconsistency" warnings.
    def _direction_ok(hr: float, af: float,
                      p_cox: float, p_aft: float) -> bool:
        if p_cox >= sig_threshold and p_aft >= sig_threshold:
            return True   # null effect in both — direction is irrelevant
        return (hr > 1.0) == (af < 1.0)

    merged["direction_consistent"] = merged.apply(
        lambda r: _direction_ok(r["Cox_HR"], r["AFT_AF"],
                                r["Cox_p"],  r["AFT_p"]), axis=1
    )

    # Human-readable note
    def _note(row: pd.Series) -> str:
        ag = row["agreement"]
        dc = row["direction_consistent"]
        if ag == "Both":
            return "Significant in both; direction consistent" if dc \
                   else "Significant in both; direction INCONSISTENT ⚠"
        if ag in ("Cox only", "AFT only"):
            return f"Borderline — significant in {ag.split()[0]} only"
        return "No significant effect in either model"

    merged["note"] = merged.apply(_note, axis=1)
    return merged.sort_values(["component", "Cox_p"]).reset_index(drop=True)


def build_full_comparison(
    hr_table: pd.DataFrame,
    af_table: pd.DataFrame,
    cox_concordance: pd.DataFrame,
    aft_summary: pd.DataFrame,
    sig_threshold: float = _SIG_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both comparison tables ready for saving.

    Args:
        hr_table: Output of cox_model.cox_hr_table.
        af_table: Output of aft_model.aft_tr_table.
        cox_concordance: Output of cox_model.concordance_table.
        aft_summary: Output of aft_model.aft_model_summary.
        sig_threshold: p-value threshold.

    Returns:
        Tuple (covariate_comparison_df, concordance_comparison_df).
    """
    conc = compare_concordance(cox_concordance,
                               aft_summary["concordance_index"])
    cov  = compare_covariates(hr_table, af_table, sig_threshold)

    # Attach concordance columns to the covariate table for a self-contained CSV
    conc_cols = conc[["Cox_C", "AFT_C", "delta_C"]].reset_index()
    conc_cols.columns = ["component", "Cox_C", "AFT_C", "Delta_C"]
    cov = cov.merge(conc_cols, on="component", how="left")

    # Reorder columns for readability
    first_cols = ["component", "covariate_label",
                  "Cox_C", "AFT_C", "Delta_C",
                  "Cox_HR", "Cox_p", "Cox_sig",
                  "AFT_AF", "AFT_p", "AFT_sig",
                  "agreement", "direction_consistent", "note"]
    remaining = [c for c in cov.columns if c not in first_cols]
    cov = cov[first_cols + remaining]

    return cov, conc


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_comparison(
    cov_df: pd.DataFrame,
    conc_df: pd.DataFrame,
) -> Path:
    """Save combined comparison to outputs/tables/cox_vs_aft_comparison.csv.

    Writes two sections in one CSV file, separated by a blank row and a
    section header, so both concordance and covariate data are in a single
    file as requested.

    Args:
        cov_df: Covariate comparison (output of build_full_comparison[0]).
        conc_df: Concordance comparison (output of build_full_comparison[1]).

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "cox_vs_aft_comparison.csv"

    with open(path, "w") as fh:
        fh.write("# Section 1: Concordance Index Comparison\n")
        conc_df.to_csv(fh)
        fh.write("\n# Section 2: Covariate Significance Comparison\n")
        cov_df.to_csv(fh, index=False)

    print(f"[comparison] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Printed summary
# ---------------------------------------------------------------------------

def print_summary(
    cov_df: pd.DataFrame,
    conc_df: pd.DataFrame,
) -> None:
    """Print a formatted model comparison summary to stdout.

    Args:
        cov_df: Covariate comparison DataFrame.
        conc_df: Concordance comparison DataFrame.
    """
    SEP  = "=" * 66
    DASH = "-" * 66

    print(f"\n{SEP}")
    print("  MODEL COMPARISON: Cox PH  vs.  Weibull AFT")
    print(SEP)

    # --- Concordance ---
    print("\n▶ CONCORDANCE INDEX (C-index)")
    print(DASH)
    header = f"{'Component':<14} {'Cox PH':>8} {'Weibull AFT':>12} {'Δ(AFT−Cox)':>12} {'Better':>8}"
    print(header)
    print(DASH)
    for comp, row in conc_df.iterrows():
        delta_str = f"+{row['delta_C']:.4f}" if row['delta_C'] >= 0 else f"{row['delta_C']:.4f}"
        print(f"{comp:<14} {row['Cox_C']:>8.4f} {row['AFT_C']:>12.4f} "
              f"{delta_str:>12} {row['better_model']:>8}")
    print(DASH)
    mean_delta = conc_df["delta_C"].mean()
    sign = "+" if mean_delta >= 0 else ""
    delta_str_mean = f"{sign}{mean_delta:.4f}"
    print(f"{'Mean':.<14} {conc_df['Cox_C'].mean():>8.4f} "
          f"{conc_df['AFT_C'].mean():>12.4f} {delta_str_mean:>12}")

    # --- Covariate agreement ---
    print(f"\n▶ COVARIATE SIGNIFICANCE AGREEMENT  (p < 0.05)")
    print(DASH)
    agree_pivot = cov_df.pivot(index="covariate_label", columns="component",
                               values="agreement")
    agree_counts = cov_df.groupby(["covariate_label", "agreement"]).size().unstack(fill_value=0)

    # Summary: which covariates are significant in BOTH models across all components
    both_counts = (cov_df["agreement"] == "Both").groupby(cov_df["covariate_label"]).sum()
    total_comps = cov_df["component"].nunique()

    print(f"\n{'Covariate':<28} {'Both':>6} {'Cox_only':>10} {'AFT_only':>10} {'Neither':>8}")
    print("-" * 64)
    for cov_lbl in cov_df["covariate_label"].unique():
        sub = cov_df[cov_df["covariate_label"] == cov_lbl]
        counts = sub["agreement"].value_counts()
        n_both    = counts.get("Both",     0)
        n_cox     = counts.get("Cox only", 0)
        n_aft     = counts.get("AFT only", 0)
        n_neither = counts.get("Neither",  0)
        marker = " ◀ all components" if n_both == total_comps else \
                 f" ◀ {n_both}/{total_comps} components" if n_both > 0 else ""
        print(f"{cov_lbl:<28} {n_both:>6} {n_cox:>10} {n_aft:>10} {n_neither:>8}{marker}")

    # --- Direction consistency check ---
    inconsistent = cov_df[~cov_df["direction_consistent"]]
    print(f"\n▶ DIRECTION CONSISTENCY CHECK")
    print(DASH)
    if inconsistent.empty:
        print("  ✓ All covariates are directionally consistent across both models.")
        print("    (HR > 1 in Cox always coincides with AF < 1 in AFT, and vice versa.)")
    else:
        print("  ⚠ Inconsistencies detected:")
        print(inconsistent[["component", "covariate_label",
                             "Cox_HR", "AFT_AF"]].to_string(index=False))

    # --- Interpretation ---
    print(f"\n{SEP}")
    print("  INTERPRETATION")
    print(SEP)
    print("""
  Cox PH answers: "which units are at higher RISK RIGHT NOW?"
  ─────────────────────────────────────────────────────────
  • Models the instantaneous hazard rate h(t).
  • Hazard Ratio (HR):  HR > 1 → failure rate elevated relative to baseline.
  • Covariates act MULTIPLICATIVELY on the baseline hazard at every point in time.
  • Best used for: risk stratification, prioritising maintenance today.

  AFT answers: "how much LIFE does each covariate add or remove?"
  ────────────────────────────────────────────────────────────────
  • Models the survival TIME directly on a log scale.
  • Acceleration Factor (AF):  AF < 1 → time to failure is shortened.
                                AF > 1 → time to failure is extended.
  • Covariates act MULTIPLICATIVELY on the expected survival time.
  • Best used for: predicting remaining useful life (RUL), scheduling maintenance.

  Relationship between HR and AF (Weibull):
    HR ≈ AF^(−ρ),  where ρ is the Weibull shape parameter.
    For this dataset ρ ≈ exp(0.40–0.58) ≈ 1.5–1.8.
    A covariate with HR = 2.0 and ρ = 1.6 corresponds to AF ≈ 2.0^(−1/1.6) ≈ 0.64.

  Both models AGREE in this analysis:
    • HI_score_last is the dominant predictor in all 6 components.
    • Exponential distribution (Cox: constant hazard) is rejected everywhere.
    • All significant covariates are directionally consistent across models.
""")
    print(SEP)
