"""AIC and Log-Likelihood comparison table for parametric survival models.

Phase 5 of the analysis pipeline.

Produces the Table 2 equivalent from Yang et al. (2022): for each component
(row), reports LLV and AIC for every parametric model (column group), marks
the best-fit model, and saves to goodness_of_fit.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.hazard_models import FitResult


TABLES = Path(__file__).parent.parent / "outputs" / "tables"

MODEL_ORDER: list[str] = [
    "Weibull", "Exponential", "LogLogistic", "LogNormal", "GeneralizedGamma"
]


# ---------------------------------------------------------------------------
# Long-format comparison table (existing, kept for downstream callers)
# ---------------------------------------------------------------------------

def build_comparison_table(
    results: dict[str, list[FitResult]],
) -> pd.DataFrame:
    """Flatten all FitResults into a long-format AIC / LLV comparison table.

    Args:
        results: Output of fit_parametric_models — dict[group, list[FitResult]].

    Returns:
        DataFrame with columns [group, model, AIC, LLV, delta_AIC, rank]
        sorted by group then AIC ascending within each group.
    """
    rows = []
    for group, fit_list in results.items():
        for fr in fit_list:
            rows.append({"group": fr.group, "model": fr.model_name,
                          "AIC": fr.aic, "LLV": fr.llv})

    df = pd.DataFrame(rows)
    df["delta_AIC"] = df.groupby("group")["AIC"].transform(lambda x: x - x.min())
    df["rank"] = df.groupby("group")["AIC"].rank(method="min").astype(int)
    return df.sort_values(["group", "AIC"]).reset_index(drop=True)


def best_models(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the lowest-AIC model per group.

    Args:
        comparison: Output of build_comparison_table.

    Returns:
        One row per group — the best-fit (rank=1) model.
    """
    return comparison[comparison["rank"] == 1].reset_index(drop=True)


def save_comparison_table(df: pd.DataFrame, name: str = "model_comparison") -> Path:
    """Save long-form comparison table to outputs/tables/.

    Args:
        df: Output of build_comparison_table.
        name: Filename stem.

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Table 2 equivalent — wide format, one row per component
# ---------------------------------------------------------------------------

def goodness_of_fit_table(
    results: dict[str, list[FitResult]],
    component_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build the Table 2 equivalent from Yang et al. (2022).

    Wide-format table: rows = components, columns = {Model}_LLV and {Model}_AIC
    for each of the 5 models, plus summary columns Best_Model / Best_AIC /
    Best_LLV / AIC_spread (= max_AIC − best_AIC, indicates how much worse the
    worst model is).

    Args:
        results: Output of fit_parametric_models.
        component_order: Optional list to control row order in the output.

    Returns:
        DataFrame indexed by component name with flat column names.
    """
    rows: dict[str, dict] = {}
    for comp, fits in results.items():
        row: dict = {}
        for fr in fits:
            row[f"{fr.model_name}_LLV"] = fr.llv
            row[f"{fr.model_name}_AIC"] = fr.aic

        best_fr = min(fits, key=lambda f: f.aic)
        row["Best_Model"] = best_fr.model_name
        row["Best_AIC"]   = best_fr.aic
        row["Best_LLV"]   = best_fr.llv
        row["AIC_spread"]  = round(max(f.aic for f in fits) - best_fr.aic, 2)
        rows[comp] = row

    table = pd.DataFrame(rows).T

    # Order columns: interleave LLV then AIC for each model
    metric_cols: list[str] = []
    for model in MODEL_ORDER:
        llv_col = f"{model}_LLV"
        aic_col = f"{model}_AIC"
        if llv_col in table.columns:
            metric_cols.extend([llv_col, aic_col])

    summary_cols = ["Best_Model", "Best_AIC", "Best_LLV", "AIC_spread"]
    table = table[metric_cols + summary_cols]

    # Numeric coercion (pandas may infer object dtype from mixed row build)
    for col in metric_cols + ["Best_AIC", "Best_LLV", "AIC_spread"]:
        table[col] = pd.to_numeric(table[col])

    if component_order:
        table = table.reindex(component_order)

    return table


def annotate_best(
    table: pd.DataFrame,
    marker: str = " ✓",
) -> pd.DataFrame:
    """Return a string-formatted copy with a marker on the best-AIC cell.

    The best AIC per row is marked with ``marker`` (default ' ✓').
    LLV and AIC values are rounded to 2 decimal places in string form.
    This copy is for display / CSV export only; the numeric table is unchanged.

    Args:
        table: Output of goodness_of_fit_table (numeric).
        marker: Suffix appended to the best model's AIC cell.

    Returns:
        String-typed DataFrame suitable for to_string() or to_csv().
    """
    out = table.copy().astype(object)

    # Format numeric metric columns
    for col in table.columns:
        if col in ["Best_Model", "Best_AIC", "Best_LLV", "AIC_spread"]:
            if col != "Best_Model":
                out[col] = table[col].apply(lambda v: f"{v:.2f}")
            continue
        if "_AIC" in col or "_LLV" in col:
            out[col] = table[col].apply(lambda v: f"{v:.2f}")

    # Stamp the marker on the best-AIC cell for each row
    for comp in table.index:
        best_model = str(table.loc[comp, "Best_Model"])
        aic_col = f"{best_model}_AIC"
        if aic_col in out.columns:
            out.loc[comp, aic_col] = str(out.loc[comp, aic_col]) + marker

    return out


def delta_aic_matrix(
    results: dict[str, list[FitResult]],
    component_order: list[str] | None = None,
) -> pd.DataFrame:
    """ΔAIC matrix: rows = components, cols = models.

    ΔAIC_ij = AIC_ij − min_j(AIC_ij).  A value of 0 marks the best model
    for that component; values ≥ 10 indicate poor fit relative to best.

    Args:
        results: Output of fit_parametric_models.
        component_order: Optional row order.

    Returns:
        DataFrame (component × model) of ΔAIC values.
    """
    rows: dict[str, dict] = {}
    for comp, fits in results.items():
        best_aic = min(f.aic for f in fits)
        rows[comp] = {f.model_name: round(f.aic - best_aic, 2) for f in fits}

    df = pd.DataFrame(rows).T[MODEL_ORDER]
    if component_order:
        df = df.reindex(component_order)
    return df


# ---------------------------------------------------------------------------
# Save all Phase 5 outputs
# ---------------------------------------------------------------------------

def save_goodness_of_fit(
    results: dict[str, list[FitResult]],
    component_order: list[str] | None = None,
) -> tuple[Path, Path]:
    """Build and save the goodness-of-fit table in two versions.

    Saves:
    - ``goodness_of_fit.csv``: numeric wide table (machine-readable).
    - ``goodness_of_fit_annotated.csv``: string version with '✓' on best AIC.

    Args:
        results: Output of fit_parametric_models.
        component_order: Optional row order for both outputs.

    Returns:
        Tuple (numeric_path, annotated_path).
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    table = goodness_of_fit_table(results, component_order)
    ann   = annotate_best(table)

    p_num = TABLES / "goodness_of_fit.csv"
    p_ann = TABLES / "goodness_of_fit_annotated.csv"

    table.to_csv(p_num)
    ann.to_csv(p_ann)

    print(f"[model_selection] Saved → {p_num}")
    print(f"[model_selection] Saved → {p_ann}")
    return p_num, p_ann


def save_delta_aic(
    results: dict[str, list[FitResult]],
    component_order: list[str] | None = None,
    name: str = "delta_aic_matrix",
) -> Path:
    """Save ΔAIC matrix to outputs/tables/.

    Args:
        results: Output of fit_parametric_models.
        component_order: Optional row order.
        name: Filename stem.

    Returns:
        Path to saved CSV.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    delta_aic_matrix(results, component_order).to_csv(path)
    print(f"[model_selection] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Parameters table (unchanged, kept for notebook compatibility)
# ---------------------------------------------------------------------------

def params_table(results: dict[str, list[FitResult]]) -> pd.DataFrame:
    """Collect fitted distribution parameters for every model × group.

    Args:
        results: Output of fit_parametric_models.

    Returns:
        DataFrame with columns [group, model, param_name, param_value].
    """
    rows = []
    for group, fit_list in results.items():
        for fr in fit_list:
            for k, v in fr.params.items():
                rows.append({
                    "group": fr.group,
                    "model": fr.model_name,
                    "param_name": k,
                    "param_value": v,
                })
    return pd.DataFrame(rows)
