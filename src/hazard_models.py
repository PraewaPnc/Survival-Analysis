"""Non-parametric kernel hazard and 5 parametric survival models.

Phase 4 of the analysis pipeline.

Kernel hazard estimator (Yang et al. 2022):

    h(t) = (1/b) × Σᵢ K((t − tᵢ)/b) × ΔH(tᵢ)

where tᵢ are Nelson-Aalen event times, ΔH(tᵢ) are their cumulative-hazard
increments, K(·) is the Epanechnikov kernel, and b is the bandwidth chosen by
least-squares cross-validation (LSCV):

    CV(b) = ∫ĥ²(t)dt  −  2·Σᵢ ĥ₋ᵢ(tᵢ)·ΔH(tᵢ)

The LSCV minimisation is fully vectorised (no Python loops over observations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import (
    ExponentialFitter,
    GeneralizedGammaFitter,
    LogLogisticFitter,
    LogNormalFitter,
    NelsonAalenFitter,
    WeibullFitter,
)


TABLES = Path(__file__).parent.parent / "outputs" / "tables"

PARAMETRIC_MODELS: dict[str, type] = {
    "Weibull": WeibullFitter,
    "Exponential": ExponentialFitter,
    "LogLogistic": LogLogisticFitter,
    "LogNormal": LogNormalFitter,
    "GeneralizedGamma": GeneralizedGammaFitter,
}


# ---------------------------------------------------------------------------
# FitResult container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Container for a single fitted parametric model.

    Attributes:
        model_name: Distribution name (e.g. 'Weibull').
        group: Grouping label (e.g. component name).
        fitter: Fitted lifelines fitter object.
        aic: Akaike Information Criterion.
        llv: Log-Likelihood Value.
        params: Fitted distribution parameters.
    """

    model_name: str
    group: str
    fitter: object
    aic: float
    llv: float
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level kernel helpers
# ---------------------------------------------------------------------------

def _epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u) = 0.75(1−u²)·𝟙(|u|≤1)."""
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    out[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return out


def _na_increments(
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Nelson-Aalen and return event times with their ΔH increments.

    Only rows where ΔH > 0 (i.e. actual failure times) are returned;
    censored observations are already accounted for inside the NA estimator.

    Args:
        durations: Observed times (failures + censored).
        events: Binary event indicator (1=failure).

    Returns:
        Tuple (event_times, dH) — arrays of the same length.
    """
    naf = NelsonAalenFitter()
    naf.fit(durations, events)
    ch = naf.cumulative_hazard_
    inc = ch.diff().dropna()
    t_ev = inc.index.values.astype(float)
    dH = inc.iloc[:, 0].values.astype(float)
    # Keep only genuine event times (ΔH > 0 guards against zero-increment rows)
    mask = dH > 1e-12
    return t_ev[mask], dH[mask]


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def silverman_bandwidth(event_times: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for kernel hazard estimation.

    Args:
        event_times: Array of event times (failures only).

    Returns:
        Bandwidth b = 1.06 × σ × n^(−1/5).
    """
    n = len(event_times)
    if n < 2:
        return float(max(event_times.ptp(), 1.0))
    return float(1.06 * event_times.std() * n ** (-0.2))


def cv_bandwidth(
    durations: np.ndarray,
    events: np.ndarray,
    n_candidates: int = 25,
    n_grid: int = 200,
    search_range: tuple[float, float] = (0.3, 2.5),
) -> float:
    """Least-squares cross-validation (LSCV) bandwidth selection.

    Minimises the LSCV objective over a grid of candidate bandwidths:

        CV(b) = ∫ĥ²(t)dt  −  2·Σᵢ ĥ₋ᵢ(tᵢ)·ΔH(tᵢ)

    The LOO term ĥ₋ᵢ(tᵢ) is computed via a vectorised kernel weight matrix
    with zeroed diagonal (no Python loop over events).

    Args:
        durations: All observed times (failures + censored).
        events: Binary event indicator.
        n_candidates: Number of bandwidth candidates to evaluate.
        n_grid: Number of points on the integration grid for ∫ĥ²dt.
        search_range: Multipliers applied to the Silverman bandwidth to define
            the candidate interval (default: [0.3, 2.5] × b_Silverman).

    Returns:
        Optimal bandwidth b*.
    """
    event_times, dH = _na_increments(durations, events)
    n_ev = len(event_times)

    if n_ev < 5:
        return silverman_bandwidth(event_times)

    b_sil = silverman_bandwidth(event_times)
    b_lo, b_hi = search_range[0] * b_sil, search_range[1] * b_sil
    candidates = np.linspace(b_lo, b_hi, n_candidates)

    # Pairwise differences (n_ev × n_ev) — computed once, reused per candidate
    diff_mat = event_times[:, None] - event_times[None, :]  # tᵢ − tⱼ

    # Integration grid
    t_grid = np.linspace(event_times.min(), event_times.max(), n_grid)

    scores = np.empty(n_candidates)
    for k, b in enumerate(candidates):
        # LOO kernel weight matrix (diagonal = 0)
        W_loo = _epanechnikov(diff_mat / b)
        np.fill_diagonal(W_loo, 0.0)

        # ĥ₋ᵢ(tᵢ) = (Σⱼ≠ᵢ K((tᵢ−tⱼ)/b)×ΔH(tⱼ)) / b
        h_loo = (W_loo @ dH) / b  # shape (n_ev,)

        # ĥ(t) on the integration grid  (n_grid × n_ev) → (n_grid,)
        U_grid = (t_grid[:, None] - event_times[None, :]) / b
        h_grid = (_epanechnikov(U_grid) @ dH) / b

        int_h2 = np.trapz(h_grid ** 2, t_grid)
        loo_term = 2.0 * float(h_loo @ dH)

        scores[k] = int_h2 - loo_term

    return float(candidates[np.argmin(scores)])


# ---------------------------------------------------------------------------
# Kernel hazard estimator
# ---------------------------------------------------------------------------

def kernel_hazard(
    durations: np.ndarray,
    events: np.ndarray,
    bandwidth: Optional[float] = None,
    n_points: int = 300,
    trim_pct: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Epanechnikov kernel hazard smoother on Nelson-Aalen increments.

    Implements the estimator from Yang et al. (2022):

        h(t) = (1/b) Σᵢ K((t − tᵢ)/b) × ΔH(tᵢ)

    Args:
        durations: All observed times (failures + censored).
        events: Binary event indicator (1=failure).
        bandwidth: Kernel bandwidth (days). If None, selected by LSCV.
        n_points: Number of evaluation points on the time grid.
        trim_pct: Trim this percentage from each tail of the duration range
            to reduce boundary bias in the hazard estimate.

    Returns:
        Tuple (t_grid, h_grid) — evaluation times and hazard estimates ≥ 0.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=float)

    event_times, dH = _na_increments(durations, events)

    if bandwidth is None:
        bandwidth = cv_bandwidth(durations, events)

    lo = np.percentile(durations, trim_pct)
    hi = np.percentile(durations, 100.0 - trim_pct)
    t_grid = np.linspace(lo, hi, n_points)

    # Vectorised: (n_points × n_ev) kernel weight matrix
    U = (t_grid[:, None] - event_times[None, :]) / bandwidth
    h_grid = (_epanechnikov(U) @ dH) / bandwidth

    return t_grid, np.clip(h_grid, 0.0, None)


def kernel_hazard_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    bandwidth: Optional[float] = None,
    n_points: int = 300,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run kernel_hazard for each group, printing the selected bandwidth.

    Args:
        df: Validated DataFrame.
        group_col: Column to group by.
        groups: Ordered list of group values.
        duration_col: Time-to-event column.
        event_col: Event indicator column.
        bandwidth: Shared bandwidth for all groups. Per-group LSCV if None.
        n_points: Evaluation grid resolution.

    Returns:
        Dict mapping group label → (t_grid, h_grid).
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        T = sub[duration_col].values
        E = sub[event_col].values
        # Per-group LSCV unless a shared bandwidth is given
        bw = bandwidth if bandwidth is not None else cv_bandwidth(T, E)
        print(f"[kernel_hazard] {g}: bandwidth = {bw:.1f} days")
        t_grid, h_grid = kernel_hazard(T, E, bandwidth=bw, n_points=n_points)
        out[g] = (t_grid, h_grid)
    return out


# ---------------------------------------------------------------------------
# Parametric models
# ---------------------------------------------------------------------------

def fit_parametric_models(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> dict[str, list[FitResult]]:
    """Fit all 5 parametric models for each group.

    Args:
        df: Validated DataFrame.
        group_col: Column to group by.
        groups: Ordered list of group values.
        duration_col: Time-to-event column.
        event_col: Binary event indicator.

    Returns:
        Dict mapping group label → list of FitResult (one per model).
        Failed fits are silently skipped with a printed warning.
    """
    results: dict[str, list[FitResult]] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        T = sub[duration_col]
        E = sub[event_col]
        group_results: list[FitResult] = []
        for name, cls in PARAMETRIC_MODELS.items():
            try:
                fitter = cls()
                fitter.fit(T, E, label=f"{name}_{g}")
                group_results.append(
                    FitResult(
                        model_name=name,
                        group=g,
                        fitter=fitter,
                        aic=round(float(fitter.AIC_), 2),
                        llv=round(float(fitter.log_likelihood_), 4),
                        params={k: round(float(v), 4) for k, v in fitter.params_.items()},
                    )
                )
            except Exception as exc:
                print(f"[hazard_models] {name} failed for '{g}': {exc}")
        results[g] = group_results
    return results


def parametric_hazard(
    fit_result: FitResult,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted parametric hazard h(t) on a time grid.

    Args:
        fit_result: A FitResult with a fitted lifelines fitter.
        t_grid: Evaluation time points (days).

    Returns:
        Array of hazard values at each point in t_grid.
    """
    return np.asarray(fit_result.fitter.hazard_at_times(t_grid)).flatten()
