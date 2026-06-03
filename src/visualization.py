"""All plotting functions — each saves a PNG to outputs/figures/."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test

from src.hazard_models import FitResult


FIGURES = Path(__file__).parent.parent / "outputs" / "figures"
PALETTE = "tab10"

# Fixed color maps used consistently across all KM/NA/hazard plots
COMP_COLORS: dict[str, str] = {
    "Conductor": "#1f77b4",
    "Damper":    "#ff7f0e",
    "Spacer":    "#2ca02c",
    "Insulator": "#d62728",
    "Fittings":  "#9467bd",
    "Arrester":  "#8c564b",
}

REGION_COLORS: dict[str, str] = {
    "Northeast": "#1f77b4",
    "North":     "#ff7f0e",
    "Central":   "#2ca02c",
    "South":     "#d62728",
}

# HI0 = worst condition (red) → HI5 = best condition (green)
HI_COLORS: dict[str, str] = {
    "HI0": "#d73027",
    "HI1": "#f46d43",
    "HI2": "#fdae61",
    "HI3": "#74add1",
    "HI4": "#4575b4",
    "HI5": "#1a9641",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plot_event_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of event vs. censored counts per component.

    Args:
        df: Validated DataFrame.

    Returns:
        Path to saved PNG.
    """
    counts = df.groupby(["component", "event_occurred"]).size().unstack(fill_value=0)
    counts.columns = ["Censored", "Failure"]

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_title("Failure vs. Censored Count by Component", fontweight="bold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Status")
    return _save(fig, "eda_event_distribution")


def plot_duration_boxplot(df: pd.DataFrame) -> Path:
    """Box plots of maintenance_period_days per component.

    Args:
        df: Validated DataFrame.

    Returns:
        Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("component")["maintenance_period_days"].median().sort_values().index.tolist()
    sns.boxplot(data=df, x="component", y="maintenance_period_days", order=order, ax=ax, palette=PALETTE)
    ax.set_title("Maintenance Period Distribution by Component", fontweight="bold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Duration (days)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "eda_duration_boxplot")


def plot_duration_histogram_overall(
    df: pd.DataFrame,
    n_bins: int = 25,
    name: str = "fig4_duration_histogram_overall",
) -> Path:
    """Overall maintenance-period histogram — censored vs. failure (Figure 4 style).

    Replicates the style of Yang et al. (2022) Figure 4: a single histogram
    of all observed durations with failures (solid) and censored observations
    (hatched) shown as overlaid bars, plus vertical lines for the mean and
    median. A summary text box mirrors the table inset used in the paper.

    Args:
        df: Validated DataFrame.
        n_bins: Number of histogram bins.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    dur_fail = df.loc[df["event_occurred"] == 1, "maintenance_period_days"]
    dur_cens = df.loc[df["event_occurred"] == 0, "maintenance_period_days"]

    all_dur = df["maintenance_period_days"]
    bin_edges = np.linspace(all_dur.min(), all_dur.max(), n_bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        dur_cens, bins=bin_edges,
        color="#4C72B0", alpha=0.55, edgecolor="white", linewidth=0.6,
        hatch="////", label=f"Censored  (n={len(dur_cens):,})",
    )
    ax.hist(
        dur_fail, bins=bin_edges,
        color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.6,
        label=f"Failure  (n={len(dur_fail):,})",
    )

    mean_val = all_dur.mean()
    med_val = all_dur.median()
    ax.axvline(mean_val, color="#2ca02c", linestyle="--", linewidth=1.6,
               label=f"Mean = {mean_val:.0f} d")
    ax.axvline(med_val, color="#d62728", linestyle=":",  linewidth=1.6,
               label=f"Median = {med_val:.0f} d")

    # Summary inset — mirrors the paper's table annotation
    n_total = len(df)
    n_ev = len(dur_fail)
    cens_rate = 100 * len(dur_cens) / n_total
    info = (
        f"N = {n_total:,}\n"
        f"Failures = {n_ev:,}  ({100 - cens_rate:.1f}%)\n"
        f"Censored = {len(dur_cens):,}  ({cens_rate:.1f}%)\n"
        f"Min = {int(all_dur.min())} d\n"
        f"Max = {int(all_dur.max())} d\n"
        f"Std = {all_dur.std():.0f} d"
    )
    ax.text(
        0.97, 0.97, info,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_title(
        "Distribution of Maintenance Periods — All Components",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Maintenance Period (days)", labelpad=8)
    ax.set_ylabel("Frequency", labelpad=8)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    return _save(fig, name)


def plot_duration_histograms_by_component(
    df: pd.DataFrame,
    n_bins: int = 20,
    name: str = "fig5_duration_histograms_by_component",
) -> Path:
    """Per-component maintenance-period histograms — censored vs. failure (Figure 5 style).

    Replicates the style of Yang et al. (2022) Figure 5: a 2×3 grid of
    histograms, one per component, each showing the failure (solid) and
    censored (hatched) duration distributions with per-panel summary
    annotations.

    Args:
        df: Validated DataFrame.
        n_bins: Number of histogram bins per panel.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.preprocessing import COMPONENTS

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    # Shared bin edges across all panels so shapes are comparable
    global_min = df["maintenance_period_days"].min()
    global_max = df["maintenance_period_days"].max()
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    for i, comp in enumerate(COMPONENTS):
        ax = axes[i]
        sub = df[df["component"] == comp]
        dur_fail = sub.loc[sub["event_occurred"] == 1, "maintenance_period_days"]
        dur_cens = sub.loc[sub["event_occurred"] == 0, "maintenance_period_days"]

        ax.hist(
            dur_cens, bins=bin_edges,
            color="#4C72B0", alpha=0.55, edgecolor="white", linewidth=0.5,
            hatch="////", label="Censored",
        )
        ax.hist(
            dur_fail, bins=bin_edges,
            color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.5,
            label="Failure",
        )

        dur_all = sub["maintenance_period_days"]
        ax.axvline(dur_all.mean(),   color="#2ca02c", linestyle="--", linewidth=1.4)
        ax.axvline(dur_all.median(), color="#d62728", linestyle=":",  linewidth=1.4)

        n_tot = len(sub)
        n_ev = len(dur_fail)
        cens_rate = 100 * len(dur_cens) / n_tot
        info = (
            f"N={n_tot:,}  Fail={n_ev} ({100 - cens_rate:.0f}%)\n"
            f"Med={dur_all.median():.0f} d  Mean={dur_all.mean():.0f} d"
        )
        ax.text(
            0.97, 0.97, info,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
        )

        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xlabel("Duration (days)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)

        if i == 0:
            ax.legend(
                fontsize=8,
                handles=[
                    plt.Rectangle((0, 0), 1, 1, fc="#DD8452", alpha=0.85, label="Failure"),
                    plt.Rectangle((0, 0), 1, 1, fc="#4C72B0", alpha=0.55,
                                  hatch="////", label="Censored"),
                    plt.Line2D([0], [0], color="#2ca02c", linestyle="--",
                               linewidth=1.4, label="Mean"),
                    plt.Line2D([0], [0], color="#d62728", linestyle=":",
                               linewidth=1.4, label="Median"),
                ],
                loc="upper left",
            )

    fig.suptitle(
        "Maintenance Period Distributions by Component — Failure vs. Censored",
        fontweight="bold", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_covariate_heatmap(df: pd.DataFrame, cols: list[str]) -> Path:
    """Correlation heatmap among numeric covariates.

    Args:
        df: Validated DataFrame.
        cols: List of numeric column names to include.

    Returns:
        Path to saved PNG.
    """
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.4)
    ax.set_title("Covariate Correlation Matrix", fontweight="bold")
    return _save(fig, "eda_covariate_heatmap")


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

def plot_km_by_component(
    fitters: dict[str, KaplanMeierFitter],
    logrank_p: float | None = None,
    name: str = "fig6_km_by_component",
) -> Path:
    """Figure 6 style: all 6 component KM curves with 95% CI and numbers-at-risk table.

    Replicates Yang et al. (2022) Figure 6. Each component is drawn in a fixed
    colour from COMP_COLORS. The shaded band is the Greenwood 95% CI.
    The numbers-at-risk table is added below the plot via lifelines'
    add_at_risk_counts at yearly intervals.

    Args:
        fitters: Dict of label → fitted KaplanMeierFitter (from fit_km_by_group).
        logrank_p: Optional pre-computed log-rank p-value to annotate in the title.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.survival_curves import AT_RISK_TIMES

    fig, ax = plt.subplots(figsize=(13, 8))

    for label, kmf in fitters.items():
        color = COMP_COLORS.get(label, "#888888")
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2.2,
        )

    # Title — append log-rank annotation when p-value is supplied
    title = "Kaplan-Meier Survival Functions — By Component"
    if logrank_p is not None:
        p_str = "p < 0.001" if logrank_p < 0.001 else f"p = {logrank_p:.4f}"
        title += f"\nLog-rank test: {p_str}"

    ax.set_title(title, fontweight="bold", pad=12, fontsize=12)
    ax.set_xlabel("Time (days)", labelpad=8, fontsize=11)
    ax.set_ylabel("Survival Probability S(t)", labelpad=8, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(AT_RISK_TIMES)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, title="Component")

    # Numbers-at-risk table — aligned to the same yearly x-ticks as the plot
    add_at_risk_counts(
        *list(fitters.values()),
        ax=ax,
        fontsize=8,
        rows_to_show=["At risk"],
    )

    plt.tight_layout()
    return _save(fig, name)


def plot_km_by_region(
    df: pd.DataFrame,
    regions: list[str],
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
) -> Path:
    """KM survival curves stratified by region — 2×2 grid, one panel per region.

    Each panel plots all 6 component KM curves (CI off for readability) for
    a single geographic region, with a log-rank p-value annotation comparing
    components within that region.

    Args:
        df: Validated DataFrame.
        regions: List of region names (length 4 expected for 2×2 layout).
        duration_col: Time-to-event column.
        event_col: Event indicator column.

    Returns:
        Path to saved PNG.
    """
    from src.preprocessing import COMPONENTS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, reg in enumerate(regions):
        ax = axes[i]
        sub_reg = df[df["region"] == reg]

        for comp in COMPONENTS:
            sub = sub_reg[sub_reg["component"] == comp]
            if sub.empty:
                continue
            color = COMP_COLORS.get(comp, "#888888")
            kmf = KaplanMeierFitter(label=comp)
            kmf.fit(sub[duration_col], sub[event_col])
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.8)

        # Per-panel log-rank test (components within this region)
        try:
            res = multivariate_logrank_test(
                sub_reg[duration_col], sub_reg["component"], sub_reg[event_col]
            )
            p_str = "p < 0.001" if res.p_value < 0.001 else f"p = {res.p_value:.4f}"
            chi2_str = f"χ²={res.test_statistic:.1f}"
        except Exception:
            p_str = ""
            chi2_str = ""

        n_ev = int(sub_reg[event_col].sum())
        info = (
            f"N={len(sub_reg):,}  Failures={n_ev} ({100*n_ev/len(sub_reg):.1f}%)\n"
            f"Log-rank: {chi2_str}, {p_str}"
        )
        ax.text(
            0.97, 0.97, info,
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88),
        )

        ax.set_title(f"Region: {reg}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("S(t)", fontsize=9)
        ax.set_ylim(0, 1.05)

        if i == 0:
            ax.legend(fontsize=7.5, loc="lower left", title="Component", title_fontsize=7.5)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    fig.suptitle(
        "Kaplan-Meier Survival Functions — Stratified by Region",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    return _save(fig, "km_by_region")


def plot_km_by_hi_class(
    df: pd.DataFrame,
    hi_col: str = "HI_class_last",
    duration_col: str = "maintenance_period_days",
    event_col: str = "event_occurred",
    name: str = "km_by_hi_class",
) -> Path:
    """KM survival curves stratified by HI_class_last with 95% CI and risk table.

    HI0 (worst health) is coloured dark red; HI5 (best health) is dark green,
    using the HI_COLORS gradient. Includes a numbers-at-risk table and a
    multivariate log-rank test annotation.

    Args:
        df: Validated DataFrame.
        hi_col: Column name for the health index class (default 'HI_class_last').
        duration_col: Time-to-event column.
        event_col: Binary event indicator.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.survival_curves import AT_RISK_TIMES

    hi_classes = sorted(df[hi_col].dropna().unique().tolist())  # ['HI0',..'HI5']

    fig, ax = plt.subplots(figsize=(13, 9))

    fitters_list: list[KaplanMeierFitter] = []
    for hi in hi_classes:
        sub = df[df[hi_col] == hi]
        if sub.empty:
            continue
        color = HI_COLORS.get(hi, "#888888")
        kmf = KaplanMeierFitter(label=hi)
        kmf.fit(sub[duration_col], sub[event_col])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.2)
        fitters_list.append(kmf)

    # Multivariate log-rank test across all HI classes
    try:
        res = multivariate_logrank_test(
            df[duration_col], df[hi_col], df[event_col]
        )
        p_str = "p < 0.001" if res.p_value < 0.001 else f"p = {res.p_value:.4f}"
        chi2_str = f"χ²={res.test_statistic:.2f}"
        title_suffix = f"\nLog-rank test: {chi2_str},  {p_str}"
    except Exception:
        title_suffix = ""

    # Per-class N and event count annotation (top-right)
    info_lines = [
        f"{hi}: N={df[df[hi_col]==hi].shape[0]:,}, "
        f"events={int(df[df[hi_col]==hi][event_col].sum())} "
        f"({100*df[df[hi_col]==hi][event_col].mean():.1f}%)"
        for hi in hi_classes
    ]
    ax.text(
        0.97, 0.97, "\n".join(info_lines),
        transform=ax.transAxes, fontsize=7.5,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", alpha=0.90),
    )

    ax.set_title(
        f"Kaplan-Meier Survival Functions — Stratified by {hi_col}" + title_suffix,
        fontweight="bold", pad=12, fontsize=12,
    )
    ax.set_xlabel("Time (days)", labelpad=8, fontsize=11)
    ax.set_ylabel("Survival Probability S(t)", labelpad=8, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(AT_RISK_TIMES)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, title="Health Index Class")

    # Numbers-at-risk table
    add_at_risk_counts(
        *fitters_list,
        ax=ax,
        fontsize=7.5,
        rows_to_show=["At risk"],
    )

    plt.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Nelson-Aalen
# ---------------------------------------------------------------------------

def plot_na_by_component(
    fitters: dict[str, NelsonAalenFitter],
    name: str = "na_cumulative_hazard",
) -> Path:
    """Overlay Nelson-Aalen cumulative hazard curves — all components on one axes.

    Shows H(t) with 95% CI for each component using fixed COMP_COLORS. The
    overlay lets inter-component differences be read at a glance.

    Args:
        fitters: Dict from fit_na_by_group.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, naf in fitters.items():
        color = COMP_COLORS.get(label, "#888888")
        naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=color, linewidth=2.0)

    ax.set_title(
        "Nelson-Aalen Cumulative Hazard H(t) — By Component",
        fontweight="bold", pad=10,
    )
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Cumulative Hazard H(t)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, title="Component")
    ax.set_xlim(left=0)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, name)


def plot_na_subplots(
    fitters: dict[str, NelsonAalenFitter],
    name: str = "na_cumulative_hazard_subplots",
) -> Path:
    """2×3 per-component Nelson-Aalen cumulative hazard plots with 95% CI.

    Each panel shows one component's H(t) curve with its Greenwood 95% CI
    shaded. Annotations show H(t) at 1 yr, 2 yr, 4 yr, and 6 yr.

    Args:
        fitters: Dict from fit_na_by_group.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.survival_curves import AT_RISK_TIMES

    components = list(fitters.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    annot_times = [365, 730, 1460, 2190]   # 1, 2, 4, 6 yr

    for i, comp in enumerate(components):
        ax = axes[i]
        naf = fitters[comp]
        color = COMP_COLORS.get(comp, "#888888")

        # Main cumulative hazard curve + CI band
        naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=color,
                                   linewidth=2.2)

        # Pull out raw arrays for annotation
        ch = naf.cumulative_hazard_
        ci = naf.confidence_interval_cumulative_hazard_

        # Annotate H(t) at selected time points
        for t in annot_times:
            idx = ch.index[ch.index <= t]
            if not len(idx):
                continue
            t_last = idx[-1]
            h_val  = float(ch.loc[t_last].iloc[0])
            ax.annotate(
                f"H={h_val:.4f}",
                xy=(t_last, h_val),
                xytext=(t_last + 50, h_val + 0.001),
                fontsize=6.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )

        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("H(t)", fontsize=9)
        ax.set_xlim(left=0)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        # Remove auto legend from lifelines inside subplots (already labelled by title)
        if ax.get_legend():
            ax.get_legend().remove()

    fig.suptitle(
        "Nelson-Aalen Cumulative Hazard — Per Component (95% CI shaded)",
        fontweight="bold", fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Kernel Hazard
# ---------------------------------------------------------------------------

def plot_kernel_hazard(
    hazard_data: dict[str, tuple[np.ndarray, np.ndarray]],
    name: str = "kernel_hazard_epanechnikov",
) -> Path:
    """Overlay Epanechnikov kernel hazard estimates for all components.

    Args:
        hazard_data: Output of kernel_hazard_by_group — dict[label → (t, h)].
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, (t, h) in hazard_data.items():
        color = COMP_COLORS.get(label, "#888888")
        ax.plot(t, h * 1_000, label=label, color=color, linewidth=2.0)
    ax.set_title(
        "Kernel Hazard Estimates (Epanechnikov, LSCV bandwidth) — By Component",
        fontweight="bold", pad=10,
    )
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Hazard h(t)  [×10⁻³ per day]", fontsize=11)
    ax.legend(fontsize=9, title="Component")
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Per-component hazard figures (Figures 7-12 style, Yang et al. 2022)
# ---------------------------------------------------------------------------

# Style spec for parametric models — colour, linestyle, linewidth
_PARAM_STYLE: dict[str, dict] = {
    "Weibull":         {"color": "#e41a1c", "ls": "--",   "lw": 1.8},
    "Exponential":     {"color": "#377eb8", "ls": "--",   "lw": 1.8},
    "LogLogistic":     {"color": "#4daf4a", "ls": ":",    "lw": 2.0},
    "LogNormal":       {"color": "#984ea3", "ls": "-.",   "lw": 1.8},
    "GeneralizedGamma":{"color": "#ff7f00", "ls": (0,(5,2,1,2)), "lw": 1.8},
}


def plot_hazard_per_component(
    component: str,
    fit_results: list[FitResult],
    t_kernel: np.ndarray,
    h_kernel: np.ndarray,
    fig_num: int | None = None,
) -> Path:
    """One figure per component: kernel hazard + 5 parametric hazard curves.

    Replicates the style of Yang et al. (2022) Figures 7-11.  The non-
    parametric Epanechnikov kernel estimate is drawn as a thick black line;
    each parametric model is a coloured dashed/dotted curve.  The y-axis is
    scaled to ×10⁻³ per day for legibility.

    Args:
        component: Component name used for the title and filename.
        fit_results: List of FitResult objects (one per parametric model).
        t_kernel: Time grid from kernel_hazard.
        h_kernel: Hazard estimate from kernel_hazard (values per day).
        fig_num: Optional figure number prefix (e.g. 7 → 'fig7_...').

    Returns:
        Path to saved PNG.
    """
    prefix = f"fig{fig_num}_" if fig_num is not None else ""
    name = f"{prefix}hazard_{component.lower()}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Non-parametric kernel hazard ---
    ax.plot(
        t_kernel, h_kernel * 1_000,
        color="black", linewidth=2.6,
        label="Kernel (Epanechnikov)", zorder=5,
    )

    # --- Parametric hazard curves ---
    for fr in fit_results:
        style = _PARAM_STYLE.get(fr.model_name, {"color": "gray", "ls": "--", "lw": 1.5})
        try:
            h_param = fr.fitter.hazard_at_times(t_kernel)
            ax.plot(
                t_kernel, np.asarray(h_param).flatten() * 1_000,
                color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
                label=fr.model_name,
            )
        except Exception as exc:
            print(f"[viz] hazard_at_times failed for {fr.model_name}/{component}: {exc}")

    ax.set_title(
        f"Component: {component} — Hazard Function h(t)",
        fontweight="bold", pad=12, fontsize=12,
    )
    ax.set_xlabel("Time (days)", fontsize=11, labelpad=6)
    ax.set_ylabel("Hazard h(t)  [×10⁻³ per day]", fontsize=11, labelpad=6)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    ax.set_xlim(t_kernel[0], t_kernel[-1])

    # Annotate the LSCV-selected bandwidth
    bw = float(np.mean(np.diff(t_kernel)) * len(t_kernel) / 10)  # rough display proxy
    # (actual bw printed during fitting; just note 'LSCV' on plot)
    ax.text(
        0.99, 0.02, "Bandwidth: LSCV",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
        color="gray",
    )

    fig.tight_layout()
    return _save(fig, name)


def plot_hazard_all_components(
    hazard_data: dict[str, tuple[np.ndarray, np.ndarray]],
    parametric_results: dict[str, list[FitResult]],
    fig_start: int = 7,
) -> list[Path]:
    """Generate one hazard figure per component (6 figures, Fig 7-12 style).

    Args:
        hazard_data: Output of kernel_hazard_by_group.
        parametric_results: Output of fit_parametric_models.
        fig_start: Starting figure number prefix.

    Returns:
        List of 6 saved PNG paths, one per component.
    """
    paths = []
    for i, (comp, (t, h)) in enumerate(hazard_data.items()):
        fits = parametric_results.get(comp, [])
        path = plot_hazard_per_component(
            component=comp,
            fit_results=fits,
            t_kernel=t,
            h_kernel=h,
            fig_num=fig_start + i,
        )
        paths.append(path)
        print(f"[viz] Saved {path.name}")
    return paths


def plot_hazard_overview(
    hazard_data: dict[str, tuple[np.ndarray, np.ndarray]],
    parametric_results: dict[str, list[FitResult]],
    best_models: pd.DataFrame,
    name: str = "hazard_overview_2x3",
) -> Path:
    """2×3 overview: kernel hazard + best parametric model per component.

    Args:
        hazard_data: Output of kernel_hazard_by_group.
        parametric_results: Output of fit_parametric_models.
        best_models: DataFrame from model_selection.best_models (one row per
            component, must have columns 'group' and 'model').
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.preprocessing import COMPONENTS

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, comp in enumerate(COMPONENTS):
        ax = axes[i]
        if comp not in hazard_data:
            ax.set_visible(False)
            continue

        t, h = hazard_data[comp]
        ax.plot(t, h * 1_000, color="black", linewidth=2.4, label="Kernel", zorder=5)

        # Best parametric model for this component
        best_row = best_models[best_models["group"] == comp]
        if not best_row.empty:
            best_name = best_row.iloc[0]["model"]
            fr = next(
                (f for f in parametric_results.get(comp, []) if f.model_name == best_name),
                None,
            )
            if fr is not None:
                style = _PARAM_STYLE.get(best_name, {"color": "#e41a1c", "ls": "--", "lw": 1.8})
                try:
                    h_p = np.asarray(fr.fitter.hazard_at_times(t)).flatten()
                    ax.plot(
                        t, h_p * 1_000,
                        color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
                        label=f"{best_name} (best AIC)",
                    )
                except Exception:
                    pass

        ax.set_title(comp, fontweight="bold", fontsize=10)
        ax.set_xlabel("Time (days)", fontsize=8)
        ax.set_ylabel("h(t) [×10⁻³/day]", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Hazard Functions — Kernel (Epanechnikov) vs. Best Parametric Model",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Parametric Models
# ---------------------------------------------------------------------------

def plot_parametric_sf_per_component(
    results: dict[str, list[FitResult]],
    km_fitters: dict[str, KaplanMeierFitter],
) -> Path:
    """For each component: overlay all parametric S(t) curves vs. KM.

    Args:
        results: Output of fit_parametric_models.
        km_fitters: Fitted KM fitters for reference.

    Returns:
        Path to saved PNG.
    """
    components = list(results.keys())
    ncols = 3
    nrows = (len(components) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = axes.flatten()

    model_colors = {
        "Weibull": "#e41a1c",
        "Exponential": "#377eb8",
        "LogLogistic": "#4daf4a",
        "LogNormal": "#984ea3",
        "GeneralizedGamma": "#ff7f00",
    }

    for i, comp in enumerate(components):
        ax = axes[i]
        if comp in km_fitters:
            km_fitters[comp].plot_survival_function(ax=ax, ci_show=True, color="black",
                                                    label="Kaplan-Meier", linewidth=2)
        for fr in results[comp]:
            color = model_colors.get(fr.model_name, "gray")
            fr.fitter.plot_survival_function(ax=ax, ci_show=False, color=color,
                                             label=fr.model_name, linestyle="--")
        ax.set_title(comp, fontsize=10)
        ax.set_xlabel("Days")
        ax.set_ylabel("S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, loc="lower left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Parametric Survival Functions vs. Kaplan-Meier", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "parametric_sf_all_components")


def plot_model_comparison_heatmap(comparison: pd.DataFrame) -> Path:
    """Heatmap of AIC by model × component.

    Args:
        comparison: Output of build_comparison_table.

    Returns:
        Path to saved PNG.
    """
    pivot = comparison.pivot(index="model", columns="group", values="AIC")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.3)
    ax.set_title("AIC by Model and Component (lower = better)", fontweight="bold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Model")
    return _save(fig, "model_aic_heatmap")


def plot_best_model_sf(
    results: dict[str, list[FitResult]],
    best: pd.DataFrame,
    km_fitters: dict[str, KaplanMeierFitter],
) -> Path:
    """Plot only the best-AIC parametric model vs. KM per component.

    Args:
        results: Output of fit_parametric_models.
        best: Output of best_models — one row per component.
        km_fitters: Fitted KM fitters for reference.

    Returns:
        Path to saved PNG.
    """
    components = best["group"].tolist()
    ncols = 3
    nrows = (len(components) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = axes.flatten()

    for i, row in enumerate(best.itertuples()):
        ax = axes[i]
        comp = row.group
        model_name = row.model
        if comp in km_fitters:
            km_fitters[comp].plot_survival_function(ax=ax, ci_show=True, color="black",
                                                    label="Kaplan-Meier", linewidth=2)
        fr = next((f for f in results[comp] if f.model_name == model_name), None)
        if fr:
            fr.fitter.plot_survival_function(ax=ax, ci_show=False, color="#e41a1c",
                                             label=f"{model_name} (best)", linestyle="--", linewidth=2)
        ax.set_title(f"{comp}\n(best: {model_name}, AIC={row.AIC:.0f})", fontsize=9)
        ax.set_xlabel("Days")
        ax.set_ylabel("S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Best-Fit Parametric Model vs. Kaplan-Meier", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "best_model_sf")


# ---------------------------------------------------------------------------
# Model Selection — AIC comparison bar charts (Phase 5)
# ---------------------------------------------------------------------------

def plot_aic_barchart(
    results: dict[str, list[FitResult]],
    name: str = "aic_comparison_faceted",
) -> Path:
    """2×3 faceted bar chart: AIC per model within each component.

    Each panel shows one component; bars represent the 5 parametric models;
    the y-axis is zoomed to [min_AIC − margin, max_AIC + margin] so small
    differences are visible. The best-AIC bar is highlighted in green; all
    others are steel blue.

    Args:
        results: Output of fit_parametric_models.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.model_selection import MODEL_ORDER
    from src.preprocessing import COMPONENTS

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    bar_color_default = "#4C72B0"
    bar_color_best    = "#2ca02c"
    bar_width = 0.65

    for i, comp in enumerate(COMPONENTS):
        ax = axes[i]
        fits = {fr.model_name: fr for fr in results.get(comp, [])}
        models = [m for m in MODEL_ORDER if m in fits]
        aics   = [fits[m].aic for m in models]
        best_aic = min(aics)

        colors = [bar_color_best if a == best_aic else bar_color_default for a in aics]
        x = np.arange(len(models))

        bars = ax.bar(x, aics, width=bar_width, color=colors, edgecolor="white",
                      linewidth=0.6, zorder=3)

        # Zoom y-axis: margin = 2% of AIC range or at least 5 units
        rng = max(aics) - min(aics)
        margin = max(rng * 0.15, 5.0)
        ax.set_ylim(min(aics) - margin, max(aics) + margin * 1.5)

        # Value labels above each bar
        for bar, aic in zip(bars, aics):
            delta = aic - best_aic
            label = f"{aic:.0f}" + (f"\n(Δ{delta:.0f})" if delta > 0.5 else "\n(best)")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + margin * 0.05,
                label,
                ha="center", va="bottom", fontsize=7,
                fontweight="bold" if delta < 0.5 else "normal",
                color=bar_color_best if delta < 0.5 else "#333333",
            )

        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("Generalized", "Gen.") for m in models],
            fontsize=8, rotation=25, ha="right",
        )
        ax.set_ylabel("AIC", fontsize=9)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "AIC Comparison Across Parametric Models — By Component\n"
        "(green = best fit; Δ = AIC difference from best)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_delta_aic(
    results: dict[str, list[FitResult]],
    name: str = "delta_aic_grouped",
) -> Path:
    """Grouped bar chart: ΔAIC per model, one bar group per component.

    All six components share a common y-axis (ΔAIC = AIC − AIC_best for that
    component), making cross-component model rankings directly comparable.
    Reference lines at ΔAIC = 2 and ΔAIC = 10 mark conventional thresholds
    for 'substantial' and 'no support' (Burnham & Anderson 2002).

    Args:
        results: Output of fit_parametric_models.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.model_selection import MODEL_ORDER
    from src.preprocessing import COMPONENTS

    # Build ΔAIC matrix (component × model)
    comp_list = [c for c in COMPONENTS if c in results]
    model_list = [m for m in MODEL_ORDER]

    delta: dict[str, list[float]] = {}
    for comp in comp_list:
        fits = {fr.model_name: fr.aic for fr in results[comp]}
        best = min(fits.values())
        delta[comp] = [fits.get(m, np.nan) - best for m in model_list]

    n_comp   = len(comp_list)
    n_models = len(model_list)
    group_width = 0.8
    bar_width   = group_width / n_models
    x_base      = np.arange(n_comp)

    model_colors = [_PARAM_STYLE[m]["color"] for m in model_list]

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, (model, color) in enumerate(zip(model_list, model_colors)):
        offsets = x_base + (j - n_models / 2 + 0.5) * bar_width
        vals = [delta[c][j] for c in comp_list]
        bars = ax.bar(
            offsets, vals,
            width=bar_width * 0.92,
            color=color, alpha=0.88,
            edgecolor="white", linewidth=0.5,
            label=model, zorder=3,
        )

        # Label bars with Δ value when > 0.5
        for bar, v in zip(bars, vals):
            if v > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333",
                )

    # Reference threshold lines
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55,
               label="ΔAIC = 2 (substantial)")
    ax.axhline(10.0, color="black", linestyle=":",  linewidth=1.0, alpha=0.55,
               label="ΔAIC = 10 (no support)")

    ax.set_xticks(x_base)
    ax.set_xticklabels(comp_list, fontsize=11)
    ax.set_ylabel("ΔAIC  (vs. best model per component)", fontsize=11, labelpad=6)
    ax.set_xlabel("Component", fontsize=11, labelpad=6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "ΔAIC — Parametric Model Comparison Across Components\n"
        "(ΔAIC = 0 is the best model; dashed/dotted lines are Burnham-Anderson thresholds)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.legend(fontsize=9, title="Model", loc="upper right",
              framealpha=0.92, ncol=2)

    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Cox Proportional Hazard — Phase 6
# ---------------------------------------------------------------------------

def plot_forest_plot(
    hr_table: pd.DataFrame,
    components: list[str],
    name: str = "cox_forest_plot",
) -> Path:
    """2×3 forest plot grid: one panel per component, showing HR and 95% CI.

    Hazard ratios are plotted on a log₂ scale.  Significant covariates
    (p < 0.05) are coloured red (HR > 1, elevated risk) or blue (HR < 1,
    protective); non-significant covariates are shown in grey.  Marker size
    encodes −log₁₀(p) so stronger signals are visually larger.

    Args:
        hr_table: Long-format output of cox_model.cox_hr_table.
        components: Ordered list of component names (defines panel order).
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.cox_model import COVARIATE_LABELS

    # Ordered covariate labels (bottom → top on y-axis)
    cov_order = list(COVARIATE_LABELS.values())
    n_cov = len(cov_order)
    y_pos = {label: i for i, label in enumerate(cov_order)}

    # Global x-axis limits across all panels
    hr_min = hr_table["HR_lower"].min()
    hr_max = hr_table["HR_upper"].max()
    x_lo = max(0.05, hr_min * 0.7)
    x_hi = min(20.0,  hr_max * 1.5)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), sharey=True)
    axes = axes.flatten()

    for i, comp in enumerate(components):
        ax = axes[i]
        sub = hr_table[hr_table["component"] == comp].set_index("covariate_label")

        for label in cov_order:
            if label not in sub.index:
                continue
            row = sub.loc[label]
            hr, lo, hi, p = row["HR"], row["HR_lower"], row["HR_upper"], row["p"]
            y = y_pos[label]

            # Colour logic
            sig = p < 0.05
            if sig and hr > 1.0:
                color = "#d62728"   # red — risk factor
            elif sig and hr < 1.0:
                color = "#1f77b4"   # blue — protective
            else:
                color = "#888888"   # grey — not significant

            marker_size = max(5, min(14, -np.log10(p + 1e-10) * 2.5))

            # CI line
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5,
                    solid_capstyle="round", zorder=3)
            # End caps
            for x_cap in (lo, hi):
                ax.plot([x_cap, x_cap], [y - 0.12, y + 0.12],
                        color=color, linewidth=1.5, zorder=3)
            # Point estimate
            ax.scatter([hr], [y], color=color, s=marker_size ** 2,
                       zorder=5, edgecolors="white", linewidth=0.5)

            # HR text on right
            stars = row["sig"]
            ax.text(
                x_hi * 1.05, y,
                f"{hr:.2f} [{lo:.2f}–{hi:.2f}] {stars}",
                va="center", ha="left", fontsize=6.5,
                color=color if sig else "#555555",
                transform=ax.get_yaxis_transform(),  # use axes x, data y
            )

        # Reference line HR=1
        ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

        # C-index from the fitted model — will be annotated if passed
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(cov_order, fontsize=8.5)
        ax.set_xlabel("Hazard Ratio (log scale)", fontsize=8)
        ax.set_title(comp, fontweight="bold", fontsize=11, pad=6)
        ax.xaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # Minor x-ticks at 0.5 and 2.0 for readability
        ax.axvline(0.5, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)
        ax.axvline(2.0, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)

    # Remove legend box and add a text legend in top-left of first panel
    axes[0].text(
        x_lo * 1.1, n_cov - 0.3,
        "● p<0.05, HR>1  ● p<0.05, HR<1  ● p≥0.05",
        fontsize=7, color="#333333",
        va="top",
    )

    fig.suptitle(
        "Cox Proportional Hazard — Forest Plot of Hazard Ratios\n"
        "(standardised covariates; 1 unit = 1 SD; HR [95% CI]; * p<0.05, ** p<0.01, *** p<0.001)",
        fontweight="bold", fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_concordance_bar(
    concordance: pd.DataFrame,
    name: str = "cox_concordance",
) -> Path:
    """Bar chart of concordance index (C-index) per component.

    Reference lines at C=0.5 (random) and C=0.7 (good discrimination).

    Args:
        concordance: Output of cox_model.concordance_table — indexed by
            component with column 'concordance_index'.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.preprocessing import COMPONENTS

    comp_order = [c for c in COMPONENTS if c in concordance.index]
    values = [concordance.loc[c, "concordance_index"] for c in comp_order]

    # Colour by performance band
    def _cindex_color(c: float) -> str:
        if c >= 0.75:
            return "#2ca02c"
        if c >= 0.65:
            return "#1f77b4"
        if c >= 0.55:
            return "#ff7f0e"
        return "#d62728"

    colors = [_cindex_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp_order))
    bars = ax.bar(x, values, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.6, zorder=3)

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Reference lines
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.2,
               label="C = 0.5 (random)", zorder=2)
    ax.axhline(0.7, color="#2ca02c", linestyle="--", linewidth=1.2,
               label="C = 0.7 (good)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_order, fontsize=11)
    ax.set_ylabel("Concordance Index (C-index)", fontsize=11, labelpad=6)
    ax.set_ylim(0.45, min(1.0, max(values) + 0.08))
    ax.set_title(
        "Cox PH Model — Concordance Index by Component",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Proportional Hazards Assumption — Phase 6 diagnostics
# ---------------------------------------------------------------------------

def plot_ph_test_heatmap(
    ph_df: pd.DataFrame,
    components: list[str],
    p_threshold: float = 0.05,
    name: str = "ph_assumption_heatmap",
) -> Path:
    """Heatmap of Schoenfeld residual p-values (component × covariate).

    Cells are coloured by −log₁₀(p): white = no concern, red = strong
    violation.  Cells with p < p_threshold are starred and outlined.

    Args:
        ph_df: Output of cox_model.check_ph_assumption.
        components: Ordered list of components (defines row order).
        p_threshold: Significance level to flag as violated.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.cox_model import COVARIATE_LABELS

    cov_order = list(COVARIATE_LABELS.values())

    # Pivot to matrix form
    p_pivot = ph_df.pivot(index="component", columns="covariate_label", values="p")
    p_pivot = p_pivot.reindex(index=components, columns=cov_order)

    color_mat = -np.log10(p_pivot.clip(lower=1e-10))

    # Annotation: p-value + star for violations
    annot = p_pivot.applymap(
        lambda p: f"{p:.3f}{'*' if p < p_threshold else ''}"
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        color_mat,
        annot=annot,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=max(3.0, color_mat.max().max() * 1.1),
        linewidths=0.4,
        linecolor="#dddddd",
        ax=ax,
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "−log₁₀(p)", "shrink": 0.8},
    )

    # Thick border on violated cells
    for i, comp in enumerate(components):
        for j, cov in enumerate(cov_order):
            p_val = p_pivot.loc[comp, cov] if comp in p_pivot.index and cov in p_pivot.columns else 1.0
            if p_val < p_threshold:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="#d62728", linewidth=2.5,
                ))

    ax.set_title(
        f"Proportional Hazards Assumption — Schoenfeld Residuals Test\n"
        f"(colour = −log₁₀(p); * and red border = p < {p_threshold}; "
        f"p < {p_threshold} indicates possible PH violation)",
        fontweight="bold", fontsize=11, pad=10,
    )
    ax.set_xlabel("Covariate", fontsize=10)
    ax.set_ylabel("Component", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    return _save(fig, name)


def plot_schoenfeld_residuals(
    schoenfeld_data: dict[str, pd.DataFrame],
    ph_df: pd.DataFrame,
    components: list[str],
    p_threshold: float = 0.05,
) -> list[Path]:
    """Per-component Schoenfeld residuals vs. time — 6 figures (one per component).

    Each figure has a 2×4 panel grid, one panel per covariate.  Dots are
    the scaled Schoenfeld residuals at each failure time; the coloured curve
    is a moving-average smoother.  Panels with p < p_threshold have a light
    red background and red title, indicating a potential PH violation.

    Args:
        schoenfeld_data: Output of cox_model.compute_schoenfeld_residuals.
        ph_df: Output of cox_model.check_ph_assumption.
        components: Ordered list of components.
        p_threshold: Significance level for flagging violation.

    Returns:
        List of 6 saved PNG paths.
    """
    from src.cox_model import COVARIATE_LABELS

    cov_cols   = list(COVARIATE_LABELS.keys())    # internal names
    cov_labels = list(COVARIATE_LABELS.values())  # display names

    # Pre-index PH p-values for quick lookup
    ph_lookup: dict[tuple[str, str], float] = {}
    for _, row in ph_df.iterrows():
        ph_lookup[(row["component"], row["covariate"])] = row["p"]

    def _moving_avg(arr: np.ndarray, window: int) -> np.ndarray:
        kernel = np.ones(window) / window
        padded = np.pad(arr, window // 2, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[:len(arr)]

    paths: list[Path] = []

    for comp in components:
        if comp not in schoenfeld_data:
            continue

        resid_df = schoenfeld_data[comp]
        event_times = resid_df["event_time"].values

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        for j, (cov, label) in enumerate(zip(cov_cols, cov_labels)):
            ax = axes[j]
            p_val = ph_lookup.get((comp, cov), 1.0)
            violated = p_val < p_threshold

            if cov not in resid_df.columns:
                ax.set_visible(False)
                continue

            r = resid_df[cov].values
            sort_idx = np.argsort(event_times)
            t_sorted = event_times[sort_idx]
            r_sorted = r[sort_idx]

            # Scatter
            dot_color = "#d62728" if violated else "#4C72B0"
            ax.scatter(t_sorted, r_sorted, s=8, alpha=0.35, color=dot_color, zorder=3)

            # Moving-average smoother (window ≈ 12% of n_events)
            n = len(r_sorted)
            window = max(7, n // 8)
            smooth_r = _moving_avg(r_sorted, window)
            ax.plot(t_sorted, smooth_r, color=dot_color, linewidth=2.2, zorder=4,
                    label="Smoothed")

            # Reference line at 0
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

            # Shaded violation background
            if violated:
                ax.set_facecolor("#fff0f0")
                title_color = "#d62728"
            else:
                title_color = "#222222"

            # p-value annotation
            p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            ax.set_title(f"{label}\n{p_str}", fontsize=8.5,
                         color=title_color, fontweight="bold" if violated else "normal")
            ax.set_xlabel("Time (days)", fontsize=7.5)
            ax.set_ylabel("Scaled residual", fontsize=7.5)
            ax.tick_params(labelsize=7)

        # Hide the 8th panel (7 covariates in 2×4 layout)
        axes[7].set_visible(False)

        p_violated = ph_df[ph_df["component"] == comp]["violated"].sum()
        fig.suptitle(
            f"Schoenfeld Residuals vs. Time — {comp}\n"
            f"({p_violated}/7 covariate(s) with p < {p_threshold}; "
            f"flat pattern = PH holds; trend = possible violation)",
            fontweight="bold", fontsize=11, y=1.01,
        )
        fig.tight_layout()
        path = _save(fig, f"ph_schoenfeld_{comp.lower()}")
        paths.append(path)
        print(f"[viz] Saved {path.name}")

    return paths


# ---------------------------------------------------------------------------
# AFT Model — Phase 7
# ---------------------------------------------------------------------------

def plot_aft_forest_plot(
    af_table: pd.DataFrame,
    components: list[str],
    name: str = "aft_forest_plot",
) -> Path:
    """2×3 Acceleration Factor forest plot — one panel per component (7.3).

    Replicates the Phase 6 HR forest plot layout but for AFT Acceleration
    Factors (AF = exp(coef) from the lambda_ sub-model).

    Colouring follows the user specification:
    - Significant (p < 0.05): **red**
    - Non-significant (p ≥ 0.05): **grey**

    Position encodes direction: AF < 1 (left of reference) = accelerates
    failure; AF > 1 (right of reference) = decelerates failure.  Marker size
    encodes −log₁₀(p) so the most significant covariates are visually largest.

    Args:
        af_table: Output of aft_model.extract_acceleration_factors — must
            contain columns AF, AF_lower_95, AF_upper_95, p, sig,
            covariate_label, component.
        components: Ordered list of component names (defines panel order).
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.cox_model import COVARIATE_LABELS

    cov_order = list(COVARIATE_LABELS.values())
    y_pos = {label: i for i, label in enumerate(cov_order)}

    # Fixed axis limits that show the full data range with margin
    x_lo, x_hi = 0.28, 1.55   # AF range observed: 0.376–1.218

    COLOR_SIG  = "#d62728"  # red — significant
    COLOR_NS   = "#aaaaaa"  # grey — not significant

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), sharey=True)
    axes = axes.flatten()

    for i, comp in enumerate(components):
        ax = axes[i]
        sub = af_table[af_table["component"] == comp].set_index("covariate_label")

        n_sig = 0
        for label in cov_order:
            if label not in sub.index:
                continue
            row = sub.loc[label]
            af, lo, hi, p = row["AF"], row["AF_lower_95"], row["AF_upper_95"], row["p"]
            y   = y_pos[label]
            sig = p < 0.05
            if sig:
                n_sig += 1

            color       = COLOR_SIG if sig else COLOR_NS
            marker_size = max(6, min(15, -np.log10(p + 1e-10) * 2.8))

            # CI line
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6,
                    solid_capstyle="round", zorder=3)
            # End caps (vertical ticks)
            for x_cap in (lo, hi):
                ax.plot([x_cap, x_cap], [y - 0.13, y + 0.13],
                        color=color, linewidth=1.6, zorder=3)
            # Point estimate
            ax.scatter([af], [y], color=color, s=marker_size ** 2,
                       zorder=5, edgecolors="white", linewidth=0.6)

            # AF value + CI + stars on right margin
            ax.text(
                x_hi * 1.06, y,
                f"{af:.3f} [{lo:.3f}–{hi:.3f}] {row['sig']}",
                va="center", ha="left", fontsize=6.5,
                color=color,
                transform=ax.get_yaxis_transform(),
            )

        # Reference line at AF = 1.0
        ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.75,
                   zorder=4)

        # Light shading: left of 1.0 = accelerates, right = decelerates
        ax.axvspan(x_lo, 1.0, alpha=0.03, color="#d62728", zorder=0)
        ax.axvspan(1.0, x_hi, alpha=0.03, color="#1f77b4", zorder=0)

        # Directional x-axis label
        ax.set_xlabel(
            "← Accelerates failure     AF (log scale)     Decelerates failure →",
            fontsize=7.5, labelpad=6,
        )

        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)

        # Fine x-ticks at 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
        ax.set_xticklabels(
            ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.4"],
            fontsize=7,
        )

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(cov_order, fontsize=8.5)
        ax.xaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        # Panel title: component + significant count
        ax.set_title(
            f"{comp}  ({n_sig}/{len(cov_order)} significant)",
            fontweight="bold", fontsize=11, pad=6,
        )

    # Shared legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIG,
               markersize=9, label="p < 0.05 (significant)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NS,
               markersize=9, label="p ≥ 0.05 (not significant)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Weibull AFT Model — Acceleration Factor Forest Plot  (Phase 7.3)\n"
        "AF = exp(coef) from λ sub-model;  marker size ∝ −log₁₀(p);  "
        "* p<0.05, ** p<0.01, *** p<0.001;  covariates are z-score standardised",
        fontweight="bold", fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_aft_vs_km(
    aft_fitters: dict,
    df: pd.DataFrame,
    components: list[str],
    name: str = "aft_vs_km",
) -> Path:
    """2×3 grid: AFT predicted S(t) for the mean profile vs. Kaplan-Meier.

    The KM curve provides the non-parametric reference. The AFT curve is
    the model prediction when all standardised covariates = 0 (i.e. the
    population-mean covariate profile for that component).

    Args:
        aft_fitters: Output of aft_model.fit_aft_by_component.
        df: Validated raw DataFrame (for refitting KM).
        components: Ordered list of component names.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from lifelines import KaplanMeierFitter
    from src.aft_model import predict_mean_survival

    t_grid = np.linspace(365, 2646, 250)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for i, comp in enumerate(components):
        ax = axes[i]
        waf = aft_fitters.get(comp)
        if waf is None:
            ax.set_visible(False)
            continue

        # Kaplan-Meier
        sub_raw = df[df["component"] == comp]
        kmf = KaplanMeierFitter(label="Kaplan-Meier")
        kmf.fit(sub_raw["maintenance_period_days"], sub_raw["event_occurred"])
        kmf.plot_survival_function(ax=ax, ci_show=True, color="black",
                                   linewidth=2.0)

        # AFT mean-profile prediction
        try:
            s_pred = predict_mean_survival(waf, t_grid)
            ax.plot(t_grid, s_pred, color=COMP_COLORS.get(comp, "#e41a1c"),
                    linewidth=2.2, linestyle="--", label="AFT (mean profile)")
        except Exception as exc:
            print(f"[viz] AFT predict failed for {comp}: {exc}")

        # Annotate with C-index and AIC
        ci  = round(float(waf.concordance_index_), 3)
        aic = round(float(waf.AIC_), 0)
        ax.text(0.97, 0.97, f"C-index = {ci}\nAIC = {aic:,.0f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.88))

        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("S(t)", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "Weibull AFT Model — Predicted Survival vs. Kaplan-Meier\n"
        "(AFT curve: mean covariate profile; dashed = AFT, solid = KM)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_aft_survival_profiles(
    aft_fitters: dict,
    components: list[str],
    name: str = "aft_hi_profiles",
) -> Path:
    """2×3 grid: AFT predicted S(t) for five HI_score_last profiles.

    Shows how the dominant predictor (Health Index) shifts the predicted
    survival curve, holding all other standardised covariates at zero.
    Profiles: HI at −2, −1, 0, +1, +2 SD from the population mean.

    Args:
        aft_fitters: Output of aft_model.fit_aft_by_component.
        components: Ordered list of component names.
        name: Output filename stem.

    Returns:
        Path to saved PNG.
    """
    from src.aft_model import build_hi_profiles

    t_grid   = np.linspace(365, 2646, 250)
    n_sd_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    profiles  = build_hi_profiles(n_sd_vals)

    # Diverging palette: red (low HI / poor health) → blue (high HI / good health)
    profile_colors = ["#d62728", "#ff7f0e", "#888888", "#1f77b4", "#2ca02c"]
    profile_labels = [f"HI = {v:+.0f} SD" for v in n_sd_vals]
    profile_lws    = [1.6, 1.6, 2.2, 1.6, 1.6]
    profile_ls     = [":", "--", "-", "--", ":"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for i, comp in enumerate(components):
        ax = axes[i]
        waf = aft_fitters.get(comp)
        if waf is None:
            ax.set_visible(False)
            continue

        for profile_row, color, label, lw, ls in zip(
            profiles.itertuples(index=False),
            profile_colors, profile_labels, profile_lws, profile_ls,
        ):
            profile_df = pd.DataFrame([profile_row._asdict()])
            try:
                sf = waf.predict_survival_function(profile_df, times=t_grid)
                s  = sf.iloc[:, 0].values
                ax.plot(t_grid, s, color=color, linewidth=lw,
                        linestyle=ls, label=label)
            except Exception as exc:
                print(f"[viz] profile predict failed for {comp}: {exc}")

        ax.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("S(t)", fontsize=9)
        ax.set_ylim(0, 1.05)

        if i == 0:
            ax.legend(fontsize=7.5, loc="lower left", title="HI Profile", title_fontsize=7.5)

    fig.suptitle(
        "Weibull AFT — Predicted Survival for HI_score_last Profiles\n"
        "(all other covariates held at population mean; "
        "HI in SD units; −2 SD = very low HI, +2 SD = very high HI)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# RUL Visualisations — Phase 7.4 / 7.5
# ---------------------------------------------------------------------------

# Traffic-light risk palette (user-specified)
_RISK_COLORS: dict[str, str] = {
    "Critical": "#d62728",   # red
    "Warning":  "#ff7f0e",   # orange
    "Monitor":  "#f4d03f",   # yellow
    "Healthy":  "#2ca02c",   # green
}
_RISK_ORDER = ["Critical", "Warning", "Monitor", "Healthy"]

# RUL thresholds in years for bin colouring
_THRESHOLDS_YR: list[float] = [1.0, 2.0, 4.0]  # boundaries between categories


def _bin_color(midpoint_yr: float) -> str:
    """Return the risk-category colour for a histogram bin by its midpoint."""
    if midpoint_yr < _THRESHOLDS_YR[0]:
        return _RISK_COLORS["Critical"]
    if midpoint_yr < _THRESHOLDS_YR[1]:
        return _RISK_COLORS["Warning"]
    if midpoint_yr < _THRESHOLDS_YR[2]:
        return _RISK_COLORS["Monitor"]
    return _RISK_COLORS["Healthy"]


def plot_rul_distribution(
    rul_df: pd.DataFrame,
    components: list[str],
) -> Path:
    """2×3 histograms of RUL_years per component, coloured by risk category.

    Each panel (one per component) shows the distribution of predicted
    Remaining Useful Life in years.  Every histogram bar is coloured by
    the risk category its midpoint falls in:
    Critical (red) < 1 yr | Warning (orange) 1–2 yr |
    Monitor (yellow) 2–4 yr | Healthy (green) ≥ 4 yr.

    A vertical dashed line marks the component's median RUL.
    Vertical dotted lines at 1, 2, and 4 yr show the category boundaries.

    Args:
        rul_df: Output of aft_model.predict_rul.
        components: Ordered list of component names (defines panel order).

    Returns:
        Path to saved PNG at outputs/figures/rul_distribution.png.
    """
    n_bins = 30

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, comp in enumerate(components):
        ax = axes[i]
        rul = rul_df.loc[rul_df["component"] == comp, "RUL_years"].values

        # X-axis capped at 99th percentile so a few extreme tails don't crush the plot
        x_max = max(np.percentile(rul, 99) * 1.08, 5.0)
        bins = np.linspace(0, x_max, n_bins + 1)

        counts, edges, patches = ax.hist(rul, bins=bins, edgecolor="white",
                                         linewidth=0.35)

        # Colour each bar by risk category
        for patch, left, right in zip(patches, edges[:-1], edges[1:]):
            patch.set_facecolor(_bin_color((left + right) / 2))
            patch.set_alpha(0.88)

        # Median RUL line
        median_yr = float(np.median(rul))
        ax.axvline(median_yr, color="black", linewidth=1.8, linestyle="--",
                   label=f"Median = {median_yr:.1f} yr")

        # Category boundary lines
        for threshold_yr, ls in zip(_THRESHOLDS_YR, [":", "--", "-."]):
            if threshold_yr <= x_max:
                ax.axvline(threshold_yr, color="#777777", linewidth=0.9,
                           linestyle=ls, alpha=0.7)

        # Per-panel risk count annotation (top-right)
        cat_counts = rul_df.loc[rul_df["component"] == comp,
                                "risk_category"].value_counts()
        lines = [f"{c}: {cat_counts.get(c, 0)}"
                 for c in _RISK_ORDER if cat_counts.get(c, 0) > 0]
        ax.text(0.97, 0.97, "\n".join(lines),
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))

        ax.set_title(comp, fontweight="bold", fontsize=11)
        ax.set_xlabel("RUL (years)", fontsize=9, labelpad=4)
        ax.set_ylabel("Count", fontsize=9, labelpad=4)
        ax.set_xlim(0, x_max)
        ax.legend(fontsize=8, loc="upper right",
                  handlelength=1.2, framealpha=0.85)

    # Shared colour legend at figure bottom
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_RISK_COLORS[c], label=f"{c}  ({t})")
        for c, t in zip(_RISK_ORDER,
                        ["< 1 yr", "1–2 yr", "2–4 yr", "≥ 4 yr"])
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, title="Risk Category",
               title_fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Predicted RUL Distribution by Component  (Phase 7.5)\n"
        "Bars coloured by risk category; dashed line = component median RUL; "
        "dotted lines = category boundaries",
        fontweight="bold", fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return _save(fig, "rul_distribution")


def plot_risk_category_by_component(
    breakdown: pd.DataFrame,
    components: list[str],
) -> Path:
    """Stacked 100% bar chart of risk-category percentages per component.

    Colours follow the traffic-light scheme specified in Phase 7.5:
    Critical = red, Warning = orange, Monitor = yellow, Healthy = green.
    Percentage labels are shown inside segments ≥ 4 % wide.

    Args:
        breakdown: Output of aft_model.rul_risk_breakdown.
        components: Ordered list of component names.

    Returns:
        Path to saved PNG at outputs/figures/risk_category_by_component.png.
    """
    pct_cols = [f"{c}_pct" for c in _RISK_ORDER]
    pct = breakdown.reindex(components)[pct_cols].fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    x       = np.arange(len(components))
    bottoms = np.zeros(len(components))

    for cat, col in zip(_RISK_ORDER, pct_cols):
        vals = pct[col].values
        bars = ax.bar(x, vals, bottom=bottoms, color=_RISK_COLORS[cat],
                      label=cat, width=0.62, edgecolor="white", linewidth=0.6)

        for bar, v, b in zip(bars, vals, bottoms):
            if v >= 4.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{v:.1f}%",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if v > 10 else "#333333",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=12)
    ax.set_ylabel("Percentage of Observations (%)", fontsize=11, labelpad=6)
    ax.set_xlabel("Component", fontsize=11, labelpad=6)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Risk Category Distribution by Component  (Phase 7.5)\n"
        "Critical < 1 yr  |  Warning 1–2 yr  |  Monitor 2–4 yr  |  Healthy ≥ 4 yr",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.legend(title="Risk Category", loc="lower right",
              fontsize=10, title_fontsize=10, framealpha=0.92)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, "risk_category_by_component")
