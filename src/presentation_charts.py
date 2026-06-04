"""Phase 8 — client-presentation analytics.

Four charts built from outputs/tables/combined_risk_scores.csv:
  1. Risk heatmap (region × event type)
  2. Risk-level distribution by component (100% stacked)
  3. Failure probability vs RUL scatter (quadrant view)
  4. Top-20 critical spans action list

All saved to outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
COMBINED_PATH = ROOT / "outputs" / "tables" / "combined_risk_scores.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Consistent risk-level palette and order.
RISK_COLORS = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#ffd92f", "Low": "#2ca02c"}
RISK_ORDER = ["Critical", "High", "Medium", "Low"]

RF_THRESHOLD = 0.11  # RF val-tuned optimal threshold

REGION_ORDER = ["Northeast", "North", "Central", "South"]
EVENT_ORDER = ["tropical_cyclone", "summer_storm", "southwest_monsoon", "heatwave"]

RISK_FOOTNOTE = "Risk score = 0.6 × failure_prob + 0.4 × (1 − RUL_norm)"


def load_risk() -> pd.DataFrame:
    """Load the combined risk-score table."""
    return pd.read_csv(COMBINED_PATH)


# ---------------------------------------------------------------------------
# Chart 1 — Risk heatmap: region × event type
# ---------------------------------------------------------------------------

def chart_risk_heatmap(df: pd.DataFrame) -> Path:
    """Mean risk_score per region × event_type heatmap."""
    pivot = df.pivot_table(index="region", columns="event_type",
                           values="risk_score", aggfunc="mean")
    pivot = pivot.reindex(index=[r for r in REGION_ORDER if r in pivot.index],
                          columns=[e for e in EVENT_ORDER if e in pivot.columns])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0, vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)

    thresh = pivot.values.max() * 0.6
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > thresh else "black", fontweight="bold")

    fig.colorbar(im, ax=ax, label="Mean risk score")
    ax.set_title("Mean Risk Score by Region and Weather Event Type", fontweight="bold")
    fig.text(0.5, 0.005, RISK_FOOTNOTE, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = FIGURES_DIR / "risk_heatmap_region_event.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 2 — Risk-level distribution by component (100% stacked)
# ---------------------------------------------------------------------------

def chart_distribution_by_component(df: pd.DataFrame) -> Path:
    """100% stacked bars of risk-level share per component, sorted by severity."""
    counts = df.groupby("component")["risk_level"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=RISK_ORDER, fill_value=0)
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100
    order = (pct["Critical"] + pct["High"]).sort_values(ascending=False).index
    pct, totals = pct.loc[order], totals.loc[order]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bottom = np.zeros(len(pct))
    for level in RISK_ORDER:
        ax.bar(pct.index, pct[level], bottom=bottom, label=level,
               color=RISK_COLORS[level], edgecolor="white", linewidth=0.5)
        for i, v in enumerate(pct[level]):
            if v >= 4:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="black")
        bottom += pct[level].values

    for i, comp in enumerate(pct.index):
        ax.text(i, 101.5, f"N={int(totals.loc[comp]):,}", ha="center", fontsize=9,
                fontweight="bold")

    ax.set_ylim(0, 107)
    ax.set_ylabel("Share of assessments (%)")
    ax.set_title("Risk Level Distribution by Component", fontweight="bold")
    ax.legend(title="Risk level", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    path = FIGURES_DIR / "risk_distribution_by_component.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 3 — Failure prob vs RUL scatter (quadrants)
# ---------------------------------------------------------------------------

def chart_failure_prob_vs_rul(df: pd.DataFrame) -> Path:
    """Quadrant scatter of failure_prob vs RUL_days, coloured by risk level."""
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    med_rul = df["RUL_days"].median()
    x_cap = df["RUL_days"].quantile(0.99)

    fig, ax = plt.subplots(figsize=(11, 7))
    for level in RISK_ORDER:
        sub = sample[sample["risk_level"] == level]
        ax.scatter(sub["RUL_days"], sub["failure_prob"], s=18, alpha=0.6,
                   color=RISK_COLORS[level], label=level, edgecolor="none")

    ax.axvline(med_rul, color="grey", ls="--", lw=1)
    ax.axhline(RF_THRESHOLD, color="grey", ls="--", lw=1)
    ax.set_xlim(0, x_cap)
    ax.set_ylim(0, sample["failure_prob"].max() * 1.08)

    y_top = ax.get_ylim()[1]
    quad = dict(ha="center", va="center", fontsize=10, fontweight="bold", alpha=0.55)
    ax.text(med_rul * 0.5, y_top * 0.92, "High Priority\n(low RUL + high prob)", **quad)
    ax.text((med_rul + x_cap) / 2, y_top * 0.92, "Weather-driven risk\n(high RUL + high prob)", **quad)
    ax.text(med_rul * 0.5, y_top * 0.06, "Age-driven risk\n(low RUL + low prob)", **quad)
    ax.text((med_rul + x_cap) / 2, y_top * 0.06, "Low risk\n(high RUL + low prob)", **quad)

    ax.set_xlabel(f"RUL_days  (dashed = median {med_rul:,.0f}; x capped at 99th pct)")
    ax.set_ylabel(f"failure_prob  (dashed = RF threshold {RF_THRESHOLD})")
    ax.set_title("Failure Probability vs Remaining Useful Life", fontweight="bold")
    ax.legend(title="Risk level", loc="center right")
    fig.tight_layout()
    path = FIGURES_DIR / "failure_prob_vs_rul.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 4 — Top 20 critical spans action list
# ---------------------------------------------------------------------------

def chart_top20_critical(df: pd.DataFrame) -> Path:
    """Horizontal bars of the 20 highest-risk units, annotated for action."""
    top = df.sort_values("risk_score", ascending=False).head(20).iloc[::-1]
    labels = [f"{r.span_id} | {r.component} | {r.region}" for r in top.itertuples()]
    colors = [RISK_COLORS[lvl] for lvl in top["risk_level"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(range(len(top)), top["risk_score"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)), labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="orange", ls="--", lw=1.2, label="High (0.5)")
    ax.axvline(0.7, color="red", ls="--", lw=1.2, label="Critical (0.7)")

    for bar, (_, r) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r.event_type} · {r.forecast_date}", ha="right", va="center",
                fontsize=7, color="white", fontweight="bold")

    ax.set_xlabel("Risk score")
    ax.set_title("Top 20 Highest Risk Spans — Immediate Action Required", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "top20_critical_spans.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> list[Path]:
    """Generate all four charts and print a key-stats summary."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_risk()

    paths = [
        chart_risk_heatmap(df),
        chart_distribution_by_component(df),
        chart_failure_prob_vs_rul(df),
        chart_top20_critical(df),
    ]

    if verbose:
        n = len(df)
        n_crit = int((df["risk_level"] == "Critical").sum())
        top = df.loc[df["risk_score"].idxmax()]
        ev = df.groupby("event_type")["risk_score"].mean().sort_values(ascending=False)
        rg = df.groupby("region")["risk_score"].mean().sort_values(ascending=False)

        print("Saved charts:")
        for p in paths:
            print(f"  {p}")
        print("\n=== Key stats summary ===")
        print(f"Total spans assessed: {n:,}")
        print(f"Critical: {n_crit} ({n_crit / n:.1%})")
        print(f"Highest risk span: {top.span_id}, {top.component}, "
              f"risk_score={top.risk_score:.3f}, {top.event_type}")
        print(f"Most dangerous event type: {ev.index[0]} (mean risk {ev.iloc[0]:.3f})")
        print(f"Highest risk region: {rg.index[0]} (mean risk {rg.iloc[0]:.3f})")

    return paths


if __name__ == "__main__":
    run()
