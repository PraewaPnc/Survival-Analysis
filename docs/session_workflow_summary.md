---
name: session-workflow-summary
description: "Complete survival analysis pipeline built in this session — phases, modules, outputs, and key results"
metadata: 
  node_type: memory
  type: project
  originSessionId: 32a14397-c1b9-4b0c-a8b8-ec092e496cad
---

Full pipeline for electrical transmission line predictive maintenance, replicating Yang et al. (2022) with Cox PH and AFT extensions.

**Why:** Academic/research project to model failure probabilities across 6 component types using survival analysis methods.

**How to apply:** Reference when continuing the project — all modules, outputs, and findings are documented here.

---

## Data
- 10,080 rows (1 per span × component), 6 components, 4 regions, 2018–2025
- Overall failure rate: 15.4%; censoring rate: 84.7%
- Highest failure: Arrester (29.5%), Insulator (23.9%)
- Lowest failure: Conductor (4.9%), Fittings (6.1%)

---

## Phases and Modules

### Phase 1 — Project Setup
- Created `src/`, `notebooks/`, `outputs/`, `requirements.txt`, all module scaffolds
- Python 3.12; lifelines 0.30.3

### Phase 2 — Preprocessing (`src/preprocessing.py`)
- `load_data → validate_data → censoring_report → summary_stats → save_summary`
- `failure_mode_breakdown`, `covariate_summary`, `run_phase2`
- Figures: fig4_duration_histogram_overall.png, fig5_duration_histograms_by_component.png
- Tables: data_summary.csv, failure_mode_breakdown.csv

### Phase 3 — Survival Curves (`src/survival_curves.py`)
- `fit_km_by_group`, `fit_na_by_group`, `km_summary_table`
- `log_rank_test` (multivariate χ²=643.9, p<0.001), `pairwise_log_rank`
- AT_RISK_TIMES = [365, 730, 1095, 1460, 1825, 2190] days
- Figures: fig6_km_by_component.png (with risk table), km_by_region.png, km_by_hi_class.png
- Tables: km_survival_summary.csv, logrank_pairwise_component.csv, logrank_pairwise_region.csv

### Phase 4 — Hazard Estimation (`src/hazard_models.py`)
- Epanechnikov kernel smoother on Nelson-Aalen increments (vectorised LSCV bandwidth)
- 5 parametric models: Weibull, Exponential, LogLogistic, LogNormal, GeneralizedGamma
- `parametric_hazard` uses `hazard_at_times()`
- Figures: fig7–fig12 (one per component), hazard_overview_2x3.png, kernel_hazard_epanechnikov.png

### Phase 5 — Model Selection (`src/model_selection.py`)
- `goodness_of_fit_table` — Table 2 equivalent (wide: component × model × [LLV, AIC])
- `annotate_best` (✓ marker), `delta_aic_matrix`
- Best model: Weibull (5/6 components), GeneralizedGamma (Insulator)
- Exponential rejected everywhere (ΔAIC >> 10)
- Figures: aic_comparison_faceted.png, delta_aic_grouped.png, model_aic_heatmap.png
- Tables: goodness_of_fit.csv, goodness_of_fit_annotated.csv, delta_aic_matrix.csv, model_comparison.csv, fitted_parameters.csv

### Phase 6 — Cox PH (`src/cox_model.py`)
- 7 covariates (z-score standardised): lightning_flash_density, avg_wind_speed_ms,
  avg_humidity_pct, pm25_annual_avg, HI_score_last, encroachment_severity (ordinal 0–3), voltage_kv
- `fit_cox_by_component` — CoxPHFitter(penalizer=0.1), EPV range 11.7–70.9
- `cox_hr_table`, `concordance_table`, `save_cox_results`
- PH check: `check_ph_assumption` + `compute_schoenfeld_residuals` (Schoenfeld rank test)
  → 1/42 marginal violation (Insulator/PM2.5, p=0.046); not significant after Bonferroni
- C-index: 0.694 (Arrester) to 0.910 (Conductor)
- Significant everywhere: HI_score_last (HR 1.65–2.22, p<0.001)
- Figures: cox_forest_plot.png, cox_concordance.png, ph_assumption_heatmap.png, ph_schoenfeld_*.png ×6
- Tables: cox_hazard_ratios.csv, cox_concordance.csv, ph_assumption_test.csv

### Phase 7 — Weibull AFT (`src/aft_model.py`)

**7.1 Fit**
- `fit_aft_by_component` — WeibullAFTFitter(penalizer=0.1), same standardised data as Cox
- `aft_tr_table`, `aft_model_summary` (AIC, LLV, C-index, rho_)
- AFT C-index consistently better than Cox (+0.003 to +0.013)

**7.2 Acceleration Factors**
- `extract_acceleration_factors` — AF=exp(coef), 95% CI, p, sig, effect label
- AF < 1: accelerates failure; AF > 1: decelerates failure
- Table: aft_acceleration_factors.csv (42 rows)

**7.3 AF Forest Plot**
- `plot_aft_forest_plot` — red=significant (p<0.05), grey=ns; directional x-axis
- Figure: aft_forest_plot.png

**7.4 RUL Prediction**
- `predict_rul(df, fitters, method='median')` — adds 4 columns:
  predicted_lifetime_days, RUL_days, RUL_years, risk_category
- Categories: Critical (<365d) / Warning (365–730d) / Monitor (730–1460d) / Healthy (≥1460d)
- `rul_summary_by_component`, `rul_risk_breakdown`, `save_rul_predictions`
- Tables: rul_predictions.csv (10,080 rows), rul_summary_by_component.csv, rul_risk_breakdown.csv

**7.5 RUL Visualisations**
- `plot_rul_distribution` — 2×3 histograms; bars coloured by risk category; median line
- `plot_risk_category_by_component` — stacked 100% bar chart
  (Critical=red, Warning=orange, Monitor=yellow, Healthy=green)
- Figures: rul_distribution.png, risk_category_by_component.png

**7.6 Cox vs AFT Comparison (`src/model_comparison.py`)**
- `compare_concordance`, `compare_covariates`, `build_full_comparison`
- Agreement: Both / Cox only / AFT only / Neither; direction consistency check
- Zero "Cox only" or "AFT only" — both models flag identical covariate sets
- `print_summary` — formatted tables + interpretation statements
- Table: cox_vs_aft_comparison.csv

**7.7 Priority Maintenance List**
- `generate_priority_list` — filters Critical, sorts by RUL_days, prints top-20
- 404 Critical units: Arrester 288, Insulator 116
- All top-20 from LINE-NE2 (Northeast, highest lightning density)
- Table: priority_maintenance_list.csv

---

## Key Results

| Finding | Value |
|---|---|
| Best parametric model | Weibull (5/6), GeneralizedGamma (Insulator) |
| Dominant predictor | HI_score_last — all 6 components, both models, p<0.001 |
| Cox C-index range | 0.694 (Arrester) – 0.910 (Conductor) |
| AFT C-index range | 0.702 (Arrester) – 0.921 (Conductor) |
| PH assumption | Holds (1/42 marginal, not significant after Bonferroni) |
| Critical units | 404/10,080 (4.0%): Arrester + Insulator only |
| Model agreement | Cox and AFT flag identical significant covariates; all directionally consistent |

## Interpretation (Cox vs AFT)
- **Cox PH:** "which units are at higher RISK RIGHT NOW?" — HR > 1 = elevated hazard
- **AFT:** "how much LIFE does each covariate add or remove?" — AF < 1 = shortened survival
- Relationship: HR ≈ AF^(−ρ), ρ ≈ 1.5–1.8 for this dataset

## File counts
- src/ modules: 7
- outputs/figures/: 25 PNGs
- outputs/tables/: 18 CSVs
- notebooks/: 1 (01_full_analysis.ipynb)
