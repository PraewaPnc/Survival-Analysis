# Transmission Line Predictive Maintenance — Survival Analysis Project

## Project Goal
Apply survival analysis (as per Yang et al. 2022, Scientific Reports) to model
failure probabilities of electrical transmission line components using simulated
maintenance records enriched with environmental and operational covariates.

## Data
- File: data/transmission_line_maintenance_data.csv
- Rows: 10,080 (one row per component per span)
- Components: Conductor, Damper, Spacer, Insulator, Fittings, Arrester
- Lines: 8 lines across 4 regions (Northeast, North, Central, South)
- Period: 2018–2025 (installation 2018-01-01 → 2021-12-29)
- Study end / administrative censoring date: **2025-03-31** (`STUDY_END_DATE` in src/preprocessing.py).
  All censored units satisfy installation_date + maintenance_period_days = 2025-03-31; also the latest observed failure date.

### Weather Forecast Data (predictive labeling pipeline)
- File: data/weather_forecast_72hr.csv — 483,840 rows (72 hourly rows per span × event);
  includes age_at_forecast_days and forecast_date (constant within a 72-hr event)
- Aggregated: data/weather_aggregated.csv — 6,720 rows (one per span_id × event_id), 1,680 spans × 4 events
- Training set: data/training_dataset.csv — 40,320 rows (span × component × event), 57 cols
- Time-based split: data/train.csv (32,052 × 51, forecast_date ≤ 2024-03-31) /
  data/test.csv (8,268 × 51, forecast_date > 2024-03-31)

## Key Columns
### Survival Analysis Columns
- maintenance_period_days : time-to-event (int)
- event_occurred          : 1=failure, 0=right-censored
- component               : component type
- failure_mode            : cause of failure (nullable)

### Covariates
- Environmental : lightning_flash_density, avg_wind_speed_ms, avg_humidity_pct,
                  pm25_annual_avg, coastal_proximity, pollution_severity
- Health Index  : HI_score_last (0–5), HI_class_last (HI0–HI5), HI_trend_per_year
- Encroachment  : encroachment_severity, vegetation_encroachment
- Outage        : total_outage_events, fault_type_most_common, permanent_faults

## Reference Paper
Yang et al. (2022). "Prognostic modeling of predictive maintenance with survival
analysis for mobile work equipment." Scientific Reports, 12, 8529.
Methods used: Kaplan-Meier, Nelson-Aalen, Kernel hazard estimator (Epanechnikov),
Parametric models: Weibull, Exponential, Log-logistic, Log-normal, Generalized-gamma
Goodness-of-fit: AIC, Log-Likelihood Value (LLV)

## Tech Stack
- Language : Python 3.12 (system python3)
- Core libs : lifelines 0.30.3, scikit-survival, pandas, numpy, matplotlib, seaborn
- ML (Phase 8.4) : scikit-learn 1.8, xgboost 3.2, lightgbm 4.6, joblib (all tree ensembles; no SMOTE/TF)
- Structure : src/ for modules, notebooks/ for EDA, outputs/ for figures and results, models/ for pickled models
- macOS note : xgboost needs OpenMP. brew's libomp was unavailable, so libomp.dylib (from sklearn's bundled
  copy) is placed at /opt/homebrew/opt/libomp/lib/libomp.dylib — xgboost's hardcoded rpath. No env var needed.

## Code Style
- Type hints on all functions
- Docstrings (Google style)
- Each analysis step saved as PNG to outputs/figures/
- Results tables saved as CSV to outputs/tables/

---

## Implementation Status

### Phase 1 — Project Setup ✓
Directory structure, all source modules, notebook scaffold, requirements.txt.

### Phase 2 — Preprocessing ✓  (src/preprocessing.py)
- `load_data` → `validate_data` → `censoring_report` → `summary_stats` → `save_summary`
- `failure_mode_breakdown`, `covariate_summary`, `run_phase2`
- Figures: fig4_duration_histogram_overall.png, fig5_duration_histograms_by_component.png

### Phase 3 — Survival Curves ✓  (src/survival_curves.py)
- `fit_km_by_group`, `fit_na_by_group`, `km_summary_table`
- `log_rank_test` (multivariate), `pairwise_log_rank`, `save_logrank_results`
- AT_RISK_TIMES = [365, 730, 1095, 1460, 1825, 2190] days
- Figures: fig6_km_by_component.png, km_by_region.png, km_by_hi_class.png

### Phase 4 — Hazard Estimation ✓  (src/hazard_models.py)
- Epanechnikov kernel smoother on Nelson-Aalen increments (correct censoring treatment)
- LSCV bandwidth selection — vectorised, no Python loops over observations
- 5 parametric models: Weibull, Exponential, LogLogistic, LogNormal, GeneralizedGamma
- `parametric_hazard` uses lifelines `hazard_at_times()`
- Figures: fig7–fig12 per component, hazard_overview_2x3.png, kernel_hazard_epanechnikov.png

### Phase 5 — Model Selection ✓  (src/model_selection.py)
- `goodness_of_fit_table` — Table 2 equivalent (wide format, component × model)
- `annotate_best` — marks best AIC per component with ✓
- `delta_aic_matrix` — ΔAIC relative to best model per component
- Figures: aic_comparison_faceted.png, delta_aic_grouped.png, model_aic_heatmap.png
- Tables: goodness_of_fit.csv, goodness_of_fit_annotated.csv, delta_aic_matrix.csv

### Phase 6 — Cox Proportional Hazard ✓  (src/cox_model.py)
- 7 covariates: lightning_flash_density, avg_wind_speed_ms, avg_humidity_pct,
  pm25_annual_avg, HI_score_last, encroachment_severity (ordinal 0–3), voltage_kv
- Z-score standardised using full-dataset stats (globally comparable HRs)
- L2 penalizer=0.1 for numerical stability (EPV range: 11.7–70.9)
- `prepare_cox_data`, `fit_cox_by_component`, `cox_hr_table`, `concordance_table`
- Figures: cox_forest_plot.png (2×3 grid), cox_concordance.png

### Phase 6 — PH Assumption Check ✓  (src/cox_model.py)
- `check_ph_assumption` — Schoenfeld residuals test (rank-transformed time), all 6 components × 7 covariates
- `compute_schoenfeld_residuals` — scaled residuals with event_time column attached for plotting
- `_component_prepared` — internal helper for correct component-wise index alignment
- Figures: ph_assumption_heatmap.png, ph_schoenfeld_{component}.png × 6

### Phase 7 — Weibull AFT Model ✓  (src/aft_model.py)
- Same 7 covariates and standardisation pipeline as Cox (reuses `prepare_cox_data`)
- `fit_aft_by_component` — WeibullAFTFitter(penalizer=0.1) per component
- `aft_tr_table` — Time Ratios (TR = exp(coef)) from lambda_ sub-model, CI, p, sig
- `aft_model_summary` — AIC, LLV, C-index, rho_ shape intercept per component
- `build_hi_profiles` — creates covariate profiles at HI ±2, ±1, 0 SD
- Figures: aft_vs_km.png, aft_hi_profiles.png

### Phase 7.2 — Acceleration Factors ✓  (src/aft_model.py)
- `extract_acceleration_factors` — AF = exp(coef), 95% CI, p, sig, effect label
  per covariate × component; sorted by component then p-value
- `save_acceleration_factors` → outputs/tables/aft_acceleration_factors.csv
- AF < 1: accelerates failure; AF > 1: decelerates failure; AF = 1: no effect

### Phase 7.3 — AF Forest Plot ✓  (src/visualization.py)
- `plot_aft_forest_plot` — 2×3 grid, one panel per component
- Red = significant (p < 0.05); Grey = not significant (p ≥ 0.05)
- Marker size ∝ −log₁₀(p); directional x-axis label with background shading
- Panel subtitle shows (n_sig / 7 significant) count
- Figure: aft_forest_plot.png

### Phase 7.4 — Remaining Useful Life (RUL) ✓  (src/aft_model.py)
- `predict_rul(df, fitters, method='median')` — adds 4 columns to df:
  `predicted_lifetime_days`, `RUL_days`, `RUL_years`, `risk_category`
- RUL = max(0, predicted_median_survival − current_age)
- Risk categories: Critical (<365 d) / Warning (365–730 d) / Monitor (730–1460 d) / Healthy (≥1460 d)
- `rul_summary_by_component` — mean, median, std, min, max RUL per component
- `rul_risk_breakdown` — count + % per risk category per component
- `save_rul_predictions` — saves rul_predictions.csv + summary + breakdown
- Figures: rul_distribution.png, rul_risk_breakdown.png

### Phase 7.5 — RUL Visualisations ✓  (src/visualization.py)
- `plot_rul_distribution(rul_df, components)` — 2×3 histograms (one per component);
  each bin coloured by risk category; median RUL as vertical dashed line;
  x-axis capped at 99th percentile per panel; count annotation per category
- `plot_risk_category_by_component(breakdown, components)` — stacked 100% bar chart;
  Critical=red, Warning=orange, Monitor=yellow, Healthy=green; % labels inside segments
- Figures: rul_distribution.png, risk_category_by_component.png

### Phase 7.6 — Cox PH vs. Weibull AFT Comparison ✓  (src/model_comparison.py)
- `compare_concordance` — C-index table: Cox vs AFT per component + delta
- `compare_covariates` — merges HR and AF tables; computes agreement
  (Both / Cox only / AFT only / Neither) and direction consistency
- `build_full_comparison` — combined DataFrame for CSV + printed summary
- `save_comparison` — two-section CSV (concordance + covariate) in one file
- `print_summary` — formatted table + interpretive statements
- Table: cox_vs_aft_comparison.csv

### Phase 7.7 — Priority Maintenance List ✓  (src/aft_model.py)
- `generate_priority_list(rul_df, output_cols, print_top_n=20)`
  Filters risk_category == "Critical", sorts by RUL_days ascending,
  saves 11 operational columns to priority_maintenance_list.csv,
  prints top-N header table to stdout.
- Output columns: span_id, line_id, component, installation_date,
  maintenance_period_days, HI_score_last, HI_class_last,
  RUL_days, predicted_lifetime_days, lightning_flash_density, region
- Table: priority_maintenance_list.csv  (404 rows)

## Phase 8: ML Risk Model (extends existing pipeline)
Add weather-aware failure risk prediction on top of the existing RUL/AFT results.
Target variable: fault_to_failure_flag (binary, newly generated by probabilistic
simulation — see Phase 8.2 for the label formula and intercept calibration).

### Phase 8 — Weather Features & Predictive Labels ✓  (src/weather_features.py)
**8.1 — Forecast aggregation**
- `load_forecast` → `aggregate_forecast` → `save_aggregated` → `run`
- Collapses 483,840 hourly rows → 6,720 rows (one per span_id × event_id) via single
  `groupby().agg()` driven by `AGG_SPEC` table (source col → (func, output name))
- 7 features: forecast_wind_max (max gust), forecast_wind_avg (mean wind),
  forecast_rain_total (sum rain), forecast_pressure_min (min), forecast_humidity_max (max),
  forecast_temp_max (max), forecast_confidence_avg (mean); region/event_type/forecast_month
  + age_at_forecast_days + forecast_date carried via `first` (constant within a 72-hr event)
- Output: data/weather_aggregated.csv

**8.2 — Probabilistic fault-to-failure labels**
- `build_training_dataset` → `save_training_dataset` → `run_labels`; `_sigmoid` helper (stable logistic)
- Joins weather_aggregated onto transmission data on `span_id` → 40,320 rows (span × component × event)
- Drops transmission's own `fault_to_failure_flag` (replaced) and weather's duplicate `region`
- Adds engineered feature `age_at_forecast_years` = age_at_forecast_days / 365
- `component_score` = 0.4·(HI_score_last/5) + 0.3·(HI_trend/HI_trend.max) + 0.2·(perm_faults/perm_faults.max)
  + 0.1·(age_at_forecast_years/age.max)
- `weather_score` = 0.4·(wind_max/wind_th) + 0.35·(rain_total/rain_th) + 0.25·(1 − pressure_min/1013)
- Region thresholds (`REGION_THRESHOLDS`): NE (20,150), North (22,180), Central (18,130), South (25,300)
- `p = sigmoid(2.5·component_score + 2.0·weather_score + SCORE_INTERCEPT)`, Bernoulli draw with RandomState(42)
- **`SCORE_INTERCEPT = -4.2`** — calibrated from spec's nominal −2.8. forecast_rain_total is a 72-hour *sum*
  that routinely exceeds region rain thresholds, so −2.8 gave 42.5%; −4.3 brought it to ~16%, but adding the
  0.1·age term to component_score lowered it to 14.8%, so the intercept was nudged to **−4.2 → 16.09%**
  (target 15–20%). Coefficients are named constants (`COMPONENT_WEIGHT`, `WEATHER_WEIGHT`, `SCORE_INTERCEPT`).
- Output: data/training_dataset.csv (40,320 × 57), label distribution 0→33,831 / 1→6,489

**8.3 — Pre-training data quality audit ✓  (src/data_quality.py)**
- `load_data` → `leakage_check` → `imbalance_check` → `distribution_check` → `split_and_check` → `run`
- **3.1 Leakage** — point-biserial corr of each numeric feature vs label; mean@label1 vs mean@label0;
  flags HIGH (|r|>0.7) / MEDIUM (0.4–0.7). Hard-leakage cols dropped on load: failure_mode,
  maintenance_date, event_occurred. `failure_probability` (label-generation artifact) dropped from clean
  set — shown in report (r=0.28) but never a real feature. Result: no HIGH/MEDIUM flags; top correlate
  forecast_pressure_min (r=−0.23).
- **3.2 Imbalance** — overall dist (16.09%, ratio 5.22, ✓ within 10–25%); per-subgroup rates flagged if
  <5% or >50% (component/region/event_type). None flagged; range heatwave 6.4% → tropical_cyclone 30.5%.
- **3.3 Distributions** — numeric: %missing (flag>5%), %zero (flag>80%), std (flag=0). Categorical
  (component, region, event_type, coastal_proximity, pollution_severity): value counts, flag <1%.
  Flags: structure_encroachment (95.3% zero), encroachment_caused_trip (81.8% zero); no missing/constants.
  **Both near-zero-variance cols dropped** after the check, before saving train/test (`NEAR_ZERO_VAR_COLS`).
- **3.4 Split — TIME-BASED (primary).** `forecast_date` cast to Timestamp; train = rows ≤ cutoff,
  test = rows > cutoff. **Cutoff = 2024-03-31** (spec's nominal 2023-06-30 gave 67/33, outside 70/30–85/15);
  2024-03-31 → train 32,052 (79.5%) / test 8,268 (20.5%). Label rate train 16.13% / test 15.94% (both within
  12–22%, no re-split). span_id leak across the time split = 983 (expected — spans recur across years).
  Rationale: matches real deployment (train on historical events, predict future events).
  Secondary **GroupShuffleSplit(groups=span_id)** run as a contrast only (NOT saved): 80/20, span leak = 0.
  Report includes a side-by-side comparison table of both methods.
- Outputs: outputs/tables/data_quality_report.md, data/train.csv (32,052 × 51), data/test.csv (8,268 × 51)

### Phase 8.4 — ML Risk Models (RF / XGBoost / LightGBM) ✓  (src/ml_models.py)
- `load_train_test` → `define_features` → `build_preprocessor` → train(RF/XGB/LightGBM)
  → tune thresholds (val slice) → `_metrics` → `plot_importance` → save
- **4.1 Features = 43.** Excludes target + IDs (span_id, line_id, tower_id, **event_id** added — 5,342-level
  ID) + forecast_date (split key), installation_date & age_at_forecast_days (absorbed by age_at_forecast_years),
  failure_probability (already dropped upstream). Split: 35 numeric + 8 categorical.
- **4.2 Preprocessing** — `ColumnTransformer`: StandardScaler (numeric) + OneHotEncoder(handle_unknown=ignore)
  (categorical). **`region` added to the one-hot list** (spec omitted it; would otherwise drop via remainder).
  Fit on train only → 71 processed columns. Saved models/preprocessor.pkl.
- **4.4 All three models are tree ensembles with built-in class weighting, so no resampling (SMOTE removed**
  — applying SMOTE *and* class weighting would double-correct): RF `class_weight="balanced"`;
  XGB `scale_pos_weight=neg/pos≈5.2` (early-stops 20 rounds on val); LightGBM `class_weight="balanced"`,
  `num_leaves=31`, `subsample=0.8`/`subsample_freq=1`, `colsample_bytree=0.8` (early-stops 20 rounds on val).
  All share `learning_rate=0.05`, `max_depth=6`, `n_estimators=200`. (**LSTM was dropped** — the aggregated
  5-feature "pseudo-sequence" had no temporal signal and was the weakest model; a third tree model fits the
  tabular data better.)
- **Clean train/val/test protocol** — `run()`: a stratified 10% validation slice is carved from train;
  preprocessor + all 3 models fit on the 90% slice; XGB & LightGBM early-stop on the val slice; **F1-optimal
  thresholds (`tune_threshold`, sweep `np.arange(0.1,0.6,0.01)`) are picked on the val slice, never on test.**
  Thresholds saved to models/thresholds.{json,pkl}: RF 0.11, XGB 0.50, LightGBM 0.55.
- **Production refit** — `refit_full_train()`: with thresholds frozen, preprocessor + all 3 models are **refit
  on the full 32,052-row train** (XGB & LightGBM reuse the 200 trees from their early-stop runs). Overwrites
  the saved models/preprocessor and ml_model_comparison_final.csv. Final metrics = test @ frozen thresholds.
- **Primary model = RandomForest** (recall 0.824 at thr 0.11 — recall prioritised since a missed failure
  outweighs a false alarm). **LightGBM best F1/ROC-AUC** (0.359 / 0.680); XGBoost in between.
- **4.6 Saved**: models/{rf_model.pkl, xgboost_model.pkl, lightgbm_model.pkl, preprocessor.pkl, thresholds.json/pkl}
- Figures: ml_feature_importance_{rf,xgboost,lightgbm}.png.
  Tables: ml_model_comparison.csv (default 0.5), ml_model_comparison_final.csv (val-tuned, authoritative)

### Phase 8.5 — Combined Risk Score & Inference ✓  (src/ml_risk_model.py)
- **Step 5 `build_combined_risk_scores`** — RF failure prob for every test row (primary model) joined to
  Phase 7.4 RUL on (span_id, component); misses filled with the component-median RUL.
  `risk_score = 0.6·failure_prob + 0.4·(1 − RUL_days/RUL_days.max())`;
  levels: ≥0.7 Critical / ≥0.5 High / ≥0.3 Medium / else Low.
  Output combined_risk_scores.csv (span_id, component, region, event_type, forecast_date, failure_prob,
  RUL_days, risk_score, risk_level). All 8,268 test rows matched RUL, **0 filled** (rul_predictions covers all
  span×component).
- **Step 5.2 `predict_risk(span_id, forecast_df, rul_days=None)`** — single-span inference: aggregates a raw
  72-hr forecast (`weather_features.aggregate_forecast`), picks the span's most severe event (max rainfall),
  broadcasts weather features onto the span's 6 component rows from the maintenance data, runs all 3 models,
  and returns **span-level risk = worst (max) component probability**. Returns dict: span_id, event_type,
  failure_prob_{rf,xgboost,lightgbm}, recommended_threshold (RF 0.11), combined_risk_score, risk_level.
  If rul_days is None → combined_risk_score = RF failure_prob alone; else blends with RUL (normalised by the
  global rul_predictions RUL_days.max()). Artifacts loaded once and cached (`_load_artifacts`).
- **`build_test_predictions(force=False)`** — per-row test predictions: `prob_{rf,xgboost,lightgbm}` +
  thresholded `pred_{...}` (frozen thresholds) + `actual`. Idempotent (skips if file exists).
  → test_predictions.csv. Note: `prob_rf` == `failure_prob` (both `rf.predict_proba(X)[:,1]` on test).
- **`build_priority_forecast_list`** — maps weather_aggregated.csv onto combined_risk_scores.csv on
  (span_id, event_type, forecast_date), ranked by risk_score desc. → priority_maintenance_list_forecast.csv
  (8,268 rows; weather context first, then RUL_days/failure_prob/risk_score/risk_level). All top rows are
  tropical_cyclone, Northeast, Insulator/Arrester/Damper.
- **`plot_roc_pr_curves`** — ROC + Precision-Recall curves for all 3 models on test, with each model's
  val-tuned operating threshold marked. → ml_roc_pr_curves.png. ROC-AUC: LightGBM 0.680, XGB 0.676, RF 0.674;
  Avg-Precision: LightGBM 0.299, XGB 0.296, RF 0.290 (PR baseline = 0.159).

---

## Key Results

### Data Summary (Phase 2)
| Component | N    | Failures | Censoring rate | Median duration |
|-----------|------|----------|----------------|-----------------|
| Overall   | 10,080 | 1,547  | 84.7%          | 1,821 days      |
| Conductor | 1,680 | 82     | 95.1%          | 1,896 days      |
| Damper    | 1,680 | 256    | 84.8%          | 1,812 days      |
| Spacer    | 1,680 | 208    | 87.6%          | 1,842 days      |
| Insulator | 1,680 | 402    | 76.1%          | 1,749 days      |
| Fittings  | 1,680 | 103    | 93.9%          | 1,877 days      |
| Arrester  | 1,680 | 496    | 70.5%          | 1,707 days      |

Top failure modes: end_of_life (382), lightning_damage (369), mechanical_fatigue (244)

### Kaplan-Meier (Phase 3)
- Component log-rank test: χ²=643.9, **p < 0.001**
- All median survival times = ∞ (>50% still surviving at study end — expected with 85% censoring)
- Arrester and Insulator diverge earliest; Conductor and Fittings have highest S(t)

### LSCV Bandwidths (Phase 4)
| Component | Events | Bandwidth |
|-----------|--------|-----------|
| Conductor | 72     | 539 days  |
| Damper    | 183    | 188 days  |
| Spacer    | 152    | 144 days  |
| Insulator | 308    | 77 days   |
| Fittings  | 77     | 591 days  |
| Arrester  | 391    | 418 days  |

### Model Selection — ΔAIC Matrix (Phase 5)
| Component | Weibull | Exponential | LogLogistic | LogNormal | Gen.Gamma | **Best**        |
|-----------|---------|-------------|-------------|-----------|-----------|-----------------|
| Conductor | **0.00** | 26.71      | 0.17        | 1.79      | 1.85      | **Weibull**     |
| Damper    | **0.00** | 64.99      | 0.87        | 0.69      | 1.61      | **Weibull**     |
| Spacer    | **0.00** | 55.68      | 1.51        | 4.08      | 0.66      | **Weibull**     |
| Insulator | 1.87    | 93.82       | 6.82        | 5.02      | **0.00**  | **Gen.Gamma**   |
| Fittings  | **0.00** | 27.71      | 0.27        | 2.03      | 1.77      | **Weibull**     |
| Arrester  | **0.00** | 206.25     | 5.98        | 17.81     | 1.18      | **Weibull**     |

Exponential rejected across all components (ΔAIC >> 10 everywhere).

### Cox PH Results (Phase 6)

**Concordance Index (C-index):**

| Component | C-index | Interpretation |
|-----------|---------|----------------|
| Conductor | 0.910   | Excellent |
| Fittings  | 0.885   | Excellent |
| Spacer    | 0.795   | Good |
| Damper    | 0.751   | Good |
| Insulator | 0.735   | Good |
| Arrester  | 0.694   | Acceptable |

**Significant covariates (p < 0.05):**
- `HI_score_last` — **significant in all 6 components** (HR 1.65–2.22); lower health index = strongly elevated failure risk
- `lightning_flash_density` — significant for Arrester only (HR=1.22, p<0.001)
- `pm25_annual_avg` — significant for Insulator only (HR=1.28, p<0.001)
- `avg_humidity_pct` — significant for Insulator only (HR=1.19, p=0.0003)
- All other covariates: not statistically significant (ns)

### PH Assumption — Schoenfeld Residuals Test (Phase 6)

Test: `proportional_hazard_test()` with rank-transformed event times, 42 combinations (6 components × 7 covariates).

| Component | Violations (p < 0.05) | Detail |
|---|---|---|
| Arrester  | 0 | all p > 0.15 |
| Conductor | 0 | all p > 0.57 |
| Damper    | 0 | all p > 0.30 |
| Fittings  | 0 | all p > 0.23 |
| Spacer    | 0 | all p > 0.45 |
| Insulator | **1** | PM2.5 Annual Avg: p = 0.046 |

**Verdict: PH assumption holds.** The single marginal result (Insulator / PM2.5, p=0.046) disappears under Bonferroni correction (threshold = 0.05/42 ≈ 0.001). `HI_score_last`, the dominant predictor, satisfies PH in all 6 components (p > 0.39 everywhere).

### Weibull AFT Results (Phase 7)

**C-index: AFT consistently slightly better than Cox (+0.003 to +0.013):**

| Component | Cox C-index | AFT C-index |
|---|---|---|
| Conductor | 0.910 | **0.921** |
| Fittings  | 0.885 | **0.898** |
| Spacer    | 0.795 | **0.802** |
| Damper    | 0.751 | **0.754** |
| Insulator | 0.735 | **0.739** |
| Arrester  | 0.694 | **0.702** |

**rho_ shape intercept** 0.40–0.58 across components — ρ < 1 indicates monotonically decreasing hazard (consistent with Phase 4 kernel estimates).

### Acceleration Factors (Phase 7.2 — significant only, p < 0.05)

All 9 significant results have AF < 1 (accelerate failure). No covariate significantly decelerates failure.

| Component | Covariate | coef | AF [95% CI] | Effect |
|---|---|---|---|---|
| All 6 | Health Index Score | −0.55 to −0.82 | **0.44–0.57** [narrow CI] | accelerates failure |
| Insulator | PM2.5 Annual Avg | −0.184 | **0.832** [0.778–0.890] | accelerates failure |
| Insulator | Avg Humidity (%) | −0.140 | **0.870** [0.811–0.933] | accelerates failure |
| Arrester | Lightning Flash Density | −0.115 | **0.892** [0.843–0.944] | accelerates failure |

### RUL Predictions (Phase 7.4)

| Component | Mean RUL | Median RUL | Critical | Warning | Monitor | Healthy |
|---|---|---|---|---|---|---|
| Conductor | 46.5 yr | 42.3 yr | 0% | 0% | 0% | **100%** |
| Fittings  | 38.9 yr | 32.9 yr | 0% | 0% | 0% | **100%** |
| Spacer    | 18.6 yr | 10.8 yr | 0% | 0% | 1.8% | 98.2% |
| Damper    | 14.7 yr |  8.1 yr | 0% | 0% | 6.6% | 93.4% |
| Insulator | 10.7 yr |  4.8 yr | **6.9%** | 10.3% | 24.0% | 58.8% |
| Arrester  |  7.2 yr |  3.1 yr | **17.1%** | 16.5% | 26.8% | 39.6% |

Conductor and Fittings: 100% Healthy — consistent with their very low event rates (4.9% and 6.1%). Arrester has the most critical/warning observations (33.6% combined), matching its 29.5% observed failure rate.

### Cox PH vs. Weibull AFT Comparison (Phase 7.6)

**Concordance Index — AFT wins across all 6 components:**

| Component | Cox C | AFT C | Δ |
|---|---|---|---|
| Conductor | 0.910 | **0.921** | +0.011 |
| Fittings  | 0.885 | **0.898** | +0.013 |
| Spacer    | 0.795 | **0.802** | +0.007 |
| Damper    | 0.751 | **0.754** | +0.003 |
| Insulator | 0.735 | **0.739** | +0.004 |
| Arrester  | 0.694 | **0.702** | +0.008 |

**Covariate agreement:** 9/42 combinations significant in BOTH models. Zero "Cox only" or "AFT only" cases — the two models flag identical covariate sets. All significant covariates are directionally consistent (HR > 1 ↔ AF < 1).

**Interpretation:**
- Cox PH: *"which units are at higher RISK RIGHT NOW?"* → HR, hazard-rate modelling
- AFT: *"how much LIFE does each covariate add or remove?"* → AF, survival-time modelling
- Relationship: HR ≈ AF^(−ρ) where ρ ≈ 1.5–1.8 for this dataset

### Priority Maintenance List (Phase 7.7)

**404 Critical units** (RUL < 1 year) out of 10,080 total (4.0%):

| Component | Count | % of component |
|---|---|---|
| Arrester  | 288 | 17.1% |
| Insulator | 116 |  6.9% |

**By region:** Northeast 184, North 132, Central 61, South 27.

**HI pattern:** All 404 Critical units have HI_score_last ≈ 5.0 (mean=4.998). This is an artifact of the simulated data: high-HI components on high-lightning-density lines (LINE-NE2 dominates) exceed the AFT-predicted lifetime sooner because HI_score_last accelerates failure in the model. In real operations, Critical units would typically have low HI.

**All top-20 most urgent units are from LINE-NE2 (Northeast)** — the highest lightning flash density line in the dataset.

### ML Risk Models (Phase 8.4) — test set (8,268 rows)

Three tree ensembles (RF / XGBoost / LightGBM). **Final comparison — post-refit on 100% of train, evaluated on
test @ frozen val-tuned thresholds.** Protocol: (1) threshold tuning done on a held-out 10% val slice (no test
leakage); (2) thresholds frozen; (3) models refit on the full 32,052-row train (XGB & LightGBM reuse their
200 early-stop trees).

| Model | Val-tuned thr | F1 | Precision | Recall | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|
| **RandomForest (primary)** | **0.11** | 0.319 | 0.197 | **0.824** | 0.674 | 0.438 |
| XGBoost | 0.50 | 0.355 | 0.258 | 0.571 | 0.676 | 0.670 |
| **LightGBM** | 0.55 | **0.359** | 0.278 | 0.505 | **0.680** | 0.712 |

All three ROC-AUCs are modest (0.67–0.68) — expected, since the target is a noisy Bernoulli draw whose strongest
single predictor only reaches point-biserial r≈0.23. **LightGBM is the best by F1 and ROC-AUC**; the three are
within ~0.006 ROC-AUC of each other.

**Primary model = RandomForest.** Rationale: in predictive maintenance a missed failure (false negative) is far
costlier than a false alarm, so **recall is the priority metric**. At its val-tuned threshold 0.11, RF catches
**82% of failures** — the highest recall — while LightGBM/XGBoost catch 51%/57%. RF's low threshold reflects its
compressed probabilities (not "broken", just calibrated low). Thresholds persisted to models/thresholds.json
(+ .pkl): RF 0.11 / XGB 0.50 / LightGBM 0.55. (LSTM was dropped — its aggregated 5-feature pseudo-sequence had
no temporal signal and trailed all tree models; a third tree model suits the tabular data better.)

**Notable findings:**
- **XGBoost is weather-dominated:** `forecast_wind_max` alone = 0.148 importance (3× the next feature),
  then `forecast_pressure_min`, `event_type=tropical_cyclone`, `forecast_rain_total`. Matches the label
  formula, where `weather_score` carries the larger (2.0) coefficient.
- **LightGBM & RandomForest spread importance more evenly** — LightGBM's top splits are mttr_hours, HI_trend,
  age_at_forecast_years, forecast_confidence, HI_score, forecast_rain; RF is similarly flat (top ~0.05).
- Engineered/health features (`age_at_forecast_years`, `HI_score_last`, `HI_trend_per_year`) surface in all
  three models' top ranks — consistent with `component_score` in label generation.

### Combined Risk Score (Phase 8.5) — test set (8,268 rows)

`risk_score = 0.6·RF_failure_prob + 0.4·(1 − RUL_norm)`. RUL join: **8,268 matched / 0 filled** (rul_predictions
covers every span×component, so no component-median fallback was needed).

| risk_level | count | % |
|---|---|---|
| Critical (≥0.7) | 14 | 0.2% |
| High (≥0.5) | 1,783 | 21.6% |
| Medium (≥0.3) | 5,663 | 68.5% |
| Low (<0.3) | 808 | 9.8% |

(Post-refit RF probabilities; built on the 100%-train model.) Most units sit in Medium — the 0.4·(1−RUL_norm)
term keeps scores off the floor even when failure_prob is low, while only 14 units combine high failure
probability with low RUL to reach Critical. **All top-10 risk_score rows are `tropical_cyclone` events**, mostly
Northeast Insulators/Arresters/Dampers — the highest-weather × most-vulnerable-component combinations.

**Inference:** `predict_risk("SP-00001", forecast_df, rul_days=...)` returns per-model probabilities, the
recommended RF threshold (0.11), and the blended `combined_risk_score` + `risk_level` for one span — the
deployment entry point (train on history, score a fresh 72-hr forecast).

---

## Output Files

### outputs/figures/
| File | Phase | Description |
|------|-------|-------------|
| fig4_duration_histogram_overall.png | 2 | Overall duration histogram (censored vs failure) |
| fig5_duration_histograms_by_component.png | 2 | Per-component histograms (2×3 grid) |
| eda_duration_boxplot.png | 2 | Duration boxplot by component (EDA) |
| eda_event_distribution.png | 2 | Event/censoring distribution (EDA) |
| fig6_km_by_component.png | 3 | KM curves, 95% CI, numbers-at-risk table |
| km_by_region.png | 3 | KM curves per region (2×2 grid) |
| km_by_hi_class.png | 3 | KM curves by health index class |
| fig7_hazard_conductor.png | 4 | Conductor: kernel + 5 parametric hazards |
| fig8_hazard_damper.png | 4 | Damper: kernel + 5 parametric hazards |
| fig9_hazard_spacer.png | 4 | Spacer: kernel + 5 parametric hazards |
| fig10_hazard_insulator.png | 4 | Insulator: kernel + 5 parametric hazards |
| fig11_hazard_fittings.png | 4 | Fittings: kernel + 5 parametric hazards |
| fig12_hazard_arrester.png | 4 | Arrester: kernel + 5 parametric hazards |
| hazard_overview_2x3.png | 4 | Overview: kernel vs. best parametric (2×3) |
| kernel_hazard_epanechnikov.png | 4 | Kernel hazard overlay, all components |
| na_cumulative_hazard.png | 4 | Nelson-Aalen cumulative hazard, all components overlay |
| na_cumulative_hazard_subplots.png | 4 | Nelson-Aalen cumulative hazard per component (2×3 grid) |
| aic_comparison_faceted.png | 5 | Faceted AIC bars per component |
| delta_aic_grouped.png | 5 | ΔAIC grouped bar chart with B-A thresholds |
| model_aic_heatmap.png | 5 | AIC heatmap (model × component) |
| cox_forest_plot.png | 6 | Forest plot: HR ± 95% CI per covariate, 2×3 grid |
| cox_concordance.png | 6 | C-index bar chart per component |
| ph_assumption_heatmap.png | 6 | PH test p-value heatmap (component × covariate) |
| ph_schoenfeld_conductor.png | 6 | Schoenfeld residuals vs. time — Conductor |
| ph_schoenfeld_damper.png | 6 | Schoenfeld residuals vs. time — Damper |
| ph_schoenfeld_spacer.png | 6 | Schoenfeld residuals vs. time — Spacer |
| ph_schoenfeld_insulator.png | 6 | Schoenfeld residuals vs. time — Insulator |
| ph_schoenfeld_fittings.png | 6 | Schoenfeld residuals vs. time — Fittings |
| ph_schoenfeld_arrester.png | 6 | Schoenfeld residuals vs. time — Arrester |
| aft_vs_km.png | 7 | AFT mean-profile prediction vs. Kaplan-Meier, 2×3 grid |
| aft_hi_profiles.png | 7 | AFT predicted S(t) for HI at ±2, ±1, 0 SD per component |
| aft_forest_plot.png | 7.3 | AF forest plot: red=significant, grey=ns, directional x-axis, 2×3 grid |
| rul_distribution.png | 7.5 | 2×3 histograms of RUL_years; bars coloured by risk category; median line |
| risk_category_by_component.png | 7.5 | Stacked 100% bar chart; Critical=red, Warning=orange, Monitor=yellow, Healthy=green |
| rul_risk_breakdown.png | 7.5 | Risk category counts/% per component |
| ml_feature_importance_rf.png | 8.4 | RandomForest feature importances (top features) |
| ml_feature_importance_xgboost.png | 8.4 | XGBoost feature importances (weather-dominated) |
| ml_feature_importance_lightgbm.png | 8.4 | LightGBM feature importances (top splits) |
| ml_roc_pr_curves.png | 8.5 | ROC + Precision-Recall curves, all 3 models, val-tuned thresholds marked |
| failure_prob_vs_rul.png | Pres | RF failure probability vs. RUL_days scatter (presentation) |
| risk_distribution_by_component.png | Pres | Combined risk-level distribution per component (presentation) |
| risk_heatmap_region_event.png | Pres | Risk-score heatmap, region × event_type (presentation) |
| top20_critical_spans.png | Pres | Top-20 highest-risk spans bar chart (presentation) |

### outputs/tables/
| File | Phase | Description |
|------|-------|-------------|
| data_summary.csv | 2 | N, events, censoring rate, duration stats per component |
| failure_mode_breakdown.csv | 2 | Component × failure mode pivot |
| km_survival_summary.csv | 3 | S(t) and 95% CI at yearly time points |
| logrank_pairwise_component.csv | 3 | Pairwise log-rank p-values (components) |
| logrank_pairwise_region.csv | 3 | Pairwise log-rank p-values (regions) |
| na_cumulative_hazard.csv | 4 | Nelson-Aalen cumulative hazard estimates per component |
| goodness_of_fit.csv | 5 | Table 2 equivalent: LLV + AIC per model × component |
| goodness_of_fit_annotated.csv | 5 | Same with ✓ marking best model per row |
| delta_aic_matrix.csv | 5 | ΔAIC matrix (component × model) |
| model_comparison.csv | 5 | Long-form AIC/LLV table with rank and ΔAIC |
| fitted_parameters.csv | 5 | Fitted distribution parameters per model × component |
| cox_hazard_ratios.csv | 6 | HR, 95% CI, p-value per covariate × component (long form) |
| cox_hr_detailed.csv   | 6 | Same with z-score and significance stars |
| cox_concordance.csv   | 6 | C-index per component |
| ph_assumption_test.csv | 6 | Schoenfeld test: test_statistic, p, violated flag per covariate × component |
| aft_time_ratios.csv | 7 | TR, 95% CI, p-value per covariate × component (long form) |
| aft_tr_detailed.csv | 7 | Same with z-score and significance stars |
| aft_model_summary.csv | 7 | AIC, LLV, C-index, rho_ intercept per component |
| aft_acceleration_factors.csv | 7.2 | AF, 95% CI, coef, p, sig, effect per covariate × component |
| rul_predictions.csv | 7.4 | Full df + predicted_lifetime_days, RUL_days, RUL_years, risk_category |
| rul_summary_by_component.csv | 7.4 | Mean/median/std/min/max RUL per component |
| rul_risk_breakdown.csv | 7.4 | Risk category counts and % per component |
| cox_vs_aft_comparison.csv | 7.6 | Two-section CSV: concordance + covariate significance agreement (42 rows) |
| priority_maintenance_list.csv | 7.7 | 404 Critical units sorted by RUL_days ascending (11 operational columns) |
| data_quality_report.md | 8.3 | Leakage / imbalance / distribution audit + train-test split comparison |
| ml_model_comparison.csv | 8.4 | RF/XGB/LightGBM metrics at default 0.5 threshold |
| ml_model_comparison_final.csv | 8.4 | RF/XGB/LightGBM metrics, val-tuned thresholds, post-refit (authoritative) |
| combined_risk_scores.csv | 8.5 | Per-test-row combined risk: failure_prob + RUL → risk_score, risk_level (8,268 rows) |
| test_predictions.csv | 8.5 | Per-test-row prob_{rf,xgboost,lightgbm} + thresholded pred_{...} + actual |
| priority_maintenance_list_forecast.csv | 8.5 | Weather-context priority list ranked by risk_score desc (8,268 rows) |
