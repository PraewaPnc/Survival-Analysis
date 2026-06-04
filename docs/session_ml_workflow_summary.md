# Session ML Workflow Summary — Phase 8: Weather-Aware Risk Model

Extends the existing survival-analysis pipeline (Phases 1–7) with a weather-aware
failure-risk classifier and a combined risk score. Target: `fault_to_failure_flag`.

## Pipeline overview

```
weather_forecast_72hr.csv
   │  8.1 aggregate (72 hourly rows → 1 per span×event)
   ▼
weather_aggregated.csv ──┐
                         │  8.2 join on span_id + probabilistic label
transmission data ───────┘
   ▼
training_dataset.csv (40,320 × 57)
   │  8.3 data-quality audit + time-based split
   ▼
train.csv (32,052) / test.csv (8,268)
   │  8.4 train RF / XGBoost / LightGBM, tune thresholds, refit on 100%
   ▼
models/  +  ml_model_comparison_final.csv
   │  8.5 blend RF prob + RUL → risk score; single-span inference
   ▼
combined_risk_scores.csv, test_predictions.csv, predict_risk()
```

## Modules (src/)

| Module | Purpose |
|---|---|
| `weather_features.py` | 8.1 forecast aggregation + 8.2 probabilistic label generation |
| `data_quality.py` | 8.3 pre-training audit (leakage, imbalance, distributions, split) |
| `ml_models.py` | 8.4 train/eval/threshold-tune (`run`) + production refit (`refit_full_train`) — RF/XGB/LightGBM |
| `ml_risk_model.py` | 8.5 combined risk (`build_combined_risk_scores`), inference (`predict_risk`), test predictions (`build_test_predictions`) |

---

## 8.1 — Forecast aggregation (`weather_features.py`)

- Collapses 483,840 hourly rows → 6,720 (one per `span_id × event_id`) via a single
  `groupby().agg()` driven by `AGG_SPEC`.
- 7 weather features: `forecast_wind_max` (max gust), `forecast_wind_avg` (mean wind),
  `forecast_rain_total` (sum rain), `forecast_pressure_min`, `forecast_humidity_max`,
  `forecast_temp_max`, `forecast_confidence_avg`. Plus `age_at_forecast_days` and
  `forecast_date` carried via `first` (constant within a 72-hr event).
- Output: `data/weather_aggregated.csv`

## 8.2 — Probabilistic labels (`weather_features.py`)

Joins weather onto transmission data on `span_id` → 40,320 rows (span × component × event).

```
component_score = 0.4·(HI_score_last/5) + 0.3·(HI_trend/HI_trend.max)
               + 0.2·(perm_faults/perm_faults.max) + 0.1·(age_yr/age_yr.max)
weather_score  = 0.4·(wind_max/wind_th) + 0.35·(rain_total/rain_th)
               + 0.25·(1 − pressure_min/1013)
p     = sigmoid(2.5·component_score + 2.0·weather_score + SCORE_INTERCEPT)
label = Bernoulli(p), RandomState(42)
```

- Region rain/wind thresholds: NE (20,150), North (22,180), Central (18,130), South (25,300).
- **`SCORE_INTERCEPT = -4.2`** — calibrated from the spec's nominal −2.8 (which gave a 42.5%
  failure rate because `forecast_rain_total` is a 72-hr sum that often exceeds thresholds).
  −4.3 → ~16%, but adding the 0.1·age term lowered it to 14.8%, so −4.2 → **16.09%**.
- Output: `data/training_dataset.csv` (40,320 × 57)

## 8.3 — Data-quality audit (`data_quality.py`)

| Check | Result |
|---|---|
| Leakage (point-biserial vs label) | No HIGH/MEDIUM; top correlate `forecast_pressure_min` r=−0.23. `failure_probability` artifact dropped. |
| Class imbalance | 16.09% positive, ratio 5.22 (within 10–25%). No subgroup flagged. |
| Distributions | Dropped 2 near-zero-variance cols: `structure_encroachment` (95.3% zero), `encroachment_caused_trip` (81.8% zero). |
| Split (TIME-BASED, primary) | Cutoff `2024-03-31` → train 79.5% / test 20.5%; label rate 16.1% / 15.9%. |

- **Time-based split** simulates real deployment (train on history, predict future). The
  span_id leak across the time split (983) is expected and realistic.
- GroupShuffleSplit run as a contrast only (not saved).
- Outputs: `outputs/tables/data_quality_report.md`, `data/train.csv`, `data/test.csv`

## 8.4 — ML risk models (`ml_models.py`)

**Features (43):** 35 numeric + 8 categorical. Excludes target, IDs (incl. `event_id`),
`forecast_date` (split key), `installation_date`/`age_at_forecast_days` (absorbed by
`age_at_forecast_years`). `region` added to one-hot (spec omitted it). → 71 processed columns.

Three tree ensembles — all have built-in class weighting, so **no SMOTE** (resampling + weighting would
double-correct):
- RF — `class_weight="balanced"`
- XGBoost — `scale_pos_weight = neg/pos ≈ 5.2` (early-stops 20 rounds on val)
- LightGBM — `class_weight="balanced"`, `num_leaves=31` (early-stops 20 rounds on val)

(LSTM was originally the 3rd model but **replaced by LightGBM**: the aggregated 5-feature "pseudo-sequence"
carried no temporal signal and trailed all tree models — a tabular dataset is better served by a 3rd tree.)

**Clean train/val/test protocol:**
1. `run()` — carve a stratified 10% validation slice from train; fit preprocessor + models on
   the 90%; tune F1-optimal thresholds on the val slice (`np.arange(0.1,0.6,0.01)`), **never on test**.
2. `refit_full_train()` — freeze thresholds, refit preprocessor + all models on the full
   32,052-row train (XGB & LightGBM reuse their 200 early-stop trees). This is the deployed artifact set.

**Frozen thresholds:** RF 0.11, XGB 0.50, LightGBM 0.55 (`models/thresholds.json`).

**Final metrics — post-refit on test:**

| Model | Thr | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **RandomForest (primary)** | 0.11 | 0.319 | 0.197 | **0.824** | 0.674 |
| XGBoost | 0.50 | 0.355 | 0.258 | 0.571 | 0.676 |
| **LightGBM** | 0.55 | **0.359** | 0.278 | 0.505 | **0.680** |

**Primary = RandomForest** — recall is the priority metric (a missed failure costs more than a
false alarm); RF catches 82% of failures. LightGBM is best by F1/ROC-AUC. ROC-AUCs are modest (~0.67–0.68) because the label is a
noisy Bernoulli draw (strongest single predictor r≈0.23). XGBoost is weather-dominated
(`forecast_wind_max` = 0.148 importance, 3× the next).

## 8.5 — Combined risk score & inference (`ml_risk_model.py`)

```
risk_score = 0.6 · failure_prob(RF) + 0.4 · (1 − RUL_days / RUL_days.max())
risk_level: ≥0.7 Critical / ≥0.5 High / ≥0.3 Medium / else Low
```

- `build_combined_risk_scores()` — RF failure prob for every test row joined to Phase 7.4 RUL on
  (span_id, component); misses filled with component-median RUL (0 needed). → `combined_risk_scores.csv`.

  | risk_level | count | % |
  |---|---|---|
  | Critical | 14 | 0.2% |
  | High | 1,783 | 21.6% |
  | Medium | 5,663 | 68.5% |
  | Low | 808 | 9.8% |

  All top-10 risk rows are `tropical_cyclone` events, mostly Northeast Insulators/Arresters/Dampers.

- `predict_risk(span_id, forecast_df, rul_days=None)` — single-span inference: aggregates a raw
  72-hr forecast, picks the most severe event (max rainfall), broadcasts onto the span's 6 component
  rows, runs all 3 models, returns **span-level risk = worst (max) component probability**. Returns
  per-model probs, recommended threshold (RF 0.11), combined_risk_score, risk_level. If `rul_days` is
  None → score = RF failure prob alone.

- `build_test_predictions(force=False)` — per-row test predictions: `prob_{rf,xgboost,lightgbm}` +
  `pred_{...}` at frozen thresholds + `actual`. Skips if file exists. → `test_predictions.csv`.

> Note: `failure_prob` (combined_risk_scores) and `prob_rf` (test_predictions) are the same value —
> both are `rf.predict_proba(X)[:, 1]` on the test set.

---

## Key design decisions

1. **Intercept calibration (−2.8 → −4.2)** to keep the simulated failure rate realistic (~16%).
2. **Time-based split as primary** (not random/grouped) — matches deployment; span recurrence
   across years is accepted.
3. **Thresholds tuned on a held-out validation slice, never on test** — then frozen and models
   refit on 100% of train.
4. **RandomForest as primary** despite XGBoost's marginally higher ROC-AUC — recall-first for the
   maintenance use case.
5. **`predict_risk` aggregates components by max** (worst component drives span risk).

## Environment notes

- ML deps: `xgboost 3.2`, `lightgbm 4.6`, `joblib` (all tree ensembles; SMOTE/TensorFlow no longer used).
- macOS/Apple-Silicon: xgboost needs OpenMP. Homebrew's libomp was unavailable, so sklearn's bundled
  `libomp.dylib` was placed at `/opt/homebrew/opt/libomp/lib/libomp.dylib` (xgboost's hardcoded rpath).

## How to reproduce

```bash
python3 -m src.weather_features          # 8.1 + 8.2 → training_dataset.csv
python3 -m src.data_quality              # 8.3 → train.csv / test.csv + report
python3 -m src.ml_models                 # 8.4 run(): train + tune thresholds
python3 -c "from src.ml_models import refit_full_train; refit_full_train()"   # refit on 100%
python3 -m src.ml_risk_model             # 8.5 → combined_risk_scores.csv
python3 -c "from src.ml_risk_model import build_test_predictions; build_test_predictions()"
python3 -c "from src.ml_risk_model import build_priority_forecast_list; build_priority_forecast_list()"
python3 -c "from src.ml_risk_model import plot_roc_pr_curves; plot_roc_pr_curves()"
```

## Artifacts

- **Models:** `models/{rf_model,xgboost_model,lightgbm_model}.pkl`,
  `models/preprocessor.pkl`, `models/thresholds.{json,pkl}`
- **Tables (`outputs/tables/`):**
  - `ml_model_comparison.csv` — default-0.5 test metrics
  - `ml_model_comparison_final.csv` — authoritative: test @ val-tuned thresholds (F1, Precision, Recall,
    ROC-AUC, Accuracy)
  - `combined_risk_scores.csv` — per test row: failure_prob, RUL_days, risk_score, risk_level
  - `test_predictions.csv` — per test row: prob/pred per model + actual
  - `priority_maintenance_list_forecast.csv` — risk scores + weather context, ranked by risk_score desc
  - `data_quality_report.md` — Phase 8.3 audit
- **Figures (`outputs/figures/`):**
  - `ml_feature_importance_{rf,xgboost,lightgbm}.png`
  - `ml_roc_pr_curves.png` — ROC + Precision-Recall for all 3 models (operating thresholds marked)
- **Data (`data/`):** `weather_aggregated.csv`, `training_dataset.csv`, `train.csv`, `test.csv`

### Model performance at a glance (test set @ val-tuned thresholds)

| Model | Thr | F1 | Precision | Recall | ROC-AUC | Accuracy | Avg-Precision |
|---|---|---|---|---|---|---|---|
| **RandomForest (primary)** | 0.11 | 0.319 | 0.197 | **0.824** | 0.674 | 0.438 | 0.290 |
| XGBoost | 0.50 | 0.355 | 0.258 | 0.571 | 0.676 | 0.670 | 0.296 |
| **LightGBM** | 0.55 | **0.359** | 0.278 | 0.505 | **0.680** | 0.712 | **0.299** |

PR baseline (positive rate) = 0.159. RF's low accuracy is by design — its 0.11 threshold maximises recall
(flags ~66% of units), the right trade-off when a missed failure costs more than a false alarm.
