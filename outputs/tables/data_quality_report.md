# Data Quality Report — Phase 8 ML Risk Model

Source: `training_dataset.csv` — 40,320 rows × 54 cols (after dropping hard-leakage cols: failure_mode, maintenance_date, event_occurred).

## 3.1 Leakage check

Risk flags: HIGH if |r| > 0.7, MEDIUM if 0.4 ≤ |r| ≤ 0.7.

| feature | correlation | mean_label1 | mean_label0 | risk |
| --- | --- | --- | --- | --- |
| failure_probability | 0.2762 | 0.2245 | 0.1479 | ok |
| forecast_pressure_min | -0.2260 | 994.6651 | 998.9411 | ok |
| forecast_wind_avg | 0.2146 | 10.4224 | 7.7427 | ok |
| forecast_rain_total | 0.2095 | 212.2218 | 153.3543 | ok |
| forecast_wind_max | 0.2063 | 20.8790 | 15.6310 | ok |
| forecast_month | 0.2041 | 7.4582 | 6.0317 | ok |
| forecast_temp_max | -0.1716 | 31.7272 | 33.2033 | ok |
| forecast_humidity_max | 0.1564 | 96.0524 | 89.9363 | ok |
| HI_score_last | 0.0967 | 4.1631 | 3.8449 | ok |
| HI_trend_per_year | 0.0875 | 1.0796 | 0.9001 | ok |
| total_outage_events | 0.0385 | 1.2493 | 1.1309 | ok |
| permanent_faults | 0.0373 | 0.8049 | 0.7440 | ok |
| age_at_forecast_years | 0.0337 | 2.6884 | 2.5428 | ok |
| age_at_forecast_days | 0.0337 | 981.2780 | 928.1300 | ok |
| transient_faults | 0.0314 | 0.4444 | 0.3870 | ok |
| lightning_flash_density | 0.0288 | 18.2842 | 17.8271 | ok |
| maintenance_period_days | -0.0256 | 1765.0760 | 1800.2200 | ok |
| lightning_caused_trip | 0.0250 | 2.0849 | 1.9895 | ok |
| lightning_strikes_5km_annual | 0.0202 | 41.8324 | 40.7009 | ok |
| avg_rainfall_mm_year | -0.0191 | 1415.2233 | 1436.2403 | ok |
| lightning_strikes_1km_annual | 0.0186 | 9.2344 | 9.0271 | ok |
| temp_fluctuation_c | 0.0144 | 12.2708 | 12.1234 | ok |
| max_peak_current_kA | 0.0133 | 40.5884 | 39.9974 | ok |
| voltage_kv | -0.0111 | 265.1472 | 269.4410 | ok |
| pm25_exceedance_days | 0.0111 | 36.1116 | 34.5662 | ok |
| forecast_confidence_avg | -0.0110 | 74.9963 | 75.0032 | ok |
| pm25_annual_avg | 0.0094 | 48.8546 | 48.1706 | ok |
| row_width_m | -0.0066 | 32.9851 | 33.1767 | ok |
| avg_temperature_c | 0.0046 | 29.7418 | 29.7116 | ok |
| structure_encroachment | 0.0044 | 0.0492 | 0.0466 | ok |
| encroachment_events_total | -0.0027 | 4.4280 | 4.4727 | ok |
| avg_humidity_pct | -0.0026 | 78.9499 | 79.0032 | ok |
| avg_wind_speed_ms | -0.0021 | 4.3498 | 4.3580 | ok |
| mttr_hours | -0.0019 | 16.3086 | 16.3796 | ok |
| max_wind_speed_ms | 0.0015 | 25.8946 | 25.8671 | ok |
| max_temperature_c | 0.0013 | 39.2393 | 39.2340 | ok |
| last_vegetation_cut_days | 0.0007 | 382.3447 | 381.9367 | ok |
| vegetation_encroachment | 0.0007 | 0.2549 | 0.2540 | ok |
| encroachment_caused_trip | 0.0005 | 0.3344 | 0.3331 | ok |

**Dropped from clean train/test:** failure_probability (`failure_probability` is the label-generation artifact; HIGH-risk features removed as leakage).

## 3.2 Class imbalance check

| label | count | pct |
| --- | --- | --- |
| 0 | 33831 | 83.91 |
| 1 | 6489 | 16.09 |

- Failure rate: **16.09%**
- Imbalance ratio (majority/minority): **5.21**
- ✓ within 10–25% target range

Subgroup failure rates (⚠ if < 5% or > 50%):

| dimension | group | n | failure_rate | flag |
| --- | --- | --- | --- | --- |
| component | Arrester | 6720 | 0.1771 |  |
| component | Conductor | 6720 | 0.1333 |  |
| component | Damper | 6720 | 0.1722 |  |
| component | Fittings | 6720 | 0.1341 |  |
| component | Insulator | 6720 | 0.1859 |  |
| component | Spacer | 6720 | 0.1631 |  |
| region | Central | 10080 | 0.1551 |  |
| region | North | 10080 | 0.1553 |  |
| region | Northeast | 10080 | 0.1816 |  |
| region | South | 10080 | 0.1518 |  |
| event_type | heatwave | 10080 | 0.0652 |  |
| event_type | southwest_monsoon | 10080 | 0.1425 |  |
| event_type | summer_storm | 10080 | 0.1316 |  |
| event_type | tropical_cyclone | 10080 | 0.3045 |  |

## 3.3 Feature distribution check

Numeric (⚠ missing>5%, zero>80%, or constant):

| feature | pct_missing | pct_zero | std | flag |
| --- | --- | --- | --- | --- |
| voltage_kv | 0.0000 | 0.0000 | 141.5280 |  |
| maintenance_period_days | 0.0000 | 0.0000 | 505.3859 |  |
| HI_score_last | 0.0000 | 0.0000 | 1.2088 |  |
| HI_trend_per_year | 0.0000 | 0.0000 | 0.7538 |  |
| lightning_strikes_1km_annual | 0.0000 | 0.0000 | 4.0917 |  |
| lightning_strikes_5km_annual | 0.0000 | 0.0000 | 20.6155 |  |
| max_peak_current_kA | 0.0000 | 0.0000 | 16.2865 |  |
| lightning_flash_density | 0.0000 | 0.0000 | 5.8327 |  |
| lightning_caused_trip | 0.0000 | 0.1155 | 1.3997 |  |
| avg_rainfall_mm_year | 0.0000 | 0.0000 | 403.8330 |  |
| max_wind_speed_ms | 0.0000 | 0.0000 | 6.9332 |  |
| avg_wind_speed_ms | 0.0000 | 0.0000 | 1.4593 |  |
| avg_humidity_pct | 0.0000 | 0.0000 | 7.4116 |  |
| avg_temperature_c | 0.0000 | 0.0000 | 2.4334 |  |
| max_temperature_c | 0.0000 | 0.0000 | 1.5153 |  |
| temp_fluctuation_c | 0.0000 | 0.0000 | 3.7595 |  |
| pm25_annual_avg | 0.0000 | 0.0000 | 26.8759 |  |
| pm25_exceedance_days | 0.0000 | 0.4542 | 50.9746 |  |
| row_width_m | 0.0000 | 0.0000 | 10.6794 |  |
| encroachment_events_total | 0.0000 | 0.3452 | 6.1168 |  |
| vegetation_encroachment | 0.0000 | 0.7458 | 0.4354 |  |
| structure_encroachment | 0.0000 | 0.9530 | 0.2117 | zero>80% |
| encroachment_caused_trip | 0.0000 | 0.8185 | 0.8975 | zero>80% |
| last_vegetation_cut_days | 0.0000 | 0.0000 | 200.8220 |  |
| total_outage_events | 0.0000 | 0.3287 | 1.1293 |  |
| transient_faults | 0.0000 | 0.6921 | 0.6733 |  |
| permanent_faults | 0.0000 | 0.3287 | 0.6002 |  |
| mttr_hours | 0.0000 | 0.0000 | 13.7775 |  |
| forecast_month | 0.0000 | 0.0000 | 2.5682 |  |
| age_at_forecast_days | 0.0000 | 0.0000 | 579.6782 |  |
| forecast_wind_max | 0.0000 | 0.0000 | 9.3475 |  |
| forecast_wind_avg | 0.0000 | 0.0000 | 4.5894 |  |
| forecast_rain_total | 0.0000 | 0.0000 | 103.2662 |  |
| forecast_pressure_min | 0.0000 | 0.0000 | 6.9521 |  |
| forecast_humidity_max | 0.0000 | 0.0000 | 14.3745 |  |
| forecast_temp_max | 0.0000 | 0.0000 | 3.1603 |  |
| forecast_confidence_avg | 0.0000 | 0.0000 | 0.2320 |  |
| age_at_forecast_years | 0.0000 | 0.0000 | 1.5882 |  |

Categorical (⚠ any category < 1% of rows):

| feature | category | count | pct | flag |
| --- | --- | --- | --- | --- |
| component | Conductor | 6720 | 16.67 |  |
| component | Damper | 6720 | 16.67 |  |
| component | Spacer | 6720 | 16.67 |  |
| component | Insulator | 6720 | 16.67 |  |
| component | Fittings | 6720 | 16.67 |  |
| component | Arrester | 6720 | 16.67 |  |
| region | Northeast | 10080 | 25.00 |  |
| region | North | 10080 | 25.00 |  |
| region | Central | 10080 | 25.00 |  |
| region | South | 10080 | 25.00 |  |
| event_type | summer_storm | 10080 | 25.00 |  |
| event_type | southwest_monsoon | 10080 | 25.00 |  |
| event_type | heatwave | 10080 | 25.00 |  |
| event_type | tropical_cyclone | 10080 | 25.00 |  |
| coastal_proximity | inland | 20160 | 50.00 |  |
| coastal_proximity | near_coast | 10080 | 25.00 |  |
| coastal_proximity | coastal | 10080 | 25.00 |  |
| pollution_severity | b | 20160 | 50.00 |  |
| pollution_severity | c | 10080 | 25.00 |  |
| pollution_severity | d | 10080 | 25.00 |  |

**Dropped near-zero-variance columns:** structure_encroachment, encroachment_caused_trip (>80% zeros — removed before saving train/test).

## 3.4 Train/test split leakage check

**Cutoff:** `2024-03-31` on `forecast_date` → train 79.5% / test 20.5%. (Spec's nominal 2023-06-30 gave 67/33, outside 70/30–85/15.)

**Label rate:** train 16.13%, test 15.94% (both within 12–22%).

- PRIMARY (time-based, cutoff 2024-03-31):
-   PASS — split ratio train=79.5% / test=20.5% (within 70/30–85/15).
-   PASS — label rate train=16.13%, test=15.94% (both within 12–22%).
-   INFO — span_id leak across time split: 983 (expected; spans recur over years).
- SECONDARY (GroupShuffleSplit, check only — not saved):
-   PASS — span leak=0; train=80.0% / test=20.0%, label rate train=16.24%, test=15.51%.

Side-by-side split comparison:

| method | train_n | train_pct | test_n | test_pct | train_label_rate | test_label_rate | span_leak | span_leak_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time-based (primary) | 32052 | 0.7949 | 8268 | 0.2051 | 0.1613 | 0.1594 | 983 | FAIL (expected) |
| Group-based (check) | 32256 | 0.8000 | 8064 | 0.2000 | 0.1624 | 0.1551 | 0 | PASS |

> **Note:** Time-based split used as primary — matches real deployment where a model trained on historical events predicts future events. The span_id leak under the time split is expected and realistic (the same span experiences forecast events across multiple years); GroupShuffleSplit is reported only as a contrast that eliminates span overlap.

Saved (time-based only): `train.csv` (32,052 rows), `test.csv` (8,268 rows).
