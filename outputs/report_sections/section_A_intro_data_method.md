# รายงานเชิงเทคนิค — การวิเคราะห์การอยู่รอดสำหรับการบำรุงรักษาเชิงพยากรณ์ของสายส่งไฟฟ้าแรงสูง
# Technical Report — Survival Analysis for Predictive Maintenance of High-Voltage Transmission Lines

> ส่วน A: บทนำ ข้อมูล และระเบียบวิธี / Section A: Introduction, Data & Methodology
> กลุ่มผู้อ่านเป้าหมาย: Senior Data Scientist — เขียนแบบ bilingual (ภาษาไทย + ศัพท์เทคนิคภาษาอังกฤษ inline)

---

## 1. บทนำและคำถามวิจัย / Introduction & Research Questions

### 1.1 บริบทของปัญหา / Problem Context

สายส่งไฟฟ้าแรงสูง (high-voltage transmission lines) เป็นโครงสร้างพื้นฐานวิกฤต (critical
infrastructure) ที่ประกอบด้วยอุปกรณ์ (components) หลายชนิดซึ่งเสื่อมสภาพและล้มเหลว (fail) ภายใต้
แรงกดดันจากสิ่งแวดล้อมและการใช้งานที่แตกต่างกัน ทั้งฟ้าผ่า (lightning) ความชื้น (humidity) มลพิษ
ฝุ่นละออง (PM2.5) ลม (wind) การรุกล้ำของพืชพรรณ (vegetation encroachment) และอายุการใช้งาน
สะสม กลยุทธ์การบำรุงรักษาแบบเดิมมักเป็นแบบ time-based (เปลี่ยนตามรอบเวลา) หรือ run-to-failure
(ใช้จนพัง) ซึ่งทั้งสองแบบไม่เหมาะกับสินทรัพย์ที่มีความเสี่ยงไม่เท่ากันในแต่ละจุด เป้าหมายของโครงการนี้
คือการย้ายจากกรอบคิดดังกล่าวไปสู่ **predictive maintenance** ที่ตัดสินใจบนพื้นฐานของความน่าจะเป็น
ของการล้มเหลว (failure probability) ในระดับ component แต่ละชิ้น

### 1.2 ทำไมจึงใช้ Survival Analysis / Why Survival Analysis

ข้อมูลบำรุงรักษามีลักษณะเฉพาะที่วิธีการถดถอยทั่วไป (ordinary regression / classification) จัดการได้ไม่ดี
นั่นคือ **right-censoring** — อุปกรณ์จำนวนมากยัง "ไม่ล้มเหลว" ณ เวลาสิ้นสุดการศึกษา (end of study)
เราจึงทราบเพียงว่า time-to-failure ของมัน "มากกว่า" เวลาที่สังเกตได้ แต่ไม่ทราบค่าจริง การทิ้งข้อมูล
เหล่านี้หรือถือว่าเป็น "ไม่ล้มเหลว" จะทำให้เกิด bias อย่างรุนแรง Survival analysis ออกแบบมาเพื่อใช้
ข้อมูล censored อย่างถูกต้องผ่านแนวคิดของ survival function S(t) และ hazard function h(t)
โดยตรง

โครงการนี้ยึดตามระเบียบวิธีของ **Yang et al. (2022). "Prognostic modeling of predictive
maintenance with survival analysis for mobile work equipment." Scientific Reports, 12, 8529.**
ซึ่งใช้ชุดเครื่องมือ: Kaplan-Meier, Nelson-Aalen, kernel hazard estimator (Epanechnikov),
parametric models (Weibull, Exponential, Log-logistic, Log-normal, Generalized-gamma) และ
ประเมินความเหมาะสมด้วย AIC และ Log-Likelihood Value (LLV) เราขยายกรอบนี้เพิ่มด้วย regression
สองตัว (Cox PH และ Weibull AFT) เพื่อ quantify ผลของ covariates และผลิต RUL ราย component

### 1.3 คำถามวิจัย / Research Questions & Objectives

โครงการนี้ตอบคำถามวิจัยหลัก 4 ข้อ:

1. **ประมาณ survival และ hazard รายชนิดอุปกรณ์ / Estimate survival & hazard per component**
   ประมาณค่า survival function S(t) และ hazard function h(t) สำหรับอุปกรณ์ทั้ง 6 ชนิด
   (Conductor, Damper, Spacer, Insulator, Fittings, Arrester) โดยใช้ทั้งวิธี non-parametric
   (Kaplan-Meier, Nelson-Aalen, kernel hazard) และระบุรูปร่างของ hazard (เพิ่มขึ้น/คงที่/ลดลง)

2. **เปรียบเทียบระหว่างอุปกรณ์และระหว่างภูมิภาค / Compare components & regions**
   ทดสอบทางสถิติว่า survival curves ของอุปกรณ์ต่างชนิด และของภูมิภาคต่างกัน (Northeast,
   North, Central, South) แตกต่างกันอย่างมีนัยสำคัญหรือไม่ ด้วย log-rank test (multivariate
   และ pairwise)

3. **เลือก parametric model ที่ดีที่สุด / Select best parametric model**
   fit distribution ทั้ง 5 ตระกูล (Weibull, Exponential, Log-logistic, Log-normal,
   Generalized-gamma) ราย component แล้วเลือกตัวที่เหมาะสมที่สุดด้วยเกณฑ์ AIC และ LLV
   (Phase 5)

4. **วัดผลของ covariates + ผลิต RUL รายชิ้นและ priority list / Quantify covariate effects +
   individual RUL & priority maintenance list**
   ประมาณผลของปัจจัยสิ่งแวดล้อม/สุขภาพ/การใช้งานที่มีต่อความเสี่ยงด้วย Cox Proportional
   Hazard (HR) และ Weibull AFT (Time Ratio / Acceleration Factor) จากนั้นพยากรณ์
   **Remaining Useful Life (RUL)** ราย component แต่ละชิ้น และจัดลำดับเป็น **priority
   maintenance list** สำหรับการตัดสินใจเชิงปฏิบัติการ

---

## 2. ข้อมูลและนิยาม / Data & Definitions

### 2.1 ภาพรวมชุดข้อมูล / Dataset Overview

- **ไฟล์ / File:** `data/transmission_line_maintenance_data.csv`
- **จำนวนแถว / Rows:** 10,080 แถว — หนึ่งแถวต่อหนึ่ง component ต่อหนึ่ง span (one row per
  component per span)
- **อุปกรณ์ / Components:** 6 ชนิด × 1,680 แถวต่อชนิด = 10,080 (Conductor, Damper, Spacer,
  Insulator, Fittings, Arrester)
- **สายส่ง / Lines:** 8 lines
- **ภูมิภาค / Regions:** 4 ภูมิภาค (Northeast, North, Central, South)
- **ช่วงเวลา / Period:** 2018–2025
- **จำนวนคอลัมน์ / Columns:** 43 คอลัมน์ (ตรวจสอบจาก CSV header โดยตรง)

### 2.2 นิยามหลักด้านการอยู่รอด / Core Survival Definitions

- **Time-to-event (`maintenance_period_days`)** — ระยะเวลาตั้งแต่ติดตั้ง (installation) จนถึง
  เหตุการณ์ (event) หรือจนถึงเวลาตัดข้อมูล (censoring) มีหน่วยเป็นวัน (days) เป็นแกนเวลา t ของ
  survival analysis
- **`event_occurred`** — ตัวบ่งชี้เหตุการณ์ (event indicator): `1` = เกิด failure (uncensored),
  `0` = **right-censored** คือยังไม่ล้มเหลว ณ เวลาที่สังเกต
- **Right-censoring** — สำหรับแถวที่ `event_occurred = 0` เราทราบเพียงว่า time-to-failure จริง
  > `maintenance_period_days` แต่ไม่ทราบค่าที่แท้จริง การประมาณทั้งหมดถูกออกแบบให้ใช้ข้อมูลนี้
  อย่างถูกต้อง (ไม่ทิ้ง และไม่นับเป็น failure)
- **อัตรา censoring / event โดยรวม / Overall censoring & event rate:** censoring rate =
  **84.65%**, event rate = **15.35%** (1,547 failures จาก 10,080 แถว) — ระดับ censoring สูง
  นี้คือเหตุผลสำคัญที่ต้องใช้ survival analysis แทน regression ปกติ

### 2.3 พจนานุกรมคอลัมน์ / Column Dictionary

ตารางต่อไปนี้ครอบคลุมทั้ง 43 คอลัมน์ในไฟล์ จัดกลุ่มตามบทบาท พร้อมความหมาย (ไทย + อังกฤษย่อ),
ชนิดข้อมูล (type), หน่วย (unit) และบทบาทในการวิเคราะห์ (role)

#### (ก) คอลัมน์ Survival / Survival Columns

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `maintenance_period_days` | ระยะเวลาถึงเหตุการณ์ / time-to-event | int | days | **Duration (t)** — แกนเวลาหลักของทุกโมเดล |
| `event_occurred` | ตัวบ่งชี้เหตุการณ์ / 1=failure, 0=right-censored | int (0/1) | — | **Event indicator (δ)** |
| `component` | ชนิดอุปกรณ์ / component type | categorical | — | ตัวแปรจัดกลุ่มหลัก (stratify ทุก phase) |
| `failure_mode` | สาเหตุการล้มเหลว / failure cause (nullable เมื่อ censored) | categorical | — | EDA / failure-mode breakdown (ไม่เข้าโมเดล) |

#### (ข) Covariates สิ่งแวดล้อม / Environmental Covariates

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `lightning_flash_density` | ความหนาแน่นฟ้าผ่า / lightning flash density | float | flashes/km²/yr | **Cox/AFT covariate** |
| `avg_wind_speed_ms` | ความเร็วลมเฉลี่ย / average wind speed | float | m/s | **Cox/AFT covariate** |
| `avg_humidity_pct` | ความชื้นเฉลี่ย / average relative humidity | float | % | **Cox/AFT covariate** |
| `pm25_annual_avg` | ค่าเฉลี่ยฝุ่น PM2.5 รายปี / annual mean PM2.5 | float | µg/m³ | **Cox/AFT covariate** |
| `coastal_proximity` | ระยะใกล้ชายฝั่ง / coastal proximity (inland/near_coast/coastal) | categorical | — | EDA / descriptive |
| `pollution_severity` | ระดับมลพิษ / pollution severity class (b/c/d) | categorical | — | EDA / descriptive |

#### (ค) Health Index

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `HI_score_last` | คะแนนสุขภาพล่าสุด / latest health index score (0–5) | float | score 0–5 | **Cox/AFT covariate** (predictor หลัก) |
| `HI_class_last` | ชั้นสุขภาพล่าสุด / health index class (HI0–HI5) | categorical | — | จัดกลุ่ม KM (km_by_hi_class) |
| `HI_trend_per_year` | แนวโน้ม HI ต่อปี / HI trend per year | float | score/yr | EDA / descriptive |

#### (ง) Encroachment (การรุกล้ำแนวสายส่ง)

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `encroachment_severity` | ระดับการรุกล้ำ / encroachment severity (none/minor/moderate/severe) | ordinal | mapped 0–3 | **Cox/AFT covariate** (เข้ารหัส ordinal 0–3) |
| `vegetation_encroachment` | การรุกล้ำของพืชพรรณ / vegetation encroachment flag | int (0/1) | — | EDA / descriptive |

#### (จ) Outage / Fault

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `total_outage_events` | จำนวนเหตุไฟดับรวม / total outage events | int | count | EDA / descriptive |
| `fault_type_most_common` | ชนิด fault ที่พบบ่อยสุด / most common fault type | categorical | — | EDA / descriptive |
| `permanent_faults` | จำนวน fault ถาวร / permanent faults | int | count | EDA / descriptive |

#### (ฉ) Identifier / Operational Columns และคอลัมน์เสริมอื่น ๆ / Auxiliary Columns

| Column | ความหมาย (Thai + brief English) | Type | Unit | Role in analysis |
|---|---|---|---|---|
| `span_id` | รหัส span / span identifier | id (string) | — | Identifier (priority list) |
| `line_id` | รหัสสายส่ง / line identifier (8 lines) | id (string) | — | Identifier / grouping |
| `tower_id` | รหัสเสา / tower identifier | id | — | Identifier |
| `voltage_kv` | ระดับแรงดัน / line voltage | int | kV | **Cox/AFT covariate** |
| `region` | ภูมิภาค / region (NE/N/Central/South) | categorical | — | จัดกลุ่ม KM (km_by_region) |
| `installation_date` | วันที่ติดตั้ง / installation date | date | YYYY-MM-DD | คำนวณอายุ / priority list |
| `maintenance_date` | วันที่บำรุงรักษา / maintenance date (nullable) | date | YYYY-MM-DD | descriptive |
| `lightning_strikes_1km_annual` | ฟ้าผ่าในรัศมี 1 กม./yr / strikes within 1 km | float | strikes/yr | EDA / สิ่งแวดล้อมเสริม |
| `lightning_strikes_5km_annual` | ฟ้าผ่าในรัศมี 5 กม./yr / strikes within 5 km | float | strikes/yr | EDA / สิ่งแวดล้อมเสริม |
| `max_peak_current_kA` | กระแสยอดสูงสุดของฟ้าผ่า / max peak lightning current | float | kA | EDA |
| `lightning_caused_trip` | trip จากฟ้าผ่า / lightning-caused trip flag | int (0/1) | — | EDA |
| `avg_rainfall_mm_year` | ปริมาณฝนเฉลี่ยรายปี / annual average rainfall | float | mm/yr | EDA |
| `max_wind_speed_ms` | ความเร็วลมสูงสุด / max wind speed | float | m/s | EDA |
| `avg_temperature_c` | อุณหภูมิเฉลี่ย / average temperature | float | °C | EDA |
| `max_temperature_c` | อุณหภูมิสูงสุด / max temperature | float | °C | EDA |
| `temp_fluctuation_c` | ช่วงการแกว่งอุณหภูมิ / temperature fluctuation range | float | °C | EDA |
| `pm25_exceedance_days` | จำนวนวันที่ PM2.5 เกินมาตรฐาน / PM2.5 exceedance days | int | days/yr | EDA |
| `row_width_m` | ความกว้างแนวเขตสายส่ง / right-of-way width | float | m | EDA |
| `encroachment_events_total` | จำนวนเหตุการณ์รุกล้ำรวม / total encroachment events | int | count | EDA |
| `structure_encroachment` | การรุกล้ำของสิ่งปลูกสร้าง / structure encroachment flag | int (0/1) | — | EDA |
| `encroachment_caused_trip` | trip จากการรุกล้ำ / encroachment-caused trip flag | int (0/1) | — | EDA |
| `last_vegetation_cut_days` | จำนวนวันตั้งแต่ตัดพืชครั้งล่าสุด / days since last vegetation cut | int | days | EDA |
| `transient_faults` | จำนวน fault ชั่วคราว / transient faults | int | count | EDA |
| `mttr_hours` | เวลาเฉลี่ยในการซ่อม / mean time to repair | float | hours | EDA |
| `fault_to_failure_flag` | ธง fault ที่ลุกลามเป็น failure / fault-to-failure flag | int (0/1) | — | EDA |

### 2.4 ชุด Covariate 7 ตัวที่ใช้ในโมเดลถดถอย / The 7 Modeling Covariates

Cox PH (Phase 6) และ Weibull AFT (Phase 7) ใช้ covariate ชุดเดียวกัน 7 ตัว เพื่อให้ผลเทียบกันได้
โดยตรง (apples-to-apples) covariate เชิงตัวเลขถูก standardise ด้วย **z-score** จากสถิติของทั้ง
dataset (full-dataset mean/std) เพื่อให้ Hazard Ratio / Acceleration Factor เทียบกันได้ทั่วทั้ง
component และมีการเพิ่ม L2 penalizer = 0.1 เพื่อเสถียรภาพเชิงตัวเลข (numerical stability)

| # | Covariate | การเข้ารหัส / Encoding | หมายเหตุ |
|---|---|---|---|
| 1 | `lightning_flash_density` | z-score standardised | สิ่งแวดล้อม |
| 2 | `avg_wind_speed_ms` | z-score standardised | สิ่งแวดล้อม |
| 3 | `avg_humidity_pct` | z-score standardised | สิ่งแวดล้อม |
| 4 | `pm25_annual_avg` | z-score standardised | สิ่งแวดล้อม |
| 5 | `HI_score_last` | z-score standardised | predictor หลัก (มีนัยสำคัญทั้ง 6 component) |
| 6 | `encroachment_severity` | ordinal map none/minor/moderate/severe → 0/1/2/3, แล้ว standardise | เชิงลำดับ (ordinal) |
| 7 | `voltage_kv` | z-score standardised | เชิงปฏิบัติการ (operational) |

> หมายเหตุ: `coastal_proximity` และ `pollution_severity` เป็น categorical เชิงพรรณนาที่ใช้ใน EDA
> เท่านั้น ไม่ได้เข้า regression เพื่อหลีกเลี่ยง collinearity กับ covariate เชิงสิ่งแวดล้อมที่เป็นตัวเลข

---

## 3. ภาพรวมระเบียบวิธี / Methodology Pipeline

### 3.1 การไหลของ Pipeline แบบ Phase ต่อ Phase / Phase-to-Phase Flow

ระเบียบวิธีถูกจัดเป็นลำดับ phase ที่แต่ละขั้น **ป้อน (feed)** ผลให้ขั้นถัดไปอย่างมีตรรกะชัดเจน:

**Phase 2 — Preprocessing / EDA (`src/preprocessing.py`)**
โหลดและ validate ข้อมูล, สร้าง censoring report, summary statistics, failure-mode breakdown
และ covariate summary ขั้นนี้กำหนดความถูกต้องของ `maintenance_period_days` /
`event_occurred` ซึ่งเป็น input ของทุก phase ถัดไป และให้ context สำคัญคือ censoring rate
84.65% ที่กำหนดทิศทางทั้งหมด → ส่งต่อ duration และ event indicator ที่ตรวจสอบแล้วไปยัง Phase 3

**Phase 3 — Population Survival: KM / NA / log-rank (`src/survival_curves.py`)**
ประมาณ S(t) ด้วย **Kaplan-Meier (KM)** และ cumulative hazard H(t) ด้วย **Nelson-Aalen (NA)**
ราย component / region / HI class แล้วทดสอบความแตกต่างด้วย **log-rank test** (multivariate +
pairwise) → **NA increments (dH) เป็น input โดยตรงของ kernel hazard ใน Phase 4** ส่วน log-rank
ตอบ RQ ข้อ 2 (ความแตกต่างระหว่างกลุ่ม)

**Phase 4 — Hazard Shape: Kernel + Parametric (`src/hazard_models.py`)**
ใช้ **Epanechnikov kernel smoother** บน NA increments (จัดการ censoring ถูกต้อง) เลือก
bandwidth ด้วย **LSCV** (vectorised) เพื่อประมาณรูปร่างของ h(t) แบบ non-parametric พร้อมกับ
fit parametric 5 ตระกูลเพื่อเทียบรูปร่าง → **หลักฐานว่า hazard มีรูปร่าง monotonically decreasing
(ρ < 1) นี้เป็นแรงจูงใจให้ใช้ parametric regression ใน Phase 6–7** และยืนยันความเหมาะสมของตระกูล
Weibull

**Phase 5 — Model Selection: AIC (`src/model_selection.py`)**
สร้าง goodness-of-fit table (LLV + AIC) และ ΔAIC matrix ราย component × model เลือก
distribution ที่ดีที่สุดต่อ component → ผลว่า **Weibull ชนะ 5 ใน 6 component** (ยกเว้น Insulator
ที่ Generalized-gamma ดีที่สุดเล็กน้อย) ยืนยันการเลือกใช้ **Weibull AFT** เป็น regression หลักใน
Phase 7

**Phase 6 — Cox Proportional Hazard (`src/cox_model.py`)**
fit Cox PH ราย component ด้วย covariate 7 ตัว ให้ **Hazard Ratio (HR)** ตอบคำถาม *"หน่วยใด
เสี่ยงสูงกว่า ณ ปัจจุบัน?"* พร้อมตรวจสอบ **PH assumption** ด้วย Schoenfeld residuals test →
ระบุ covariate ที่มีนัยสำคัญ (HI_score_last มีนัยสำคัญทั้ง 6 component) ซึ่งชี้ทิศทางการตีความใน
Phase 7

**Phase 7 — Weibull AFT: RUL + Priority List (`src/aft_model.py`)**
fit **Weibull AFT** ด้วย covariate ชุดเดียวกัน ให้ **Time Ratio / Acceleration Factor (AF)**
ตอบคำถาม *"แต่ละปัจจัยเพิ่ม/ลดอายุเท่าใด?"* จากนั้นพยากรณ์ **RUL** ราย component แต่ละชิ้น จัด
risk category และผลิต **priority maintenance list** (404 Critical units) → ปิด loop กลับสู่การ
ตัดสินใจเชิงปฏิบัติการ (RQ ข้อ 4) ความสัมพันธ์เชิงทฤษฎี HR ≈ AF^(−ρ) เชื่อมโยง Phase 6 และ 7 เข้าด้วยกัน

### 3.2 ความสัมพันธ์เชิงตรรกะ (สรุป) / Logical Dependency Summary

```
Phase 2  duration + event (validated)
   └─► Phase 3  KM S(t) / NA H(t) / log-rank
          └─ NA increments (dH) ──► Phase 4  kernel hazard + parametric shape
                                       └─ shape = decreasing (ρ<1) ──► motivates parametric regression
                                       └─► Phase 5  AIC selection ─ Weibull best ──┐
                                                                                   ▼
   Phase 6  Cox PH (HR, "risk now") ◄── covariate set (7) ──► Phase 7  Weibull AFT (AF, "life added/removed")
                                                                       └─► RUL per unit ─► priority list
```

### 3.3 สินค้าคงคลังของโมเดล (8 core models + 2 regression) / 8-Model Inventory

| Model | Family | Phase | บทบาท / Role — คำถามที่ตอบ |
|---|---|---|---|
| Kaplan-Meier | Non-parametric | 3 | ประมาณ S(t) เชิงประชากร — "สัดส่วนที่ยังอยู่รอด ณ เวลา t?" |
| Nelson-Aalen | Non-parametric | 3 | ประมาณ cumulative hazard H(t); ป้อน increments ให้ kernel |
| Kernel (Epanechnikov) hazard | Non-parametric | 4 | ประมาณรูปร่าง h(t) แบบเรียบ — "hazard เพิ่ม/คง/ลด?" |
| Weibull | Parametric | 4–5 | distribution ยืดหยุ่น (ρ คุมรูป hazard); best model 5/6 component |
| Exponential | Parametric | 4–5 | สมมติ hazard คงที่ (baseline เปรียบเทียบ); ถูกปฏิเสธทุก component |
| Log-logistic | Parametric | 4–5 | รองรับ hazard แบบ non-monotonic (เพิ่มแล้วลด) |
| Log-normal | Parametric | 4–5 | ทางเลือก non-monotonic hazard อีกแบบ |
| Generalized-gamma | Parametric | 4–5 | superfamily ครอบ Weibull/LogNormal/Exponential; best ของ Insulator |
| **Cox PH** *(regression ext.)* | Semi-parametric | 6 | quantify ผล covariate ด้วย **HR** — "หน่วยใดเสี่ยงกว่า?" |
| **Weibull AFT** *(regression ext.)* | Parametric regression | 7 | quantify **AF** + พยากรณ์ **RUL** — "ปัจจัยเพิ่ม/ลดอายุเท่าใด?" |

### 3.4 Tech Stack

| ชั้น / Layer | เครื่องมือ / Tools |
|---|---|
| ภาษา / Language | Python 3.12 (system `python3`) |
| Survival core | lifelines 0.30.3, scikit-survival |
| Data wrangling | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| โครงสร้าง / Structure | `src/` (modules), `notebooks/` (EDA), `outputs/figures/` + `outputs/tables/` (artifacts) |

---

*สิ้นสุดส่วน A — ส่วนถัดไป (B) จะครอบคลุมผลลัพธ์ Phase 2–4 (EDA, KM/NA, hazard estimation).*
*End of Section A — Section B continues with Phase 2–4 results (EDA, KM/NA, hazard estimation).*
