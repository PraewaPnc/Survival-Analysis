# PART B — Individual-Level Analysis / การวิเคราะห์ระดับรายชิ้น

ส่วนนี้เปลี่ยนมุมมองจากการวิเคราะห์ระดับกลุ่ม (population-level: KM, Nelson-Aalen, hazard, model selection) ไปสู่การวิเคราะห์ระดับรายชิ้นส่วน (individual-level) โดยอาศัยแบบจำลองที่มี covariate เพื่อตอบคำถามเชิงปฏิบัติการสองข้อ คือ (1) ปัจจัยใดเร่ง/ชะลอการเสื่อมสภาพ และ (2) อุปกรณ์แต่ละชิ้นเหลืออายุการใช้งานเท่าใด (Remaining Useful Life)

---

## 9. Phase 7 — Weibull AFT Model & Remaining Useful Life / แบบจำลอง Weibull AFT และอายุการใช้งานคงเหลือ

### 9.1 วัตถุประสงค์และความต่างจาก Cox / Purpose & contrast with Cox

แบบจำลอง Accelerated Failure Time (AFT) สร้างแบบจำลอง **เวลาการอยู่รอด (survival time) โดยตรง** ไม่ใช่ hazard rate แบบ Cox PH จุดนี้คือความแตกต่างเชิงแนวคิดที่สำคัญ

| | Cox PH | Weibull AFT |
|---|---|---|
| สร้างแบบจำลองของ | hazard rate (อัตราเสี่ยง) | survival time (เวลาอยู่รอด) |
| ค่าหลักที่รายงาน | Hazard Ratio (HR) | Time Ratio / Acceleration Factor (TR=AF) |
| คำถามที่ตอบ | "ชิ้นใดเสี่ยง **มากกว่า ณ ตอนนี้**?" | "covariate แต่ละตัวเพิ่ม/ลด **อายุ** เท่าใด?" |
| การใช้งานปลายทาง | จัดอันดับความเสี่ยง | ทำนาย RUL รายชิ้น (per-unit) |

เพราะ AFT ทำนายเวลาการอยู่รอดโดยตรง จึงเป็นรากฐานที่ทำให้สามารถคำนวณ **RUL รายชิ้น** ได้ รูปแบบสมการ Weibull AFT คือ

$$\log(T) = \beta_0 + \beta^\top x + \sigma \cdot \varepsilon, \qquad S(t\mid x) = \exp\!\big(-(t\cdot\lambda(x))^{\rho}\big)$$

โดย lambda_ sub-model เข้ารหัสผลของ covariate บน (log) scale parameter และ rho_ เป็น shape sub-model (ค่าคงที่ intercept เท่านั้นโดย default)

### 9.2 การเตรียมข้อมูล / Data preparation

ใช้ pipeline การ standardise และ covariate ชุดเดียวกับ Phase 6 (Cox) ทุกประการ โดยเรียกใช้ `prepare_cox_data` ซ้ำ (ดู `src/aft_model.py` บรรทัด 62) เพื่อให้ผลลัพธ์เทียบกันได้โดยตรง

- **7 covariates (z-score standardised):** `lightning_flash_density`, `avg_wind_speed_ms`, `avg_humidity_pct`, `pm25_annual_avg`, `HI_score_last`, `encroachment_severity_enc` (ordinal 0–3), `voltage_kv`
- **โมเดล:** `WeibullAFTFitter(penalizer=0.1)` แยกฟิตรายชิ้นส่วน (per-component) ผ่าน `fit_aft_by_component`

### 9.3 Time Ratio / Acceleration Factor

TR = AF = exp(coef) ตีความ **ต่อการเปลี่ยน 1 ค่าเบี่ยงเบนมาตรฐาน (1 SD)**

- **AF < 1** → เร่งการเสื่อม (accelerates failure, ลดอายุ)
- **AF > 1** → ชะลอการเสื่อม (decelerates failure, เพิ่มอายุ)
- **AF = 1** → ไม่มีผล

ผลลัพธ์มีนัยสำคัญ (significant, p < 0.05) ทั้งหมด **9/42 combinations** และ **ทั้ง 9 ตัวมี AF < 1 (เร่งการเสื่อมทั้งหมด ไม่มีตัวใดชะลอ)** ตรวจสอบจาก `outputs/tables/aft_time_ratios.csv` และ `aft_acceleration_factors.csv`

| Component | Covariate | coef | AF (TR) | 95% CI | p | Effect |
|---|---|---|---|---|---|---|
| Conductor | Health Index Score | −0.5537 | **0.575** | 0.513–0.644 | <0.0001 | เร่งการเสื่อม |
| Damper | Health Index Score | −0.6841 | **0.505** | 0.442–0.576 | <0.0001 | เร่งการเสื่อม |
| Spacer | Health Index Score | −0.6425 | **0.526** | 0.468–0.591 | <0.0001 | เร่งการเสื่อม |
| Insulator | Health Index Score | −0.7464 | **0.474** | 0.411–0.547 | <0.0001 | เร่งการเสื่อม |
| Fittings | Health Index Score | −0.5635 | **0.569** | 0.508–0.638 | <0.0001 | เร่งการเสื่อม |
| Arrester | Health Index Score | −0.8178 | **0.441** | 0.376–0.518 | <0.0001 | เร่งการเสื่อม |
| Insulator | PM2.5 Annual Avg | −0.1839 | **0.832** | 0.778–0.890 | <0.0001 | เร่งการเสื่อม |
| Insulator | Avg Humidity (%) | −0.1395 | **0.870** | 0.811–0.933 | 0.0001 | เร่งการเสื่อม |
| Arrester | Lightning Flash Density | −0.1146 | **0.892** | 0.843–0.944 | 0.0001 | เร่งการเสื่อม |

**ตีความ:** `HI_score_last` เป็นปัจจัยเด่นในทุกชิ้นส่วน — การเพิ่มขึ้น 1 SD ของ Health Index ลดเวลาอยู่รอดลงเหลือ 44–57% Arrester ไวต่อ HI มากที่สุด (AF=0.441) ส่วนปัจจัยสิ่งแวดล้อมเฉพาะตัว ได้แก่ PM2.5 และความชื้นมีผลต่อ Insulator (สอดคล้องกลไก flashover/การปนเปื้อนพื้นผิว) และฟ้าผ่ามีผลต่อ Arrester (อุปกรณ์ที่ออกแบบมารับ surge โดยตรง)

![AFT acceleration-factor forest plot / แผนภาพ forest ของ Acceleration Factor](outputs/figures/aft_forest_plot.png)
*รูปที่ 23 — Forest plot ของ Acceleration Factor รายชิ้นส่วน (2×3 grid): สีแดง = significant (p<0.05), สีเทา = not significant / AF forest plot per component.*

### 9.4 สรุปคุณภาพแบบจำลอง / AFT model summary

ตรวจสอบจาก `outputs/tables/aft_model_summary.csv`

| Component | AIC | LLV | C-index | rho_ intercept |
|---|---|---|---|---|
| Conductor | 1795.56 | −888.78 | **0.921** | 0.403 |
| Fittings | 2222.68 | −1102.34 | **0.898** | 0.399 |
| Spacer | 4254.38 | −2118.19 | **0.802** | 0.437 |
| Damper | 5161.17 | −2571.58 | **0.754** | 0.418 |
| Insulator | 7676.82 | −3829.41 | **0.739** | 0.447 |
| Arrester | 9227.96 | −4604.98 | **0.702** | 0.575 |

**rho_ intercept อยู่ในช่วง 0.40–0.58** หมายเหตุ: ค่านี้คือ log-shape parameterisation ของ AFT ใน lifelines (ρ ที่นี่ < 1) ซึ่งสอดคล้องเชิงทิศทางกับ wear-out behaviour ที่พบในเฟสก่อนหน้า (kernel hazard และ Weibull shape) C-index สูงสุดที่ Conductor (0.921) และต่ำสุดที่ Arrester (0.702) สะท้อนว่า Arrester มีอัตราการเสีย/ความผันแปรสูงสุด ทำให้ทำนายยากที่สุด

### 9.5 อัลกอริทึมทำนาย RUL / RUL prediction algorithm

(`predict_rul`, `src/aft_model.py` บรรทัด 332–397)

1. ทำนาย **median survival time** ต่อชิ้น ตาม covariate profile ของชิ้นนั้น (`predict_median`) — เลือก median เพราะ robust กว่า mean ในข้อมูลที่ censoring สูง
2. `RUL_days = max(0, predicted_lifetime_days − maintenance_period_days)` (clip ที่ 0)
3. `RUL_years = RUL_days / 365`

**Risk categories:**

| หมวด | เกณฑ์ RUL |
|---|---|
| Critical | < 365 วัน |
| Warning | 365–730 วัน |
| Monitor | 730–1460 วัน |
| Healthy | ≥ 1460 วัน |

### 9.6 การกระจาย RUL และหมวดความเสี่ยง / RUL distribution & risk breakdown

ตรวจสอบจาก `outputs/tables/rul_risk_breakdown.csv` และ `rul_summary_by_component.csv`

| Component | Median RUL | Critical | Warning | Monitor | Healthy |
|---|---|---|---|---|---|
| Conductor | ~42.3 ปี (15451 วัน) | 0% | 0% | 0% | **100%** |
| Fittings | ~32.9 ปี (12002 วัน) | 0% | 0% | 0% | **100%** |
| Spacer | ~10.8 ปี (3934 วัน) | 0% | 0% | 1.8% | 98.2% |
| Damper | ~8.1 ปี (2958 วัน) | 0% | 0% | 6.6% | 93.4% |
| Insulator | ~4.8 ปี (1751 วัน) | **6.9%** | 10.3% | 24.0% | 58.8% |
| Arrester | ~3.1 ปี (1115 วัน) | **17.1%** | 16.5% | 26.8% | 39.6% |

**ตีความ:** Conductor และ Fittings เป็น Healthy 100% สอดคล้องกับอัตราการเสียที่ต่ำมาก (4.9% และ 6.1%) ส่วน Arrester มีสัดส่วน Critical+Warning รวม 33.6% สูงสุด ตรงกับอัตราการเสียที่สังเกตได้ 29.5% ลำดับความเร่งด่วนเรียงตาม median RUL: Arrester < Insulator < Damper < Spacer < Fittings < Conductor

![RUL distribution by component / การกระจาย RUL รายชิ้นส่วน](outputs/figures/rul_distribution.png)
*รูปที่ 24 — ฮิสโทแกรม RUL_years รายชิ้นส่วน (2×3): แท่งระบายสีตามหมวดความเสี่ยง เส้นประ = median / RUL histograms coloured by risk category.*

![Risk category by component / สัดส่วนหมวดความเสี่ยงรายชิ้นส่วน](outputs/figures/risk_category_by_component.png)
*รูปที่ 25 — Stacked 100% bar chart: Critical=แดง, Warning=ส้ม, Monitor=เหลือง, Healthy=เขียว / 100% stacked bar of risk categories.*

![RUL risk breakdown / สรุปหมวดความเสี่ยง RUL](outputs/figures/rul_risk_breakdown.png)
*รูปที่ 26 — สรุปจำนวนและสัดส่วนหมวดความเสี่ยงต่อชิ้นส่วน / Risk-category counts and percentages per component.*

### 9.7 รายการบำรุงรักษาเร่งด่วน / Priority maintenance list

(`generate_priority_list`) ตรวจสอบจาก `outputs/tables/priority_maintenance_list.csv` — **404 Critical units (4.0% ของ 10,080)**

| มิติ | รายละเอียด |
|---|---|
| ชิ้นส่วน | Arrester 288 + Insulator 116 |
| ภูมิภาค | Northeast 184, North 132, Central 61, South 27 |
| Health Index | ~98.3% เป็น HI5 (HI_score เฉลี่ย = 4.998) |
| Top-20 เร่งด่วนสุด | มาจาก LINE-NE2 (Northeast) ทั้งหมด |

ตัวอย่าง 8 แถวแรก (เรียงตาม RUL_days น้อยสุดก่อน):

| span_id | line_id | component | install_date | age (days) | HI | HI_class | RUL_days | pred_life (days) | lightning | region |
|---|---|---|---|---|---|---|---|---|---|---|
| SP-00356 | LINE-NE2 | Arrester | 2018-10-15 | 2359 | 5.0 | HI5 | 0.0 | 2249.7 | 21.70 | Northeast |
| SP-00289 | LINE-NE2 | Insulator | 2019-09-07 | 2032 | 5.0 | HI5 | 0.0 | 2001.5 | 27.92 | Northeast |
| SP-00290 | LINE-NE2 | Insulator | 2018-04-29 | 2528 | 5.0 | HI5 | 0.0 | 2156.3 | 29.28 | Northeast |
| SP-00293 | LINE-NE2 | Arrester | 2018-03-02 | 2586 | 5.0 | HI5 | 0.0 | 2291.8 | 21.87 | Northeast |
| SP-00296 | LINE-NE2 | Arrester | 2018-02-05 | 1991 | 5.0 | HI5 | 0.0 | 1951.6 | 29.30 | Northeast |
| SP-00303 | LINE-NE2 | Arrester | 2018-01-08 | 2639 | 5.0 | HI5 | 0.0 | 1947.2 | 29.84 | Northeast |
| SP-00305 | LINE-NE2 | Arrester | 2019-08-04 | 2066 | 5.0 | HI5 | 0.0 | 2039.0 | 25.94 | Northeast |
| SP-00306 | LINE-NE2 | Insulator | 2018-06-15 | 2481 | 5.0 | HI5 | 0.0 | 2382.1 | 21.25 | Northeast |

> **⚠️ ข้อควรระวัง — Artifact ของข้อมูลจำลอง / Simulated-data artifact**
> การที่ Critical units เกือบทั้งหมด (~98%) เป็น **HI5** เป็นผลจากธรรมชาติของข้อมูลจำลอง ไม่ใช่พฤติกรรมที่คาดในงานจริง กลไกคือ ชิ้นส่วน HI สูงที่อยู่บนสายฟ้าผ่าหนาแน่น (LINE-NE2 ครอง top-20) มี predicted lifetime สั้น เพราะใน AFT ค่า `HI_score_last` สูง **เร่งการเสื่อม** (AF<1) ทำให้ predicted median lifetime ต่ำกว่าอายุปัจจุบัน → RUL=0 → Critical **ในงานจริง** Critical units มักมี HI **ต่ำ** (อุปกรณ์ที่เสื่อมจริงจะมีสุขภาพต่ำ) จึงควรตีความรายการนี้ในเชิงโครงสร้างเมธอด ไม่ใช่ข้อสรุปทางวิศวกรรมโดยตรง

---

## 10. Cox PH vs Weibull AFT — Interplay / การเปรียบเทียบ Cox PH กับ Weibull AFT

(`src/model_comparison.py`) ตรวจสอบจาก `outputs/tables/cox_vs_aft_comparison.csv`

### 10.1 Concordance Index

**AFT เอาชนะ Cox ทั้ง 6 ชิ้นส่วน** (Δ +0.003 ถึง +0.013) แม้ส่วนต่างจะเล็ก แต่สม่ำเสมอทุกชิ้น

| Component | Cox C | AFT C | Δ |
|---|---|---|---|
| Conductor | 0.910 | **0.921** | +0.011 |
| Fittings | 0.885 | **0.898** | +0.013 |
| Spacer | 0.795 | **0.802** | +0.007 |
| Damper | 0.751 | **0.754** | +0.003 |
| Insulator | 0.735 | **0.739** | +0.004 |
| Arrester | 0.694 | **0.702** | +0.008 |

(ค่าใน CSV: Conductor 0.0113, Fittings 0.0129, Spacer 0.0065, Damper 0.0034, Insulator 0.0042, Arrester 0.0077 — ปัดเป็นทศนิยม 3 ตำแหน่งตามตาราง)

### 10.2 ความสอดคล้องของ covariate / Covariate agreement

- **9/42 combinations มีนัยสำคัญใน BOTH (ทั้งสองโมเดล)**
- **ZERO กรณี "Cox only" หรือ "AFT only"** — ทั้งสองโมเดลตั้งธง (flag) ชุด covariate เดียวกันเป๊ะ
- ทุก covariate ที่มีนัยสำคัญ **สอดคล้องเชิงทิศทาง (direction_consistent = True)**: HR > 1 ⟺ AF < 1 (เพิ่มความเสี่ยง = ลดอายุ)

9 ตัวที่ตรงกันคือ HI_score_last ทั้ง 6 ชิ้นส่วน + Insulator (PM2.5, Humidity) + Arrester (Lightning) — ตรงกับ 9 significant AFs ในหัวข้อ 9.3 พอดี

### 10.3 ความสัมพันธ์เชิงทฤษฎีและคู่มือการเลือกใช้ / Relationship & usage guide

ความสัมพันธ์ระหว่างสองโมเดลสำหรับ Weibull โดยประมาณคือ

$$\text{HR} \approx \text{AF}^{-\rho}, \qquad \rho \approx 1.5\text{–}1.8 \text{ (สำหรับชุดข้อมูลนี้)}$$

จึงทำให้ HR>1 และ AF<1 เกิดคู่กันเสมอ และเป็นเหตุผลว่าทำไมสองโมเดลให้ชุด covariate ที่มีนัยสำคัญเหมือนกัน

**เมื่อใดควรใช้แบบใด / When to use which:**

| ใช้ Cox PH เมื่อ | ใช้ Weibull AFT เมื่อ |
|---|---|
| ถามว่า "ชิ้นใดเสี่ยง **ณ ตอนนี้**" | ถามว่า "covariate เพิ่ม/ลด **อายุ** เท่าใด" |
| จัดอันดับความเสี่ยงเชิงสัมพัทธ์ (HR) | ทำนายเวลาอยู่รอด/RUL รายชิ้น (AF) |
| ไม่ต้องสมมติรูปการแจกแจง (semi-parametric) | ต้องการค่าทำนายเป็นวัน/ปี (parametric, fully predictive) |
| วางแผนจัดลำดับการตรวจ | วางแผนตารางเปลี่ยนอุปกรณ์และงบประมาณล่วงหน้า |

![AFT vs Kaplan-Meier / เปรียบเทียบ AFT กับ KM](outputs/figures/aft_vs_km.png)
*รูปที่ 27 — การทำนาย S(t) ด้วย AFT (mean-profile) เทียบกับเส้น Kaplan-Meier รายชิ้นส่วน (2×3) / AFT mean-profile prediction vs KM.*

![AFT HI profiles / โปรไฟล์ Health Index ตาม AFT](outputs/figures/aft_hi_profiles.png)
*รูปที่ 28 — เส้น S(t) ที่ AFT ทำนายเมื่อ HI_score_last อยู่ที่ ±2, ±1, 0 SD รายชิ้นส่วน — แสดงผลเชิงปริมาณของ Health Index ต่ออายุการใช้งาน / AFT predicted survival at varying HI levels.*

**สรุป Part B:** AFT ให้ความสามารถทำนายระดับรายชิ้น (RUL, predicted lifetime) ที่ Cox ทำไม่ได้โดยตรง ขณะที่ทั้งสองโมเดลยืนยันข้อค้นพบเดียวกันว่า **Health Index คือปัจจัยขับเคลื่อนหลัก** ในทุกชิ้นส่วน เสริมด้วยฟ้าผ่า (Arrester) และมลพิษ/ความชื้น (Insulator) — ให้ผลที่สอดคล้องและเสริมกัน (consistent and complementary)
