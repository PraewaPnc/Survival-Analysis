# PART A — Population-Level Analysis / การวิเคราะห์ระดับประชากร

ส่วนนี้วิเคราะห์ข้อมูลในระดับประชากร (population-level) กล่าวคือมองภาพรวมของชุดข้อมูลทั้งหมดและพฤติกรรมการอยู่รอด (survival behaviour) ของแต่ละกลุ่ม component ก่อนที่จะลงรายละเอียดในระดับ covariate และระดับหน่วยรายตัวในส่วนถัดไป จุดมุ่งหมายคือสร้างฐานความเข้าใจเชิงพรรณนา (descriptive) และเชิงนอนพาราเมตริก (non-parametric) ตามแนวทางของ Yang et al. (2022)

---

## 4. Phase 2 — Exploratory Data Analysis & Preprocessing / การสำรวจข้อมูลและการเตรียมข้อมูล

### 4.1 วัตถุประสงค์ของเฟส / Purpose
เฟสนี้มีหน้าที่ตรวจสอบความถูกต้องของข้อมูล (data validation) ทำความสะอาดข้อมูล (cleaning) และสร้างภาพรวมเชิงพรรณนา (descriptive overview) ของชุดข้อมูลก่อนเข้าสู่การวิเคราะห์การอยู่รอด เป้าหมายหลักคือยืนยันว่าโครงสร้างข้อมูลแบบ time-to-event ถูกต้อง (คอลัมน์ `maintenance_period_days` และ `event_occurred`) เข้าใจสัดส่วนการเซ็นเซอร์ (censoring) และระบุรูปแบบการชำรุด (failure mode) เด่นของแต่ละ component เพื่อเป็นบริบทให้กับการตีความผลในเฟสถัดไป

### 4.2 ขั้นตอนการตรวจสอบและเตรียมข้อมูล / Data Validation Steps
ฟังก์ชัน `validate_data` ใน `src/preprocessing.py` ดำเนินการตรวจสอบดังนี้:

1. **Required-columns check** — ตรวจว่ามีคอลัมน์จำเป็นครบ ได้แก่ `maintenance_period_days`, `event_occurred`, `component`, `region` หากขาดจะ raise `ValueError` ทันที
2. **Binary event check** — ตรวจว่า `event_occurred` มีค่าเฉพาะในเซต {0, 1} เท่านั้น (1 = failure, 0 = right-censored) หากพบค่าอื่นจะ raise `ValueError`
3. **Positive-duration filter** — ทิ้งแถวที่ `maintenance_period_days ≤ 0` เนื่องจากเวลาอยู่รอดที่ไม่เป็นบวกไม่มีความหมายในเชิง survival analysis (พิมพ์จำนวนแถวที่ถูกตัดออก)
4. **Censoring-consistency warning** — ตรวจความสอดคล้องว่าหากเป็นแถวที่ถูกเซ็นเซอร์ (`event_occurred == 0`) ก็ไม่ควรมี `failure_mode` ที่ไม่ใช่ค่าว่าง หากพบความไม่สอดคล้องจะแสดงคำเตือน (warning) แต่ยังคงเก็บแถวไว้ตามเดิม (kept as-is)

นอกจากนี้ฟังก์ชัน `load_data` จะ parse คอลัมน์ `installation_date` และ `maintenance_date` เป็น datetime ตั้งแต่ขั้นโหลด

### 4.3 ตารางสรุปข้อมูล / Data Summary Table
อ้างอิงจาก `outputs/tables/data_summary.csv` (ตรวจสอบแล้วตรงกับค่าด้านล่าง)

| Component | N | Events | Event rate | Censored | Censoring rate | Median (days) | Mean (days) | Std (days) |
|-----------|------:|------:|----------:|--------:|---------------:|--------------:|------------:|-----------:|
| **Overall** | 10,080 | 1,547 | 15.35% | 8,533 | 84.65% | 1,820.5 | 1,794.6 | 505.4 |
| Conductor | 1,680 | 82 | 4.88% | 1,598 | 95.12% | 1,895.5 | 1,880.5 | 448.2 |
| Fittings | 1,680 | 103 | 6.13% | 1,577 | 93.87% | 1,876.5 | 1,865.3 | 457.3 |
| Spacer | 1,680 | 208 | 12.38% | 1,472 | 87.62% | 1,842.0 | 1,818.9 | 495.9 |
| Damper | 1,680 | 256 | 15.24% | 1,424 | 84.76% | 1,811.5 | 1,786.3 | 505.6 |
| Insulator | 1,680 | 402 | 23.93% | 1,278 | 76.07% | 1,748.5 | 1,718.8 | 554.0 |
| Arrester | 1,680 | 496 | 29.52% | 1,184 | 70.48% | 1,706.5 | 1,697.6 | 535.3 |

**การตีความ / Interpretation:** ชุดข้อมูลมีการเซ็นเซอร์สูงมากในระดับ 84.65% ซึ่งเป็นลักษณะปกติของข้อมูลบำรุงรักษาอุปกรณ์ที่อายุการใช้งานยาวกว่าระยะเวลาเก็บข้อมูล (study window 2018–2025) มีลำดับความเสี่ยง (risk ranking) ที่ชัดเจน: Arrester (29.5%) และ Insulator (23.9%) มีอัตราการชำรุดสูงสุด ขณะที่ Conductor (4.9%) และ Fittings (6.1%) มีอัตราต่ำสุด ลำดับนี้สอดคล้องกับค่า median duration ที่ลดหลั่นกัน (Arrester สั้นสุดที่ 1,706.5 วัน, Conductor ยาวสุดที่ 1,895.5 วัน) และจะปรากฏซ้ำในทุกเฟสถัดไป

### 4.4 การจำแนกตามรูปแบบการชำรุด / Failure-Mode Breakdown
อ้างอิงจาก `outputs/tables/failure_mode_breakdown.csv`

| Component | Failure modes เด่น (count) |
|-----------|----------------------------|
| **Arrester** | lightning_damage (233), overload (139), end_of_life (124) |
| **Insulator** | lightning_damage (136), contamination_flashover (113), end_of_life (85), mechanical_fatigue (68) |
| **Damper** | corrosion (80), end_of_life (64), mechanical_fatigue (63), wind_damage (49) |
| **Spacer** | mechanical_fatigue (76), end_of_life (72), physical_damage (60) |
| **Fittings** | corrosion (58), mechanical_fatigue (23), end_of_life (22) |
| **Conductor** | corrosion (35), physical_damage (18), end_of_life (15), mechanical_fatigue (14) |

**การตีความ / Interpretation:** รูปแบบการชำรุดสะท้อนกลไกความเสียหายเชิงกายภาพอย่างชัดเจน อุปกรณ์ที่เกี่ยวกับการป้องกันและฉนวน (Arrester, Insulator) ถูกครอบงำด้วย `lightning_damage` ซึ่งเชื่อมโยงโดยตรงกับ covariate `lightning_flash_density` ที่จะมีนัยสำคัญในเฟส Cox/AFT ขณะที่ชิ้นส่วนโลหะ (metal parts: Damper, Fittings, Conductor) มี `corrosion` เป็นสาเหตุหลัก สำหรับ Insulator การมี `contamination_flashover` สอดคล้องกับผลที่ `pm25_annual_avg` และ `avg_humidity_pct` มีนัยสำคัญเฉพาะกับ Insulator ในเฟสถัดไป ภาพรวมทั้งชุดข้อมูล สาเหตุที่พบมากที่สุดคือ end_of_life, lightning_damage และ mechanical_fatigue ตามลำดับ

### 4.5 รูปประกอบ / Reference Figures

![รูปที่ 1 การกระจายสถานะ event/censored / Distribution of event vs. censored status](outputs/figures/eda_event_distribution.png)
*รูปที่ 1 — การกระจายของสถานะ event/censored ทั้งภาพรวมและรายอุปกรณ์ / Distribution of event vs. censored observations overall and per component.*

![รูปที่ 2 Boxplot ของ duration รายอุปกรณ์ / Duration boxplot by component](outputs/figures/eda_duration_boxplot.png)
*รูปที่ 2 — Boxplot ของ `maintenance_period_days` แยกตามอุปกรณ์ / Boxplot of duration grouped by component.*

![รูปที่ 3 ฮิสโทแกรม duration ภาพรวม / Overall duration histogram](outputs/figures/fig4_duration_histogram_overall.png)
*รูปที่ 3 — ฮิสโทแกรมของ duration ภาพรวม แยกสีระหว่าง censored และ failure / Overall duration histogram (censored vs. failure).*

![รูปที่ 4 ฮิสโทแกรม duration รายอุปกรณ์ / Per-component duration histograms](outputs/figures/fig5_duration_histograms_by_component.png)
*รูปที่ 4 — ฮิสโทแกรมของ duration แยกตามอุปกรณ์ (กริด 2×3) / Per-component duration histograms (2×3 grid).*

---

## 5. Phase 3 — Non-parametric Survival / การวิเคราะห์การอยู่รอดแบบนอนพาราเมตริก

### 5.1 วัตถุประสงค์และบทบาทของแต่ละแบบจำลอง / Purpose & Role of Each Model
เฟสนี้ประมาณค่าการอยู่รอดโดย **ไม่กำหนดรูปแบบการแจกแจง** ล่วงหน้า (non-parametric) เพื่อให้ข้อมูล "พูดเอง" ก่อนนำไปเทียบกับแบบจำลองพาราเมตริกในเฟสถัดไป ใช้เวลาอยู่รอดจาก `maintenance_period_days` และตัวบ่งชี้เหตุการณ์จาก `event_occurred` แบ่งวิเคราะห์เป็น 3 เครื่องมือ โดยแต่ละตัวตอบคำถามต่างกัน:

- **Kaplan-Meier — KM (Model 1):** ตอบคำถาม *"ความน่าจะเป็นในการอยู่รอด S(t) ตามเวลาเป็นอย่างไร?"* เป็น **product-limit estimator** ที่ประมาณ S(t) จากผลคูณสะสมของสัดส่วนที่รอดผ่านแต่ละช่วงเวลาที่มีเหตุการณ์เกิดขึ้น แนวคิดเชิงสูตรคือ S(t) = Π (1 − d_i / n_i) สำหรับทุกเวลาเหตุการณ์ t_i ≤ t โดย d_i คือจำนวนการชำรุด ณ เวลา t_i และ n_i คือจำนวนหน่วยที่ยังเสี่ยง (at-risk) ช่วงความเชื่อมั่นใช้สูตร Greenwood (Greenwood CI) ทุกหน่วยที่ถูกเซ็นเซอร์จะถูกนำออกจากชุดเสี่ยงอย่างถูกต้องโดยไม่นับเป็นเหตุการณ์

- **Nelson-Aalen — NA (Model 2):** ตอบคำถาม *"อัตราความเสี่ยงสะสม (cumulative hazard) H(t) = Σ d_i / n_i เป็นเท่าใด?"* เป็นการสะสมความเสี่ยง (accumulated risk) ตลอดเวลา ค่าความชัน (slope) ของ H(t) สะท้อนอัตราความเสี่ยงทันที (instantaneous hazard) ส่วนเพิ่ม (increments) ของ NA จะถูกส่งต่อให้ kernel hazard estimator ในเฟส 4 เพื่อประมาณ hazard function แบบเรียบ (smoothed)

- **Log-rank test:** ตอบคำถาม *"เส้นการอยู่รอดของแต่ละกลุ่มแตกต่างกันอย่างมีนัยสำคัญหรือไม่?"* โดยเปรียบเทียบข้ามกลุ่ม component / region / HI class ภายใต้สมมติฐานหลัก H₀ ว่าเส้น survival ของทุกกลุ่มเหมือนกัน ใช้ทั้งแบบ multivariate (รวมทุกกลุ่ม) และ pairwise (ทีละคู่)

### 5.2 ตารางสรุป Kaplan-Meier / KM Survival Table
อ้างอิงจาก `outputs/tables/km_survival_summary.csv` ประเมินที่ AT_RISK_TIMES = [365, 730, 1095, 1460, 1825, 2190] วัน (1–6 ปี)

| Component | S(1yr) | S(2yr) | S(3yr) | S(4yr) | S(5yr) | **S(6yr)** | Median |
|-----------|-------:|-------:|-------:|-------:|-------:|-----------:|:------:|
| Conductor | 0.994 | 0.991 | 0.984 | 0.973 | 0.958 | **0.939** | ∞ |
| Fittings | 0.992 | 0.985 | 0.978 | 0.966 | 0.946 | **0.922** | ∞ |
| Spacer | 0.983 | 0.971 | 0.948 | 0.928 | 0.897 | **0.851** | ∞ |
| Damper | 0.979 | 0.963 | 0.932 | 0.909 | 0.862 | **0.803** | ∞ |
| Insulator | 0.963 | 0.932 | 0.895 | 0.848 | 0.794 | **0.731** | ∞ |
| Arrester | 0.969 | 0.944 | 0.890 | 0.828 | 0.738 | **0.630** | ∞ |

**การตีความ / Interpretation:** ที่ปีที่ 6 (2,190 วัน) Conductor ยังมีโอกาสรอด 93.9% และ Fittings 92.2% ขณะที่ Arrester เหลือเพียง 63.0% และ Insulator 73.1% — ยืนยันลำดับความเสี่ยงเดียวกับเฟส 2 ทุกกลุ่มมี **median survival = ∞** เพราะไม่มีกลุ่มใดที่ S(t) ลดต่ำกว่า 0.50 ภายในกรอบเวลาการศึกษา ซึ่งเป็นผลที่ **คาดหมายได้** ภายใต้อัตราการเซ็นเซอร์สูงถึง 84.7% (กว่าครึ่งของหน่วยยังคงอยู่รอดเมื่อสิ้นสุดการเก็บข้อมูล) ค่า ∞ จึงไม่ใช่ความผิดพลาด แต่บ่งชี้ว่าต้องอาศัยแบบจำลองพาราเมตริก (เฟส 5–7) เพื่อ extrapolate อายุการใช้งานที่แท้จริง

### 5.3 ผลการทดสอบ Log-rank / Log-rank Test Results

**ตาม component (multivariate):** χ² = 643.9, **p < 0.001** — ปฏิเสธ H₀ อย่างชัดเจน เส้น survival ของ component ต่างกันจริง

**Pairwise ตาม component** (`logrank_pairwise_component.csv`): ทุกคู่แตกต่างกันอย่างมีนัยสำคัญ (p < 0.05) **ยกเว้น** คู่ Conductor–Fittings (p = 0.104) ซึ่งแยกกันไม่ออกในเชิงสถิติ — สอดคล้องกับการที่ทั้งสองมีอัตราการชำรุดต่ำสุดและเส้น KM ที่อยู่ติดกัน

**Pairwise ตาม region** (`logrank_pairwise_region.csv`):

| Pair | p-value | นัยสำคัญ |
|------|--------:|:--------:|
| Central – North | 1.3 × 10⁻¹⁰ | แตกต่างมาก |
| Central – South | 4.7 × 10⁻⁹ | แตกต่างมาก |
| Central – Northeast | 6.5 × 10⁻⁸ | แตกต่างมาก |
| North – Northeast | 0.318 | ไม่แตกต่าง |
| North – South | 0.552 | ไม่แตกต่าง |
| Northeast – South | 0.672 | ไม่แตกต่าง |

**การตีความ / Interpretation:** ภูมิภาค **Central แตกต่างจากทุกภูมิภาคอื่นอย่างมีนัยสำคัญสูง** (p < 1 × 10⁻⁷ ทุกคู่) ขณะที่ North, Northeast และ South **แยกกันไม่ออก** ในเชิงสถิติ (p = 0.318–0.672) ชี้ว่ามีปัจจัยเชิงภูมิศาสตร์/สิ่งแวดล้อมเฉพาะของ Central ที่ทำให้พฤติกรรมการอยู่รอดต่างจากภูมิภาคอื่น ซึ่งจะถูกอธิบายเชิงปริมาณผ่าน covariate ในเฟส Cox/AFT

### 5.4 รูปประกอบ / Reference Figures

![รูปที่ 5 เส้นโค้ง Kaplan-Meier รายอุปกรณ์ / Kaplan-Meier survival curves by component](outputs/figures/fig6_km_by_component.png)
*รูปที่ 5 — เส้นโค้ง KM แยกตามอุปกรณ์ พร้อมแถบ 95% CI และตาราง numbers-at-risk / KM survival curves per component with 95% CI and numbers-at-risk table.*

![รูปที่ 6 เส้นโค้ง KM รายภูมิภาค / Kaplan-Meier curves by region](outputs/figures/km_by_region.png)
*รูปที่ 6 — เส้นโค้ง KM แยกตามภูมิภาค (กริด 2×2) — Central แยกออกชัดเจนจากภูมิภาคอื่น / KM curves per region (2×2 grid); Central separates clearly from the others.*

![รูปที่ 7 เส้นโค้ง KM ตามชั้น Health Index / Kaplan-Meier curves by health index class](outputs/figures/km_by_hi_class.png)
*รูปที่ 7 — เส้นโค้ง KM แยกตามชั้น Health Index (HI0–HI5) แสดงการลดลงของ S(t) เมื่อ HI สูงขึ้น / KM curves by health index class (HI0–HI5).*

### 5.5 เชื่อมโยงสู่เฟสถัดไป / Link to Next Phase
Nelson-Aalen ให้ภาพ **cumulative hazard H(t)** ที่มีรูปทรงเว้า (concave) บ่งชี้ว่าอัตราความเสี่ยงทันที (instantaneous hazard) มีแนวโน้มลดลงตามเวลา รูปทรงของเส้นสะสมความเสี่ยงนี้เป็นแรงจูงใจโดยตรงให้เฟส 4 ประมาณ **instantaneous hazard h(t)** แบบเรียบด้วย Epanechnikov kernel smoother บนส่วนเพิ่ม (increments) ของ NA เพื่อเปิดเผยว่าความเสี่ยงสูงสุดเกิดในช่วงใดของอายุการใช้งาน และเพื่อเปรียบเทียบกับแบบจำลองพาราเมตริก (Weibull, Log-logistic ฯลฯ) ต่อไป
