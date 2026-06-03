# รายงานผลการวิเคราะห์การอยู่รอด (Survival Analysis) เพื่อการบำรุงรักษาเชิงพยากรณ์ของสายส่งไฟฟ้า

**โครงการ:** Transmission Line Predictive Maintenance — Prognostic Modeling
**วันที่:** 2 มิถุนายน 2026
**ระเบียบวิธีอ้างอิง:** Yang et al. (2022). *"Prognostic modeling of predictive maintenance with survival analysis for mobile work equipment."* Scientific Reports, 12, 8529.
**ชุดข้อมูล:** `data/transmission_line_maintenance_data.csv` (10,080 ระเบียน)

---

## 1. บทสรุปผู้บริหาร (Executive Summary)

การศึกษานี้ประยุกต์ใช้ survival analysis เพื่อจำลองความน่าจะเป็นของความเสียหาย (failure probability) ของอุปกรณ์สายส่งไฟฟ้า 6 ชนิด จากข้อมูลบันทึกการบำรุงรักษา 10,080 ระเบียน (8 สายส่ง, 4 ภูมิภาค, ปี 2018–2025) ผลลัพธ์สำคัญสรุปได้ดังนี้

- **ลำดับความเสี่ยงของอุปกรณ์ (จากอัตราการเกิดความเสียหาย):**
  **Arrester (29.5%) > Insulator (23.9%) > Damper (15.2%) > Spacer (12.4%) > Fittings (6.1%) > Conductor (4.9%)**
  Arrester และ Insulator เป็นอุปกรณ์ที่ควรให้ความสำคัญสูงสุดในแผนบำรุงรักษา โดยที่ 6 ปี (2,190 วัน) ความน่าจะเป็นที่ Arrester จะยังทำงานได้เหลือเพียง **63%** เทียบกับ Conductor ที่ยังสูงถึง **94%**

- **อัตราความเสี่ยงเพิ่มขึ้นตามอายุการใช้งาน (wear-out / aging):** โมเดล Weibull ให้ค่าพารามิเตอร์รูปร่าง (shape, ρ) มากกว่า 1 ในทุกอุปกรณ์ (1.63–1.94) ยืนยันว่าความเสียหายเป็นแบบ **เสื่อมสภาพตามเวลา** ซึ่ง **สนับสนุนการบำรุงรักษาเชิงป้องกันตามรอบเวลา (time-based preventive maintenance)** มากกว่าการซ่อมเมื่อเสีย

- **โมเดลพาราเมตริกที่เหมาะสมที่สุด:** **Weibull** เป็นโมเดลที่ดีที่สุดสำหรับ 5 ใน 6 อุปกรณ์ (ตามเกณฑ์ AIC) ส่วน Insulator เหมาะกับ **Generalized-Gamma** โดยแบบจำลอง **Exponential ถูกปฏิเสธอย่างชัดเจน** ในทุกอุปกรณ์ — ตอกย้ำว่าสมมติฐาน "อัตราความเสี่ยงคงที่" ไม่เป็นจริง

- **ความแตกต่างเชิงพื้นที่:** ภูมิภาค **ภาคกลาง (Central) มีความเสี่ยงแตกต่างจากทุกภูมิภาคอย่างมีนัยสำคัญ** ขณะที่ภาคเหนือ ภาคตะวันออกเฉียงเหนือ และภาคใต้ มีพฤติกรรมการอยู่รอดที่ **แยกความแตกต่างทางสถิติไม่ได้**

---

## 2. บทนำและวัตถุประสงค์

การบำรุงรักษาเชิงพยากรณ์ (Predictive Maintenance, PdM) มุ่งคาดการณ์เวลาที่อุปกรณ์จะเสียหาย เพื่อวางแผนซ่อมบำรุงล่วงหน้าก่อนเกิดเหตุขัดข้องจริง ลดทั้งต้นทุนและความเสี่ยงด้านความมั่นคงของระบบไฟฟ้า งานวิจัยของ Yang et al. (2022) แสดงให้เห็นว่าเทคนิค survival analysis สามารถประเมิน "เวลาจนเกิดเหตุการณ์" (time-to-event) ของอุปกรณ์ได้อย่างมีประสิทธิภาพ แม้ในกรณีที่ข้อมูลถูกตัดปลาย (censored)

**วัตถุประสงค์ของรายงาน:**
1. ประเมินฟังก์ชันการอยู่รอด (survival function) และอัตราความเสี่ยง (hazard) ของอุปกรณ์สายส่งแต่ละชนิด
2. เปรียบเทียบความแตกต่างของการอยู่รอดระหว่างชนิดอุปกรณ์และระหว่างภูมิภาค
3. คัดเลือกแบบจำลองพาราเมตริกที่อธิบายข้อมูลได้ดีที่สุดตามเกณฑ์ AIC และ Log-Likelihood Value (LLV)
4. ให้ข้อเสนอแนะเชิงปฏิบัติสำหรับการวางแผนบำรุงรักษา

---

## 3. ข้อมูลและระเบียบวิธี

### 3.1 ชุดข้อมูล
- **จำนวนระเบียน:** 10,080 (หนึ่งระเบียนต่ออุปกรณ์ต่อช่วงสาย/span)
- **ชนิดอุปกรณ์ (6):** Conductor, Damper, Spacer, Insulator, Fittings, Arrester (อุปกรณ์ละ 1,680 ระเบียน)
- **ขอบเขต:** 8 สายส่ง ใน 4 ภูมิภาค (Northeast, North, Central, South); ช่วงเวลา 2018–2025
- **ตัวแปรหลัก:** `maintenance_period_days` (เวลาจนเกิดเหตุการณ์), `event_occurred` (1 = เสียหาย, 0 = ถูกตัดปลาย/censored), `component`, `failure_mode`
- **ตัวแปรร่วม (covariates):** ปัจจัยสิ่งแวดล้อม (lightning, wind, humidity, PM2.5, coastal proximity, pollution), ดัชนีสุขภาพ (HI score/class/trend), การรุกล้ำแนวสาย (encroachment), เหตุการณ์ไฟดับ (outage)

### 3.2 นิยาม Time-to-Event และ Censoring
ตัวแปรเวลา `maintenance_period_days` วัดจำนวนวันจนกระทั่งเกิดความเสียหาย (event) หรือจนถึงวันสิ้นสุดการสังเกต (right-censored หากยังไม่เสียหาย) อัตราการตัดปลายรวม **84.65%** (อัตราการเกิดเหตุการณ์ 15.35%) ซึ่งเป็นลักษณะปกติของข้อมูลความเชื่อถือได้ของอุปกรณ์ที่มีอายุการใช้งานยาว

### 3.3 วิธีการวิเคราะห์ (ตาม Yang et al. 2022)
| ขั้นตอน | วิธี |
|---|---|
| ประมาณฟังก์ชันการอยู่รอด (non-parametric) | Kaplan-Meier (KM) |
| ประมาณอัตราความเสี่ยงสะสม | Nelson-Aalen |
| ประมาณอัตราความเสี่ยงแบบต่อเนื่อง | Kernel hazard estimator (Epanechnikov) |
| ทดสอบความแตกต่างระหว่างกลุ่ม | Log-rank test (pairwise) |
| แบบจำลองพาราเมตริก | Weibull, Exponential, Log-logistic, Log-normal, Generalized-Gamma |
| เกณฑ์คัดเลือกแบบจำลอง | AIC, Log-Likelihood Value (LLV) |

เครื่องมือ: Python 3.11+, `lifelines`, `scikit-survival`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 4. ผลการวิเคราะห์

### 4.1 การสำรวจข้อมูลและสรุปเหตุการณ์ (EDA)

ตารางสรุปเหตุการณ์รายอุปกรณ์ (จาก `data_summary.csv`):

| อุปกรณ์ | N | จำนวนเสียหาย | จำนวน censored | อัตราเสียหาย | มัธยฐานวัน (median) | ค่าเฉลี่ยวัน (mean) |
|---|---:|---:|---:|---:|---:|---:|
| **Arrester** | 1,680 | 496 | 1,184 | **29.5%** | 1,706.5 | 1,697.6 |
| **Insulator** | 1,680 | 402 | 1,278 | **23.9%** | 1,748.5 | 1,718.8 |
| Damper | 1,680 | 256 | 1,424 | 15.2% | 1,811.5 | 1,786.3 |
| Spacer | 1,680 | 208 | 1,472 | 12.4% | 1,842.0 | 1,818.9 |
| Fittings | 1,680 | 103 | 1,577 | 6.1% | 1,876.5 | 1,865.3 |
| Conductor | 1,680 | 82 | 1,598 | 4.9% | 1,895.5 | 1,880.5 |
| **รวม** | **10,080** | **1,547** | **8,533** | **15.35%** | 1,820.5 | 1,794.6 |

ช่วงเวลาสังเกตอยู่ที่ 365–2,646 วัน ทุกอุปกรณ์ Arrester และ Insulator เกิดความเสียหายบ่อยที่สุดและมีมัธยฐานเวลาสั้นที่สุด

**รูปประกอบ:** [การกระจายเหตุการณ์เสียหาย/censored](outputs/figures/eda_event_distribution.png) · [Boxplot ระยะเวลารายอุปกรณ์](outputs/figures/eda_duration_boxplot.png) · [Histogram ระยะเวลารวม](outputs/figures/fig4_duration_histogram_overall.png) · [Histogram รายอุปกรณ์](outputs/figures/fig5_duration_histograms_by_component.png)

### 4.2 การวิเคราะห์ Kaplan-Meier

ความน่าจะเป็นในการอยู่รอด S(t) ที่จุดเวลาสำคัญ (จาก `km_survival_summary.csv`):

| อุปกรณ์ | S(1 ปี) | S(2 ปี) | S(3 ปี) | S(4 ปี) | S(5 ปี) | S(6 ปี) |
|---|---:|---:|---:|---:|---:|---:|
| Conductor | 0.994 | 0.991 | 0.984 | 0.973 | 0.958 | **0.939** |
| Fittings | 0.992 | 0.985 | 0.978 | 0.966 | 0.946 | **0.922** |
| Spacer | 0.983 | 0.971 | 0.948 | 0.928 | 0.897 | **0.851** |
| Damper | 0.979 | 0.963 | 0.932 | 0.909 | 0.862 | **0.803** |
| Insulator | 0.963 | 0.932 | 0.895 | 0.848 | 0.794 | **0.731** |
| Arrester | 0.969 | 0.944 | 0.890 | 0.828 | 0.738 | **0.630** |

*(1 ปี = 365 วัน ... 6 ปี = 2,190 วัน)*

**ข้อสังเกตสำคัญ:** มัธยฐานเวลาการอยู่รอด (median survival) ของทุกอุปกรณ์เป็น **∞ (ยังไม่ถึง)** — กล่าวคือ ภายในกรอบเวลาสังเกต ยังไม่มีอุปกรณ์ชนิดใดที่ความน่าจะเป็นในการอยู่รอดลดต่ำกว่า 50% ซึ่งสอดคล้องกับอัตรา censoring ที่สูง เส้นโค้ง KM ของ Arrester ลดลงชันที่สุด ตามด้วย Insulator

**รูปประกอบ:** [เส้นโค้ง KM รายอุปกรณ์](outputs/figures/fig6_km_by_component.png) · [KM รายภูมิภาค](outputs/figures/km_by_region.png) · [KM ตามระดับ Health Index](outputs/figures/km_by_hi_class.png)

### 4.3 การทดสอบ Log-rank (เปรียบเทียบกลุ่ม)

**เปรียบเทียบระหว่างอุปกรณ์** (จาก `logrank_pairwise_component.csv`): ทุกคู่แตกต่างอย่างมีนัยสำคัญ (p < 0.05) **ยกเว้น** คู่เดียวคือ **Conductor vs Fittings (p = 0.104)** ซึ่งเป็นสองอุปกรณ์ความเสี่ยงต่ำสุด จึงมีพฤติกรรมการอยู่รอดคล้ายกัน คู่ที่แตกต่างชัดเจนที่สุดคือ Arrester vs Conductor (χ² = 369.5, p ≈ 2.4×10⁻⁸²)

**เปรียบเทียบระหว่างภูมิภาค** (จาก `logrank_pairwise_region.csv`):

| คู่ภูมิภาค | χ² | p-value | นัยสำคัญ |
|---|---:|---:|:---:|
| Central vs North | 41.32 | 1.3×10⁻¹⁰ | ✓ แตกต่าง |
| Central vs Northeast | 29.21 | 6.5×10⁻⁸ | ✓ แตกต่าง |
| Central vs South | 34.31 | 4.7×10⁻⁹ | ✓ แตกต่าง |
| North vs Northeast | 1.00 | 0.318 | ✗ ไม่ต่าง |
| North vs South | 0.35 | 0.552 | ✗ ไม่ต่าง |
| Northeast vs South | 0.18 | 0.672 | ✗ ไม่ต่าง |

**สรุป:** ภาคกลางโดดเด่นแตกต่างจากทุกภูมิภาคอย่างมีนัยสำคัญ ขณะที่อีก 3 ภูมิภาคเป็นกลุ่มเดียวกันทางสถิติ

### 4.4 การประมาณอัตราความเสี่ยง (Hazard)

การประมาณ Nelson-Aalen (cumulative hazard) และ kernel hazard ด้วยฟังก์ชัน Epanechnikov แสดงให้เห็นว่าอัตราความเสี่ยงของทุกอุปกรณ์ **เพิ่มขึ้นตามเวลา** (increasing hazard) ซึ่งสอดคล้องกับลักษณะการเสื่อมสภาพ (wear-out) ไม่ใช่ความเสียหายแบบสุ่มที่อัตราคงที่

**รูปประกอบ:** [ภาพรวม hazard 2×3](outputs/figures/hazard_overview_2x3.png) · [Kernel hazard (Epanechnikov)](outputs/figures/kernel_hazard_epanechnikov.png) · รายอุปกรณ์: [Conductor](outputs/figures/fig7_hazard_conductor.png) · [Damper](outputs/figures/fig8_hazard_damper.png) · [Spacer](outputs/figures/fig9_hazard_spacer.png) · [Insulator](outputs/figures/fig10_hazard_insulator.png) · [Fittings](outputs/figures/fig11_hazard_fittings.png) · [Arrester](outputs/figures/fig12_hazard_arrester.png)

### 4.5 แบบจำลองพาราเมตริกและการคัดเลือกแบบจำลอง

ตารางเปรียบเทียบ AIC ของแบบจำลองทั้ง 5 (ค่า AIC ต่ำสุด = ดีที่สุด, เครื่องหมาย ✓ จาก `goodness_of_fit_annotated.csv`):

| อุปกรณ์ | Weibull | Exponential | Log-Logistic | Log-Normal | Gen-Gamma | **โมเดลที่ดีที่สุด** |
|---|---:|---:|---:|---:|---:|:---|
| Conductor | **1,870.99** ✓ | 1,897.70 | 1,871.16 | 1,872.78 | 1,872.84 | Weibull |
| Damper | **5,246.08** ✓ | 5,311.07 | 5,246.95 | 5,246.77 | 5,247.69 | Weibull |
| Spacer | **4,353.84** ✓ | 4,409.52 | 4,355.35 | 4,357.92 | 4,354.50 | Weibull |
| Insulator | 7,853.15 | 7,945.10 | 7,858.10 | 7,856.30 | **7,851.28** ✓ | Generalized-Gamma |
| Fittings | **2,306.83** ✓ | 2,334.54 | 2,307.10 | 2,308.86 | 2,308.60 | Weibull |
| Arrester | **9,375.42** ✓ | 9,581.67 | 9,381.40 | 9,393.23 | 9,376.60 | Weibull |

**ตาราง ΔAIC** (ส่วนต่างจากโมเดลที่ดีที่สุดในแต่ละอุปกรณ์; จาก `delta_aic_matrix.csv`):

| อุปกรณ์ | Weibull | Exponential | Log-Logistic | Log-Normal | Gen-Gamma |
|---|---:|---:|---:|---:|---:|
| Conductor | 0.00 | 26.71 | 0.17 | 1.79 | 1.85 |
| Damper | 0.00 | 64.99 | 0.87 | 0.69 | 1.61 |
| Spacer | 0.00 | 55.68 | 1.51 | 4.08 | 0.66 |
| Insulator | 1.87 | 93.82 | 6.82 | 5.02 | 0.00 |
| Fittings | 0.00 | 27.71 | 0.27 | 2.03 | 1.77 |
| Arrester | 0.00 | 206.25 | 5.98 | 17.81 | 1.18 |

**การตีความ:**
- **Weibull** เป็นโมเดลที่ดีที่สุดสำหรับ 5/6 อุปกรณ์ และสำหรับ Insulator ก็ยังใกล้เคียงผู้ชนะมาก (ΔAIC = 1.87 < 2) → **Weibull ใช้เป็นโมเดลรวมได้อย่างสมเหตุสมผลทั้งระบบ**
- **Exponential ถูกปฏิเสธชัดเจน** (ΔAIC = 26.7 ถึง 206.3 — เกินเกณฑ์ ΔAIC > 10 มาก) ยืนยันว่าอัตราความเสี่ยงไม่คงที่
- Log-Logistic, Log-Normal, Generalized-Gamma ให้ผลใกล้เคียง Weibull (ΔAIC < ~7) แต่ไม่ดีกว่าอย่างมีนัยสำคัญสำหรับอุปกรณ์ส่วนใหญ่

**รูปประกอบ:** [Heatmap AIC รายโมเดล×อุปกรณ์](outputs/figures/model_aic_heatmap.png) · [เปรียบเทียบ AIC แบบ facet](outputs/figures/aic_comparison_faceted.png) · [ΔAIC แบบกลุ่มแท่ง](outputs/figures/delta_aic_grouped.png)

### 4.6 พารามิเตอร์ที่ฟิตได้ (Weibull)

จาก `fitted_parameters.csv` — ค่า scale (λ) และ shape (ρ) ของ Weibull:

| อุปกรณ์ | λ (scale, วัน) | ρ (shape) | การตีความ |
|---|---:|---:|:---|
| Conductor | 9,793.6 | 1.857 | ρ > 1 → เสื่อมตามเวลา |
| Fittings | 9,431.8 | 1.747 | ρ > 1 → เสื่อมตามเวลา |
| Spacer | 6,294.9 | 1.720 | ρ > 1 → เสื่อมตามเวลา |
| Damper | 5,593.4 | 1.689 | ρ > 1 → เสื่อมตามเวลา |
| Insulator | 4,272.7 | 1.630 | ρ > 1 → เสื่อมตามเวลา |
| Arrester | 3,325.9 | 1.944 | ρ > 1 → เสื่อมตามเวลา (ชันที่สุด) |

**ค่า ρ > 1 ในทุกอุปกรณ์ (1.63–1.94)** ยืนยันอัตราความเสี่ยงเพิ่มขึ้นแบบ monotonic — เป็นหลักฐานทางสถิติที่สนับสนุนกลยุทธ์บำรุงรักษาเชิงป้องกันตามรอบเวลา ค่า λ ที่ต่ำของ Arrester และ Insulator สะท้อนอายุการใช้งานที่สั้นกว่าอุปกรณ์อื่น

### 4.7 รูปแบบความเสียหาย (Failure Modes)

จาก `failure_mode_breakdown.csv` — สาเหตุความเสียหายเด่นรายอุปกรณ์:

| อุปกรณ์ | สาเหตุหลัก (จำนวนครั้ง) | รวม |
|---|:---|---:|
| **Arrester** | ฟ้าผ่า lightning (233), overload (139), end-of-life (124) | 496 |
| **Insulator** | ฟ้าผ่า lightning (136), contamination flashover (113), end-of-life (85), mechanical fatigue (68) | 402 |
| Damper | corrosion (80), end-of-life (64), mechanical fatigue (63), wind damage (49) | 256 |
| Spacer | mechanical fatigue (76), end-of-life (72), physical damage (60) | 208 |
| Fittings | corrosion (58), mechanical fatigue (23), end-of-life (22) | 103 |
| Conductor | corrosion (35), physical damage (18), end-of-life (15), mechanical fatigue (14) | 82 |

**ข้อสังเกต:** **ฟ้าผ่า (lightning)** เป็นสาเหตุหลักของอุปกรณ์ความเสี่ยงสูงสุด 2 ชนิด (Arrester, Insulator) สอดคล้องกับบทบาทป้องกันแรงดันเกินของอุปกรณ์เหล่านี้ ส่วน **corrosion** เป็นสาเหตุเด่นของอุปกรณ์โลหะ (Damper, Fittings, Conductor)

---

## 5. อภิปรายผล

1. **สนับสนุนการบำรุงรักษาเชิงป้องกันตามเวลา:** การที่ค่า shape ของ Weibull มากกว่า 1 ในทุกอุปกรณ์ และการที่ Exponential (อัตราคงที่) ถูกปฏิเสธทุกกรณี เป็นหลักฐานชัดเจนว่าความเสียหายมีลักษณะ "เสื่อมสภาพสะสม" การเปลี่ยน/ตรวจสอบตามรอบเวลาจึงมีประสิทธิผลมากกว่าการรอจนเสีย

2. **การจัดลำดับความสำคัญทรัพยากร:** Arrester และ Insulator ควรได้รับการตรวจสอบถี่กว่าและมีอะไหล่สำรองพร้อมกว่า เนื่องจากทั้งความถี่และความเร็วของการเสื่อมสภาพสูงกว่าอุปกรณ์อื่นอย่างมีนัยสำคัญ

3. **ปัจจัยเชิงพื้นที่:** ความแตกต่างของภาคกลางบ่งชี้ว่ามีปัจจัยสิ่งแวดล้อม/การใช้งานเฉพาะถิ่น (เช่น มลพิษ ฟ้าผ่า ภาระโหลด) ที่ควรนำเข้าสู่แบบจำลองหลายตัวแปร (multivariate) ต่อไป

4. **ความสอดคล้องกับ Yang et al. (2022):** ผลการศึกษายืนยันว่า Weibull เป็นแบบจำลองที่ทนทานและเหมาะสมที่สุดสำหรับข้อมูลความเชื่อถือได้ของอุปกรณ์ ซึ่งสอดคล้องกับข้อสรุปหลักของงานอ้างอิง

---

## 6. ข้อจำกัด

- **ข้อมูลจำลอง (simulated):** ชุดข้อมูลถูกสร้างขึ้นพร้อม covariates ด้านสิ่งแวดล้อมและการใช้งาน ผลลัพธ์จึงเหมาะสำหรับการสาธิตระเบียบวิธีมากกว่าการตัดสินใจเชิงปฏิบัติการจริง
- **มัธยฐานเวลาการอยู่รอดยังไม่ถึง:** เนื่องจาก censoring สูง (84.65%) จึงไม่สามารถประมาณมัธยฐานอายุการใช้งานได้โดยตรงในกรอบเวลาสังเกต ค่าประมาณระยะยาวต้องอาศัยการคาดการณ์จากแบบจำลองพาราเมตริก
- **ยังไม่ได้ใช้ covariates ในแบบจำลอง:** การวิเคราะห์นี้เป็นแบบ univariate (แยกตามชนิดอุปกรณ์/ภูมิภาค) ยังไม่ได้รวมผลของตัวแปรร่วมเข้าในแบบจำลองความเสี่ยงโดยตรง

---

## 7. สรุปและข้อเสนอแนะ

**สรุป:** survival analysis สามารถจัดลำดับความเสี่ยงและอธิบายพฤติกรรมการเสื่อมสภาพของอุปกรณ์สายส่งได้อย่างชัดเจน Arrester และ Insulator คือกลุ่มเสี่ยงสูงสุด อัตราความเสี่ยงเพิ่มตามอายุ และ Weibull เป็นแบบจำลองที่เหมาะสมที่สุดเกือบทั้งระบบ

**ข้อเสนอแนะ:**
1. **จัดลำดับบำรุงรักษาตามความเสี่ยง:** เพิ่มความถี่การตรวจสอบ Arrester และ Insulator; ลดความถี่สำหรับ Conductor และ Fittings เพื่อใช้ทรัพยากรอย่างคุ้มค่า
2. **กำหนดรอบเปลี่ยนตามอายุ:** ใช้พารามิเตอร์ Weibull (λ, ρ) ประมาณช่วงเวลาที่อัตราความเสี่ยงเริ่มเร่งตัว เพื่อกำหนดรอบเปลี่ยนเชิงป้องกัน
3. **ต่อยอดด้วยแบบจำลองหลายตัวแปร:** พัฒนา **Cox Proportional Hazards** หรือ **Accelerated Failure Time (AFT) แบบมี covariates** เพื่อวัดผลกระทบเชิงปริมาณของปัจจัยสิ่งแวดล้อม (ฟ้าผ่า มลพิษ ความชื้น) และดัชนีสุขภาพ (HI) ต่อความเสี่ยง
4. **ตรวจสอบปัจจัยเฉพาะภาคกลาง:** สืบหาสาเหตุที่ทำให้ภาคกลางแตกต่าง เพื่อมาตรการเฉพาะถิ่น

---

## 8. ภาคผนวก — รายการรูปและตาราง

### 8.1 ตารางผลลัพธ์ (`outputs/tables/`)
| ไฟล์ | คำอธิบาย |
|---|---|
| [data_summary.csv](outputs/tables/data_summary.csv) | สรุปจำนวนเหตุการณ์/censoring รายอุปกรณ์ |
| [km_survival_summary.csv](outputs/tables/km_survival_summary.csv) | S(t) ที่จุดเวลาสำคัญ พร้อมช่วงความเชื่อมั่น |
| [goodness_of_fit.csv](outputs/tables/goodness_of_fit.csv) · [goodness_of_fit_annotated.csv](outputs/tables/goodness_of_fit_annotated.csv) | AIC/LLV ทุกโมเดล + ผู้ชนะ |
| [model_comparison.csv](outputs/tables/model_comparison.csv) | ตารางเปรียบเทียบ AIC/LLV จัดอันดับ |
| [delta_aic_matrix.csv](outputs/tables/delta_aic_matrix.csv) | เมทริกซ์ ΔAIC |
| [fitted_parameters.csv](outputs/tables/fitted_parameters.csv) | พารามิเตอร์ที่ฟิตได้ทุกโมเดล |
| [logrank_pairwise_component.csv](outputs/tables/logrank_pairwise_component.csv) | Log-rank รายคู่อุปกรณ์ |
| [logrank_pairwise_region.csv](outputs/tables/logrank_pairwise_region.csv) | Log-rank รายคู่ภูมิภาค |
| [failure_mode_breakdown.csv](outputs/tables/failure_mode_breakdown.csv) | จำนวนความเสียหายตามสาเหตุ |

### 8.2 รูป (`outputs/figures/`)
| ไฟล์ | คำอธิบาย |
|---|---|
| [eda_event_distribution.png](outputs/figures/eda_event_distribution.png) | การกระจายเหตุการณ์เสียหาย/censored |
| [eda_duration_boxplot.png](outputs/figures/eda_duration_boxplot.png) | Boxplot ระยะเวลารายอุปกรณ์ |
| [fig4_duration_histogram_overall.png](outputs/figures/fig4_duration_histogram_overall.png) | Histogram ระยะเวลารวม |
| [fig5_duration_histograms_by_component.png](outputs/figures/fig5_duration_histograms_by_component.png) | Histogram ระยะเวลารายอุปกรณ์ |
| [fig6_km_by_component.png](outputs/figures/fig6_km_by_component.png) | เส้นโค้ง Kaplan-Meier รายอุปกรณ์ |
| [km_by_region.png](outputs/figures/km_by_region.png) | KM รายภูมิภาค |
| [km_by_hi_class.png](outputs/figures/km_by_hi_class.png) | KM ตามระดับ Health Index |
| [hazard_overview_2x3.png](outputs/figures/hazard_overview_2x3.png) | ภาพรวมอัตราความเสี่ยง 2×3 |
| [kernel_hazard_epanechnikov.png](outputs/figures/kernel_hazard_epanechnikov.png) | Kernel hazard (Epanechnikov) |
| [fig7_hazard_conductor.png](outputs/figures/fig7_hazard_conductor.png) | Hazard — Conductor |
| [fig8_hazard_damper.png](outputs/figures/fig8_hazard_damper.png) | Hazard — Damper |
| [fig9_hazard_spacer.png](outputs/figures/fig9_hazard_spacer.png) | Hazard — Spacer |
| [fig10_hazard_insulator.png](outputs/figures/fig10_hazard_insulator.png) | Hazard — Insulator |
| [fig11_hazard_fittings.png](outputs/figures/fig11_hazard_fittings.png) | Hazard — Fittings |
| [fig12_hazard_arrester.png](outputs/figures/fig12_hazard_arrester.png) | Hazard — Arrester |
| [model_aic_heatmap.png](outputs/figures/model_aic_heatmap.png) | Heatmap AIC รายโมเดล×อุปกรณ์ |
| [aic_comparison_faceted.png](outputs/figures/aic_comparison_faceted.png) | เปรียบเทียบ AIC แบบ facet |
| [delta_aic_grouped.png](outputs/figures/delta_aic_grouped.png) | ΔAIC แบบกลุ่มแท่ง |

---

*รายงานนี้สร้างจากผลลัพธ์ใน `outputs/` ของการวิเคราะห์ใน `notebooks/01_full_analysis.ipynb` และโมดูลใน `src/`*
