## 8. Phase 6 — แบบจำลอง Cox Proportional Hazards / Cox Proportional Hazards Model

### 8.1 จุดหมุนของงานวิจัย / The Pivot of the Study

Phase 6 คือ **จุดหมุน (pivot)** ของทั้งงานวิจัยนี้ ใน Phase 3–5 เราวิเคราะห์ **เวลาเพียงอย่างเดียว (time alone)** — เส้นโค้งการรอด (survival) และอัตราอันตราย (hazard) ถูกมองเป็นฟังก์ชันของเวลาโดยไม่มีตัวแปรร่วม (covariate) เข้ามาเกี่ยวข้อง เราตอบได้แค่ว่า *"ประชากรของอุปกรณ์เสื่อมสภาพไปตามเวลาอย่างไร?"*

Cox PH เปลี่ยนคำถามนั้นโดยใส่ตัวแปรร่วม **เข้าไปในฟังก์ชัน hazard โดยตรง**:

$$h(t \mid \mathbf{x}) = h_0(t)\cdot \exp(\beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p)$$

โดยที่ $h_0(t)$ คือ **baseline hazard** ที่ไม่ได้ถูกกำหนดรูปแบบไว้ล่วงหน้า (unspecified) — นี่คือเหตุผลที่เรียก Cox ว่าแบบจำลอง **กึ่งพารามิเตอร์ (semi-parametric)**: ส่วนของเวลา $h_0(t)$ ปล่อยให้เป็นอิสระ ส่วนของผลกระทบจากตัวแปรร่วม $\exp(\beta \mathbf{x})$ มีรูปแบบพารามิเตอร์ชัดเจน คำถามจึงเลื่อนจาก *"ประชากรเสื่อมไปตามเวลาอย่างไร?"* มาเป็น **"ปัจจัยใดเป็นตัวขับเคลื่อนความเสี่ยงการล้มเหลว และมากน้อยเพียงใด? / WHICH FACTORS drive failure risk, and BY HOW MUCH?"**

**สมมติฐาน Proportional Hazards (PH assumption):** หัวใจของแบบจำลองคือ ผลของตัวแปรร่วมแต่ละตัวจะ **คูณ (multiply)** baseline hazard ด้วยค่าคงที่ $\exp(\beta x)$ ที่ **ไม่เปลี่ยนแปลงตามเวลา (constant over time)** กล่าวคือ อัตราส่วนอันตรายระหว่างสองหน่วยที่มีค่าตัวแปรร่วมต่างกันจะคงที่ตลอดช่วงเวลาที่ศึกษา (สมมติฐานนี้ตรวจสอบใน §8.6)

**การประมาณค่าด้วย Partial Likelihood:** เนื่องจาก $h_0(t)$ ไม่ถูกระบุ Cox จึงประมาณค่าสัมประสิทธิ์ $\beta$ ด้วย **partial likelihood** ซึ่งพิจารณาเฉพาะ *ลำดับการเกิดเหตุการณ์ (order of events)* ในแต่ละชุดเสี่ยง (risk set) ณ เวลาที่เกิดความล้มเหลว โดยไม่ต้องอาศัยรูปแบบของ baseline hazard เลย — ทำให้ Cox ยืดหยุ่นและทนทานต่อรูปร่างของ hazard ที่ไม่ทราบ

---

### 8.2 การเตรียมข้อมูล / Data Preparation
*(ดู `src/cox_model.py` — `prepare_cox_data`, `fit_cox_by_component`)*

แบบจำลองใช้ **ตัวแปรร่วม 7 ตัว (7 covariates)** ดังนี้:

| Covariate | คำอธิบาย / Description | การเข้ารหัส / Encoding |
|---|---|---|
| `lightning_flash_density` | ความหนาแน่นฟ้าผ่า | continuous, z-score |
| `avg_wind_speed_ms` | ความเร็วลมเฉลี่ย (m/s) | continuous, z-score |
| `avg_humidity_pct` | ความชื้นเฉลี่ย (%) | continuous, z-score |
| `pm25_annual_avg` | ค่าฝุ่น PM2.5 เฉลี่ยรายปี | continuous, z-score |
| `HI_score_last` | Health Index Score (0–5) | continuous, z-score |
| `voltage_kv` | แรงดันไฟฟ้า (kV) | continuous, z-score |
| `encroachment_severity` | ระดับการรุกล้ำ | **ordinal** none/minor/moderate/severe → 0/1/2/3 |

**Z-score standardisation:** ตัวแปรต่อเนื่องทั้ง 6 ตัวถูกแปลงเป็นคะแนนมาตรฐาน $(x-\mu)/\sigma$ โดยใช้ **ค่าสถิติจากชุดข้อมูลทั้งหมด (full-dataset $\mu, \sigma$)** ไม่ใช่จากชุดย่อยรายอุปกรณ์ ทำให้ค่า **Hazard Ratio (HR) มีหน่วยเป็น "ต่อ 1 ส่วนเบี่ยงเบนมาตรฐาน (per 1 SD)"** และ **เทียบกันได้ข้ามอุปกรณ์ทุกประเภท (globally comparable)** ส่วน `encroachment_severity` เข้ารหัสแบบ ordinal เพื่อรักษาลำดับความรุนแรง

**กลยุทธ์การ fit:** fit แบบจำลอง Cox **หนึ่งตัวต่อหนึ่งอุปกรณ์ (one model PER component)** รวม 6 แบบจำลอง โดยใช้ `CoxPHFitter(penalizer=0.1)` ซึ่งเป็น **L2 ridge regularisation** เพื่อความเสถียรเชิงตัวเลข (numerical stability) เนื่องจาก events-per-variable (EPV) ค่อนข้างต่ำ อยู่ในช่วง **11.7–70.9** (Conductor มี 82 เหตุการณ์ / 7 ตัวแปร ≈ 11.7 EPV ซึ่งต่ำที่สุด) แถวที่มีค่าว่าง (NA) ถูกตัดทิ้งด้วย `dropna`

---

### 8.3 ตรรกะของแบบจำลองและการตีความ Hazard Ratio / Model Logic & HR Interpretation

จากสัมประสิทธิ์ $\beta$ ที่ประมาณได้ เราคำนวณ **Hazard Ratio**:

$$\text{HR} = \exp(\text{coef}), \qquad \text{CI}_{95\%} = \exp(\text{coef} \pm 1.96\cdot se)$$

การตีความ (ต่อการเพิ่มขึ้น 1 SD ของตัวแปรร่วม):

- **HR > 1** → เพิ่มอัตราอันตราย เป็น **ปัจจัยเสี่ยง (risk factor)** ทำให้รอดสั้นลง
- **HR < 1** → ลดอัตราอันตราย เป็น **ปัจจัยป้องกัน (protective)**
- **HR = 1** → ไม่มีผล

ระดับนัยสำคัญแสดงด้วยดาว: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` ไม่มีนัยสำคัญ

---

### 8.4 ความสามารถในการจำแนก — Concordance Index / Discrimination (C-index)
*(ตรวจสอบจาก `outputs/tables/cox_concordance.csv`)*

C-index วัดความสามารถของแบบจำลองในการ **จัดอันดับ (rank)** ว่าหน่วยใดจะล้มเหลวก่อน: 0.5 = สุ่ม (random), 0.7 = ดี (good), > 0.8 = ดีเยี่ยม (excellent)

| Component | C-index | การตีความ / Interpretation |
|---|---|---|
| Conductor | **0.910** | ดีเยี่ยม / Excellent |
| Fittings  | **0.885** | ดีเยี่ยม / Excellent |
| Spacer    | 0.795 | ดี / Good |
| Damper    | 0.751 | ดี / Good |
| Insulator | 0.735 | ดี / Good |
| Arrester  | 0.694 | พอใช้ / Acceptable |

**ข้อสังเกตสำคัญ:** การจำแนกดีที่สุดในอุปกรณ์ที่ **ล้มเหลวน้อยและสะอาดที่สุด (rarest/cleanest failures)** คือ Conductor (อัตราล้มเหลว 4.9%) และ Fittings (6.1%) — เมื่อความล้มเหลวถูกขับเคลื่อนโดยตัวแปร HI อย่างชัดเจน แบบจำลองจึงแยกแยะได้คม ในทางกลับกัน Arrester (อัตราล้มเหลว 29.5%) มีความสามารถจำแนกอ่อนที่สุด (0.694) เพราะความล้มเหลวเกิดบ่อยและกระจายตัว สะท้อนความผันผวนจากปัจจัยภายนอก (เช่น ฟ้าผ่า) ที่ทำนายได้ยากกว่า

![C-index ของ Cox PH รายอุปกรณ์ / Cox PH concordance index per component](outputs/figures/cox_concordance.png)
*รูปที่ 19 — ดัชนีความสอดคล้อง (C-index) ของแบบจำลอง Cox PH แยกตามอุปกรณ์ / Concordance index of the Cox PH model per component.*

---

### 8.5 ผลลัพธ์ Hazard Ratio / Hazard Ratio Results
*(ตรวจสอบจาก `outputs/tables/cox_hr_detailed.csv`, `cox_hazard_ratios.csv`)*

**ผลลัพธ์หลัก / KEY FINDINGS:**

**`HI_score_last` มีนัยสำคัญในทุกอุปกรณ์ทั้ง 6 ประเภท (significant in ALL 6 components)** ด้วย HR อยู่ในช่วง **1.65–2.22** และ p < 1e-13 ทั้งหมด — นี่คือ **ตัวทำนายที่ทรงพลังที่สุด (dominant predictor)** ของทั้งแบบจำลอง การที่ Health Index แย่ลง 1 SD เพิ่มความเสี่ยงล้มเหลวขึ้น 65–122%

ตารางต่อไปนี้แสดงเฉพาะ **HR ที่มีนัยสำคัญ (p < 0.05)** ทั้งหมด 9 ค่า:

| Component | Covariate | HR | 95% CI | p | sig |
|---|---|---|---|---|---|
| Conductor | Health Index Score | **1.646** | 1.449–1.870 | 1.9e-14 | *** |
| Damper    | Health Index Score | **1.914** | 1.655–2.215 | 2.6e-18 | *** |
| Spacer    | Health Index Score | **1.836** | 1.613–2.089 | 2.9e-20 | *** |
| Insulator | Health Index Score | **2.073** | 1.772–2.425 | 8.2e-20 | *** |
| Fittings  | Health Index Score | **1.665** | 1.467–1.889 | 2.7e-15 | *** |
| Arrester  | Health Index Score | **2.218** | 1.856–2.650 | 1.7e-18 | *** |
| Insulator | PM2.5 Annual Avg   | **1.284** | 1.173–1.405 | 6.5e-08 | *** |
| Insulator | Avg Humidity (%)   | **1.187** | 1.082–1.303 | 2.9e-04 | *** |
| Arrester  | Lightning Flash Density | **1.221** | 1.120–1.331 | 5.3e-06 | *** |

**การตีความเชิงกายภาพ (physical interpretation) ของผลรอง:**
- **Insulator** ไวต่อสภาพแวดล้อมมลพิษ — ทั้ง **PM2.5** (HR=1.284) และ **ความชื้น (humidity)** (HR=1.187) เร่งความล้มเหลว สอดคล้องกับกลไก *pollution flashover* ที่ฝุ่นเกาะผิวฉนวนแล้วความชื้นทำให้เกิดการนำไฟฟ้าตามผิว
- **Arrester** ไวต่อ **ความหนาแน่นฟ้าผ่า (lightning flash density)** (HR=1.221) ตามหน้าที่หลักของกับดักเสิร์จที่ต้องรับพลังงานฟ้าผ่าซ้ำ ๆ จนเสื่อม

**ตัวแปรร่วมอื่นทั้งหมดไม่มีนัยสำคัญ (ns):** avg_wind_speed, voltage_kv และ encroachment_severity ไม่มีผลที่มีนัยสำคัญในอุปกรณ์ใดเลย

![Forest plot ของ Cox PH — HR ± 95% CI ต่อตัวแปรร่วม / Cox PH forest plot, HR ± 95% CI per covariate](outputs/figures/cox_forest_plot.png)
*รูปที่ 20 — Forest plot ของ Hazard Ratio พร้อมช่วงความเชื่อมั่น 95% แยกตามอุปกรณ์ (กริด 2×3) เส้นแนวตั้งที่ HR=1 คือเส้นไม่มีผล / Forest plot of hazard ratios with 95% CI per component (2×3 grid); vertical line at HR=1 marks the no-effect reference.*

---

### 8.6 การตรวจสอบสมมติฐาน PH / PH Assumption Diagnostics
*(ตรวจสอบจาก `outputs/tables/ph_assumption_test.csv` — `check_ph_assumption`)*

เนื่องจาก Cox ตั้งอยู่บนสมมติฐาน proportional hazards เราตรวจสอบด้วย **Schoenfeld residuals test** (`proportional_hazard_test` ด้วยการแปลงเวลาแบบ **rank-transformed time**) ครอบคลุมทั้ง **42 ชุดค่า (6 อุปกรณ์ × 7 ตัวแปรร่วม)** หลักการ: Schoenfeld residuals นิยามเฉพาะ ณ เวลาที่เกิดความล้มเหลว และภายใต้สมมติฐาน PH ควรไม่มีแนวโน้มตามเวลา (no time trend); p < 0.05 ชี้ว่าผลของตัวแปรนั้นอาจเปลี่ยนตามเวลา (ละเมิด PH)

**คำตัดสิน / VERDICT: สมมติฐาน PH เป็นจริง (PH HOLDS).** จากทั้ง 42 ชุดค่า มีเพียง **1 ชุดที่ใกล้เส้นนัยสำคัญ** คือ **Insulator / PM2.5 (p = 0.0457)** ซึ่ง **หายไปทันทีภายใต้การแก้ไข Bonferroni** (เกณฑ์ = 0.05/42 ≈ 0.0012) ส่วนที่เหลือทั้งหมด p > 0.14

สิ่งสำคัญที่สุด: **`HI_score_last` ซึ่งเป็นตัวทำนายที่ทรงพลังที่สุด ผ่านสมมติฐาน PH ในทุกอุปกรณ์** (p > 0.39 ทุกกรณี: Conductor 0.573, Damper 0.393, Spacer 0.456, Insulator 0.630, Fittings 0.561, Arrester 0.585) — ผลของ Health Index ต่อความเสี่ยงจึงคงที่ตลอดอายุการใช้งาน ทำให้การตีความ HR ของมันมีความน่าเชื่อถือสูง

| Component | จำนวนการละเมิด (p<0.05) / Violations | รายละเอียด / Detail |
|---|---|---|
| Arrester  | 0 | ทุกตัว p > 0.14 |
| Conductor | 0 | ทุกตัว p > 0.57 |
| Damper    | 0 | ทุกตัว p > 0.29 |
| Fittings  | 0 | ทุกตัว p > 0.23 |
| Spacer    | 0 | ทุกตัว p > 0.45 |
| Insulator | **1** | PM2.5 Annual Avg: p = 0.046 (หายไปหลัง Bonferroni) |

![Heatmap ค่า p ของการทดสอบ PH / PH assumption test p-value heatmap](outputs/figures/ph_assumption_heatmap.png)
*รูปที่ 21 — Heatmap ของค่า p จากการทดสอบ Schoenfeld (อุปกรณ์ × ตัวแปรร่วม) เซลล์เดียวที่เข้าใกล้เกณฑ์คือ Insulator / PM2.5 / Heatmap of Schoenfeld test p-values (component × covariate); the only borderline cell is Insulator / PM2.5.*

![Schoenfeld residuals ของ Insulator / Schoenfeld residuals for Insulator](outputs/figures/ph_schoenfeld_insulator.png)
*รูปที่ 22 — Schoenfeld residuals เทียบกับเวลา สำหรับ Insulator (กรณีที่ใกล้เส้นนัยสำคัญที่สุด) เส้น smoother ที่แบนราบยืนยันว่าไม่มีแนวโน้มตามเวลาที่มีนัยสำคัญหลังแก้ไขการทดสอบหลายครั้ง / Scaled Schoenfeld residuals vs. time for Insulator (the most borderline case); the flat smoother confirms no meaningful time trend after multiple-testing correction.*

---

### 8.7 บทสรุปและการเชื่อมโยงสู่ Phase ถัดไป / Synthesis & Link to Next Phase

Cox PH เปลี่ยนการศึกษาจาก "เวลาเพียงอย่างเดียว" ไปสู่ **"ปัจจัยขับเคลื่อนความเสี่ยง"** ได้สำเร็จ และให้ข้อสรุปเชิงปฏิบัติการที่ชัดเจน:

1. **Health Index คือปัจจัยหลักสากล** — HI_score_last ทรงพลังในทุกอุปกรณ์ (HR 1.65–2.22) และผ่าน PH ทุกกรณี เป็นสัญญาณเตือนภัยที่น่าเชื่อถือที่สุดสำหรับการบำรุงรักษาเชิงป้องกัน
2. **ปัจจัยสิ่งแวดล้อมเฉพาะอุปกรณ์** — PM2.5/ความชื้นต่อ Insulator และฟ้าผ่าต่อ Arrester ชี้เป้าหมายการลงทุน (เช่น ฉนวนทนมลพิษในพื้นที่ฝุ่นสูง, การป้องกันฟ้าผ่าในพื้นที่ฟ้าผ่าหนาแน่น)
3. **แบบจำลองเชื่อถือได้** — C-index ดีถึงดีเยี่ยม (0.69–0.91) และสมมติฐาน PH เป็นจริง

อย่างไรก็ตาม Cox ตอบเพียงว่า *"หน่วยใดมีความเสี่ยงสูงกว่า ณ ตอนนี้ (which units are at higher RISK RIGHT NOW)"* ผ่านอัตราส่วนอันตราย แต่ **ไม่ให้คำทำนายเวลารอดชีวิตรายหน่วยโดยตรง** เพราะ baseline hazard $h_0(t)$ ไม่ถูกกำหนดรูปแบบ

นี่คือสะพานสู่ **Phase 7 — Weibull AFT Model** ซึ่งกำหนดรูปแบบ baseline ให้ชัดเจน (parametric) และเปลี่ยนคำถามไปสู่ *"แต่ละปัจจัยเพิ่มหรือลด **อายุการใช้งาน (LIFE)** เท่าไร?"* ผ่าน **Time Ratio / Acceleration Factor** ทำให้สามารถทำนายเวลารอดและ **Remaining Useful Life (RUL)** รายหน่วยได้ ซึ่งจำเป็นต่อการจัดลำดับความสำคัญการบำรุงรักษา (priority maintenance list)
