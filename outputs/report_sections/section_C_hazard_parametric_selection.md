## 6. Phase 4 — การประมาณอัตราความเสี่ยง / Hazard Estimation

### 6.1 จุดประสงค์ของเฟสนี้ / Purpose

ใน Phase 3 เราได้ประมาณ **cumulative hazard H(t)** (อัตราความเสี่ยงสะสม) ผ่าน Nelson-Aalen estimator ซึ่งบอก "ความเสี่ยงรวมที่สะสมมาถึงเวลา t" แต่ยังไม่บอก **รูปร่างของความเสี่ยงในแต่ละช่วงเวลา** โดยตรง

เป้าหมายของ Phase 4 คือการก้าวจาก H(t) ไปสู่ **instantaneous hazard rate h(t)** (อัตราความเสี่ยง ณ ขณะหนึ่ง) เพื่อตอบคำถามเชิงรูปร่าง (failure shape) ว่า ความเสี่ยงของการเสียหาย **เพิ่มขึ้น / ลดลง / คงที่** ตามอายุการใช้งานหรือไม่ ซึ่งมีนัยทางวิศวกรรมโดยตรง:

- **h(t) เพิ่มขึ้นตามเวลา** ⇒ พฤติกรรมเสื่อมสภาพ (wear-out) — ยิ่งเก่ายิ่งเสี่ยง สมเหตุสมผลกับการบำรุงรักษาเชิงป้องกัน
- **h(t) คงที่** ⇒ ความเสียหายแบบสุ่ม (random / memoryless) — การเปลี่ยนตามอายุไม่ช่วยลดความเสี่ยง
- **h(t) ลดลงตามเวลา** ⇒ พฤติกรรมเสียช่วงต้น (infant mortality / burn-in)

นอกจากนี้ Phase 4 ยังทำการ fit **parametric distributions** เพื่อให้ h(t) อยู่ในรูป **closed-form formula** ที่สามารถใช้ทำนาย (prediction) และคาดการณ์นอกช่วงข้อมูล (extrapolation) ได้ ซึ่ง non-parametric estimator ทำไม่ได้

ในเฟสนี้มี **6 โมเดล** ได้แก่ kernel hazard estimator (Model 3) และ parametric models อีก 5 ตัว (Models 4–8)

### 6.2 Kernel Hazard / Epanechnikov (Model 3)

Kernel hazard estimator ตามแนวทาง Yang et al. (2022) เป็นวิธี **non-parametric** ที่ปรับ (smooth) ความเสี่ยงจาก increment ของ Nelson-Aalen ให้เป็นเส้นโค้งต่อเนื่อง (ดู `src/hazard_models.py`):

$$\hat{h}(t) = \frac{1}{b} \sum_i K\!\left(\frac{t - t_i}{b}\right) \cdot \Delta H(t_i)$$

โดยที่:
- $t_i$ คือ **Nelson-Aalen event times** (เวลาที่เกิดความเสียหายจริง)
- $\Delta H(t_i)$ คือ **cumulative-hazard increments** — ส่วนต่างของ H(t) ที่แต่ละ event time (จุดนี้คือวิธีที่ผลลัพธ์ Phase 3 ป้อนต่อเข้า Phase 4 โดยตรง) โดยการนับ censored observations ถูกจัดการอย่างถูกต้องภายใน NA estimator แล้ว
- $K(\cdot)$ คือ **Epanechnikov kernel**: $K(u) = 0.75\,(1 - u^2)\cdot\mathbb{1}(|u| \le 1)$ — kernel ที่มี support จำกัด ($|u|\le 1$) และให้น้ำหนักมากที่สุดที่จุดศูนย์กลาง

**การเลือก bandwidth ด้วย LSCV (least-squares cross-validation):**

bandwidth $b$ ควบคุมความเรียบ (smoothness) ของเส้น — ค่ามากเกินไปทำให้เส้นเรียบเกินจริง (oversmooth) ค่าน้อยเกินไปทำให้เส้นกระตุก (undersmooth) วิธีเลือก $b$ ในโค้ดเป็นดังนี้:

1. **Silverman seed**: $b_{\text{sil}} = 1.06\,\sigma\,n^{-1/5}$ (rule-of-thumb เริ่มต้น)
2. สร้าง **candidate grid 25 ค่า** ในช่วง $[0.3, 2.5] \times b_{\text{sil}}$
3. ใช้ **leave-one-out (LOO)** ผ่าน kernel weight matrix ที่ zeroed diagonal (no Python loop — fully vectorised) เพื่อหาค่า $\hat{h}_{-i}(t_i)$
4. minimise objective:
   $$\text{CV}(b) = \int \hat{h}^2(t)\,dt \;-\; 2\sum_i \hat{h}_{-i}(t_i)\,\Delta H(t_i)$$
   โดยใช้ **trapezoidal integration** บน grid 200 จุด (`np.trapz`) สำหรับเทอม $\int \hat{h}^2 dt$
5. เลือก $b^*$ ที่ทำให้ CV(b) ต่ำสุด

หมายเหตุ: มีการ **trim ปลายหาง 2%** (`trim_pct=2.0`) ในขั้นตอนการประเมิน h(t) เพื่อลด boundary bias ที่ขอบของช่วงข้อมูล และมี fallback ไปใช้ Silverman bandwidth กรณี event น้อยกว่า 5 ตัว

**Per-component bandwidths** (จาก LSCV; bandwidth แปรผกผันกับจำนวน event — ยิ่ง event มาก ข้อมูลยิ่งแน่น ใช้ bandwidth แคบได้):

| Component / ส่วนประกอบ | Events | Bandwidth b* (วัน) |
|---|---|---|
| Conductor | 72 | 539 |
| Damper | 183 | 188 |
| Spacer | 152 | 144 |
| Insulator | 308 | 77 |
| Fittings | 77 | 591 |
| Arrester | 391 | 418 |

จะเห็นว่า Insulator (308 events) และ Arrester (391 events) ซึ่งมี event มากที่สุด ได้ bandwidth แคบที่สุดและกว้างปานกลางตามลำดับ ส่วน Fittings และ Conductor ที่มี event น้อย ได้ bandwidth กว้างที่สุด (591 และ 539 วัน) เพื่อชดเชยความเบาบางของข้อมูล

**ผลลัพธ์ / Result:** ทุกส่วนประกอบแสดงรูปร่าง hazard ที่ **เพิ่มขึ้นแบบ monotonic แล้ว plateau** (monotonically increasing then plateauing) ⇒ ยืนยันพฤติกรรม **wear-out** ไม่ใช่ความเสียหายแบบ random/constant ซึ่งเป็นหลักฐานเชิงประจักษ์ตัวแรกที่จะถูกยืนยันซ้ำด้วย parametric models ใน Phase 5

![รูปที่ 8 Kernel hazard (Epanechnikov) ทุกอุปกรณ์ / Epanechnikov kernel hazard overlay](outputs/figures/kernel_hazard_epanechnikov.png)
*รูปที่ 8 — เส้น kernel hazard (Epanechnikov) ซ้อนทับทุกอุปกรณ์ ทุกเส้นเพิ่มขึ้นแบบ monotonic ⇒ wear-out / Epanechnikov kernel hazard overlay for all components; all curves rise monotonically (wear-out).*

![รูปที่ 9 ภาพรวม hazard: kernel เทียบ parametric ที่ดีที่สุด / Hazard overview: kernel vs. best parametric](outputs/figures/hazard_overview_2x3.png)
*รูปที่ 9 — ภาพรวมรายอุปกรณ์ (กริด 2×3): kernel hazard เทียบกับ parametric ที่เหมาะสมที่สุด / Per-component overview (2×3): kernel vs. best-fit parametric hazard.*

![รูปที่ 10 Hazard Conductor / Conductor hazard](outputs/figures/fig7_hazard_conductor.png)
*รูปที่ 10 — Conductor: kernel hazard + 5 parametric hazards / Conductor: kernel + 5 parametric hazard curves.*

![รูปที่ 11 Hazard Damper / Damper hazard](outputs/figures/fig8_hazard_damper.png)
*รูปที่ 11 — Damper: kernel hazard + 5 parametric hazards / Damper: kernel + 5 parametric hazard curves.*

![รูปที่ 12 Hazard Spacer / Spacer hazard](outputs/figures/fig9_hazard_spacer.png)
*รูปที่ 12 — Spacer: kernel hazard + 5 parametric hazards / Spacer: kernel + 5 parametric hazard curves.*

![รูปที่ 13 Hazard Insulator / Insulator hazard](outputs/figures/fig10_hazard_insulator.png)
*รูปที่ 13 — Insulator: kernel hazard + 5 parametric hazards (best fit = Generalized-gamma) / Insulator: kernel + 5 parametric hazard curves.*

![รูปที่ 14 Hazard Fittings / Fittings hazard](outputs/figures/fig11_hazard_fittings.png)
*รูปที่ 14 — Fittings: kernel hazard + 5 parametric hazards / Fittings: kernel + 5 parametric hazard curves.*

![รูปที่ 15 Hazard Arrester / Arrester hazard](outputs/figures/fig12_hazard_arrester.png)
*รูปที่ 15 — Arrester: kernel hazard + 5 parametric hazards (shape ρ สูงสุด) / Arrester: kernel + 5 parametric hazard curves (highest shape ρ).*

### 6.3 Parametric Models (Models 4–8)

หลังจากเห็นรูปร่าง hazard เชิงประจักษ์จาก kernel แล้ว เรา fit **parametric distributions 5 ตัว** ผ่าน **maximum likelihood estimation (MLE)** ด้วย lifelines โดยอ่านค่า hazard ออกมาด้วยเมธอด `hazard_at_times()` (ดู `parametric_hazard()` ใน `src/hazard_models.py`)

**ทำไมต้อง fit parametric ทับ kernel?** — เพราะ parametric model ให้:
- **smooth** — เส้นเรียบ ไม่ขึ้นกับ bandwidth
- **extrapolatable** — คาดการณ์นอกช่วงข้อมูล (เกินอายุที่สังเกตได้) ผ่านสูตร closed-form
- **parameter-interpretable** — พารามิเตอร์ (เช่น shape $\rho$) ตีความเชิงฟิสิกส์ได้โดยตรง
- **AIC-comparable** — เปรียบเทียบ goodness-of-fit เชิงปริมาณข้ามโมเดลได้ (นำไปใช้ใน Phase 5)

| # | Model / แบบจำลอง | Parameters | รูปร่าง hazard ที่แทนได้ / Hazard shape | lifelines Fitter |
|---|---|---|---|---|
| 4 | **Weibull** | scale $\lambda$, shape $\rho$ | monotonic — เพิ่ม ($\rho>1$), คงที่ ($\rho=1$), ลด ($\rho<1$) | `WeibullFitter` |
| 5 | **Exponential** | rate $\lambda$ | **คงที่เท่านั้น** (constant) — memoryless | `ExponentialFitter` |
| 6 | **Log-logistic** | scale $\alpha$, shape $\beta$ | unimodal — เพิ่มแล้วลด (hump) หรือ ลดอย่างเดียว | `LogLogisticFitter` |
| 7 | **Log-normal** | $\mu$, $\sigma$ | unimodal — เพิ่มแล้วลด | `LogNormalFitter` |
| 8 | **Generalized-gamma** | $\mu$, $\ln\sigma$, $\lambda$ | ยืดหยุ่นสูง — ครอบคลุม Weibull, log-normal, gamma เป็น special cases | `GeneralizedGammaFitter` |

Exponential เป็นกรณีพิเศษของ Weibull ที่ $\rho=1$ จึงเป็น "ตัวควบคุม" สำหรับทดสอบสมมติฐาน constant-hazard ส่วน Generalized-gamma เป็นโมเดลที่ยืดหยุ่นที่สุด (3 พารามิเตอร์) ครอบคลุมหลายตระกูลเป็นกรณีย่อย

---

## 7. Phase 5 — การคัดเลือกแบบจำลอง / Model Selection

### 7.1 จุดประสงค์และวิธีการ / Purpose & Method

เมื่อ fit ครบ 5 parametric models ต่อส่วนประกอบแล้ว เราต้องเลือกโมเดลที่ **อธิบายข้อมูลได้ดีที่สุดโดยไม่ overfit** เกณฑ์ที่ใช้คือ **Akaike Information Criterion (AIC)** ซึ่ง trade-off ระหว่าง goodness-of-fit กับความซับซ้อนของโมเดล:

$$\text{AIC} = 2k + 2\cdot(-\text{LLV})$$

โดย $k$ = จำนวนพารามิเตอร์ และ LLV = Log-Likelihood Value (ค่า log-likelihood ของโมเดลที่ fit แล้ว) — AIC ต่ำกว่า = ดีกว่า

จากนั้นคำนวณ **ΔAIC** เทียบกับโมเดลที่ดีที่สุดของแต่ละส่วนประกอบ:

$$\Delta\text{AIC} = \text{AIC} - \min(\text{AIC})$$

**เกณฑ์การตัดสิน (Burnham & Anderson):**
- **ΔAIC < 2** — มี substantial support (โมเดลแข่งขันได้ ใกล้เคียงตัวที่ดีที่สุด)
- **ΔAIC 4–7** — considerably less support (ด้อยกว่าอย่างเห็นได้ชัด)
- **ΔAIC > 10** — essentially no support (ปฏิเสธได้)

### 7.2 ตาราง ΔAIC เต็ม / Full ΔAIC Matrix

(ตรวจสอบจาก `outputs/tables/delta_aic_matrix.csv`)

| Component / ส่วนประกอบ | Weibull | Exponential | LogLogistic | LogNormal | GenGamma | **Best** |
|---|---|---|---|---|---|---|
| Conductor | **0.00** | 26.71 | 0.17 | 1.79 | 1.85 | **Weibull** |
| Damper | **0.00** | 64.99 | 0.87 | 0.69 | 1.61 | **Weibull** |
| Spacer | **0.00** | 55.68 | 1.51 | 4.08 | 0.66 | **Weibull** |
| Insulator | 1.87 | 93.82 | 6.82 | 5.02 | **0.00** | **GenGamma** |
| Fittings | **0.00** | 27.71 | 0.27 | 2.03 | 1.77 | **Weibull** |
| Arrester | **0.00** | 206.25 | 5.98 | 17.81 | 1.18 | **Weibull** |

### 7.3 การตีความ / Interpretation

**1. Weibull ชนะ 5 ใน 6 ส่วนประกอบ** (Conductor, Damper, Spacer, Fittings, Arrester) สำหรับ **Insulator** แม้ Generalized-gamma จะดีที่สุด แต่ Weibull มี ΔAIC = 1.87 < 2 ซึ่งยังอยู่ในเกณฑ์ **substantial support** ดังนั้นเลือก **Weibull เป็น unified model** เดียวสำหรับทุกส่วนประกอบได้อย่างมีเหตุผล — ลดความซับซ้อนของการตีความและการนำไปใช้ในเฟสถัดไป (Cox PH ใช้ baseline ร่วม, AFT ใช้ Weibull โดยตรง)

**2. Exponential ถูกปฏิเสธในทุกส่วนประกอบ** (ΔAIC อยู่ในช่วง 26.71–206.25 ≫ 10) — แสดงว่าสมมติฐาน **constant hazard เป็นเท็จ** ผลนี้ยืนยันซ้ำ (cross-validate) พฤติกรรม wear-out ที่ kernel estimator ใน Phase 4 ตรวจพบ โดยเฉพาะ Arrester ที่ ΔAIC ของ Exponential สูงถึง 206.25 — ปฏิเสธ constant-hazard อย่างเด็ดขาดที่สุด สอดคล้องกับการเป็นส่วนประกอบที่มี event มากที่สุดและเสื่อมเร็วที่สุด

**3. โมเดล unimodal (LogLogistic, LogNormal)** แข่งขันได้ในบางส่วนประกอบ (เช่น LogLogistic ΔAIC 0.17–1.51 ใน Conductor/Damper/Spacer/Fittings) แต่ไม่มีตัวใดเอาชนะ Weibull ได้ ยืนยันว่า hazard เป็นแบบ monotonic-increasing ไม่ใช่ hump-shaped

### 7.4 พารามิเตอร์ Weibull ที่ fit ได้ / Fitted Weibull Parameters

(ตรวจสอบจาก `outputs/tables/fitted_parameters.csv`)

| Component / ส่วนประกอบ | scale $\lambda$ (วัน) | shape $\rho$ | การตีความ shape |
|---|---|---|---|
| Conductor | 9793.6 | 1.857 | $\rho>1$ ⇒ increasing hazard |
| Damper | 5593.5 | 1.689 | $\rho>1$ ⇒ increasing hazard |
| Spacer | 6294.9 | 1.720 | $\rho>1$ ⇒ increasing hazard |
| Fittings | 9431.8 | 1.747 | $\rho>1$ ⇒ increasing hazard |
| Arrester | 3325.9 | 1.944 | $\rho>1$ ⇒ increasing hazard (สูงสุด) |
| Insulator* | — | — | GenGamma: $\mu$=8.416, $\ln\sigma$=−1.775, $\lambda$=3.891 |

\* Insulator best-fit เป็น Generalized-gamma (Weibull เป็นทางเลือกที่ ΔAIC=1.87)

**การตีความพารามิเตอร์:**

- **ทุกค่า $\rho > 1$** (1.689–1.944) ⇒ **increasing hazard (wear-out)** ทุกส่วนประกอบ — ยืนยันข้อสรุปจาก Phase 4 ในเชิงพารามิเตอร์อย่างชัดเจน ไม่มีส่วนประกอบใดแสดง random failure ($\rho=1$) หรือ infant mortality ($\rho<1$)
- **Arrester มี $\rho$ สูงสุด (1.944)** ⇒ hazard ไต่ขึ้นชันที่สุด สอดคล้องกับการเป็นส่วนประกอบเสี่ยงสูงสุด
- **scale $\lambda$ แปรผกผันกับลำดับความเสี่ยง (risk ranking)** — $\lambda$ คือ characteristic life (มาตราเวลาของการเสียหาย): Arrester $\lambda$=3325.9 (ต่ำสุด ⇒ เสียเร็วสุด/เสี่ยงสุด) ในขณะที่ Conductor $\lambda$=9793.6 และ Fittings $\lambda$=9431.8 (สูงสุด ⇒ ทนทานสุด/เสี่ยงต่ำสุด) ลำดับนี้สอดคล้องกับอัตราการเกิด event ที่สังเกตได้ใน Phase 2 อย่างสมบูรณ์

![รูปที่ 16 AIC รายอุปกรณ์ (faceted) / Faceted AIC bars per component](outputs/figures/aic_comparison_faceted.png)
*รูปที่ 16 — แผนภูมิแท่ง AIC แยกตามอุปกรณ์ (faceted) สำหรับทั้ง 5 distribution / Faceted AIC bar chart per component across the 5 distributions.*

![รูปที่ 17 ΔAIC grouped bar พร้อมเกณฑ์ B&A / Grouped ΔAIC bars with Burnham-Anderson thresholds](outputs/figures/delta_aic_grouped.png)
*รูปที่ 17 — ΔAIC grouped bar พร้อมเส้นเกณฑ์ Burnham & Anderson (ΔAIC = 2, 4, 10) / Grouped ΔAIC bars with B&A support thresholds.*

![รูปที่ 18 Heatmap AIC (model × component) / AIC heatmap](outputs/figures/model_aic_heatmap.png)
*รูปที่ 18 — Heatmap ของ AIC (distribution × อุปกรณ์) ค่าต่ำ (เข้ม) = ดีกว่า / AIC heatmap (model × component); lower (darker) is better.*

### 7.5 เชื่อมโยงสู่เฟสถัดไป / Link to Next Phase

Phase 3–5 ทั้งหมดวิเคราะห์ **เวลาเพียงอย่างเดียว (time alone)** โดยยังไม่นำ covariates (ปัจจัยสภาพแวดล้อม/สุขภาพ/การบุกรุก) เข้ามาในแบบจำลองเลย ผลที่ได้บอกเราชัดเจน 2 ประการ:

1. **รูปร่างของความเสี่ยงคือ wear-out** (kernel + Weibull $\rho>1$ ยืนยันตรงกัน) — การบำรุงรักษาเชิงป้องกันตามอายุมีเหตุผลรองรับ
2. แต่ละส่วนประกอบมี **characteristic life ($\lambda$) ต่างกันชัดเจน** — บางตัวเสี่ยงสูงกว่าตัวอื่นมาก

อย่างไรก็ตาม คำถามสำคัญที่ time-only models **ตอบไม่ได้** คือ: **ปัจจัยใดเป็นตัวขับเคลื่อนความเสี่ยง?** (ฟ้าผ่า ความชื้น PM2.5 health index การบุกรุก ฯลฯ มีผลเชิงปริมาณเท่าไร?) การจะตอบคำถามนี้ — และเพื่อนำไปสู่การทำนาย RUL รายหน่วยและจัดลำดับความสำคัญในการบำรุงรักษา — ต้องใช้ **regression survival models** ที่รวม covariates ได้แก่ **Cox Proportional Hazards (Phase 6)** และ **Weibull Accelerated Failure Time / AFT (Phase 7)** ซึ่งจะนำเสนอในส่วนถัดไป โดย Weibull baseline ที่คัดเลือกได้ใน Phase 5 จะถูกใช้เป็นรากฐานของโมเดล AFT โดยตรง
