# Research Workflow — Transmission Line Predictive Maintenance (Survival Analysis)

แผนภาพขั้นตอนการทำวิจัย อ้างอิงระเบียบวิธีจาก Yang et al. (2022), *Scientific Reports*.

---

## 1. ภาพรวมขั้นตอนการวิจัย (Research Process)

```mermaid
flowchart TD
    A["1 ตั้งคำถาม &amp; วัตถุประสงค์<br/>โมเดลความน่าจะเป็นการเสียหาย<br/>ของอุปกรณ์สายส่งไฟฟ้า"]
    B["2 ทบทวนวรรณกรรม<br/>Yang et al. 2022<br/>กรอบ Survival Analysis"]
    C["3 เตรียมข้อมูล (Phase 2)<br/>นิยาม time-to-event + censoring"]
    D["4 วิเคราะห์เชิงสำรวจ (Phase 2-3)<br/>Descriptive + KM curves"]
    E["5 ทดสอบสมมติฐาน (Phase 3)<br/>Log-rank test"]
    F["6 สร้างแบบจำลอง (Phase 4)<br/>Kernel + Parametric hazard"]
    G["7 ประเมิน &amp; เลือกโมเดล (Phase 5)<br/>AIC / LLV / Delta-AIC"]
    H["8 สรุป &amp; ตีความเชิงวิศวกรรม<br/>จัดลำดับความเสี่ยงเพื่อบำรุงรักษา"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#e3f2fd,stroke:#1565c0
    style C fill:#fff3e0,stroke:#e65100
    style D fill:#fff3e0,stroke:#e65100
    style E fill:#fff3e0,stroke:#e65100
    style F fill:#f3e5f5,stroke:#6a1b9a
    style G fill:#f3e5f5,stroke:#6a1b9a
    style H fill:#e8f5e9,stroke:#2e7d32
```

---

## 2. รายละเอียด Pipeline (Data → Analysis → Results)

```mermaid
flowchart TD
    subgraph IN["ข้อมูลนำเข้า"]
        RAW["transmission_line_maintenance_data.csv<br/>10,080 records | 6 components | 4 regions"]
    end

    subgraph P2["Phase 2 — Preprocessing (src/preprocessing.py)"]
        L["load_data"]
        V["validate_data"]
        CR["censoring_report<br/>censoring 84.7%"]
        SS["summary_stats + failure_mode_breakdown"]
        L --> V --> CR --> SS
    end

    subgraph P3["Phase 3 — Survival Curves (src/survival_curves.py)"]
        KM["Kaplan-Meier + Nelson-Aalen<br/>by component / region / HI class"]
        LR["Log-rank test<br/>Chi2=643.9, p&lt;0.001"]
        KM --> LR
    end

    subgraph P4["Phase 4 — Hazard Estimation (src/hazard_models.py)"]
        KER["Epanechnikov kernel hazard<br/>LSCV bandwidth"]
        PAR["5 Parametric models<br/>Weibull, Exponential,<br/>LogLogistic, LogNormal, Gen-Gamma"]
        KER --- PAR
    end

    subgraph P5["Phase 5 — Model Selection (src/model_selection.py)"]
        GOF["Goodness-of-fit<br/>AIC + LLV"]
        DA["Delta-AIC matrix<br/>+ annotate best"]
        GOF --> DA
    end

    subgraph OUT["ผลลัพธ์"]
        FIG["outputs/figures/ (PNG)"]
        TAB["outputs/tables/ (CSV)"]
        REP["report.md / report.docx"]
    end

    RAW --> P2 --> P3 --> P4 --> P5
    P2 --> FIG
    P3 --> FIG
    P4 --> FIG
    P5 --> FIG
    P2 --> TAB
    P3 --> TAB
    P5 --> TAB
    FIG --> REP
    TAB --> REP

    style IN fill:#eceff1,stroke:#455a64
    style P2 fill:#fff3e0,stroke:#e65100
    style P3 fill:#fff8e1,stroke:#f9a825
    style P4 fill:#f3e5f5,stroke:#6a1b9a
    style P5 fill:#e1f5fe,stroke:#0277bd
    style OUT fill:#e8f5e9,stroke:#2e7d32
```

---

## 3. ตรรกะการตัดสินใจ — การเลือกโมเดล (Model Selection Logic)

```mermaid
flowchart TD
    START["ฟิต 5 distributions<br/>ต่อแต่ละ component"]
    AIC["คำนวณ AIC + LLV"]
    BEST["หา Delta-AIC = AIC - min(AIC)"]
    Q{"Delta-AIC ของ<br/>โมเดล &lt; 2 ?"}
    GOOD["โมเดลเหมาะสม<br/>(แทบแยกไม่ออกจากตัวที่ดีสุด)"]
    Q2{"Delta-AIC &gt; 10 ?"}
    REJECT["ปฏิเสธโมเดล<br/>(เช่น Exponential ทุก component)"]
    WEAK["หลักฐานสนับสนุนอ่อน"]
    PICK["เลือกโมเดลที่ดีที่สุด<br/>Weibull (ส่วนใหญ่)<br/>Gen-Gamma (Insulator)"]

    START --> AIC --> BEST --> Q
    Q -- ใช่ --> GOOD --> PICK
    Q -- ไม่ --> Q2
    Q2 -- ใช่ --> REJECT
    Q2 -- ไม่ --> WEAK

    style START fill:#e3f2fd,stroke:#1565c0
    style PICK fill:#e8f5e9,stroke:#2e7d32
    style REJECT fill:#ffebee,stroke:#c62828
    style GOOD fill:#f1f8e9,stroke:#558b2f
```

---

## วิธีดูแผนภาพ
- **VSCode:** ติดตั้ง extension *Markdown Preview Mermaid Support* แล้วเปิด preview (Cmd+Shift+V)
- **GitHub:** เรนเดอร์ Mermaid อัตโนมัติเมื่อ push ไฟล์ `.md`
- **Export เป็นรูป:** วางโค้ดที่ [mermaid.live](https://mermaid.live) แล้วดาวน์โหลด PNG/SVG
