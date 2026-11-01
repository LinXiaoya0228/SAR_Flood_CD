# ChangeGuard Work Package E11: Pre-Test Evidence and Wording Gap Audit

**Document Identifier:** `docs/e11_pretest_evidence_gap_audit.md`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E11 (Section 16)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `AUDIT_COMPLETED — GAPS_IDENTIFIED_AND_BOUNDED`  

---

## 1. Audit Scope & Objectives

Prior to freezing the final evaluation matrix and proceeding to Gate T rehearsal, this audit systematically reviews all evidence, experimental telemetry, numerical representations, and claim formulations across Work Packages E0 through E10.

The audit focuses specifically on four critical areas:
1. **Second-Backbone Recipe History:** Clarifying the relationship between Recipes A, B, and C during source-inner screening.
2. **Synthetic-Shift Numerical Domain:** Tracing the exact tensor domain, inversion operations, and mathematical interpretation of `MultiplicativeBackscatterShift`.
3. **Claim-Strength Audit:** Replacing overbroad or universal claims with rigorously bounded scientific statements.
4. **Statistical Accounting Audit:** Verifying metric definitions (macro vs. pooled), sample counts, seed dispersion metrics, and cluster-bootstrap limitations.

---

## 2. Second-Backbone Recipe History Audit

### 2.1 Screening Execution Record
In Work Package E10 Phase 2, three candidate training recipes were defined for `SiamUnet_diff` on the 22-event inner-train / 5-event inner-validation split:
- **Recipe A (`siam_diff_screen_recipe_a.json`):** AdamW, $\eta = 1.0 \times 10^{-3}$, weight decay $0.0$, effective batch size 32.
- **Recipe B (`siam_diff_screen_recipe_b.json`):** AdamW, $\eta = 5.0 \times 10^{-4}$, weight decay $1.0 \times 10^{-4}$, effective batch size 32.
- **Recipe C (`siam_diff_screen_recipe_c.json`):** SGD with Nesterov momentum ($0.9$), $\eta = 1.0 \times 10^{-2}$, weight decay $1.0 \times 10^{-4}$, effective batch size 32.

### 2.2 Storage I/O Contention & Process Cancellation
When Recipes A, B, and C were initially launched concurrently, concurrent reading of 19,178 raw GeoTIFF rasters from the physical spinning SATA hard disk (`/dev/sda`, 10 ms seek latency) caused severe disk head thrashing, reducing cumulative throughput to 4.4 MB/s and projecting an infeasible 18+ hour screening window.

To prevent system exhaustion and establish deterministic data caching in Linux RAM page cache:
- Recipes B and C were intentionally **terminated before producing comparable validation results** (manifested by empty `training_history.csv` files of 0 bytes in `outputs/e10_second_backbone/screen_recipe_b` and `screen_recipe_c`).
- Recipe A proceeded to complete all 10 epochs (19,178 train / 2,284 val samples per epoch).
- Within Recipe A, Epoch 5 achieved peak inner-validation Event-Macro Flood F1 of **0.7888** (Worst-Event F1: 0.7284, Macro IoU: 0.6546), establishing strong convergence matching SNUNet-ECAM's baseline (0.7817).

### 2.3 Wording Correction & Protocol Stance
- **Corrected Claim:** We do **not** claim Recipe A was empirically selected as superior to Recipes B and C.
- **Factually Accurate Statement:** Recipe A was the completed, computationally feasible training recipe. Within Recipe A, Epoch 5 ($E^* = 5$) was selected using the frozen source-inner validation metric.
- **Consequence:** Recipes B and C remain preserved as canceled attempts. Because the purpose of the second backbone was to provide a functional, well-converged baseline to test whether TTA failures generalize across architectures—and Recipe A achieved Macro F1 parity with SNUNet—rerunning Recipes B and C is scientifically unnecessary.

---

## 3. Synthetic-Shift Numerical Domain Audit

### 3.1 Input Pipeline and Representation Trace
To determine the exact mathematical domain of `MultiplicativeBackscatterShift`:
1. **Raw Sensor Data:** Dual-polarization Sentinel-1 GRD imagery provided in KuroSiwo consists of **linear amplitude** backscatter ($A = \sqrt{\sigma^0}$) in VV and VH channels. Raw values span $[0.0, \sim 0.8]$, with over 95% of pixels $\le 0.15$. Values are **not** in decibels (dB) and **not** power ($\sigma^0$).
2. **Preprocessing Pipeline (`src/data/evaluation_dataset.py`):**
   $$x_{\text{clamped}} = \text{clip}(x_{\text{raw}}, 0.0, 0.15)$$
   $$x_{\text{norm}} = \frac{x_{\text{clamped}} - \mu}{\sigma}, \quad \mu = [0.0953, 0.0264], \; \sigma = [0.0427, 0.0215]$$
3. **Shift Inversion & Perturbation (`src/changeguard/synthetic_shift.py`):**
   $$x_{\text{linear}} = x_{\text{norm}} \cdot \sigma + \mu \quad (\text{recovering } x_{\text{clamped}} \in [0.0, 0.15])$$
   $$x_{\text{shifted}} = x_{\text{linear}} \cdot (1 + s)$$
   $$x_{\text{out\_clamped}} = \text{clip}(x_{\text{shifted}}, 0.0, 0.15)$$
   $$x_{\text{out}} = \frac{x_{\text{out\_clamped}} - \mu}{\sigma}$$

### 3.2 Evaluation of Physical Grounding
- **Domain Verification:** The multiplication $x' = (1+s)x$ operates on recovered linear amplitude backscatter ($A$), **not** in standardized space and **not** in dB.
- **Physical Caveat:** In radar physics, a radiometric calibration drift $\Delta G$ scales received power $\sigma^0 \to (1 + s_{\text{pow}}) \sigma^0$. In amplitude domain, this corresponds to $A \to \sqrt{1 + s_{\text{pow}}} A \approx (1 + \frac{1}{2}s_{\text{pow}}) A$. Furthermore, clipping at $0.15$ truncates any shifted amplitude exceeding the fixed dynamic ceiling.
- **Action & Bounded Nomenclature:** The synthetic shift is rigorously designated as a **controlled linear-amplitude radiometric gain scaling stress test with dynamic range preservation**. It is not claimed to capture unconstrained real-world environmental shifts (e.g., wind-induced water surface roughening or speckle phase decoherence).

---

## 4. Claim-Strength Audit

To maintain publication integrity and prevent overclaiming, the following claim adjustments are strictly codified:

| Overbroad / Unsupported Phrasing | Corrected, Bounded Scientific Phrasing | Rationale |
| :--- | :--- | :--- |
| "TTA failure is an intrinsically pathological defect across all architectures" | "Harmful and unstable TTA behaviour was observed across both evaluated architectures under the studied development-event protocols." | Prevents universal generalization beyond evaluated models and protocols. |
| "Proves that test-time adaptation cannot work in remote sensing" | "Highlights fundamental stability challenges for unsupervised batch-norm and entropy adaptation under temporal class imbalance in SAR change detection." | Restricts conclusion to the evaluated unsupervised paradigms and specific task constraints. |
| "All domain shifts lead to catastrophic collapse" | "Evaluated radiometric gain shifts up to 20% exacerbated TTA degradation rates relative to frozen source models." | Accurately reflects that source models retained reasonable robustness while TTA degraded faster. |
| "SiamUnet proves architecture-independent failure" | "Evidence from SiamUnet_diff demonstrates that TTA vulnerability is not solely attributable to SNUNet-ECAM's high parameter capacity or attention mechanism." | Accurately frames SiamUnet_diff as a counterexample to parameter-capacity explanations. |

---

## 5. Statistical and Accounting Audit

### 5.1 Metric Formulations
1. **Event-Macro Flood F1:**
   $$\text{Macro F1} = \frac{1}{N} \sum_{i=1}^N \text{F1}_i, \quad N=6 \text{ (dev events)}$$
   Each event is weighted equally, preventing the largest event (Event 279, 2,381 samples) from dominating smaller events (Event 1111003, 19 samples).
2. **Pooled Flood F1:**
   $$\text{Pooled F1} = \frac{2 \sum_{i=1}^N \text{TP}_i}{2 \sum_{i=1}^N \text{TP}_i + \sum_{i=1}^N \text{FP}_i + \sum_{i=1}^N \text{FN}_i}$$
   Micro-averaged metric dominated by sample-rich events.
3. **Worst-Event Flood F1:**
   $$\text{Worst-Event F1} = \min_{i \in \{1, \dots, N\}} \text{F1}_i$$
   Critical safety metric measuring catastrophic localized failure.

### 5.2 Seed Dispersion & Cluster Bootstrap
- **Sample Standard Deviation:** With $K=3$ independent random seeds, dispersion is reported as sample standard deviation ($s$, with Bessel's correction $\text{ddof}=1$). Standard error of the mean is $s / \sqrt{3}$.
- **Paired Event-Cluster Bootstrap:** Because spatial pixels within an event are strongly auto-correlated, statistical significance tests must resample at the event level. On $N=6$ development events, the discrete resample space is $6^6 = 46,656$ combinations. All bootstrap estimates must acknowledge this finite-cluster resolution.

### 5.3 Quarantine and Sample Count Verification
- **Development Partition:** Exactly 6 events (`[279, 437, 520, 559, 1111003, 1111008]`), totaling exactly 2,826 samples.
- **Source Partition:** Exactly 27 source events, totaling 21,462 samples.
- **Sealed Test Partition:** Exactly 10 events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`). Confirmed **zero reads, zero evaluations, zero unpickling**.

---

## 6. Audit Verdict

All identified evidence gaps have been documented, bounded, and resolved without invalidating prior experimental data or requiring redundant GPU reruns.
- Second-backbone recipe history is accurately attributed to Recipe A feasible lock.
- Synthetic shift is mathematically certified in linear amplitude domain.
- Claim strengths are strictly bounded to observed empirical evidence.
- Ready to proceed to Phase 2 (Final Claim Freeze).

