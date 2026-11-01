# ChangeGuard Work Package E11: Final Claim Freeze (Branch C)

**Document Identifier:** `docs/e11_final_claim_freeze.md`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E11  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `FINAL_CLAIMS_FROZEN — READY_FOR_TEST_SPECIFICATION`  

---

## 1. Primary Framed Contribution

In accordance with the formal confirmation established in Block E8 (`outputs/e8_confirmation_v2`) and the diagnostic freezing in Block E9 (`docs/e9_final_scope_freeze.md`), the primary scientific contribution of ChangeGuard is defined as:

> **A controlled, reproducible empirical characterization of test-time adaptation instability in paired SAR flood change detection, including the role of normalization context, checkpoint/event variability, strong generic baselines, a second lightweight architecture and a bounded synthetic stress test.**

ChangeGuard does **not** claim to provide an autonomous, deployable controller, selective gate, or universal adaptation remedy. The project delivers a rigorous empirical investigation establishing why conventional computer-vision TTA paradigms destabilize under the physical and temporal realities of satellite radar change detection.

---

## 2. Partitioned Claims Matrix

### 2.1 Supported Claims (Empirically Verified on Development Partition)
1. **Unsupervised TTA Instability:** Standard test-time adaptation methods (Tent entropy minimization, BN test-batch statistics, and continual CoTTA) degrade segmentation performance relative to frozen source checkpoints across multi-temporal SAR flood streams.
2. **Normalization Context as the Primary Mechanism:** Replacing training running statistics with current-batch test statistics accounts for the majority of the observed performance degradation, even in the complete absence of parameter gradient updates (`bn_no_update`).
3. **Cross-Architecture Generalization of Failure:** TTA vulnerability is not confined to SNUNet-ECAM (12.06M parameters); on `SiamUnet_diff` (1.35M parameters, Siamese difference), TTA induces an even more severe performance collapse ($\Delta = -15.93\%$ Macro F1 vs. $-1.12\%$ on SNUNet).
4. **Radiometric Sensitivity Amplification:** Under controlled linear-amplitude radiometric calibration shifts ($s \in \{0.0, 0.05, 0.10, 0.20\}$), source models maintain high inherent stability ($< 1.4\%$ drop at 20% shift), whereas test-time adaptation methods degrade at approximately twice the rate ($d\text{F1}/ds \approx -0.10$ vs $-0.057$).

### 2.2 Development-Only Evidence
- **Temporal Event Ordering Effects:** Order A vs. Order B continual trajectory divergence was characterized across the 6 development events.
- **Component Removal Ablations:** Filtered entropy, temperature scaling, and learning rate sensitivity ablations conducted in E9 are development-event findings.
- **Second-Backbone Source Recipe:** Selection of $E^* = 5$ epochs was performed on the 22-event inner-train / 5-event inner-val split.

### 2.3 Negative Findings (Documented Negative Results)
1. **Unsupervised Entropy as a Quality Indicator:** Test-batch entropy exhibits weak or negative rank correlation with ground-truth segmentation accuracy, precluding entropy-thresholded gating.
2. **Speckle Null-Probes:** Synthetic identical-pair null probes $(x_A, x_A)$ fail to isolate boundary corruption due to SAR speckle activations.
3. **Source Anchor Regularization:** L2 parameter pull toward source checkpoints suppresses drift magnitude but does not restore lost flood boundaries.

### 2.4 Withdrawn Claims (Explicitly Retired)
- **ChangeGuard as an Adaptive Controller:** Completely withdrawn at Gate E (E6). No runtime heuristic gate, rollback policy, or confidence filter is proposed or deployed.
- **Selective Adaptation Gate:** Withdrawn following E4/E5 temporal risk telemetry demonstrating negative selectivity.

### 2.5 Limitations
1. **Event Partition Size:** Development evaluations were conducted across 6 historical flood events (2,826 patches). While geographically diverse, 6 clusters impose finite-sample granularity on paired cluster bootstrap intervals.
2. **Sensor Constancy:** Evaluated data are C-band Sentinel-1 GRD imagery; findings do not claim empirical verification on L-band or X-band SAR.
3. **Synthetic Shift Scope:** The synthetic shift captures linear amplitude gain drift; it does not simulate non-linear surface dielectric changes (e.g. soil moisture saturation).

### 2.6 Claims Requiring Final-Test Confirmation
1. **Main Matrix Confirmation:** Confirmation that the performance ranking $\text{source\_only} > \text{tent} \approx \text{bn\_no\_update}$ holds across the 10 completely unseen, sealed Kuro Siwo test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).
2. **Cross-Architecture Test Parity:** Confirmation that `SiamUnet_diff` also experiences significant TTA degradation on the 10 sealed test events.
3. **Protection Baseline Persistence:** Confirmation that CoTTA fails to outperform the frozen source baseline on the sealed test partition.

---

## 3. Protocol Invariance Guarantees

The following operational protocols are permanently locked for all downstream validation and final testing:
- **Prediction Timing:** `PRE_UPDATE` primary protocol (predictions generated using current-state model before applying any gradient updates from the current batch).
- **Label Access Timing:** Strict post-persistence deferred disk read (primary prediction shards must be durably committed before reading labels).
- **Spatial Masking:** Double-masking validity `(valid_mask == 1) & (label != 3)`.
- **Primary Metric:** Event-Macro Flood F1 across all events.
- **Secondary Metrics:** Worst-Event Flood F1, Event-Macro IoU, Pooled Flood F1.

