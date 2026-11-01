# ChangeGuard Plan V3 E9: Final Empirical Scope Freeze

**Status:** FROZEN — READY FOR E10  
**Date:** 2026-10-25  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E9 (Section 14)  

---

## 1. Permanent Scientific Route Freeze

ChangeGuard is permanently frozen under **Branch C**:
- **No Controller or Selective Acceptance:** No threshold, heuristic, or learned gating controller is claimed, deployed, or tuned.
- **Controlled Empirical Study:** The primary contribution is a rigorous, reproducible, negative-result empirical study demonstrating the limitations of continuous test-time adaptation on multi-temporal SAR change detection.
- **Normalization Attribution:** Observed adaptation performance is primarily driven by feature normalization statistics (AdaBN / BN-No-Update), with gradient-based adaptation (Tent, EATA, SAR, CoTTA) providing no statistically significant improvement.

---

## 2. Partition & Data Governance Freeze

1. **Development Set (Audited and Frozen):**
   - 6 Events: `[279, 437, 520, 559, 1111003, 1111008]` (N = `[2381, 169, 92, 42, 19, 123]`; Total = 2826 samples).
   - All diagnostic, confirmation, and ablation runs are restricted to this set.
2. **Sealed-Test Set (Quarantined and Frozen):**
   - 10 Events: `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.
   - Access remains strictly prohibited until Gate M final evaluation.
   - Any access prior to final evaluation constitutes an immediate research integrity violation.

---

## 3. Method & Baseline Specification Freeze

| Method | Family | Parameters Updated | Key Hyperparameters | Reset Mode | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **source_only** | Baseline | None (0) | Frozen source BN statistics | Invariant | **FROZEN** |
| **bn_no_update** | Normalization | None (0) | Batch statistics ($m = 1.0$) | Invariant / Reset | **FROZEN** |
| **adabn** | Normalization | None (0) | Streaming momentum ($m = 0.10, 0.05, 0.01$) | Reset / Continual | **FROZEN** |
| **tent** | Entropy Min | Affine BN ($\gamma, \beta$) | $\text{lr} = 10^{-4}$ | Reset / Continual | **FROZEN** |
| **eata** | Regularized TTA | Affine BN ($\gamma, \beta$) | $\text{lr} = 10^{-4}, E_0 = 0.4, \beta = 0.05$ | Reset / Continual | **FROZEN** |
| **sar** | Robust TTA | Affine BN ($\gamma, \beta$) | $\text{lr} = 10^{-4}, \rho = 0.05$ | Reset / Continual | **FROZEN** |
| **cotta** | Continual TTA | All trainable weights | $\text{lr} = 10^{-4}, \alpha = 0.999, p = 0.01$ | Reset / Continual | **FROZEN** |

---

## 4. Planned Extensions for Work Package E10

Work Package E10 will execute the planned publication-strength robustness extensions:
1. **Second Lightweight Backbone Architecture:**
   - Evaluated under the same information and timing assumptions.
   - Trained on the same 27 source events across 3 seeds.
2. **Controlled Synthetic-Shift Family:**
   - Physical radar/speckle/calibration noise stress test across clean and small ordered severities.
   - Applied symmetrically across source, normalization controls, and adaptation methods.

