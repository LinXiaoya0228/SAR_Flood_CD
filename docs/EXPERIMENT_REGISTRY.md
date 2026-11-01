# ChangeGuard V3: Experiment Registry

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md)  
**Active Work Package:** `E0` (Inventory & Planning/Measurement Gap Closure)  
**Status:** Canonical Experiment Cell Registry across Work Packages `E0`–`E11`  

---

## 1. Registry Structure & Status Taxonomy

Every planned, completed, or excluded experimental stream in the ChangeGuard V3 plan is indexed below.

### Status Definitions:
- **`COMPLETED`:** Experiment executed to terminal state, artifacts cryptographically pinned, and analysis report completed.
- **`PARTIAL`:** Specific subsets of the grid executed (e.g. single seed or single hyperparameter), while complementary cells remain scheduled.
- **`PLANNED`:** Pre-registered experiment with specified protocol, factors, and target outputs; ready for execution upon dependency resolution.
- **`BLOCKED`:** Dependent on an upstream decision (e.g. Gate E) or prerequisite software artifact (e.g. extended recorder).
- **`EXCLUDED`:** Formally evaluated and omitted with explicit scientific justification (e.g. unphysical RGB jitter, uncalibrated development GT threshold tuning).

---

## 2. Master Experiment Registry (`E0` – `E11`)

### 2.1 Block E0: Scope, Matching & Measurement Contract (Window: Oct 20–22)

| Cell ID | Topic / Scope | Factors & Configuration | Target Dataset | Status | Output Artifacts & Reports |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E0-01** | Inventory & Gap Closure | Metadata inspection, schema audit, reuse rules | All completed runs | `COMPLETED` | `docs/REQUIREMENT_COVERAGE.md`<br>`docs/MATCHED_REUSE_LEDGER.md`<br>`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`<br>`docs/EXPERIMENT_REGISTRY.md`<br>`docs/CURRENT_STATUS.md` |

---

### 2.2 Block E1: Extended Recorder, Matched Baselines & Normalization Checks (Window: Oct 23 – Nov 1)

| Subtask ID | Topic / Scope | Factors (Seed / Order / Reset) | Target Events | Status | Dependencies / Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E1.1-REC** | Extended Recorder Implementation | Extended schema unit tests | Synthetic / CPU | `COMPLETED` | Implemented in `src/changeguard/recorder.py`; verified by `tests/test_extended_recorder.py` (10/10 tests pass). |
| **E1.2-INT** | Integration & Synthetic Orchestration | Full orchestration test with mock data | Synthetic / CPU | `COMPLETED` | Implemented in `src/changeguard/e1_orchestrator.py`; verified by `tests/test_e1_orchestration_synthetic.py` (13/13 tests pass; 23 total). |
| **E1.3-SMK** | Real-Data Bounded Smoke | S0 / Order A / Reset / bs=16 | Event 520 (head 50) | `COMPLETED` | Bounded real smoke passed (`docs/e1_3_smoke_closeout.md`). Physical file writes validated under `vfm_env`. |
| **E1.4-BASE**| Matched Baseline Coverage Matrix | 51 Streams: Source (3), AdaBN (12), Tent (12), EATA (12), SAR (12) | Dev 6 events ($N=2,826$) | `COMPLETED` | 51 streams completed & audited (2 reused, 49 executed new). Master tables in `outputs/e1_matched_baselines/v1/`. Closeout in `docs/e1_4_matched_baselines_closeout.md`. |
| **E1.5-BN**  | Matched BN Normalization Confirmation | 8 Streams: S0, S1, S2 × Orders A, B (bs=16) + bs $\in \{1, 4\}$ (S0, OA) | Dev 6 events ($N=2,826$) | `COMPLETED` | 8 streams completed & audited (all 8 executed new). Master tables in `outputs/e1_bn_normalization/v1/`. Closeout in `docs/e1_5_bn_normalization_closeout.md`. |
| **E1.6-CLOSE**| E1 Analysis & Closeout | Multi-seed baseline synthesis | Dev 6 events | `COMPLETED` | Offline synthesis of 59 streams completed & audited (354/354 evaluations verified). Master tables in `outputs/e1_synthesis/v1/`. Closeout in `docs/e1_6_baseline_synthesis_closeout.md`. Block E1 terminally closed. |

---

### 2.3 Block E2: Confidence and Augmentation Comparison (Window: Oct 26 – Nov 8)

| Cell ID | Condition / Grid Dimension | Configuration & Hyperparameters | Factors | Status | Output / Historical Reuse |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E2-CONF-0** | C1: No Confidence Filter | Weak/Strong Consistency, No filter | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh with extended telemetry; replicates Week 12 C1 (`outputs/e2_confidence_grid/v1/E2_CONF_NONE`). |
| **E2-CONF-70**| Consistency $\tau=0.70$ | Weak/Strong Consistency, Conf $\ge 0.70$ | S0 / Ord A / Dev6 | `COMPLETED` | Finite grid sensitivity point (`outputs/e2_confidence_grid/v1/E2_CONF_70`). |
| **E2-CONF-90**| C2: Primary Reference $\tau=0.90$| Weak/Strong Consistency, Conf $\ge 0.90$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh with extended telemetry; replicates Week 12 C2 (`outputs/e2_confidence_grid/v1/E2_CONF_90`). |
| **E2-CONF-95**| Consistency $\tau=0.95$ | Weak/Strong Consistency, Conf $\ge 0.95$ | S0 / Ord A / Dev6 | `COMPLETED` | Finite grid sensitivity point (`outputs/e2_confidence_grid/v1/E2_CONF_95`). Closeout in `docs/e2_1_confidence_grid_closeout.md`. |
| **E2-AUG-A0** | A0: Horizontal Flip Strong | Identity weak vs H-flip strong | S0 / Ord A / Dev6 | `COMPLETED` | Reused from E2.1 `E2_CONF_90`. Closeout in `docs/e2_2_augmentation_grid_closeout.md`. |
| **E2-AUG-A1** | A1: Vertical Flip Strong | Identity weak vs V-flip strong | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh with exact inverse coordinate realignment. Macro F1 = 0.5733, Pooled F1 = 0.4419. |
| **E2-AUG-A2** | A2: Small Integer Translation| Fixed shift budget, overlap mask only | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh with synchronized paired geometry and padding strictly excluded. Macro F1 = 0.5733, Pooled F1 = 0.4419. |
| **E2-AUG-A3** | A3: Independent Weak Views | Two distinct weak geometric views | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh with independent weak geometric views ($W_1$ vs $W_2$). Macro F1 = 0.5718, Pooled F1 = 0.4501. |
| **E2-AUG-A4** | A4: SAR Intensity Perturbation| Multiplicative/additive SAR noise | S0 / Ord A / Dev6 | `BLOCKED` | Formally BLOCKED: multiplicative noise is undefined on z-score standardized inputs, lacking calibration metadata for raw-domain simulation. |
| **E2-CONF-REP**| Retained Recipe Repeat | Retained base recipe ($\tau=0.90$) vs contrast ($\tau=0.00$) | S1 / Ord B / Dev6 | `COMPLETED` | Executed fresh in E2.3 (`outputs/e2_repeat_matrix/v1/`, `docs/e2_3_promotion_and_repeat_closeout.md`). Directionally and numerically consistent $+0.00006$ macro F1 gain and 50% gradient reduction across the two tested seed/order settings. |
| **E2.4-CLOSE**| Block E2 Synthesis & Closeout | Multi-cell synthesis & component freeze | Dev 6 events | `COMPLETED` | Offline synthesis of 11 cells completed & audited (60/60 evaluations verified, 54 physical). Standalone method promotion: `NONE RETAINED`; base consistency recipe retained for E3 ($\tau=0.90$, H-flip). Master tables in `outputs/e2_synthesis/v1/`. Closeout in `docs/e2_4_synthesis_closeout.md`. Block E2 terminally closed. |

---

### 2.4 Block E3: Bounded Source-Anchor Comparison (Window: Nov 2 – 15)

| Cell ID | Anchor Family | Weight ($\lambda_{\text{anchor}}$) | Factors | Status | Output / Historical Reuse |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **E3-PROB-01**| Probability Anchor (KL) | $\lambda = 0.10$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573343, Worst F1 = 0.265045. |
| **E3-PROB-05**| C3: Probability Anchor (KL)| $\lambda = 0.50$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2 with full telemetry. Macro F1 = 0.573441, Worst F1 = 0.265045. |
| **E3-PROB-10**| Probability Anchor (KL) | $\lambda = 1.00$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573556, Worst F1 = 0.265045. |
| **E3-LOGIT-01**| Logit Anchor ($L_2$ / MSE) | $\lambda = 0.10$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573368, Worst F1 = 0.265045. |
| **E3-LOGIT-05**| Logit Anchor ($L_2$ / MSE) | $\lambda = 0.50$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573541, Worst F1 = 0.265045. |
| **E3-LOGIT-10**| Logit Anchor ($L_2$ / MSE) | $\lambda = 1.00$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573800, Worst F1 = 0.265045. |
| **E3-FEAT-01** | Feature Anchor (Bottleneck) | $\lambda = 0.10$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573327, Worst F1 = 0.265045. |
| **E3-FEAT-05** | Feature Anchor (Bottleneck) | $\lambda = 0.50$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573357, Worst F1 = 0.265045. |
| **E3-FEAT-10** | Feature Anchor (Bottleneck) | $\lambda = 1.00$ | S0 / Ord A / Dev6 | `COMPLETED` | Executed fresh in E3.2. Macro F1 = 0.573391, Worst F1 = 0.265045. |
| **E3-REP**     | Retained Anchor Repeat | Promoted family and weight | S1 / Ord B / Dev6 | `NOT_REQUIRED` | Promotion verdict: `NONE RETAINED`. All 9 cells failed promotion criteria. Repeats not required per Plan V3 Section 8.3 / Section 1.3. |
| **E3.3-CLOSE** | Block E3 Terminal Synthesis & Closeout | Negative-result route synthesis & reconciliation | Dev 6 events | `COMPLETED` | Offline synthesis completed and audited (54/54 evaluations verified). Matched-control reconciliation in `docs/e3_control_provenance_reconciliation.md`. Master tables in `outputs/e3_synthesis/v1/`. Terminal closeout in `docs/e3_3_terminal_closeout.md`. Block E3 terminally closed. |

---

### 2.5 Block E4: Class-Aware Interventions (Window: Nov 16 – 22)

| Cell ID | Condition | Mechanism & Description | Factors | Status | Notes & Scope Boundaries |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E4-PREFLIGHT** | E4.1/E4.1b Preflight & Audit | Contracts, adapters, unit tests, and smoke test | S0 / Ord A / Evt520 | `COMPLETED` | Verified in `tests/test_e4_class_aware_contracts.py` (15/15 tests pass) and GPU smoke on Event 520 (32 samples). Audited & corrected in E4.1b (`docs/e4_preflight_and_contracts.md`). |
| **E4-DIAG** | Offline $3 \times 3$ Diagnosis | Decomposes Land vs Permwater errors | S0 / Ord A / Dev6 | `COMPLETED` | Completed in Week 14 (`docs/week14_class_error_diagnostics_v1.md`). |
| **E4-BASE** | Fixed Consistency Base | Base recipe ($\tau=0.90$, H-flip, no auxiliary loss) | S0 / Ord A / Dev6 | `COMPLETED` | Exact matched control reconstructed via `--reuse-base` from `E2_CONF_90`. Macro F1 = 0.573318, Worst F1 = 0.265045. |
| **E4-K1**   | Class-Balanced Selection | Uniform per-class quota cap $\lfloor N_{eligible}/3 \rfloor$ | S0 / Ord A / Dev6 | `COMPLETED` | Evaluated on GPU 0. Macro F1 = 0.573947 ($\Delta=+0.000628$), Worst F1 = 0.256616 ($\Delta=-0.008429$). Failed promotion rules. |
| **E4-K2**   | Flood-Aware Loss Weighting | Loss weighting favoring flood pixels ($w=4.0$) | S0 / Ord A / Dev6 | `COMPLETED` | Evaluated on GPU 1. Macro F1 = 0.574597 ($\Delta=+0.001278$), Worst F1 = 0.256616 ($\Delta=-0.008429$). Failed promotion rules. |
| **E4-K3**   | Permanent-Water Exclusion | Source-predicted permanent water exclusion | S0 / Ord A / Dev6 | `COMPLETED` | Evaluated on GPU 2. Legacy ID: `E4_K3_PERM_PROTECTED`. Macro F1 = 0.575518 ($\Delta=+0.002200$), Worst F1 = 0.256616 ($\Delta=-0.008429$). Failed promotion rules. Loss masking does not guarantee prediction invariance due to global BN affine drift. |
| **E4-K4**   | Flood/Change Preservation| Penalty for modifying source flood predictions | S0 / Ord A / Dev6 | `COMPLETED` | Evaluated on GPU 3. Macro F1 = 0.571677 ($\Delta=-0.001641$), Worst F1 = 0.256616 ($\Delta=-0.008429$). Failed promotion rules. |
| **E4-REP**  | Promoted K-Variant Repeat| Decisive K-condition replication | S1 / Ord B / Dev6 | `NOT_REQUIRED`| Promotion verdict: `NONE ELIGIBLE`. All 4 K cells failed promotion criteria. Repeats not required per Plan V3 Section 9.3. |
| **E4.3-CLOSE**| Block E4 Terminal Closeout & Synthesis | Provenance reconciliation and negative-result terminal closeout | Dev 6 events | `COMPLETED` | Master tables, reports, SHA-256 manifests in `outputs/e4_class_aware/v1/`. Closeout in `docs/e4_2_screening_closeout.md`. Block E4 terminally closed. |

---

### 2.6 Block E5: Temporal & Generic Risk Evidence (Window: Nov 23 – 29)

| Cell ID | Topic | Specific Signal Evaluated | Factors | Status | Objective / Decision Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E5-NULL** | Primary Null Risk Score | $s_{\text{null}}$ over pre2/pre2 and post/post pairs | S0 / Ord A / Dev6 | `PLANNED` | Assesses if identical-pair flood predictions detect risk. |
| **E5-GEN**  | Generic Risk Scores | Entropy, confidence, source KL divergence | S0 / Ord A / Dev6 | `PLANNED` | Baseline risk discriminators under PRE_UPDATE timing. |
| **E5-DRIFT**| Parameter Drift Score | BN affine parameter displacement vector | S0 / Ord A / Dev6 | `PLANNED` | Evaluates parameter distance from source. |
| **E5-EVAL** | Causal Discrimination | AUROC / AUPRC on batch harm ($F_1 \le -0.01$) | S0, S1 / Dev6 | `BLOCKED` | Evaluates incremental information of $s_{\text{null}}$ over generic risk. |

---

### 2.7 Block E6: Feasibility Decision & Method Definition (Window: Nov 30 – Dec 6)

| Cell ID | Topic | Description | Status | Decision Paths |
| :--- | :--- | :--- | :---: | :--- |
| **E6-GATE-E** | Feasibility Gate E | Formal review of E2–E5 evidence | `PLANNED` | **Branch A:** Temporal evidence has incremental value $\to$ develop policy.<br>**Branch B:** Stable TTA works without temporal risk $\to$ develop stable TTA.<br>**Branch C:** No method beats controls $\to$ publication-grade empirical failure study. |
| **E6-CLOSE**  | Closest Baselines | Compatibility matrix for CoTTA, PA-TTA, CIM-TTDA | `PLANNED` | Analyzes applicability for dense paired SAR change detection. |

---

### 2.8 Block E7: Policy Implementation or Empirical Framework (Window: Dec 7 – 13)

| Cell ID | Topic | Description | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **E7-POL** | Policy Implementation | Online PRE_UPDATE accept/reject & rollback | `BLOCKED` | Conditional on Gate E Branch A or B. |
| **E7-SMK** | Real Policy Smoke | Single bounded smoke verifying state rollback | `BLOCKED` | Validates rollback and source-fallback semantics. |
| **E7-EMP** | Empirical Suite | Expanded diagnostic suite if Branch C chosen | `BLOCKED` | Alternative path if Gate E selects Branch C. |

---

### 2.9 Block E8: Cross-Event Evaluation & Full Confirmation (Window: Dec 14 – Jan 3)

| Cell ID | Stream ID Pattern | Factors (Seed $\times$ Order $\times$ Reset) | Target Events | Status |
| :--- | :--- | :---: | :--- | :---: |
| **E8-LOEO** | `loeo_fold_{1..6}` | 6 held-out folds (U-track primary) | Dev 6 events | `BLOCKED` |
| **E8-CONF-M**| `stream_{method}_s{0,1,2}_{ord}_{rst}` | 3 Seeds $\times$ 2 Orders $\times$ 2 Reset Modes (12 streams) | Dev 6 events | `BLOCKED` |
| **E8-CONF-C**| `stream_{controls}_s{0,1,2}_{ord}_{rst}` | Matched controls across identical 12 streams | Dev 6 events | `BLOCKED` |

---

### 2.10 Block E9: Required Ablations & Method Freeze (Window: Dec 21 – Jan 3)

| Cell ID | Ablation Target | Condition Description | Status |
| :--- | :--- | :--- | :---: |
| **E9-ABL-01** | Confidence Ablation | Method without confidence filtering | `BLOCKED` |
| **E9-ABL-02** | Consistency Ablation | Method without consistency objective | `BLOCKED` |
| **E9-ABL-03** | Anchor Ablation | Method without source anchor | `BLOCKED` |
| **E9-ABL-04** | Temporal Ablation | Method with generic risk replacing temporal risk | `BLOCKED` |
| **E9-ABL-05** | Decision Ablation | Update-rejection-only vs source-fallback | `BLOCKED` |
| **E9-FREEZE** | Final Protocol Lock | Freeze method code, configs, thresholds, and seeds | `BLOCKED` |

---

### 2.11 Block E10: Robustness & Publication Extensions (Window: Jan 4 – 31)

| Cell ID | Topic | Description | Target Events | Status |
| :--- | :--- | :--- | :--- | :---: |
| **E10-ARCH** | Second Lightweight Backbone | Train alternative architecture on 27 source events (3 seeds) | Source 27 | `BLOCKED` |
| **E10-EVAL** | Second Backbone TTA | Evaluate source, generic TTA, and proposed method on 2nd model | Dev 6 events | `BLOCKED` |
| **E10-SHIFT**| Synthetic Shift Family | Physically defensible SAR perturbation across ordered severities | Dev 6 events | `BLOCKED` |
| **E10-EXT**  | External Dataset Assessment | Assess compatibility of public paired SAR datasets | External | `BLOCKED` |

---

### 2.12 Block E11: Frozen Final Evaluation & Delivery (Window: Feb 1 – May 19)

| Cell ID | Milestone | Description | Window | Status |
| :--- | :--- | :--- | :--- | :---: |
| **E11-GATE-T**| Gate T Release | Independent audit and authorization to unseal test data | Feb 22–28, 2027 | `BLOCKED` |
| **E11-TEST**  | Final Evaluation | Single frozen execution across 10 reserved test events | Mar 1–7, 2027 | `QUARANTINED` |
| **E11-DRAFT** | Draft Final Report | Submission of complete Draft FYP Report | Mar 25, 2027 | `PLANNED` |
| **E11-FINAL** | Final Report | Submission of revised, proofed Final FYP Report | Apr 9, 2027 | `PLANNED` |
| **E11-DEMO**  | Project Demonstration | Working short repeatable demonstration + static backup | Apr 12–16, 2027| `PLANNED` |
| **E11-ORAL**  | Oral Presentation | 3 complete timed rehearsals + formal defense | May 10–12, 2027| `PLANNED` |
| **E11-DRNTU** | Archive & DR-NTU | Full text, code archive, and repository preservation | May 19, 2027 | `PLANNED` |

