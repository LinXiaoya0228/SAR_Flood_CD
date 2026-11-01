# ChangeGuard Plan V3 Work Package E2.1: Preflight & Reuse Equivalence Audit

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `167b172da2241ac2b6e7795a7cfc1ace224823d4` (`main`)  
**Work Package:** `E2.1` (Confidence Grid Screening)  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) Section 7.1  
**Measurement Contract:** [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  

---

## 1. Executive Summary & Purpose

This preflight document establishes the formal operational parameters and historical equivalence assessment for **Work Package E2.1: Confidence Grid Screening** under ChangeGuard Master Execution Plan V3.

Work Package E2.1 executes a rigorous one-factor component screen over the confidence threshold parameter ($\tau$) within the weak-strong consistency test-time adaptation framework on the Development 6-event benchmark ($N=2,826$ samples). 

Before launching GPU compute, this preflight:
1. Verifies the clean starting repository state;
2. Formally audits existing Week 12 exploratory ablation artifacts (`C1` and `C2`) against the Plan V3 measurement contract;
3. Determines cell reusability strictly according to Plan V3 rules;
4. Locks the complete threshold grid before inspecting any development performance;
5. Confirms the strict quarantine of the reserved sealed-test partition.

---

## 2. Git Status and Working Tree Verification

- **Branch:** `main`
- **Starting Commit:** `167b172da2241ac2b6e7795a7cfc1ace224823d4`
- **Commit Subject:** `docs(e1.6): complete multi-seed baseline synthesis and close Block E1`
- **Tracked Working Tree State:** Clean (zero unstaged modifications on tracked files).
- **Environment:** Miniconda `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`).
- **Hardware Target:** Logical GPU 0 (`cuda:0`, NVIDIA RTX 3090, 24 GiB VRAM).

---

## 3. Historical Artifact Audit & Reuse Equivalence Analysis

Section 7.1 of Plan V3 specifies that existing `C1` and `C2` cells from Week 12 may be reused if and only if complete equivalence is demonstrated across all measurement, protocol, and code dimensions.

### 3.1 Identification of Existing Artifacts
- **Condition C1 (No Confidence Filter):**  
  `outputs/week12_ablation/approved_dev6_s0_v1/C1_consistency_no_conf/`
- **Condition C2 (Confidence $\tau \ge 0.90$ Primary Reference):**  
  `outputs/week12_ablation/approved_dev6_s0_v1/C2_consistency_conf90/`
- **Governing Config:**  
  `configs/changeguard/week12_ablation_dev6_v1.json` (`e2e882bb4d290ec5d5d1c38e205ec96ce76c9c000d92f2a098f8bb890b1d12e0`)

### 3.2 Thirteen-Dimension Equivalence Comparison Matrix

| # | Dimension | Week 12 Historical Artifacts (`C1`, `C2`) | Plan V3 Work Package E2.1 Specification | Equivalence Status |
| :-: | :--- | :--- | :--- | :---: |
| 1 | **Model Architecture & Checkpoint** | SNUNet_ECAM Seed 0 (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`), SHA-256: `247d3584...` | SNUNet_ECAM Seed 0, identical checkpoint and SHA-256 (`247d3584...`) | **MATCH** |
| 2 | **Seed & Stream Order** | Seed 0, Order A (`[279, 437, 520, 559, 1111003, 1111008]`) | Seed 0, Order A (`[279, 437, 520, 559, 1111003, 1111008]`) | **MATCH** |
| 3 | **Reset Mode** | `reset_per_event` | `reset_per_event` | **MATCH** |
| 4 | **Target Events & Samples** | Dev 6 events, exactly $N = 2,826$ patches | Dev 6 events, exactly $N = 2,826$ patches | **MATCH** |
| 5 | **Batch Sizing & Tail Handling** | Batch size 16, tail processed as-is | Batch size 16, tail processed as-is | **MATCH** |
| 6 | **Consistency Objective** | One-way KL divergence from detached weak to aligned strong log-probs | One-way KL divergence from detached weak to aligned strong log-probs | **MATCH** |
| 7 | **Augmentation Strategy** | Weak: Identity; Strong: Horizontal Flip (`torch.flip(..., dims=[-1])`) | Weak: Identity; Strong: Horizontal Flip (`torch.flip(..., dims=[-1])`) | **MATCH** |
| 8 | **Prediction Timing** | `untransformed_pre_update_candidate` (`PRE_UPDATE`) | `PRE_UPDATE` primary prediction | **MATCH** |
| 9 | **Confidence Definition** | $\max_{c} P(Y=c) \ge \tau$ over sensory-valid pixels | $\max_{c} P(Y=c) \ge \tau$ over sensory-valid pixels | **MATCH** |
| 10 | **Screening Thresholds** | C1: No filter ($\tau=0.0$); C2: $\tau=0.90$ | C1: $\tau=0.0$; C2: $\tau=0.90$ (part of 4-cell grid) | **MATCH** |
| 11 | **Masking Semantics** | Adaptation: $V = (\text{MK0\_MNA} == 1)$; Eval: $V = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$ | Adaptation: $V = (\text{MK0\_MNA} == 1)$; Eval: $V = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$ | **MATCH** |
| 12 | **Recorder Schema & Telemetry** | **Legacy Week 12 schema:**<br>- Missing `selected_pixels_c0/c1/c2`<br>- Missing `gradient_norm_l2`<br>- Missing `bn_weight_displacement`<br>- Missing `bn_bias_displacement`<br>- Missing float16 probability maps (`_prob_c2`)<br>- Missing post-update diagnostic predictions<br>- Missing `cache_schema.json` | **Extended Plan V3 Contract (`CACHE_SCHEMA_V1`):**<br>- Complete per-class pixel counts<br>- Gradient L2 norm before step<br>- BN affine L2 displacement from source<br>- Float16 Class-2 probability maps<br>- Full diagnostic post-predictions<br>- Atomic progress and schema persistence | **CRITICAL MISMATCH** |
| 13 | **Git Commit & Code Hash** | Commit `0a8b63cbb12b81f299bfedfeb5fabd1159714d70` | Commit `167b172da2241ac2b6e7795a7cfc1ace224823d4` | **DIFFERENT (V3)** |

### 3.3 Reuse Determination
While the mathematical modeling and primary segmentation outputs of Week 12 C1 and C2 match the required experimental recipe, **their persisted artifacts omit critical telemetry dimensions explicitly mandated by Plan V3 Section 7.1 and the E2.1 prompt**:
1. Selected-pixel count and fraction per predicted class ($c_0, c_1, c_2$);
2. Pre-step gradient L2 norm telemetry;
3. Post-step parameter displacement magnitudes (BN weight and bias L2 norms);
4. Full-resolution float16 Class-2 flood probability maps (`_prob_c2`).

**Decision:**
Under Rule 5 ("Reuse an existing cell only if equivalence is demonstrated. Otherwise mark it non-reusable and execute the required matched cell"):
- The existing Week 12 C1 and C2 artifacts are classified as **`NON_REUSABLE_FOR_PRIMARY_TABLES`** due to telemetry and cache schema deficiencies.
- They will be retained in the repository as **`HISTORICAL_CONSISTENCY_BASELINE`** to cross-validate that newly generated metrics replicate the historical findings.
- All four cells of the locked E2.1 confidence grid ($\tau \in \{0.0, 0.70, 0.90, 0.95\}$) will be **executed fresh** under the unified Plan V3 E2.1 runner (`tools/run_e2_1_confidence_grid.py`) to guarantee 100% telemetry completeness and zero schema gaps.

---

## 4. Locked Predefined Threshold Grid

To prevent post-hoc data dredging or threshold tuning on evaluation data, the complete set of confidence thresholds is locked prior to viewing any development metrics:

1. **`E2_CONF_NONE`:** No confidence filtering ($\tau = 0.0$). All sensory-valid pixels ($V = (\text{MK0\_MNA} == 1)$) contribute to the one-way KL consistency loss.
2. **`E2_CONF_70`:** Moderate confidence threshold ($\tau = 0.70$). Only sensory-valid pixels with max weak probability $\ge 0.70$ contribute to the loss.
3. **`E2_CONF_90`:** Predefined primary reference threshold ($\tau = 0.90$). Consistent with historical C2 and literature defaults.
4. **`E2_CONF_95`:** High confidence threshold ($\tau = 0.95$). Highly selective filtering.

**Inviolable Constraint:** No additional threshold (e.g. 0.80, 0.85, 0.99) will be evaluated or introduced after inspecting development set performance.

---

## 5. Mandatory Screening Telemetry Specifications

For every confidence cell and every event stream, the runner must record and persist:
- `valid_pixels`: Total sensory-valid pixels ($MK0\_MNA == 1$);
- `selected_pixels`: Total pixels meeting confidence threshold within sensory mask;
- `selected_pixel_fraction`: $\text{selected\_pixels} / \text{valid\_pixels}$;
- `selected_pixels_c0`, `selected_pixels_c1`, `selected_pixels_c2`: Exact pixel counts per predicted class;
- `selected_fraction_c0`, `selected_fraction_c1`, `selected_fraction_c2`: Class selection fractions relative to total valid pixels and relative to selected pixels;
- `batches_considered`, `batches_updated`, `batches_skipped`, `batches_failed`: Exact batch lifecycle counts;
- `zero_selection_batches`: Number of batches where `selected_pixels == 0`;
- `loss_mean`, `loss_min`, `loss_max`: Raw loss statistics over updated batches;
- `grad_norm_l2_mean`, `grad_norm_l2_min`, `grad_norm_l2_max`: Euclidean gradient norm of trainable BN affine parameters;
- `bn_weight_displacement_l2`, `bn_bias_displacement_l2`: L2 distance from initial source model weights;
- Full $3 \times 3$ confusion matrices ($c_{00}$ through $c_{22}$) and derived binary metrics (TP, FP, FN, TN, F1, Recall, Precision, IoU);
- Peak GPU VRAM utilization and runtime duration.

---

## 6. Sealed Test Partition Quarantine Notice

The 10 reserved test events are strictly quarantined:
```
FORBIDDEN_SEALED_TEST_EVENTS = {
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
}
```
Any attempt to access, read, or process samples from these events will trigger an immediate fail-fast termination before execution.

