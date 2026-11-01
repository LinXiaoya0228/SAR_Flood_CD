# ChangeGuard Plan V3: Subtask E1.3 Closeout Report (Bounded Real-Data Smoke)

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Execution Environment:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`  
**Target Hardware:** Physical GPU 0 (`NVIDIA GeForce RTX 3090`, isolated via `CUDA_VISIBLE_DEVICES=0` as logical `cuda:0`)  
**Launch Session:** Detached tmux session `cg_e1_3_smoke_event520_v1`  
**Git HEAD at Execution:** [`32ab2d503ec2ac8fe06c387c21901b1612fe37b0`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CURRENT_STATUS.md)  
**Governing Documents:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) & [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  
**Verification Provenance:** Implementer Self-Verification  
**Subtask Outcome:** **`E1.3 PASS`**  

---

## 1. Executive Summary & Verification Purpose

Work Package `E1.3` executed a strictly bounded, non-destructive real-data smoke test to validate the physical integration of the Plan V3 extended experiment recorder ([`ExtendedExperimentRecorder`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/recorder.py)) and orchestration pipeline ([`E1Orchestrator`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/e1_orchestrator.py)) on real SAR GeoTIFF rasters and canonical source checkpoint weights.

The run confirmed the following technical milestones:
1. **Physical File Writing & Cache Contract:** Successful serialization of Tier 1 progressive telemetry (`step_records.json`), Tier 2 spatial arrays (`frozen_predictions.npz`, 300 arrays for 50 samples), Tier 3 confusion metrics (`event_metrics.json`), cache schema specification (`cache_schema.json`), and atomic completion status.
2. **Measurement Plumbing:** Empirical capture of gradient $L_2$ norms prior to stepping, BatchNorm affine displacement from immutable source weights, classwise pseudo-label selection counts, and full-precision predictive summaries.
3. **Double-Masking Invariant:** Verified exact conservation $\sum C_{ij} = TP + FP + FN + TN = \text{valid\_pixels}$ on double-masked pixels ($MK0\_MNA == 1 \land MK0\_MLU \ne 3$) with complete exclusion of invalid pixels from all confusion cells including TN.
4. **Strict Isolation & Zero Sealed-Test Access:** Executed exclusively on Event 520 (development split). Sealed test events were strictly quarantined.

This smoke run was conducted solely for infrastructure validation and does **not** constitute a formal scientific baseline comparison.

---

## 2. Execution Command & Environment

```bash
# Executed in detached tmux session cg_e1_3_smoke_event520_v1:
CUDA_VISIBLE_DEVICES=0 /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_e1_3_real_smoke.py \
  --config configs/changeguard/e1_3_smoke_event520_v1.json \
  --output_dir outputs/e1_real_smoke/approved_smoke_event520_v1 \
  --log_dir logs/e1_real_smoke/approved_smoke_event520_v1 \
  --device cuda:0
```

- **Interpreter:** Python 3.10.19 (`vfm_env`)
- **PyTorch:** 2.1.2+cu121
- **CUDA Device:** Logical `cuda:0` mapping to Physical GPU 0 (`NVIDIA GeForce RTX 3090`, Bus-Id `0000:19:00.0`, 24.5 GB total VRAM).
- **GPU Preflight Verification:** Verified 24,560 MiB free, 0% utilization, and zero conflicting user compute processes prior to launch. Peak allocation during execution was approximately 20.4 GB.

---

## 3. Cryptographic Provenance & Input Asset Digests

| Asset Description | File Path | Verified SHA-256 Digest |
| :--- | :--- | :--- |
| **Git Snapshot** | Repository HEAD | `32ab2d503ec2ac8fe06c387c21901b1612fe37b0` |
| **Configuration** | `configs/changeguard/e1_3_smoke_event520_v1.json` | `7f38f2510eaf1032df706dbe64bb1961f0bb89348cbd1f50c0a291af49987879` |
| **Canonical Checkpoint** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| **Dataset Manifest** | `../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` | `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad` |
| **Event Split Config** | `configs/kuro_event_split.json` | `fbf30fe0bab7b90c468aaf0fbdab7960a5cf76dd49dfb3ca51f1e01668fba85f` |

---

## 4. Sample Coverage & Execution Structure

- **Target Event:** Event `520` (Development partition).
- **Sample Selection Rule:** Deterministic ascending alphanumeric sort on `sample_id`, taking head 50 samples.
- **Sample Count:** Exactly 50 unique samples (zero duplicate, zero missing).
- **Batch Partitioning:** Batches executed: `[16, 16, 16, 2]` (4 batches total).
- **Reset Mode:** `reset_per_event` (candidate weights initialized from canonical Seed 0 checkpoint).
- **Sealed Test Boundary:** 0 samples accessed from forbidden test events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.

---

## 5. Telemetry & Per-Batch Measurement Verification

Extracted directly from `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/step_records.json`:

| Step | Batch Size | Action | Total Loss | Gradient $L_2$ Norm | BN Weight Disp ($L_2$) | BN Bias Disp ($L_2$) | Selected Pixels ($c_0 / c_1 / c_2$) | Selected Fraction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 16 | `update` | `0.004799` | `0.004367` | `4.043e-7` | `3.564e-7` | 711,161 (607,643 / 71,130 / 32,388) | 88.58% |
| **2** | 16 | `update` | `0.004145` | `0.002670` | `8.450e-7` | `7.539e-7` | 694,669 (602,771 / 55,760 / 36,138) | 86.53% |
| **3** | 16 | `update` | `0.004416` | `0.004930` | `1.537e-6` | `1.510e-6` | 683,067 (590,075 / 53,424 / 39,568) | 85.08% |
| **4** | 2 | `update` | `0.004797` | `0.019657` | `2.670e-6` | `2.714e-6` | 89,737 (81,461 / 2,059 / 6,217) | 89.28% |

**Measurement Plumbing Audit:**
- Gradient $L_2$ norm was successfully sampled prior to `optimizer.step()` on active BatchNorm affine parameters.
- BatchNorm affine displacement vectors showed monotonic parameter adjustment from source ($4.04 \times 10^{-7} \to 2.67 \times 10^{-6}$).
- Classwise selection sum invariant holds exactly: $c_0 + c_1 + c_2 = \text{selected\_pixels}$ for every step.
- All recorded values are finite IEEE floating-point numbers; no values were defaulted to fabricated zeros.

---

## 6. Offline Evaluation Metrics & Conservation Invariant

Extracted directly from `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/event_metrics.json`:

- **Execution Runtime:** 20.88 seconds
- **Sensory-Valid Evaluated Pixels:** 1,904,634
- **Updates Applied / Skipped / Failed:** 4 / 0 / 0

### Binary Flood Metrics
- **True Positives (TP):** 155,993
- **False Positives (FP):** 56,585
- **False Negatives (FN):** 58,904
- **True Negatives (TN):** 1,633,152
- **Precision:** 0.733815
- **Recall:** 0.725897
- **F1 Score:** 0.729834
- **IoU:** 0.574598

### Double-Masking Conservation Verification
$$\text{Cell Sum} = 155,993 + 56,585 + 58,904 + 1,633,152 = 1,904,634 \equiv \text{valid\_pixels}$$

### Full $3 \times 3$ Class Confusion Matrix ($C_{ij}$)
| Ground Truth \ Predicted | Class 0 (Land) | Class 1 (Perm Water) | Class 2 (Flood Water) | Row Total |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Land)** | $C_{00} = 1,422,900$ | $C_{01} = 7,903$ | $C_{02} = 38,092$ | 1,468,895 |
| **Class 1 (Perm Water)** | $C_{10} = 12,149$ | $C_{11} = 190,200$ | $C_{12} = 18,493$ | 220,842 |
| **Class 2 (Flood Water)** | $C_{20} = 28,137$ | $C_{21} = 30,767$ | $C_{22} = 155,993$ | 214,897 |
| **Total** | 1,463,186 | 228,870 | 212,578 | **1,904,634** |

- $\text{FP} = C_{02} + C_{12} = 38,092 + 18,493 = 56,585$
- $\text{FN} = C_{20} + C_{21} = 28,137 + 30,767 = 58,904$
- $\text{TP} = C_{22} = 155,993$
- $\text{TN} = C_{00} + C_{01} + C_{10} + C_{11} = 1,422,900 + 7,903 + 12,149 + 190,200 = 1,633,152$
- Matrix Sum $= \sum_{i=0}^2 \sum_{j=0}^2 C_{ij} = 1,904,634 \equiv \text{valid\_pixels}$. Conservation verified.

---

## 7. Raw Artifact Inventory & Cryptographic SHA-256 Digest

Recorded in [`outputs/e1_real_smoke/approved_smoke_event520_v1/artifact_sha256.txt`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e1_real_smoke/approved_smoke_event520_v1/artifact_sha256.txt):

| File Path | Size | SHA-256 Digest |
| :--- | :---: | :--- |
| `logs/e1_real_smoke/approved_smoke_event520_v1/smoke_execution.log` | 1.2 KB | `13de1deb13ffa6c0f4df7835f1562b16bb5990e6c18c5476118963f82bce211f` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/cache_schema.json` | 1.4 KB | `23a60f951c020b14fa4736f5f8eae80026ecb20063a2c69262b0a8a3cd132a68` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/initial_provenance.json` | 407 B | `8efff379b1ee0a0d34e9cf87852d395898666cc6331238332779bd57ca43a7c6` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/initial_provenance_manifest.json` | 3.1 KB | `ae073189f160635a08f2f3dff83c035b281804c79ae8247dea6a5b69dd425b19` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/completion_status.json` | 324 B | `efbcac1cda6a0a6eafba5245fe49923cfe18a313acc1112b50cf7ef6bd4edd3c` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/condition_summary.json` | 659 B | `6adbb42d398e076bbc9b867405e8dfc1cef11efc88760432896e489925588b40` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/step_records.json` | 5.8 KB | `5a2b7841f22cd1152769cb8e5005cc0823af9189b71eccd53234dbcb3b5886b7` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/adaptation_progress.json` | 2.1 KB | `ef31beae52efe3d38ee3cc62c189a98ca1acc5bbade6101e86d9b0088ca14cf5` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/event_metrics.json` | 659 B | `6adbb42d398e076bbc9b867405e8dfc1cef11efc88760432896e489925588b40` |
| `outputs/e1_real_smoke/approved_smoke_event520_v1/event_520/frozen_predictions.npz` | 13.4 MB | `eef964d0e917bda1e45b7ad70c3761f585aec5ca52b169a5cfd4a400f9123d77` |

---

## 8. Deviations & Failed Launch Attempts

- **Deviations:** None. The run conformed strictly to the declared E1.3 specification.
- **Failed Launches:** Zero. The detached tmux session executed on the first attempt and exited cleanly with returncode 0.
- **Historical Assets:** All historical Week 8–14 outputs, configurations, and reports remained entirely untouched.

---

## 9. Conclusion & Proposed Scope for Subtask E1.4

**Subtask Verdict:** **`E1.3 PASS`**  
All physical file operations, cryptographic preflights, hardware assignments, and numerical measurement contracts have been verified on real SAR data without data corruption or label leakage.

### Proposed Minimal Scope for Subtask E1.4 (Pending Future Authorization):
- **Objective:** Complete matched baseline coverage (Source-only, matched BN-no-update, AdaBN, Tent, EATA, SAR) under unified PRE_UPDATE protocol with spatial prediction and probability caching.
- **Target Partitions:** 6 development events ($N=2,826$).
- **Multi-Factor Coverage:** Seeds 0, 1, 2; Orders A, B; Reset modes `reset_per_event` and `continual`.
- **Reuse Application:** Maximize reuse of matched historical Seed 0 cells recorded in [`docs/MATCHED_REUSE_LEDGER.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MATCHED_REUSE_LEDGER.md). Re-execute only genuine measurement/protocol gaps.
