# ChangeGuard Work Package E10: Second Backbone Training Closeout & Verification

**Document Version:** `e10_second_backbone_training_closeout_v1`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E10 (Section 15)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `ALL_SOURCE_TRAINING_COMPLETED_AND_VERIFIED`  

---

## 1. Executive Summary

This closeout report provides the definitive record, artifact provenance, and cryptographic verification for the training of ChangeGuard's second lightweight backbone (`SiamUnet_diff`; FC-Siam-diff, Daudt et al., IEEE ICIP 2018).

In strict accordance with Block E10 of Master Execution Plan V3:
1. **Source-Inner Recipe Screening (Phase 2):** Was conducted exclusively on the frozen 22-event inner-train / 5-event inner-validation partition (`configs/splits/kuro_source_inner_split_v1.json`) at Seed 0. Zero development or sealed-test events were accessed. The inner-validation Event-Macro Flood F1 peaked at **Epoch 5 (`0.7888`)**, locking the source-training budget at $E^* = 5$ epochs.
2. **All-Source Training (Phase 3):** All three pre-registered seeds (Seeds 0, 1, 2) were trained from scratch on all 27 approved source events (21,462 patches) for exactly $E^* = 5$ fixed epochs in full FP32 precision.
3. **Verification:** All final checkpoints have been committed atomically, cryptographically hashed (SHA-256), confirmed 100% numerically finite, and verified loadable into evaluation adapters.

---

## 2. Source-Inner Recipe Screening Telemetry (Seed 0)

Screening executed under `tools/train_second_backbone.py --mode source_inner_screen` using config `configs/source/siam_diff_screen_recipe_a.json`:
- **Architecture:** `SiamUnet_diff` (1,348,771 parameters)
- **Inner-Train Corpus:** 22 events (19,178 patches)
- **Inner-Validation Corpus:** 5 events (`[267, 275, 497, 518, 1111005]`, 2,284 patches)
- **Optimization:** Adam, $\text{lr} = 1.0 \times 10^{-3}$, effective batch size 32 (microbatch 16, accumulation steps 2).

### 2.1 Complete 10-Epoch Screening History

| Epoch | Train Loss | Val Loss | Event-Macro Flood F1 | Worst-Event Flood F1 | Macro Flood IoU | Macro Flood Recall | Pooled Flood F1 | Runtime | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.3148 | 0.3560 | 0.7353 | 0.6321 | 0.5884 | 0.8002 | 0.7657 | 3729.7s | Initial best saved |
| **2** | 0.2482 | 0.3315 | 0.7594 | 0.6816 | 0.6168 | 0.8223 | 0.7719 | 187.5s | Updated best |
| **3** | 0.2345 | 0.3204 | 0.7552 | 0.6608 | 0.6126 | 0.8384 | 0.7807 | 128.1s | Retained Epoch 2 |
| **4** | 0.2278 | 0.3172 | 0.7811 | 0.7217 | 0.6435 | 0.8040 | 0.7681 | 127.1s | Updated best |
| **5** | **0.2153** | **0.2542** | **0.7888** | **0.7284** | **0.6546** | **0.8437** | **0.7944** | 126.8s | **LOCKED BEST CHECKPOINT** |
| **6** | 0.2102 | 0.2403 | 0.7879 | 0.7404 | 0.6531 | 0.8314 | 0.7845 | 135.2s | Retained Epoch 5 |
| **7** | 0.2071 | 0.2574 | 0.7877 | 0.7354 | 0.6526 | 0.8122 | 0.7767 | 164.7s | Retained Epoch 5 |
| **8** | 0.2035 | 0.2374 | 0.7875 | 0.7373 | 0.6528 | 0.8303 | 0.7835 | 126.4s | Retained Epoch 5 |
| **9** | 0.1986 | 0.2072 | 0.7827 | 0.7181 | 0.6469 | 0.8456 | 0.7860 | 125.1s | Retained Epoch 5 |
| **10**| 0.1973 | 0.2421 | 0.7876 | 0.7240 | 0.6525 | 0.8060 | 0.7723 | 124.4s | Retained Epoch 5 |

### 2.2 Screening Decision and Recipe Lock
In accordance with the pre-registered decision rule:
- Primary selection criterion: Maximum Event-Macro Flood F1 on inner-validation partition.
- Outcome: **Epoch 5** achieved the global peak Event-Macro Flood F1 of **0.7888** (with Worst-Event F1 of 0.7284 and Macro IoU of 0.6546).
- **Locked Recipe Parameters:**
  - $E^* = 5$ fixed epochs
  - $\eta^* = 1.0 \times 10^{-3}$
  - Optimizer: Adam $(\beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 0.0)$
  - Effective Batch Size: 32 (microbatch 16, accumulation steps 2)
  - Loss: ControlledSourceLoss (unweighted CE + Soft Dice, class 3 ignored)
  - Record Lock Path: [`configs/source/siam_diff_recipe_lock.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/siam_diff_recipe_lock.json)

---

## 3. All-Source Three-Seed Training Ledger (Phase 3)

Training executed on all 27 source events (21,462 patches) for exactly 5 fixed epochs without validation tuning.

| Metric / Property | Seed 0 (`all27_seed0`) | Seed 1 (`all27_seed1`) | Seed 2 (`all27_seed2`) |
| :--- | :--- | :--- | :--- |
| **Configuration ID** | `siam_diff_all27_seed0_fp32` | `siam_diff_all27_seed1_fp32` | `siam_diff_all27_seed2_fp32` |
| **Random Seed** | `0` | `1` | `2` |
| **Hardware** | GPU 1 (RTX 3090) | GPU 2 (RTX 3090) | GPU 3 (RTX 3090) |
| **Epoch Budget** | Exactly 5 fixed epochs | Exactly 5 fixed epochs | Exactly 5 fixed epochs |
| **Epoch 1 Train Loss** | `0.32071` (116.5s) | `0.31660` (120.7s) | `0.32906` (115.4s) |
| **Epoch 2 Train Loss** | `0.25026` (116.9s) | `0.25769` (121.3s) | `0.25621` (116.2s) |
| **Epoch 3 Train Loss** | `0.23529` (114.9s) | `0.23708` (120.1s) | `0.23470` (115.3s) |
| **Epoch 4 Train Loss** | `0.22272` (114.6s) | `0.22493` (119.6s) | `0.22412` (115.5s) |
| **Epoch 5 Final Loss** | **`0.21611`** (116.8s) | **`0.21798`** (120.7s) | **`0.21515`** (117.0s) |
| **Total Duration** | 579.7s (9.66m) | 602.4s (10.04m) | 579.4s (9.66m) |
| **Optimizer Steps** | 3,355 ($5 \times 671$) | 3,355 ($5 \times 671$) | 3,355 ($5 \times 671$) |
| **Final Checkpoint SHA-256** | `8e23da9a4d5f1ad01889075d6c98a7197ce295f77ea8d377f1ef26ac814b1532` | `a92f52153fda9e966d8f8e5ffe5d92c36783d1323d44a0c855d6464ae63e60ea` | `d96bb4e8d74b6e7d0163129d8990769f5e1650ed9d744cafa2a291638b251aab` |
| **Final Checkpoint Path** | [`outputs/e10_second_backbone/all27_seed0/final_checkpoint.pt`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e10_second_backbone/all27_seed0/final_checkpoint.pt) | [`outputs/e10_second_backbone/all27_seed1/final_checkpoint.pt`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e10_second_backbone/all27_seed1/final_checkpoint.pt) | [`outputs/e10_second_backbone/all27_seed2/final_checkpoint.pt`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e10_second_backbone/all27_seed2/final_checkpoint.pt) |
| **Numerical Integrity** | 100% Finite (0 NaN / Inf) | 100% Finite (0 NaN / Inf) | 100% Finite (0 NaN / Inf) |

---

## 4. Architectural Comparison: SiamUnet_diff vs SNUNet-ECAM

| Characteristic | Primary Backbone (SNUNet-ECAM) | Second Backbone (SiamUnet_diff) | Scientific Role in E10 |
| :--- | :--- | :--- | :--- |
| **Paradigm** | Nested dense skip connections with ECAM attention | Dual-branch Siamese encoder with difference skip connections | Structural contrast (dense attention vs absolute difference) |
| **Parameter Count** | 12,056,067 (~12.06M) | 1,348,771 (~1.35M) | Parameter scale contrast (8.9x reduction) |
| **BatchNorm Layers** | 56 layers | 19 layers | Affine adaptation capacity contrast |
| **Source-Inner F1** | `0.7817` (Seed 0) | `0.7888` (Seed 0) | Matched baseline performance on identical source data |
| **Source All-27 Final Loss** | `0.245` -> `0.116` (16 epochs) | `0.321` -> `0.216` (5 epochs) | Bounded, convergent optimization trajectories |

---

## 5. Affirmation of Scientific Integrity

1. **Zero Access to Test Partitions:** The 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) were strictly quarantined and never accessed, read, or probed during recipe screening or all-source training.
2. **Deterministic Seed Parity:** All three seeds are retained in equal standing. No "best seed" has been cherry-picked.
3. **Execution Readiness:** Checkpoints are frozen and ready for Phase 4 development-event evaluation and Phase 6 synthetic-shift stress testing.

