# Week 3 Source-Model Replication Protocol

**Date:** 2026-10-10
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Standard:** IEEE / ChangeGuard Controlled Reproducibility Standard
**Parent Protocol:** `docs/source_training_protocol_v2.md` (SHA-256: `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144`)
**Immutable Lock Manifest:** `configs/source/source_training_week3_replication_lock.json`

---

## 1. Scientific Objective & Scope

Week 2 established the completed seed-0 canonical source models:
- **Model A Seed 0:** SNUNet-CD `[VV, VH]` (Best Epoch 8, Inner-Val Macro F1 `0.781741`)
- **Model B Seed 0:** SNUNet-CD `[VV, VH, slope]` (Best Epoch 15, Inner-Val Macro F1 `0.797057`)

While Model B showed a $+0.0153$ Macro F1 improvement over Model A on seed 0, single-seed ($N=1$) evidence is strictly insufficient to determine whether physical slope provides a statistically stable or generalizable improvement.

### Week 3 Core Mandate:
1. **Replicate Model A `[VV, VH]` with Seed 1:** Complete an identical, from-scratch source training run on seed 1.
2. **Replicate Model B `[VV, VH, slope]` with Seed 1:** Complete an identical, from-scratch source training run on seed 1.
3. **Preregister Matched Seed 2 for Both Models:** Formally preregister production configurations for seed 2, to be executed if computational schedule permits.
4. **Produce Preliminary Multi-Seed Comparison:** Combine seed 0 and seed 1 (and seed 2 if available) to compute mean performance, variance, and confidence intervals between `[VV, VH]` and `[VV, VH, slope]`.
5. **Strict Scope Prohibition:** **Zero Test-Time Adaptation (TTA).** Zero access to quarantined TTA-validation events or sealed-test events.

---

## 2. Scientific Invariance Principles

All scientific parameters established and locked in Protocol v2 are strictly preserved across all Week 3 replication runs.

### 2.1 Allowed Variations
- **Between Seeds of the Same Model:** The **only** permitted variation is the declared pseudo-random number generator seed (`seed=1` or `seed=2`) and associated run identification metadata (`run_id`, `configuration_id`, `description`, `output_dir`).
- **Between Paired Model A and Model B:** The **only** permitted variation is the preregistered input-channel and slope policy (`in_channels_per_branch`: 2 vs 3; inclusion of metric slope normalized using train-only statistics).

### 2.2 Strictly Forbidden Variations
The following parameters are frozen and must **not** be modified:
1. **Data Split:** `configs/splits/kuro_source_inner_split_v1.json` (SHA-256: `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3`). Exactly 22 train events (19,178 samples) and 5 inner-val events (2,284 samples).
2. **SAR Preprocessing & Clamping:** Linear amplitude clamped to `[0.0, 0.15]`, nonfinite replacement, z-score normalization (`mean=[0.0953, 0.0264]`, `std=[0.0427, 0.0215]`).
3. **Slope Preprocessing:** Physical metric slope computed dynamically from projected DEM transforms; nodata sanitization; train-only z-score normalization using `artifacts/week1/source_train_slope_statistics.json` (SHA-256: `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`).
4. **Architecture:** SNUNet_ECAM with base_channel=32, out_channels=3.
5. **Loss Function:** `unweighted_ce_plus_soft_dice` (CE weight 0.5, Soft Dice weight 0.5, epsilon=1.0, ignore_index=3).
6. **Optimizer & Learning Rate:** Standard PyTorch Adam (lr=0.001, weight_decay=0.0, betas=[0.9, 0.999]). AdamW is explicitly prohibited.
7. **Batch Size Policy:** Physical microbatch size 8, gradient accumulation steps 4, effective batch size 32.
8. **Numerical Precision:** Full 32-bit floating point (`fp32`). AMP autocast and GradScaler must remain disabled.
9. **Stopping & Checkpoint Policy:** Maximum budget 150 epochs, patience 20 epochs on inner-validation event-macro flood F1. Checkpoint saved atomically on new inner-validation best.
10. **Data Leakage Safeguards:** Absolute prohibition on reading quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) or sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 3. Preregistered Model Registry

### 3.1 Parent Seed-0 Reference Models (Completed)
- **Model A Seed 0:**
  - Config: `configs/source/snunet_vvvh_v2_fp32.json` (SHA-256: `528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3`)
  - Run ID: `source_snunet_vvvh_seed0_v3_fp32`
  - Output: `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2`
  - Status: Completed (Epoch 8 best checkpoint: `750360fca8fdbef0ecfda8529bfdc3d815f93dd04686e41c10b928087b79ca9e`)
- **Model B Seed 0:**
  - Config: `configs/source/snunet_vvvh_slope_v2_fp32.json` (SHA-256: `e99e980a9afc0f72b526ccff5484e4aadaabece12c5a542bd20f94be74ef6b56`)
  - Run ID: `source_snunet_vvvh_slope_seed0_v3_fp32`
  - Output: `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2`
  - Status: Completed (Epoch 15 best checkpoint: `321e80a07a2af2e37e624ca97808efec3856488be6f71318f4b6b53f2cf3d57a`)

### 3.2 Required Seed-1 Replication Models
- **Model A Seed 1 (Radar Only):**
  - Config: `configs/source/snunet_vvvh_seed1_v2_fp32.json`
  - Configuration ID: `model_a_radar_only_seed1_v2_fp32`
  - Channels: `[VV, VH]` (2 channels)
  - Random Seed: `1`
  - Approved Run ID: `source_snunet_vvvh_seed1_v3_fp32`
  - Output Directory: `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32`
  - Status: **REQUIRED**
- **Model B Seed 1 (Radar + Slope):**
  - Config: `configs/source/snunet_vvvh_slope_seed1_v2_fp32.json`
  - Configuration ID: `model_b_topographic_ablation_seed1_v2_fp32`
  - Channels: `[VV, VH, slope]` (3 channels)
  - Random Seed: `1`
  - Approved Run ID: `source_snunet_vvvh_slope_seed1_v3_fp32`
  - Output Directory: `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32`
  - Status: **REQUIRED**

### 3.3 Preregistered Seed-2 Replication Models
- **Model A Seed 2 (Radar Only):**
  - Config: `configs/source/snunet_vvvh_seed2_v2_fp32.json`
  - Configuration ID: `model_a_radar_only_seed2_v2_fp32`
  - Channels: `[VV, VH]` (2 channels)
  - Random Seed: `2`
  - Approved Run ID: `source_snunet_vvvh_seed2_v3_fp32`
  - Output Directory: `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32`
  - Status: **PREREGISTERED_RECOMMENDED**
- **Model B Seed 2 (Radar + Slope):**
  - Config: `configs/source/snunet_vvvh_slope_seed2_v2_fp32.json`
  - Configuration ID: `model_b_topographic_ablation_seed2_v2_fp32`
  - Channels: `[VV, VH, slope]` (3 channels)
  - Random Seed: `2`
  - Approved Run ID: `source_snunet_vvvh_slope_seed2_v3_fp32`
  - Output Directory: `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32`
  - Status: **PREREGISTERED_RECOMMENDED**

---

## 4. Anti-P-Hacking Policy on Seed 2 Execution

> [!CRITICAL]
> **PREVENTING OUTCOME-DEPENDENT EXECUTION:**
> Seed 2 must **never** be run selectively based on whether seed 1 results appear favorable, unfavorable, or ambiguous.
> If seed 2 is executed, both Model A and Model B must be run.
> If seed 2 is omitted from the final source analysis, the omission must be justified strictly by recorded hardware constraints, cluster queue availability, or overall project schedule deadlines—and that decision must be documented **before** inspecting any hypothetical seed-2 result.

---

## 5. Event Integrity & Data Isolation

### 5.1 Permitted Source Training & Validation Events (27 Events Total)
- **Inner-Train Events (22):** `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`
- **Inner-Validation Events (5):** `[267, 275, 497, 518, 1111005]`

### 5.2 Quarantined & Sealed Events (Strict Zero-Access)
- **Quarantined TTA-Validation Events (6):** `[279, 437, 520, 559, 1111003, 1111008]`
- **Sealed-Test Events (10):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`

Any access to quarantined or sealed events will invalidate the entire run immediately.

---

## 6. Evaluation & Model Selection Protocol

1. **Selection Metric:** Inner-validation event-macro flood F1 (`event_macro_flood_f1`), evaluated across the 5 inner-val events at the close of each training epoch.
2. **Checkpoint Selection:** The model checkpoint (`best_checkpoint.pt`) is selected strictly by the highest event-macro flood F1 achieved during training.
3. **Deterministic Tie-Breaking:**
   - Higher event-macro flood F1
   - Higher worst-event flood F1
   - Higher event-macro flood recall
   - Smaller epoch integer
4. **Prohibition on TTA-Validation Selection:** TTA-validation data must **never** be used to select, compare, or validate source models. All source model selection decisions are confined strictly to inner-validation metrics.

---

## 7. Multi-Seed Comparison Methodology

Following completion of seed 1 (and seed 2 if executed), a multi-seed comparison report will be produced:
- **Per-Seed Reporting:** Best epoch, final epoch, event-macro F1, pooled F1, worst-event F1, validation loss.
- **Statistical Aggregation:** For each architecture (Model A vs Model B):
  $$\mu = \frac{1}{S} \sum_{s=1}^S \text{MacroF1}_s$$
  $$\sigma = \sqrt{\frac{1}{S-1} \sum_{s=1}^S (\text{MacroF1}_s - \mu)^2} \quad (S \ge 2)$$
- **Effect Size:** Mean difference $\Delta = \mu_B - \mu_A$.
- **Interpretation Standard:** If $\mu_B > \mu_A$ across replicated seeds with non-overlapping variance, the hypothesis that topographic slope aids SAR flood segmentation is supported; otherwise, the null hypothesis (or lack of stability) must be honestly disclosed.
