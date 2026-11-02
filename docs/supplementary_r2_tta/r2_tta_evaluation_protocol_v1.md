# Supplementary Evaluation Protocol: Test-Time Adaptation on Robust Source Models (R2 + TTA)

**Document ID:** `docs/supplementary_r2_tta/r2_tta_evaluation_protocol_v1.md`  
**Creation Date:** 2026-11-02  
**Status:** APPROVED AND FROZEN  
**Research Question:** Does Tent or CoTTA provide additional benefit when initialized from R2, compared with deploying the same R2 checkpoint without adaptation?

---

## 1. Scope and Rationale

This supplementary evaluation investigates whether unsupervised test-time adaptation (TTA)—specifically **Tent** (entropy minimization over batch normalization affine parameters) and **CoTTA** (continual adaptation with mean teacher and stochastic source restoration)—confers additional performance improvements or stability benefits when deployed on top of the robust source-trained model **Recipe R2** (SNUNet-ECAM trained on all 27 source events with event-balanced sampling M1 and flood-aware class weighting M2).

### Evaluation Context and Disclaimers
1. **Supplementary Status:** This evaluation is conducted on previously evaluated historical Kuro Siwo target events. It does not constitute a newly unseen confirmatory benchmark, and its results are not used to select or tune hyperparameters.
2. **Fixed Settings:** Hyperparameters for Tent and CoTTA are frozen identically to their matched R0 configurations in Work Package E11.
3. **No Target Leaks:** Target event labels are isolated and never accessed during adaptation. Physical shard persistence precedes ground-truth label reading.

---

## 2. Experimental Contrasts

### Primary Contrasts
1. **R2 + Tent vs. Fixed R2:** $\Delta = \text{F1}(\text{R2}+\text{Tent}) - \text{F1}(\text{Fixed R2})$
2. **R2 + CoTTA vs. Fixed R2:** $\Delta = \text{F1}(\text{R2}+\text{CoTTA}) - \text{F1}(\text{Fixed R2})$

### Secondary Matched Contrasts
1. **R2 + Tent vs. R0 + Tent:** $\Delta = \text{F1}(\text{R2}+\text{Tent}) - \text{F1}(\text{R0}+\text{Tent})$
2. **R2 + CoTTA vs. R0 + CoTTA:** $\Delta = \text{F1}(\text{R2}+\text{CoTTA}) - \text{F1}(\text{R0}+\text{CoTTA})$

### Contextual Contrasts (Reused Baseline Evidence)
1. **Fixed R2 vs. Fixed R0**
2. **R0 + Tent vs. Fixed R0**
3. **R0 + CoTTA vs. Fixed R0**

---

## 3. Frozen Settings and Reused Controls

### 3.1 Frozen Target Events and Population
- **Ten Target Events:** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (identical 15,896 test samples; 787,338,783 valid pixels).
- **Stream Order:** `order_a_ascending` (canonical sequential order by sample index).
- **Batch Size:** 16 (FP32 precision).
- **Sensory-Valid Mask:** $M_{\text{NA}} = 1$.
- **Positive Class:** Class 2 (newly inundated flood water).
- **Deployment Mode:** `reset_per_event` (complete episodic reset at event boundaries).
- **Prediction Timing:** Strict `PRE_UPDATE` convention (predictions durably recorded prior to parameter mutation).

### 3.2 Frozen R2 Source Checkpoints
Identities verified from `configs/robust_source_training/rst9_ledger_v1.json`:
- **Seed 0:** `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1/final_checkpoint.pt`  
  SHA-256: `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7`
- **Seed 1:** `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s1_v1/final_checkpoint.pt`  
  SHA-256: `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2`
- **Seed 2:** `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s2_v1/final_checkpoint.pt`  
  SHA-256: `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893`

### 3.3 Algorithm Hyperparameters (Unchanged from E11)
- **Tent:**
  - Optimizer: Adam ($\text{lr} = 1\times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\text{weight\_decay} = 0.0$).
  - Trainable parameters: BatchNorm affine $(\gamma, \beta)$ only.
  - Convolution and encoder/decoder weights frozen.
  - `track_running_stats = False`.
  - Loss: Shannon entropy over valid sensory pixels ($M_{\text{NA}} = 1$).
- **CoTTA:**
  - Optimizer: Adam ($\text{lr} = 1\times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\text{weight\_decay} = 0.0$).
  - Mean Teacher: EMA tracking $\alpha = 0.999$.
  - Augmentations: 4 SAR spatial transforms (Identity, H-Flip, V-Flip, Rot180) inverted and averaged.
  - Selection: Teacher max confidence $\ge 0.90$ on valid sensory pixels ($M_{\text{NA}} = 1$).
  - Stochastic Source Restoration: per-element reset probability $p = 0.01$ to initial source checkpoint.
  - **CRITICAL R2 INTEGRATION:** Student, teacher, source anchor model, and restoration targets are all initialized bit-exactly from the **SAME R2 checkpoint**. Zero R0 parameters or predictions enter R2 + CoTTA.

### 3.4 Reused Control Evidence
- **Fixed R0 & Fixed R2:** Archived in `outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv` (SHA-256 verified).
- **R0 + Tent & R0 + CoTTA:** Archived in `outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json` (SHA-256 verified).

---

## 4. Execution Matrix: Exactly Six New Physical Streams

| Stream Index | Stream ID | Method | Source Recipe | Seed | Target Events | Assigned GPU | Output Directory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `r2_snunet_tent_seed0` | Tent | R2 | 0 | 10 events | `cuda:0` | `outputs/supplementary_r2_tta/r2_snunet_tent_seed0` |
| 2 | `r2_snunet_cotta_seed0` | CoTTA | R2 | 0 | 10 events | `cuda:0` | `outputs/supplementary_r2_tta/r2_snunet_cotta_seed0` |
| 3 | `r2_snunet_tent_seed1` | Tent | R2 | 1 | 10 events | `cuda:1` | `outputs/supplementary_r2_tta/r2_snunet_tent_seed1` |
| 4 | `r2_snunet_cotta_seed1` | CoTTA | R2 | 1 | 10 events | `cuda:1` | `outputs/supplementary_r2_tta/r2_snunet_cotta_seed1` |
| 5 | `r2_snunet_tent_seed2` | Tent | R2 | 2 | 10 events | `cuda:2` | `outputs/supplementary_r2_tta/r2_snunet_tent_seed2` |
| 6 | `r2_snunet_cotta_seed2` | CoTTA | R2 | 2 | 10 events | `cuda:2` | `outputs/supplementary_r2_tta/r2_snunet_cotta_seed2` |

Total expected rows: 6 streams $\times$ 10 events = 60 event-result rows.

