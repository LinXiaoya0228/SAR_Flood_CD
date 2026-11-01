# RST5 Frozen Confirmatory Specification: Recipe R2

## 1. Executive Summary

This document establishes the frozen, immutable scientific specification for **Recipe R2** under the ChangeGuard Robust Source-Training Route.

Recipe R2 combines:
- **M1: Event-Balanced Sampling** (uniform draw across training events)
- **M2: Flood-Aware Class-Weighted Loss** (training-only class weighting applied to cross-entropy)
- **M3: Paired SAR Augmentation**: **PERMANENTLY DISABLED**

Recipe R2 was selected deterministically during RST4 under Route D2 (+0.023907 bottom-3 mean F1 improvement on Discovery Fold C) and parsimony selection within 0.005 bandwidth of top candidate R3 (macro F1 0.765146 vs 0.769657, gap 0.004511 <= 0.005000), avoiding the severe per-event risks observed with paired augmentation (M3 caused a -0.2405 F1 collapse on Event 497).

All parameters, loss formulations, optimizer schedules, fold configurations, and evaluation metrics defined herein are **permanently frozen for RST6 confirmatory testing**.

---

## 2. Model Architecture & Sensory Input Pipeline

### 2.1 Siamese Architecture
- **Model Class**: `SNUNetWrapper` (wrapping `SNUNet_ECAM`).
- **Input Channels**: 2 per temporal branch (VV, VH dual-polarization SAR).
- **Output Channels**: 3 (Class 0: No Water, Class 1: Permanent Water, Class 2: Flood Water).
- **Base Channels**: 32.
- **Precision**: Single Precision FP32 (`torch.float32`).
- **Mixed Precision / Autocast / Gradient Scaling**: None (strictly FP32 to ensure bitwise reproducibility across GPUs).

### 2.2 Input Preprocessing & Validity Masking
- **Temporal Pass Pairing**: Reference pass $x_A = \text{pre\_event\_2}$, emergency disaster pass $x_B = \text{post\_event}$.
- **Radar Preprocessing**:
  1. Clamp raw amplitude rasters to `[0.0, 0.15]`.
  2. Replace NaNs and infinite values: `nan=0.15, posinf=0.15, neginf=0.0`.
  3. Standardization with frozen population statistics:
     - Mean: VV = `0.0953`, VH = `0.0264`
     - Std: VV = `0.0427`, VH = `0.0215`
- **Double-Masking Rule**:
  Valid pixels are strictly:
  $$\text{valid} = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$$
  Sensor invalidity (`MK0_MNA == 0`) is mapped unconditionally to label 3 at dataset loading.

---

## 3. Method Components

### 3.1 M1: Event-Balanced Epoch Sampler
- **Class**: `src.robust_source_training.sampler.EventBalancedSampler`.
- **Sample Quotas**:
  $$\text{quota}_e = \lfloor M / E \rfloor + \mathbb{I}(e < M \pmod E)$$
  where $M$ is the matched R0 epoch sample budget, and $E$ is the number of training events in the fold.
- **Queue Cycling**: Local event sample indices are shuffled deterministically per seed and epoch, cycling with replacement when an event contains fewer samples than its quota.
- **Reproducibility**: Seeded with `base_seed + epoch - 1`.

### 3.2 M2: Training-Only Flood-Aware Loss
- **Objective Function**:
  $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{wce}} + 0.5 \cdot \mathcal{L}_{\text{dice}}$$
- **Cross-Entropy Component**:
  Weighted Cross-Entropy applied with `ignore_index = 3`:
  $$\mathcal{L}_{\text{wce}} = - \frac{1}{\sum_i w_{y_i}} \sum_{i: y_i \ne 3} w_{y_i} \log p_{i, y_i}$$
- **Soft Dice Component**:
  Macro soft Dice loss computed strictly unweighted over valid pixels to maintain gradient calibration.
- **Per-Fold Class Weight Computation**:
  Weights are computed strictly and exclusively from each confirmation fold's training events prior to launch:
  $$p_c = \frac{N_c}{N_{\text{total}}}$$
  $$\text{raw}_c = \frac{1}{\sqrt{\max(p_c, 10^{-12})}}$$
  $$\text{norm}_c = \frac{\text{raw}_c}{\text{mean}(\text{raw})}$$
  $$\text{clipped}_c = \text{clip}(\text{norm}_c, 0.5, 2.0)$$
  $$\text{final}_c = \frac{\text{clipped}_c}{\text{mean}(\text{clipped})}$$

### 3.3 M3: Paired Augmentation
- **Status**: **OFF** (`m3_paired_augmentation = false`).
- No geometric flips, rotations, or radiometric jitter are applied during confirmatory training.

---

## 4. Confirmatory Folds & Computed Class Weights

### 4.1 Confirmation Fold A
- **Training Set (Groups B + C)**: 18 events, 13,302 patches (655,369,643 valid pixels).
  - Training Events: `[130, 147, 267, 275, 324, 417, 427, 470, 492, 497, 502, 518, 555, 1111004, 1111005, 1111006, 1111010, 1111011]`
  - Valid Pixel Distribution: Class 0: 538,262,742 (82.13%), Class 1: 44,557,222 (6.80%), Class 2: 72,549,679 (11.07%).
  - **Computed Class Weights**: `[0.486491, 1.409167, 1.104342]`
  - Artifact: `configs/robust_source_training/rst5_folda_class_weights_v1.json` (SHA-256: `c3bbb6bae14d98710798883132aab623c2cc554c226f796c1e15ff87591640bd`)
- **Evaluation Set (Group A)**: 9 events, 8,160 patches.
  - Evaluation Events: `[118, 174, 273, 421, 496, 498, 554, 567, 1111009]`

### 4.2 Confirmation Fold B
- **Training Set (Groups A + C)**: 18 events, 14,655 patches (728,573,125 valid pixels).
  - Training Events: `[118, 130, 147, 174, 273, 275, 421, 427, 492, 496, 497, 498, 554, 567, 1111005, 1111006, 1111009, 1111011]`
  - Valid Pixel Distribution: Class 0: 501,089,528 (68.78%), Class 1: 65,829,766 (9.04%), Class 2: 161,653,831 (22.19%).
  - **Computed Class Weights**: `[0.543519, 1.499552, 0.956929]`
  - Artifact: `configs/robust_source_training/rst5_foldb_class_weights_v1.json` (SHA-256: `d84e797fd63a097124231c130a9728950d2bf3e676c937f99790fe4dcb239a83`)
- **Evaluation Set (Group B)**: 9 events, 6,807 patches.
  - Evaluation Events: `[267, 324, 417, 470, 502, 518, 555, 1111004, 1111010]`

---

## 5. Optimization & Tail-Accumulation Invariants

- **Optimizer**: Adam (`torch.optim.Adam`).
- **Learning Rate**: `0.001` (constant, no learning rate scheduler).
- **Betas**: `(0.9, 0.999)`.
- **Weight Decay**: `0.0`.
- **Batching**: Microbatch size `8`, gradient accumulation `4`, effective batch size `32`.
- **Training Budget**: Exactly 16 fixed epochs (no early stopping, no validation-driven selection).
- **Tail Accumulation**: Microbatch accumulation is epoch-local (`accum_microbatches = 0` upon each update), flushing remaining gradients at the end of each epoch without inter-epoch accumulator bleed.
- **Checkpoint Selection**: Final epoch checkpoint (`final_checkpoint.pt` at Epoch 16).

---

## 6. Scientific Immutability Rule

Under no circumstances may any hyperparameter, learning rate, loss formulation, or data boundary be altered during RST6 confirmatory evaluation. Re-tuning on confirmatory evaluation groups (Groups A and B) is strictly prohibited.
