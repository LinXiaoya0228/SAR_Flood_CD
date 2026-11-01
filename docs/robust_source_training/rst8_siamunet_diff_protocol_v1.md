# RST8 SiamUnet_diff Architecture Transfer Protocol

## 1. Scope and Governing Principles

This document defines the frozen scientific protocol, architectural specifications, configuration matrix, and Stage-A continuation rule for **Work Package RST8 -- Second-Backbone Verification** of the ChangeGuard robust source-training route (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`).

RST8 evaluates whether the benefits of Recipe R2 (M1 event-balanced sampling + M2 training-only inverse-square-root class-balanced cross-entropy), validated on SNUNet-ECAM during RST6 and RST7, transfer to an independent second change-detection architecture: **SiamUnet_diff** (FC-Siam-diff).

This protocol is permanently frozen before executing any training or inspecting any evaluation results on SiamUnet_diff.

---

## 2. Architecture Specification: SiamUnet_diff

### 2.1 Provenance and Model Structure
- **Reference**: Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP), pp. 4063-4067. IEEE.
- **Source Implementation**: Native PyTorch implementation in `src/models/siam_diff_wrapper.py`.
- **Architectural Mechanics**:
  - Two parallel encoding streams with shared weights processing pre-event observation `xA` and post-event observation `xB`.
  - Four hierarchical convolutional stages in the encoder (16, 32, 64, 128 channels) using 3x3 convolutions, `BatchNorm2d`, ReLU activations, and `Dropout2d(p=0.2)`, followed by 2x2 max-pooling.
  - Four hierarchical decoder stages utilizing transposed convolutions (`ConvTranspose2d`) for 2x spatial upsampling.
  - **Skip-Difference Mechanism**: At each decoder stage, the upsampled feature map is concatenated with the absolute element-wise difference of corresponding pre- and post-event encoder features:
    `torch.abs(x_encoder_pre - x_encoder_post)`.
  - Dense prediction head producing unnormalized 3-class logits.

### 2.2 Parameter Count and Normalization
- **Total Parameters**: Exactly 1,350,147 parameters (~1.35M parameters, compared to 12,034,819 parameters in SNUNet-ECAM).
- **Normalization**: Batch normalization (`BatchNorm2d`) is used across all convolutional blocks.
- **Regularization**: 2D spatial dropout (`Dropout2d(p=0.2)`) is active during training and disabled during evaluation (`model.eval()`).

### 2.3 Semantic Labels and Output Invariants
The model outputs dense 3-class logits matching the ChangeGuard label contract:
- Class 0: Dry ground / unflooded land
- Class 1: Permanent water body
- Class 2: Newly inundated flood water (**positive evaluation class**)
- Class 3: Ignore / invalid data mask (strictly excluded from loss and evaluation metrics)

---

## 3. Confirmatory Experimental Matrix (Stage A: Seed 0)

Stage A executes four formal training runs across the two frozen confirmatory folds using seed 0:

| Run ID | Backbone | Recipe | Fold | Seed | Train Events | Eval Events | Optimizer Steps | GPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rst8_r0_siamunet_diff_folda_s0_v1` | SiamUnet_diff | R0 | Fold A | 0 | 18 events (13,302 patches) | 9 events (8,160 patches) | 6,656 | GPU 0 |
| `rst8_r2_siamunet_diff_folda_s0_v1` | SiamUnet_diff | R2 | Fold A | 0 | 18 events (13,302 patches) | 9 events (8,160 patches) | 6,656 | GPU 1 |
| `rst8_r0_siamunet_diff_foldb_s0_v1` | SiamUnet_diff | R0 | Fold B | 0 | 18 events (14,655 patches) | 9 events (6,807 patches) | 7,328 | GPU 2 |
| `rst8_r2_siamunet_diff_foldb_s0_v1` | SiamUnet_diff | R2 | Fold B | 0 | 18 events (14,655 patches) | 9 events (6,807 patches) | 7,328 | GPU 3 |

### Strict Scientific Invariants
1. **Fold Partitions**:
   - Fold A evaluates Group A (`118, 174, 273, 421, 496, 498, 554, 567, 1111009`).
   - Fold B evaluates Group B (`267, 324, 417, 470, 502, 518, 555, 1111004, 1111010`).
   - The 16 historical target events remain strictly excluded.
2. **Hyperparameter Parity**:
   - Optimizer: Adam (`lr = 1e-3`, `betas = [0.9, 0.999]`, `weight_decay = 0.0`).
   - Microbatch size: 8; gradient accumulation: 4; effective batch size: 32.
   - Epoch budget: 16 epochs.
   - Loss components:
     - R0: Unweighted CE (weight 0.5) + Dice (weight 0.5).
     - R2: Training-only inverse-square-root class-balanced CE (weight 0.5, using frozen fold class weights) + Dice (weight 0.5).
     - Fold A class weights: `[0.613865, 1.487843, 0.898292]` (`rst5_folda_class_weights_v1.json`).
     - Fold B class weights: `[0.543519, 1.499552, 0.956929]` (`rst5_foldb_class_weights_v1.json`).
   - M1 event-balanced sampling active for R2; disabled for R0.
   - M3 paired augmentation disabled for both R0 and R2.

---

## 4. Frozen Stage-A Continuation Rule

Stage A evaluates whether the architecture transfer direction is supported before committing resources to multi-seed expansion (seeds 1 and 2).

### 4.1 Comparison Protocol
1. Calculate per-event F1 for Class 2 across all 18 unique held-out events for seed-0 R0 and seed-0 R2.
2. Calculate paired per-event deltas:
   `Delta_F1(e) = F1(e, R2, s0) - F1(e, R0, s0)`.
3. Calculate grand mean event-macro delta:
   `Delta_GrandMean = (1/18) * sum_{e=1}^18 Delta_F1(e)`.
4. Calculate tail metrics:
   - `Delta_Bottom3Mean = Bottom3Mean(R2, s0) - Bottom3Mean(R0, s0)`.
   - `Delta_WorstEvent = WorstEvent(R2, s0) - WorstEvent(R0, s0)`.
5. Check single-event driver condition:
   `Delta_GrandMean_minus_max = (1/17) * sum_{e != e_max} Delta_F1(e) > 0.0`.
6. Check precision/recall collapse conditions:
   - Macro precision drop `> -0.050`.
   - Macro recall drop `> -0.050`.
   - Count of events with dual drops >= 0.030 is `<= 4`.

### 4.2 Continuation Decision Tree
- **Condition P1 (Average Improvement)**:
  `Delta_GrandMean >= +0.010` AND `Delta_GrandMean_minus_max > 0.0` AND no precision/recall collapse.
- **Condition P2 (Tail Reliability Improvement)**:
  (`Delta_Bottom3Mean >= +0.020` OR `Delta_WorstEvent >= +0.020`) AND `Delta_GrandMean >= -0.005`.
- **Condition P3 (Seed Stability)**:
  **NOT EVALUABLE AT STAGE A** (requires >= 2 seeds).

### 4.3 Branching Action
- **If P1 or P2 is SATISFIED**:
  - Issue verdict: `TRANSFER DIRECTION SUPPORTED -- THREE-SEED EXPANSION AUTHORIZED`.
  - Prepare (but do not launch) seed 1 and seed 2 configs.
  - Await separate user authorization for expansion.
- **If neither P1 nor P2 is SATISFIED**:
  - Issue verdict: `TRANSFER DIRECTION NOT SUPPORTED -- STOP AT SEED 0`.
  - Conclude RST8. Report architectural dependence honestly and scope robust source training to SNUNet-ECAM.
