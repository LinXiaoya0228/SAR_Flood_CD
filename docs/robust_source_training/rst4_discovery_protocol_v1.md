# RST4 Discovery Staircase Protocol: Event-Disjoint Robust Source Training

**Protocol Identifier**: `RST4-DISCOVERY-V1`  
**Date**: 2026-10-26  
**Status**: FROZEN PRIOR TO DISCOVERY EXECUTION  
**Target Repository**: `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`

---

## 1. Executive Purpose & Scope

The objective of work package **RST4** is to execute the complete, frozen Discovery Staircase across four controlled recipes:
- **R0**: Matched-R0 Source Baseline (M1=off, M2=off, M3=off)
- **R1**: Component M1 enabled (Event-Balanced Epoch Sampling)
- **R2**: Components M1 + M2 enabled (Event-Balanced Sampling + Training-Only Flood-Aware Loss)
- **R3**: Components M1 + M2 + M3 enabled (Full Composite: M1 + M2 + Synchronized Paired SAR Augmentation)

All four recipes are trained under identical architecture, optimizer, hyperparameter, and seed-0 initialization conditions on the combined training population of **Group A + Group B** (18 events, 14,967 patches). Model predictions are evaluated strictly and independently on **Group C** (9 events, 6,495 patches).

RST4 serves as the **discovery and candidate selection phase**. At most one candidate recipe will be selected for subsequent confirmatory testing in RST6-RST7. No confirmatory claims or statistical hypothesis testing are asserted from RST4 alone.

---

## 2. Strict Data Boundaries & Partitions

### Training Partition (Groups A + B)
- **Total Events**: 18 events
- **Group A Events**: `[1111009, 273, 118, 498, 567, 554, 421, 174, 496]` (8,160 patches)
- **Group B Events**: `[1111004, 555, 502, 470, 1111010, 518, 267, 324, 417]` (6,807 patches)
- **Total Training Patches per Epoch**: 14,967 patches

### Evaluation Partition (Group C)
- **Total Events**: 9 events
- **Group C Events**: `[427, 1111011, 275, 1111005, 497, 147, 1111006, 130, 492]` (6,495 patches)
- **Total Evaluation Patches**: 6,495 patches (evaluated per-event with zero pooling bias)

### Integrity Prohibitions
1. **Forbidden Target Events**: Under no circumstances shall any of the 16 forbidden target events (`[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`) be accessed, inspected, loaded, trained upon, or evaluated.
2. **Confirmatory Group Evaluation Prohibition**: Neither Group A nor Group B may be evaluated for new-method performance during RST4. Evaluation on Groups A and B is reserved strictly for confirmatory work (RST6/RST7).
3. **No Target Leakage**: Group C labels and valid masks are accessed solely by the offline evaluation harness; zero Group C statistics or rasters may enter the training dataloader or class-weight calculation.

---

## 3. Fixed Model Architecture & Training Hyperparameters

All four recipes adhere strictly to identical model and optimization configurations:

| Parameter | Specification | Invariant / Enforcement |
| :--- | :--- | :--- |
| **Backbone Architecture** | `SNUNet_ECAM` | in_channels=2, out_ch=3, base_channel=32 |
| **Model Initialization Seed** | `0` | Deterministic initialization verified |
| **Initial State SHA-256** | `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1` | Verified identical across all 4 recipes |
| **Optimizer** | Adam | lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0 |
| **Learning Rate Schedule** | None | Fixed learning rate for entire 16 epochs |
| **Precision** | FP32 | Full float32 tensors, no mixed precision |
| **Microbatch Size** | 8 | 8 patches per forward/backward pass |
| **Gradient Accumulation** | 4 | Step taken every 4 microbatches |
| **Effective Batch Size** | 32 | $8 \times 4 = 32$ patches per optimizer step |
| **Epoch Budget** | 16 | Fixed 16 complete epochs |
| **Microbatches per Epoch** | 1,871 | $\lceil 14,967 / 8 \rceil = 1,871$ |
| **Optimizer Steps per Epoch** | 468 | $\lceil 1,871 / 4 \rceil = 468$ |
| **Total Optimizer Steps** | 7,488 | $16 \times 468 = 7,488$ |
| **Early Stopping** | None | Prohibited; runs run to epoch 16 completion |
| **Checkpoint Evaluated** | Final Epoch 16 | No checkpoint selection based on Group C metrics |
| **RNG Separation** | Independent | Model init, sampler, and augmentation RNGs do not cross-talk |

---

## 4. Staircase Recipe Definitions

### Recipe R0: Matched Source Baseline
- **Components**: M1=off, M2=off, M3=off
- **Sampling**: Standard random patch-uniform sampling across all 14,967 patches in Groups A+B (`shuffle=True`, seeded deterministically per epoch as `seed + epoch - 1`).
- **Loss**: Unweighted `ControlledSourceLoss` ($0.5 \times \text{CE} + 0.5 \times \text{Dice}_{\text{macro}}$) with unit class weights $[1.0, 1.0, 1.0]$.
- **Augmentation**: None. Raw linear amplitude clipped to $[0.0, 0.15]$ and standardized by frozen dataset constants.

### Recipe R1: Event-Balanced Epoch Sampling (M1)
- **Components**: M1=on, M2=off, M3=off
- **Sampling**: `EventBalancedSampler`.
  - Quota per event $q_e \in \{831, 832\}$ across 18 training events ($\sum q_e = 14,967$).
  - Per-event cyclic draw queues with deterministic random permutations seeded by `seed + 10000 * epoch + event_id`.
- **Loss**: Unweighted loss ($[1.0, 1.0, 1.0]$).
- **Augmentation**: None.

### Recipe R2: M1 + Training-Only Flood-Aware Loss (M2)
- **Components**: M1=on, M2=on, M3=off
- **Sampling**: `EventBalancedSampler` (identical to R1).
- **Loss**: `RobustSourceLoss` ($0.5 \times \text{CE}_{\text{weighted}} + 0.5 \times \text{Dice}_{\text{macro}}$).
  - Class weights $w_c^{\text{final}}$ computed strictly from all 14,967 patches of Groups A+B.
  - Formula:
    $$p_c = \frac{N_c}{N_{\text{total}}}, \quad \text{raw\_weight}_c = \frac{1}{\sqrt{\max(p_c, 10^{-12})}}, \quad w_c = \frac{\text{raw\_weight}_c}{\overline{\text{raw}}}$$
    $$w_c^{\text{clipped}} = \text{clip}(w_c, 0.5, 2.0), \quad w_c^{\text{final}} = \frac{w_c^{\text{clipped}}}{\overline{w}^{\text{clipped}}}$$
  - Applied exclusively to cross-entropy loss during training.
- **Augmentation**: None.

### Recipe R3: Full Composite (M1 + M2 + M3)
- **Components**: M1=on, M2=on, M3=on
- **Sampling**: `EventBalancedSampler` (identical to R1 and R2).
- **Loss**: `RobustSourceLoss` with class weights **bit-identical to R2** (same Groups A+B training set).
- **Augmentation**: `SynchronizedPairedSARAugmentation`.
  - Geometric: Horizontal flip ($p=0.5$), vertical flip ($p=0.5$), orthogonal 90-degree rotations.
  - Radiometric: Common-mode linear amplitude scaling factor $\text{gain} \sim \mathcal{U}(0.95, 1.05)$ applied identically to pre-event VV/VH and post-event VV/VH.
  - Applied prior to clipping $[0.0, 0.15]$ and standardization.
  - Bypassed during evaluation.

---

## 5. Output Accounting & Manifest Guarantees

Every recipe produces a structured output folder containing:
1. `preflight_manifest.json`: Written prior to compute initialization.
2. `training_class_weights.json`: (For R2 and R3) Records exact pixel counts and weights.
3. `last_checkpoint.pt` & `final_checkpoint.pt`: Checkpoints containing full model weights, optimizer state, epoch metadata, and configuration hashes.
4. `event_metrics.json`: Per-event confusion matrices and macro metrics for all 9 Group C events.
5. `final_manifest.json`: Terminal execution summary referencing preflight manifest hash, total steps, runtime, peak VRAM, and checkpoint hashes.
6. `terminal_status.json`: Atomic exit status (`COMPLETED_SCIENTIFIC_RUN` or `FAILED_SCIENTIFIC_RUN`).

Expected aggregate evaluation results:
- 4 recipes $\times$ 9 evaluation events = **36 event-result rows**.
- Zero missing events, zero dropped recipes.

---

## 6. Offline Promotion and Selection Rule

Following completion of all four runs, candidate selection is executed offline via a deterministic script adhering to `configs/robust_source_training/rst4_promotion_rule_v1.json`:
- **Primary Metric**: Event-Macro F1 across 9 Group C events.
- **Reliability Metrics**: Worst-event F1, Bottom-3 event mean F1.
- **Event Outcome Counting**:
  - Improved event: $\Delta \text{F1} \ge +0.01$ relative to R0.
  - Harmed event: $\Delta \text{F1} \le -0.01$ relative to R0.
  - Neutral event: $|\Delta \text{F1}| < 0.01$.
- **Eligibility Routes**:
  - *Route D1 (Mean Improvement)*: $\Delta \text{Macro F1} > 0$ AND Improved Events $\ge$ Harmed Events.
  - *Route D2 (Reliability Improvement)*: $\Delta \text{Bottom-3 F1} \ge +0.02$ AND $\Delta \text{Macro F1} \ge -0.005$.
- **Tie-Breaking / Parsimony**: If eligible recipes are within $0.005$ Macro F1, select the simplest: $\text{R1} \succ \text{R2} \succ \text{R3}$.
- **Fallback**: If no robust recipe satisfies eligibility, declare `NO_RECIPE_PROMOTED`.
