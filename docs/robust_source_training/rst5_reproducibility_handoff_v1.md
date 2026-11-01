# ChangeGuard Robust Source-Training: Comprehensive Reproducibility and Continuation Handoff Guide

## 1. Project Objective and Route Context

The ChangeGuard project investigates whether principled, robust source-domain training strategies can improve out-of-distribution (OOD) generalization in bi-temporal synthetic aperture radar (SAR) flood extent mapping, without relying on unstable test-time adaptation (TTA) or requiring real-time target data updates.

This route transition arose following historical experiments E0 through E11 (`PROJECT_PROPOSAL.md`), which demonstrated that online test-time adaptation (e.g., Tent, EATA) suffers from systematic sensitivity and risk of catastrophic collapse on adverse SAR event streams. The robust source-training route (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`) instead focuses on building an invariant, resilient source model by addressing:
1. Severe event-size imbalance across training events (via **M1: Event-Balanced Sampling**).
2. Extreme land-cover class imbalance between background terrain, permanent water, and flood inundation (via **M2: Flood-Aware Class-Weighted Loss**).
3. Speckle noise and geometric variance (evaluated via **M3: Synchronized Paired Augmentation**).

---

## 2. Historical Boundary and Immutability Protection

1. **Historical E0-E11 Protection**:
   All historical code, configs, logs, and artifacts under E0 through E11 are permanently frozen and immutable. Specifically:
   - `docs/e11_final_scientific_conclusions.md` (`26832a93be8efd85ada522c39e9334a678c6b3860f3379efe479567d879b3f36`)
   - `docs/e11_final_test_closeout.md` (`2e0d460ce088a8a6883ed5cdfadb61da01d9baaa3bf2b28249efcd6bcfd87ff8`)
   must remain byte-identically preserved and untouched.
2. **Forbidden Target Events ($N=16$)**:
   The following 16 target events from historical experiments are strictly prohibited from being loaded, read, accessed, or evaluated during source training and confirmation:
   `[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`.
3. **Authorized Source Corpus ($N=27$)**:
   Only the 27 authorized source events (`21,462` patches) defined in `configs/robust_source_training/rst1_source_event_inventory_v1.csv` may be accessed.

---

## 3. Summary of Completed Phases (RST0 to RST4)

- **RST0** (`9c73e84e85...`): Route transition, namespace setup, and historical freeze.
- **RST1** (`d3e3c052b2...`): Data audit, KuroSiwo input domain verification, and deterministic generation of 3 balanced 9-event groups:
  - **Group A** (9 events, 8,160 patches)
  - **Group B** (9 events, 6,807 patches)
  - **Group C** (9 events, 6,495 patches)
- **RST2** (`35b48875a4...`, addendum `b704596e79...`): Implementation and verification of event-disjoint training, checkpointing, and evaluation infrastructure.
- **RST3** (`6bc10103e6...`, reconciliation `80ab1758c0...`): Implementation and smoke testing of components M1, M2, and M3; reconciliation of M2 loss and M3 augmentation formulas.
- **RST4** (`ce76b80288...`): 16-epoch discovery staircase on Discovery Fold C (Groups A+B train, Group C eval) across recipes R0, R1, R2, R3:
  - R0 (Baseline): Event-Macro F1 = 0.763134, Bottom-3 Mean F1 = 0.562984
  - R1 (+M1): Event-Macro F1 = 0.756085, Bottom-3 Mean F1 = 0.542608
  - R2 (+M1+M2): Event-Macro F1 = 0.765146, Bottom-3 Mean F1 = 0.586891 (+0.023907 tail gain)
  - R3 (+M1+M2+M3): Event-Macro F1 = 0.769657, Bottom-3 Mean F1 = 0.582430 (-0.2405 collapse on Event 497)
  - **Selected Recipe**: **R2** promoted under Route D2 (+0.0239 tail gain) and parsimony preference within 0.005 bandwidth of R3, avoiding M3's catastrophic downside risk on Event 497.

---

## 4. Exact Frozen Recipe R2 Specification

- **Architecture**: `SNUNet_ECAM` (in=2, out=3, base=32, FP32).
- **Sensory Inputs**: Dual-polarization Sentinel-1 SAR (`pre_event_2`, `post_event`), VV/VH, clamped to 0.15, normalized by population statistics (VV: 0.0953/0.0427; VH: 0.0264/0.0215).
- **Double Masking**: Valid pixels defined as `(MK0_MNA == 1) & (MK0_MLU != 3)`.
- **M1 Component**: Event-Balanced Sampler (uniform event quota draws with cyclic local replacement).
- **M2 Component**: Training-only class-weighted cross-entropy ($0.5 \cdot \text{WCE} + 0.5 \cdot \text{Dice}$) with weights recomputed strictly per fold from that fold's training events.
- **M3 Component**: **Permanently disabled** (`m3_paired_augmentation = false`).
- **Optimization**: Adam, LR 0.001, betas (0.9, 0.999), weight decay 0, microbatch 8, grad accum 4 (effective batch 32), 16 fixed epochs, epoch-local tail flushing without inter-epoch accumulator bleed.
- **Checkpoint Rule**: Final epoch checkpoint (`final_checkpoint.pt` at Epoch 16).

---

## 5. Confirmatory Folds and 12-Run Matrix

### 5.1 Folds
- **Fold A**: Train on Groups B + C (18 events, 13,302 patches); Evaluate on Group A (9 events, 8,160 patches).
  - Training Events: `[130, 147, 267, 275, 324, 417, 427, 470, 492, 497, 502, 518, 555, 1111004, 1111005, 1111006, 1111010, 1111011]`
  - Evaluation Events: `[118, 174, 273, 421, 496, 498, 554, 567, 1111009]`
  - Fold A R2 Class Weights: `[0.486491, 1.409167, 1.104342]` (`configs/robust_source_training/rst5_folda_class_weights_v1.json`)
- **Fold B**: Train on Groups A + C (18 events, 14,655 patches); Evaluate on Group B (9 events, 6,807 patches).
  - Training Events: `[118, 130, 147, 174, 273, 275, 421, 427, 492, 496, 497, 498, 554, 567, 1111005, 1111006, 1111009, 1111011]`
  - Evaluation Events: `[267, 324, 417, 470, 502, 518, 555, 1111004, 1111010]`
  - Fold B R2 Class Weights: `[0.543519, 1.499552, 0.956929]` (`configs/robust_source_training/rst5_foldb_class_weights_v1.json`)

### 5.2 Matrix Accounting
- **Methods**: R0 (Matched Baseline) and R2 (Robust Recipe).
- **Folds**: Fold A and Fold B.
- **Seeds**: 0, 1, 2.
- **Total Runs**: $2 \times 2 \times 3 = 12$ physical runs.
- **Unique Confirmatory Events**: $9 + 9 = 18$ events.
- **Total Event-Result Rows**: $12 \times 9 = 108$ event rows.

Configs reside in: `configs/robust_source_training/rst5_confirmation/`.

---

## 6. Pre-Registered Confirmatory Decision Rules

- **Criterion P1 (Average Improvement)**:
  Grand mean event-macro F1 delta $\ge +0.010$; gain not driven by only one event (excluding max event keeps delta $> 0.0$); no widespread precision/recall collapse (prec/rec delta $\ge -0.050$).
- **Criterion P2 (Tail Reliability Improvement)**:
  Bottom-3 event mean F1 delta $\ge +0.020$ (or worst-event delta $\ge +0.020$); grand mean macro F1 delta $\ge -0.005$.
- **Criterion P3 (Seed Stability)**:
  Across-seed standard deviation reduction $\ge 20\%$; standard deviation of R2 lower than R0 in both folds independently; grand mean macro F1 delta $\ge -0.005$.

---

## 7. Execution Environment & Launch Instructions for RST6

### 7.1 Environment Setup
```bash
source ~/miniconda3/envs/sar_cd/bin/activate
cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard
export PYTHONPATH="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard:$PYTHONPATH"
```

### 7.2 Authorization Transition
In RST5, all 12 configs have:
```json
"launch_authorized": false,
"preparation_only": true
```
At the formal start of **RST6**, authorized operators transition configs to:
```json
"launch_authorized": true,
"preparation_only": false
```

### 7.3 Three-Wave Launch Script Pattern
To schedule across the 4 GPUs:
- **Wave 1** (Seeds 0):
  - GPU 0: `rst6_r0_snunet_folda_s0_v1`
  - GPU 1: `rst6_r2_snunet_folda_s0_v1`
  - GPU 2: `rst6_r0_snunet_foldb_s0_v1`
  - GPU 3: `rst6_r2_snunet_foldb_s0_v1`
- **Wave 2** (Seeds 1): launched as Wave 1 completes.
- **Wave 3** (Seeds 2): launched as Wave 2 completes.

Command pattern for each run:
```bash
tmux new-session -d -s cg_rst6_<RUN_ID> \
  "python tools/robust_source_training/run_rst2_infrastructure.py \
     --config configs/robust_source_training/rst5_confirmation/<RUN_ID>.json \
   >> logs/robust_source_training/rst6/<RUN_ID>.log 2>&1"
```

---

## 8. Prohibited Actions

1. Do NOT modify model architectures, loss formulas, or class weights.
2. Do NOT evaluate on Groups A or B prior to official RST6 testing.
3. Do NOT delete or overwrite any `outputs/` or `logs/` directory.
4. Do NOT access any of the 16 forbidden target events.
5. Do NOT alter P1/P2/P3 decision thresholds post-launch.
6. Do NOT attempt `--resume` for confirmatory runs; adhere strictly to the fresh-run recovery policy (`docs/robust_source_training/rst6_fresh_run_recovery_policy_v1.md`). Interrupted jobs must be preserved and retried fresh under a versioned run ID.
