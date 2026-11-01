# Reproduction Guide: Model Training and Resumption

This document describes how to execute source training for R0 (baseline), M2_ONLY (weighting ablation), and R2 (frozen ChangeGuard recipe) across random seeds.

---

## 1. Recipe Definitions

| Recipe ID | Patch Sampler | Loss Function | Augmentation |
|---|---|---|---|
| **R0** | Uniform random patch selection across pooled dataset | Standard Cross-Entropy + Dice ($1:1$) | None |
| **M2_ONLY** | Uniform random patch selection across pooled dataset | Inverse-Square-Root Class-Weighted CE + Dice | None |
| **R1** | Uniform event sampling (M1: equal probability per event) | Standard Cross-Entropy + Dice ($1:1$) | None |
| **R2** | Uniform event sampling (M1: equal probability per event) | Inverse-Square-Root Class-Weighted CE + Dice (M2) | None |
| **R3** | Uniform event sampling (M1) | Inverse-Square-Root Class-Weighted CE + Dice (M2) | Paired SAR speckle & spatial jitter (M3) |

---

## 2. Training Execution Commands

### Executing Confirmatory Runs (SNUNet-ECAM, Folds A & B)

```bash
# Example: Train R2 on Fold A, Seed 0
python tools/robust_source_training/run_rst5_confirmation.py \
    --config configs/robust_source_training/rst5_confirmation/rst6_r2_snunet_folda_s0_v2.json \
    --device cuda:0

# Example: Train Baseline R0 on Fold A, Seed 0
python tools/robust_source_training/run_rst5_confirmation.py \
    --config configs/robust_source_training/rst5_confirmation/rst6_r0_snunet_folda_s0_v1.json \
    --device cuda:0
```

### Executing Supplementary M2_ONLY Ablation Runs

```bash
# Example: Train M2_ONLY on Fold A, Seed 0
python tools/robust_source_training/run_ablation_m2_only.py \
    --config configs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_folda_s0_v1.json \
    --device cuda:0
```

### Executing All-27 Source Runs (Production Models)

```bash
# Train R2 on all 27 events (Seed 0)
python tools/robust_source_training/run_rst9_all27.py \
    --config configs/robust_source_training/rst9_all27/rst9_r2_snunet_all27_s0_v1.json \
    --device cuda:0
```

---

## 3. Resumption and Checkpoint Lifecycle

All training runs implement fail-closed, atomic checkpoint contracts:
- Checkpoints are saved atomically at `last_checkpoint.pt` at the end of each epoch.
- If interrupted by hardware failure or preemption, relaunching with the identical config command automatically detects `last_checkpoint.pt`, restores optimizer/scheduler states and PRNG seeds, and continues seamlessly to step 10,736 (or 6,656/7,328 for fold runs).
- When training reaches the target epoch, `final_checkpoint.pt` and `final_manifest.json` are written, and the run enters `COMPLETED_SCIENTIFIC_RUN` status.

