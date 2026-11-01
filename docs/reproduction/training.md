# Reproduction Guide: Model Training and Resumption

This document describes how to execute source training for R0 (baseline), M2_ONLY (weighting ablation), and R2 (frozen ChangeGuard recipe) across random seeds using the committed runners and configurations.

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

## 2. Committed Training Runners

ChangeGuard provides two verified entrypoints for source training:
1. `tools/robust_source_training/run_rst2_infrastructure.py`: The high-level execution engine that manages contract validation, pre-flight checks, deterministic logging, and post-training evaluation.
2. `tools/train_source_model.py`: The direct PyTorch training harness supporting explicit checkpoint resumption and batch-level controls.

Both runners read the frozen JSON configuration files located under `configs/robust_source_training/`.

---

## 3. Training Execution Commands

### A. Confirmatory 18-Event Cross-Validation Runs (SNUNet-ECAM, Folds A & B)

```bash
# Pre-flight static validation (no GPU or data required)
python tools/robust_source_training/run_rst2_infrastructure.py \
    --config configs/robust_source_training/rst5_confirmation/rst6_r2_snunet_folda_s0_v2.json \
    --validate_only

# Execute R2 on Fold A, Seed 0
python tools/robust_source_training/run_rst2_infrastructure.py \
    --config configs/robust_source_training/rst5_confirmation/rst6_r2_snunet_folda_s0_v2.json \
    --execute

# Execute Baseline R0 on Fold A, Seed 0
python tools/robust_source_training/run_rst2_infrastructure.py \
    --config configs/robust_source_training/rst5_confirmation/rst6_r0_snunet_folda_s0_v1.json \
    --execute
```

### B. Supplementary M2_ONLY Ablation Runs

```bash
# Execute M2_ONLY ablation on Fold A, Seed 0
python tools/robust_source_training/run_rst2_infrastructure.py \
    --config configs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_folda_s0_v1.json \
    --execute
```

### C. Full-Source Production Runs (All 27 Events)

```bash
# Execute R2 on all 27 events (Seed 0) via infrastructure runner
python tools/robust_source_training/run_rst2_infrastructure.py \
    --config configs/robust_source_training/rst9_all27/rst9_r2_snunet_all27_s0_v1.json \
    --execute

# Alternative: Direct execution via train_source_model.py
python tools/train_source_model.py \
    --config configs/robust_source_training/rst9_all27/rst9_r2_snunet_all27_s0_v1.json \
    --device cuda:0 \
    --all_source
```

---

## 4. Checkpoint Lifecycle & Explicit Resumption Mechanism

Automatic resumption without verification is strictly prohibited in ChangeGuard to prevent state corruption and silent configuration drift. The actual checkpointing and resumption behaviors operate under explicit contracts:

### A. Fresh-Run Recovery Policy for Confirmatory and Ablation Runs
For `CONFIRMATORY_TRAINING` and `SUPPLEMENTARY_ABLATION` execution classes, the governing contract (`src/robust_source_training/runner.py`) enforces a **fresh-run-only recovery policy**:
- Checkpoints are saved atomically as `last_checkpoint.pt` at the completion of each epoch.
- If a run is interrupted by hardware failure, preemption, or network timeout, supplying `--resume` intentionally raises a `ContractViolationError`.
- To preserve deterministic pseudo-random sequences and prevent mid-trajectory state distortion across seeds, interrupted confirmatory or ablation runs must be cleanly relaunched from epoch 0.

### B. Explicit Resumption with Verification (`tools/train_source_model.py`)
When resuming an interrupted long-running training job with `tools/train_source_model.py`:
- Resumption requires explicitly passing either:
  - `--resume <path/to/checkpoint.pt>` specifying the exact checkpoint file, or
  - `--resume_auto` to target `<output_dir>/last_checkpoint.pt`.
- If neither flag is provided, the trainer starts cleanly from epoch 0.
- When resuming, the trainer executes strict verification checks:
  1. Cryptographic validation that `config_hash` in the checkpoint exactly matches the current configuration.
  2. Verification that the random `seed` matches the declared run specification.
  3. Restoration of model weights, optimizer state, learning rate scheduler state, early stopping counter, and predictive collapse detector state.
  4. Restoration of Python, NumPy, PyTorch CPU, and CUDA PRNG states.
  5. Automatic pruning of metric CSV history (`training_history.csv`, `event_validation_metrics.csv`) back to the resumed epoch to ensure duplicate-free logs.
