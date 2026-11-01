# RST6 Fresh-Run Recovery Policy: Option 1 Protocol

## 1. Context and Rationale

The RST6 confirmatory matrix executes 12 physical runs (6 under baseline Recipe R0, 6 under robust Recipe R2) across Fold A and Fold B with seeds 0, 1, and 2.

In Phase B of RST6 preparation, an audit of `src/robust_source_training/checkpoint.py`, `src/robust_source_training/runner.py`, and `src/robust_source_training/sampler.py` confirmed that while epoch-level checkpoint restoration of model parameters, buffers, Adam optimizer states, and RNG states functions correctly, **within-epoch batch position and sampler draw progression are not serialized**.

Rather than introducing complex intra-epoch sampler serialization and replay logic immediately prior to confirmatory execution, the user and research integrity guidelines mandate **Option 1: Fresh-Run-Only Recovery**:

```text
CONFIRMATORY RESUME POLICY:
- Each formal run starts from epoch 0.
- Mid-epoch and epoch-level resume are disabled for confirmatory runs.
- An interrupted run is preserved permanently as a failed/incomplete attempt.
- A retry starts from scratch under a new versioned run ID.
- The retry uses identical scientific settings, fold, seed and initialization.
- Successful runs are never repeated.
```

### 1.1 Why Mid-Epoch Resume is Unnecessary
Each confirmatory run executes 16 epochs with effective batch size 32 on an NVIDIA RTX 3090 GPU:
- Fold A: 13,302 patches/epoch (~2.0 - 2.5 hours total runtime).
- Fold B: 14,655 patches/epoch (~2.2 - 2.8 hours total runtime).
Given the short, bounded duration of these jobs, the engineering complexity and integrity risks of intra-epoch state restoration far outweigh the cost of restarting a failed job from scratch.

### 1.2 Why Fresh Restart is Safer
Introducing untested intra-epoch state serialization and iterator splicing right before confirmation introduces risk of subtle gradient or sample ordering discrepancies. A clean start from Epoch 0 using the pre-registered seed guarantees:
1. Exact bit-identical initialization across seeds.
2. Identical deterministic sample draw sequences throughout the entire 16 epochs.
3. Perfect adherence to the frozen R0/R2 batch accounting (6,656 steps in Fold A, 7,328 steps in Fold B).

---

## 2. Recovery Rules and Invariants

### 2.1 Complete Failure Preservation
- If a process encounters an unexpected termination, signal, CUDA fault, or system reboot, the output directory (`outputs/robust_source_training/rst6/<run_id>/`) and log file (`logs/robust_source_training/rst6/<run_id>.log`) must **never be deleted or cleaned**.
- The existing failed directory is preserved as permanent negative/interrupted evidence.
- `terminal_status.json` must record `status: "FAILED"` or `status: "INTERRUPTED_RUN"` with timestamp and failure context.

### 2.2 Prohibition on Directory Overwrite
- `src/robust_source_training/runner.py` enforces that if `output_root` exists and contains files, execution fails closed with `ContractViolationError`.
- No new run may point to an existing non-empty directory.

### 2.3 Versioned Retries
- If an interrupted run must be retried, a new configuration file and run ID must be created using a strict version suffix (e.g. `rst6_r2_snunet_folda_s0_v2`).
- The retry config must copy the identical scientific parameters:
  - identical recipe (`R0` or `R2`);
  - identical components (`m1_event_balance`, `m2_flood_aware_loss`, `m3_paired_augmentation=False`);
  - identical fold (`train_event_ids`, `evaluation_event_ids`);
  - identical seed;
  - identical architecture, optimizer, learning rate, batch size, gradient accumulation, and 16-epoch budget.
- The retry run creates its own fresh directory:
  `outputs/robust_source_training/rst6/<run_id_v2>/`
- The retry must begin execution from Epoch 0, Step 0.

### 2.4 Prohibition on Partial State Reuse
- Retries must never load weights, buffers, optimizer states, or partial sample histories from an interrupted run.
- Model weights are initialized strictly from `seed_everything(seed)` and fresh constructor calls.

### 2.5 Prohibition on Rerunning Completed Runs
- A completed run marked `COMPLETED_SCIENTIFIC_RUN` with verified checkpoints, manifests, and event metrics is permanently immutable and final.
- Repeating, rerunning, or cherry-picking completed runs is strictly prohibited.

---

## 3. Enforcement Implementation

In `src/robust_source_training/runner.py`:
```python
if exec_class == "CONFIRMATORY_TRAINING" and resume_checkpoint_path is not None:
    raise ContractViolationError(
        "Confirmatory resume rejected: execution_class is 'CONFIRMATORY_TRAINING' "
        "and resume_checkpoint_path was supplied. In RST6, confirmatory training runs "
        "must execute fresh from epoch 0. Mid-epoch or epoch-level resume is disabled "
        "under the frozen fresh-run recovery policy."
    )
```
This fail-closed guard rejects any attempt to pass `--resume` to a confirmatory run before model construction or dataset allocation, preventing accidental partial state contamination.
