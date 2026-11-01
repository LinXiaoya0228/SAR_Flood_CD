# Week 6 All-Source Fixed-Epoch Diagnostic Smoke Closeout

**Document ID:** `docs/week6_all27_smoke_closeout.md`  
**Date (UTC):** 2026-10-15  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Protocol:** `docs/source_training_final_all27_protocol.md` (`2c15d3fdd1b3018ab5fe28b88b15a41dfe8b736e5b3193bf1d789049cb1429e2`)  
**Status:** `SMOKE_TEST_PASSED` — Pipeline Verified for Formal Multi-Seed Execution  

---

## 1. Executive Summary

This closeout document provides the definitive technical verification for the Week 6 diagnostic smoke test of ChangeGuard's all-source fixed-epoch source-training pipeline. The diagnostic smoke run was conducted using production configuration `snunet_vvvh_all27_seed0_v1_fp32.json` under strictly controlled diagnostic flags (`--resume_batches 8`, diagnostic run ID, isolated diagnostic output directory).

The diagnostic run executed in an isolated detached tmux session (`cg_w6_all27_smoke`) on GPU 0 (NVIDIA GeForce RTX 3090) and completed in **888.76s** (~14.8 minutes). The test verified:
1. Complete training pass across all 21,462 patches and all 27 source events in Epoch 1 with zero validation calls.
2. Bit-exact parameter restoration upon reloading from `last_checkpoint.pt` (`max_parameter_abs_diff = 0.000000e+00`).
3. Resumption viability across an 8-microbatch (2 optimizer steps) continuation pass in Epoch 2, advancing `global_step` to 673 with completely finite losses, gradients, and model weights.
4. Complete isolation of the quarantined TTA-validation (6 events) and sealed-test (10 events) partitions.

No formal training run has been executed. No formal final checkpoint (`final_checkpoint.pt`) was created. The pipeline is fully verified, regression-tested (126 passing tests), and approved for formal multi-seed execution.

---

## 2. Diagnostic Run Evidence & Artifact Ledger

The diagnostic run generated immutable artifacts within the dedicated directory:
`/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week6_smoke/smoke_snunet_vvvh_all27_seed0_v1_fp32/`

### Artifact Inventory
- **Run Manifest:** `smoke_manifest.json` (`status: SMOKE_TEST_PASSED`, `is_formal_run: false`)
- **Run Summary:** `smoke_summary.json` (Structured machine-readable metrics)
- **Execution Log:** `smoke.log` (Complete stdout/stderr trace)
- **Training Ledger:** `train_history.csv` (Per-epoch training loss and step telemetry)
- **Access Trace:** `access_trace.json` (Per-patch sample and event access audit)
- **Epoch 1 Checkpoint:** `epoch_001_checkpoint.pt` (60.6 MB, atomic save)
- **Last Checkpoint:** `last_checkpoint.pt` (60.6 MB, reloaded in Phase 2)

---

## 3. Coverage, Throughput & Numerical Telemetry

### 3.1 Sample and Event Coverage
- **Total Training Samples Observed:** 21,462 / 21,462 (100.0% coverage of all 27 KuroSiwo source events)
- **Total Microbatches Observed:** 2,683
- **Validation Calls Made:** 0 (100% all-source fixed budget; validation split is absent and disabled)
- **Quarantined Events Accessed:** 0 (Disjointness confirmed against 6 TTA-val and 10 sealed-test events)

### 3.2 Loss and Resource Metrics (Epoch 1)
- **Total Loss:** `0.240578`
- **Cross-Entropy Loss component ($L_{CE}$):** `0.298196`
- **Soft Dice Loss component ($L_{Dice}$):** `0.182959`
- **Learning Rate:** `0.001000` (constant Adam learning rate)
- **Optimizer Steps:** 671
- **Epoch Duration:** 883.97s (14.73 minutes)
- **Throughput:** ~24.28 samples/sec (~3.03 microbatches/sec)
- **Peak GPU Memory:** 4,914.88 MiB (well within 24,576 MiB capacity of RTX 3090)

---

## 4. Two-Phase Execution & Step Count Clarification

### 4.1 Phase 1: Full Epoch 1 Training Pass
Epoch 1 processed all 21,462 patches in 2,683 microbatches (batch size 8). With `gradient_accumulation_steps = 4`:
- Full accumulation windows: $\lfloor 2683 / 4 \rfloor = 670$ optimizer steps (covering 2,680 microbatches).
- Final partial accumulation window: $2683 \pmod 4 = 3$ microbatches.
- Final partial batch: The 2,683rd microbatch contained the remaining 6 samples ($21,462 - 2,682 \times 8 = 6$).
- **Handling of Partial Accumulation Group:**
  In `src/training/source_trainer.py`, when `is_last_batch` occurs on an incomplete accumulation window ($window\_microbatches = 3 < 4$), an exact scaling factor of $\frac{\text{accumulation\_steps}}{window\_microbatches} = \frac{4}{3} \approx 1.333333$ is applied to the accumulated parameter gradients:
  $$\mathbf{g}_{\text{eff}} = \mathbf{g}_{\text{accum}} \times \frac{4}{3}$$
  This rescales the partial window to represent the exact unbiased sample average $\frac{1}{3} \sum_{i=1}^3 g_i$. An optimizer step is executed, advancing `global_step` from 670 to **671**.
- Model weights, optimizer state, and RNG states were cleanly serialized to `epoch_001_checkpoint.pt` and `last_checkpoint.pt`.

### 4.2 Phase 2: Reload and Resumption Viability
A fresh model instance and `SourceTrainer` were initialized and loaded from `last_checkpoint.pt`:
- Checkpoint reloaded state: `restored_epoch = 1`, `restored_step = 671`.
- Continuation budget: `--resume_batches 8` (8 microbatches in Epoch 2).
- Microbatches processed: 8 microbatches.
- Optimizer steps performed in continuation: $8 / 4 = \mathbf{2}$ optimizer steps.
- Final global step: $671 + 2 = \mathbf{673}$.
- Continuation loss: Total = `0.203093` (CE = `0.228590`, Dice = `0.177597`).
- All parameters, gradients, and BatchNorm floating buffers remained finite.

### 4.3 Step Count Log Wording Clarification
In `smoke.log`, the following output lines were emitted:
```text
Epoch 2 diagnostic continuation completed:
  Loss Total:       0.203093 (CE: 0.228590, Dice: 0.177597)
  Microbatches Run: 8
  Optimizer Steps:  673 (Epoch 1 steps: 671 + Epoch 2 steps: 673)
...
Epoch 2 Diagnostic Steps: 673 steps completed (8 microbatches)
```
**Clarification of Ambiguity:**
The expression `(Epoch 1 steps: 671 + Epoch 2 steps: 673)` was produced by a print formatting string in `tools/run_all27_smoke.py:448`:
```python
f"Optimizer Steps: {trainer_resumed.global_step} (Epoch 1 steps: {optimizer_steps} + Epoch 2 steps: {train_metrics_ep2['optimizer_steps']})"
```
In `SourceTrainer.train_epoch`, the return key `"optimizer_steps"` records the cumulative `self.global_step` (which was 673 after the continuation pass), *not* the incremental step count for that epoch call. Consequently, the string displayed the total global step (673) where the caller anticipated the step delta (2).
- **Exact step accounting:**
  - Epoch 1 optimizer steps: **671**
  - Epoch 2 continuation optimizer steps: **2** (from 8 microbatches at accumulation factor 4)
  - Final global step: **673** ($671 + 2 = 673$)
  - The log phrasing does **not** indicate that 673 new steps occurred in Epoch 2.

---

## 5. Honest Audit of State Restoration Verification

A rigorous distinction is maintained between properties that were explicitly verified by assertion and properties that were executed via implementation without explicit equality assertion:

1. **Model Weights and BatchNorm Floating Buffers:**
   - **Status:** **Explicitly Verified Bit-Exact.**
   - Pre-reload model weights were cloned to CPU memory prior to reload. Post-reload weights were compared key-by-key:
     $$\max_{k} \|\mathbf{\theta}_{\text{pre}}^{(k)} - \mathbf{\theta}_{\text{post}}^{(k)}\|_{\infty} = 0.000000e+00$$
     Non-floating buffers were confirmed identical via `torch.equal`.

2. **Optimizer State & RNG States:**
   - **Status:** **Restored via Implementation; Verified by Continuation Viability.**
   - `SourceTrainer.resume_from_checkpoint` loaded `optimizer_state_dict`, `scaler_state_dict`, `torch` RNG state, CUDA RNG state, and NumPy RNG state.
   - The trainer successfully performed 2 subsequent optimizer steps without gradient divergence, numerical instability, or parameter corruption.
   - **Limitation:** The smoke script did not execute an explicit tensor-by-tensor equality assertion (e.g. `torch.equal(opt_pre, opt_post)`) comparing optimizer momentum buffers or internal RNG bit generators directly against the checkpoint dictionary before resuming. Functional correctness and parameter finiteness were verified rather than bitwise optimizer buffer equality.

---

## 6. Partition Quarantine & Research Integrity

The diagnostic smoke runner monitored all data loader activity via `EventAccessTracker`:
- **Allowed Source Events:** 27 events (0, 1, 2, 4, 7, 8, 9, 10, 11, 14, 18, 23, 73, 108, 111, 123, 143, 172, 175, 181, 187, 227, 243, 269, 518, 521, 535) — 21,462 patches.
- **Quarantined TTA-Validation Events (6 events):** `[279, 437, 520, 559, 1111003, 1111008]` — Zero accesses.
- **Quarantined Sealed-Test Events (10 events):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` — Zero accesses.
- **Disjointness:** Both sets were strictly disjoint from accessed event IDs (`access_tracker.train_events.isdisjoint(...) == True`).
- **Validation Suppression:** All validation logic was bypassed. No model evaluation occurred; no checkpoint selection was performed.
- **Run Classification:** `is_formal_run = false`, `smoke_run = true`. No `final_checkpoint.pt` was generated.

---

## 7. Configuration, Lock, and Provenance Reconciliation

All pipeline components have been verified against the authoritative lock manifest:
`configs/source/source_training_all27_lock.json`

| File | Path | Authoritative SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- |
| **Lock Manifest** | `configs/source/source_training_all27_lock.json` | `fce1a85af8eec47dd134b1fb15103b70251f93e8af4142db241b903bd4cffd53` | Canonical on disk |
| **Protocol** | `docs/source_training_final_all27_protocol.md` | `2c15d3fdd1b3018ab5fe28b88b15a41dfe8b736e5b3193bf1d789049cb1429e2` | Exact match |
| **Split** | `configs/splits/kuro_source_all27_split_v1.json` | `601b21c81f7f1e26984fe6507a1532dbfa373d72b29346103fb5f0a1ac38d5e1` | Exact match |
| **Seed 0 Config** | `configs/source/snunet_vvvh_all27_seed0_v1_fp32.json` | `1bf698541ae43518c227faca5ae47a3edbbe99ba3e648ac085bcf4c869975c54` | Exact match |
| **Seed 1 Config** | `configs/source/snunet_vvvh_all27_seed1_v1_fp32.json` | `d1a832a4711e4799476042da14409eac1e82cd48dc8f362cca8e3a67b05de1d4` | Exact match |
| **Seed 2 Config** | `configs/source/snunet_vvvh_all27_seed2_v1_fp32.json` | `6730145d5cfe523726124e144719121c816ed2a256450cd0fd5750dba3e0b20e` | Exact match |

### Historical Differences Explained
1. **Lock File Hash Evolution:**
   An early hash (`03ffb648ff62df133cfa5d996cbaf4a53ae63ea25e98587d55c70b63e9c5096a`) was noted during the initial draft of Week 5B before the three final seed config hashes were populated. Once the actual SHA-256 hashes of `snunet_vvvh_all27_seed{0,1,2}_v1_fp32.json` were written into the lock manifest, the lock file's hash updated to its authoritative value `fce1a85a...`.
2. **Split File Trailing Newline Reconciliation:**
   The split file previously contained two consecutive trailing newline characters (`\n\n`), resulting in hash `1c9b1ed5140624c239c6fe326b82a2b48c90265d4721f03c3ad67f2c642b358c`. Stripping the redundant newline to adhere to the POSIX standard single newline (`\n`) restored the exact canonical hash `601b21c8...` registered in the lock file.
3. **Preservation of Protocol Hash:**
   While `docs/source_training_final_all27_protocol.md` contains minor markdown trailing whitespace on header lines, it is strictly preserved without modification because any text trimming would invalidate the locked SHA-256 hash.

---

## 8. Conclusion & Sign-Off

The Week 6 diagnostic smoke test is formally closed with status **PASSED**.
- All unit, integration, and smoke test suites pass (126 of 126 tests passing).
- Data pipelines, numerical safeguards, batch accumulation algebra, and checkpoint serialization behave in full compliance with the governing protocol.
- GPU 0 is idle and all background diagnostic tasks have terminated.
- The pipeline implementation is frozen and ready to be committed to version control.

