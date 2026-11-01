# Week 2 FP32 Source-Training Recovery Protocol & Smoke Closeout Report

**Date:** 2026-10-08  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22`  
**Governing Protocol:** `docs/source_training_protocol_v2.md`  
**Status:** Verification Complete — Production Code and Bounded Smoke Tests Verified; Full Canonical Runs Held for Approval.

---

## 1. Executive Summary & Rationale

Following the numerical diagnostic closeout of Week 2 seed-0 source training:
1. **Prior Failure Context:** Both Model A (`source_snunet_vvvh_seed0_v2`) and Model B (`source_snunet_vvvh_slope_seed0_v2`) canonical FP16/AMP runs developed numerical corruption in later epochs, terminating via early stopping at Epoch 40 with `NaN` losses, invalid metrics, or predictive collapse.
2. **Empirical Diagnostic Evidence:**
   - In matched causal diagnostics, FP16 failed fast at Epoch 23, Batch 353 with nonfinite scaled gradients in the backward path.
   - An isolated FP32 diagnostic retry ran cleanly through Epoch 25, Batch 2398 (11,990 consecutive batches) with **zero** nonfinite values across all inputs, forward activations, logits, losses, gradients, and BatchNorm buffers.
3. **Approved Protocol Decision:**
   - Full FP32 source training is approved for canonical recovery of Model A `[VV, VH]` and Model B `[VV, VH, slope]`.
   - Both models rerun from Epoch 1.
   - Mixed precision (autocast and GradScaler) is strictly disabled in governing configs and runtime trainer.
   - BF16 is not approved for canonical runs.
   - All core scientific settings (22/5 split, Policy S2 slope normalization, Adam optimizer, loss formulation, batch size 32, max 150 epochs, patience 20) remain 100% frozen.
   - Fail-fast safeguards, predictive collapse detection, and atomic checkpointing are integrated.

---

## 2. Governing Specifications, Acyclic Provenance Lock & Config Hashes

### 2.1 Resolution of Circular Dependency via One-Way Immutable Lock
During pre-commit auditing of Protocol v2, a circular dependency was identified between the protocol document and the model configuration files:
- **Discovered Inconsistency:** The FP32 configurations recorded `governing_protocol_sha256: 4cd47ade...`, while the actual Protocol v2 document on disk hashed to `75cefca8...`.
- **Root Cause:** When `docs/source_training_protocol_v2.md` was drafted, its initial hash was `4cd47ade...`, which was placed into the configs. Subsequently, Protocol v2 was edited to record the final config hashes (`c7cd3105...` and `317182bd...`). That edit altered Protocol v2's own content and mutated its hash to `75cefca8...`. Updating the configs to record `75cefca8...` would change the configs' own hashes, creating an endless hash mutation cycle.
- **One-Way Acyclic Lock Architecture:** The circular dependency was resolved by establishing an acyclic, one-way dependency chain:
  1. **Protocol v2 is finalized first:** Config file paths and architecture are defined, but final configuration hashes are omitted from the protocol document.
  2. **Configs reference the finalized Protocol v2 hash:** Both Model A and Model B configurations record the immutable Protocol v2 hash (`b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144`) and point to the lock manifest path.
  3. **Independent Lock Manifest Created:** An independent lock manifest `configs/source/source_training_v2_lock.json` binds the SHA-256 digests of the finalized Protocol v2 document, both finalized configuration files, the frozen split, and the slope statistics.
  4. **Strict Isolation:** Neither Protocol v2 nor the configurations contain the lock manifest's hash, and the lock manifest does not contain its own hash.

### 2.2 Authoritative Artifact Hashes
The governing specifications and immutable lock manifest are finalized with the following SHA-256 checksums:

| Artifact Role | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Governing Protocol v2** | `docs/source_training_protocol_v2.md` | `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144` |
| **Model A FP32 Config** | `configs/source/snunet_vvvh_v2_fp32.json` | `528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3` |
| **Model B FP32 Config** | `configs/source/snunet_vvvh_slope_v2_fp32.json` | `e99e980a9afc0f72b526ccff5484e4aadaabece12c5a542bd20f94be74ef6b56` |
| **Frozen Split v1** | `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` |
| **Slope Statistics v1** | `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` |
| **Immutable Lock Manifest** | `configs/source/source_training_v2_lock.json` | `7139dbdd2031fd5654b4414299bf80e20295ebdba13bb507b64a59f10a50fbd3` |

### 2.3 Validation-Only Entry Point Verification
The launcher provides a dedicated `--validate_only` mode that executes provenance and configuration verification against the immutable lock manifest without allocating GPU memory, creating output directories, or loading dataset samples:

```bash
# Model A Validation-Only Verification
python tools/train_source_model.py \
  --config configs/source/snunet_vvvh_v2_fp32.json \
  --run_id source_snunet_vvvh_seed0_v3_fp32 \
  --seed 0 \
  --validate_only
# Result: PASSED (100% VALIDATED AGAINST IMMUTABLE PROVENANCE LOCK)

# Model B Validation-Only Verification
python tools/train_source_model.py \
  --config configs/source/snunet_vvvh_slope_v2_fp32.json \
  --run_id source_snunet_vvvh_slope_seed0_v3_fp32 \
  --seed 0 \
  --validate_only
# Result: PASSED (100% VALIDATED AGAINST IMMUTABLE PROVENANCE LOCK)
```

### 2.4 Integrity Confirmations
- **Zero Output Modifications:** All numerical smoke artifacts under `outputs/smoke_model_a_fp32/` and `outputs/smoke_model_b_fp32/` remain 100% untouched and preserved.
- **Zero Dataset Access:** During this provenance lock refactoring and validation-only verification, zero dataset samples or protected events were loaded or accessed.

### Core Frozen Parameters Preserved
- **Architecture:** SNUNet (32 base channels), 2-branch siamese. Model A: 2 in-channels per branch; Model B: 3 in-channels per branch.
- **Data Split:** 22 inner-train events (19,178 patches), 5 inner-val events (2,284 patches).
- **Optimization:** Adam, learning rate $1.0 \times 10^{-3}$, $\beta=(0.9, 0.999)$, weight decay $0.0$, no scheduler.
- **Batch Policy:** Effective batch size 32 (physical microbatch 8, accumulation steps 4).
- **Precision:** Full FP32 (`mixed_precision: false`, `precision: "fp32"`, `autocast: false`, `grad_scaler: false`).
- **Loss:** ControlledSourceLoss ($0.5 \times \text{CE} + 0.5 \times \text{SoftDice}$, ignore index 3).
- **Stopping Budget:** Maximum 150 epochs, early stopping patience 20 on inner-validation event-macro Flood F1.

---

## 3. Implementation of Fail-Fast Safeguards

Safeguards have been implemented in `src/training/source_trainer.py` and `tools/train_source_model.py`:

### 3.1 Immediate `FAILED_NUMERICAL` Guards
- **Inputs:** Asserts `torch.isfinite(xA).all()` and `torch.isfinite(xB).all()`.
- **Logits:** Asserts `torch.isfinite(logits).all()` after forward pass.
- **Loss:** Asserts `torch.isfinite(loss)` before backward pass.
- **Gradients:** Audits all `p.grad` across parameters before `optimizer.step()`.
- **Parameters:** Audits all parameters `p` after `optimizer.step()`.
- **BatchNorm Buffers:** Audits all floating-point buffers (`running_mean`, `running_var`) across modules at the conclusion of every epoch.
- **Validation:** Asserts validation inputs, logits, and losses remain finite.
- **Failure Manifest:** On violation, immediately writes `failure_manifest.json` recording status `FAILED_NUMERICAL`, offending tensor name, stage, epoch, batch index, global step, sample IDs, event IDs, and system provenance.

### 3.2 Predictive Collapse Distinction (`FAILED_COLLAPSE`)
- **Detector Class:** `PredictiveCollapseDetector` implemented with `warmup_epochs: 5` and `patience: 3`.
- **Validation Target:** Evaluated strictly on known-positive inner-validation set (18,946,970 genuine flood pixels under `label == 2`).
- **Warmup Rule:** During epochs 1 through 5, collapse is not triggered.
- **Patience Rule:** Requires 3 consecutive validation epochs of zero predicted flood pixels across the validation set.
- **Reset Rule:** Resets counter to 0 immediately upon observing nonzero predicted flood pixels.
- **Logging:** Logged in `validation_history.csv` under the `collapse_counter` column and displayed in stdout banners.
- **Exit Handling:** On trigger, exits immediately as `FAILED_COLLAPSE` without saving corrupted models or marking `COMPLETED`.

### 3.3 Atomic Checkpoint Integrity
- **Pre-Save Finiteness Audit:** Verifies all model parameters and floating-point buffers are strictly finite before touching disk.
- **Atomic Replace:** Writes checkpoint state to `.tmp.<pid>_<timestamp>` before invoking `os.replace` to guarantee atomic file replacement.
- **Resumption Validation:** `resume_from_checkpoint` checks loaded weights for finiteness and rejects checkpoints marked with failure statuses.

### 3.4 Valid Completion Statuses
- **`COMPLETED_EARLY_STOP`:** Triggered when early stopping patience (20 epochs) expires while all metrics and tensors remain finite.
- **`COMPLETED_MAX_EPOCH`:** Triggered when maximum 150 epochs complete cleanly.

---

## 4. Gate 1: Automated Unit & Safeguard Verification

### 4.1 Development Iteration & Intermediate Unit-Test Failures
During initial implementation of the FP32 fail-fast safeguards, intermediate unit-test runs encountered two distinct failures before achieving a 100% pass rate:

1. **`test_approved_optimizer_protocol_and_runtime_agreement` (in `tests/test_source_configs.py`):**
   - **Failure:** `NameError: name 'betas' is not defined` at `src/training/source_trainer.py:270`.
   - **Context:** Occurred during initial implementation of the FP32 precision configuration logic in `SourceTrainer.from_config`, where the `betas` tuple extraction variable had been accidentally omitted during refactoring.
   - **Code Correction:** Restored `betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))` in `src/training/source_trainer.py`.

2. **`test_resume_rejects_corrupted_checkpoint` (in `tests/test_fp32_safeguards.py`):**
   - **Failure:** `_pickle.UnpicklingError: Weights only load failed. Unsupported global: GLOBAL numpy._core.multiarray._reconstruct was not an allowed global by default.`
   - **Context:** In PyTorch 2.6, `torch.load` defaults to `weights_only=True`, which rejects numpy RNG state tuples saved inside checkpoint dictionaries when loaded without explicit exemption.
   - **Code Correction:** Passed `weights_only=False` to the CPU checkpoint load call in `tests/test_fp32_safeguards.py:350` (`corrupt_state = torch.load(clean_ckpt, map_location="cpu", weights_only=False)`).

### 4.2 Final Automated Unit Verification
Following these corrections, the full test suite was executed using the project virtual environment:
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest discover -s tests
```

**Results:**
- Total tests executed: **97 tests**
- Total passing: **97 tests (100% PASS)**
- Duration: 12.431s

### Dedicated Safeguard Test Suite (`tests/test_fp32_safeguards.py`)
| Test Case | Objective | Result |
| :--- | :--- | :--- |
| `test_fp32_config_disables_amp_and_scaler` | Confirms FP32 configs disable AMP and GradScaler | **PASS** |
| `test_clean_fp32_training_step` | Confirms clean execution and weight update | **PASS** |
| `test_forward_logits_nonfinite_fails_fast` | Confirms nonfinite logits raise `NumericalInstabilityError` | **PASS** |
| `test_nonfinite_loss_fails_fast` | Confirms nonfinite loss raises `NumericalInstabilityError` | **PASS** |
| `test_nonfinite_gradient_fails_fast` | Confirms nonfinite gradients raise `NumericalInstabilityError` | **PASS** |
| `test_nonfinite_bn_buffer_fails_fast` | Confirms nonfinite BN buffers fail fast at epoch end | **PASS** |
| `test_predictive_collapse_detector_warmup_and_patience` | Confirms warmup=5, patience=3, and reset on positive flood | **PASS** |
| `test_atomic_checkpoint_saving_and_finiteness_check` | Confirms atomic `.tmp` replace and rejection of NaN weights | **PASS** |
| `test_resume_rejects_corrupted_checkpoint` | Confirms resumption rejects corrupted/failed checkpoints | **PASS** |
| `test_failure_manifest_generation` | Confirms `failure_manifest.json` schema and metadata capture | **PASS** |
| `test_protocol_v2_hashes_and_validity` | Confirms Protocol v2 config hashes match frozen specification | **PASS** |
| `test_provenance_lock_negative_rejections` | Confirms production validation rejects altered protocol, config, split, slope, or precision | **PASS** |

---

## 5. Gate 2: Bounded FP32 Smoke Test Evidence

### 5.1 Development Smoke Attempt, Launcher Defect, and Directory Reset
Before the final reported smoke runs, an initial development smoke run of Model A was launched to test end-to-end launcher execution:
- **Initial Smoke Command:**
  ```bash
  /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py \
    --config configs/source/snunet_vvvh_v2_fp32.json \
    --output_dir outputs/smoke_model_a_fp32 \
    --run_id smoke_model_a_fp32 \
    --device cuda:1 \
    --seed 0 \
    --max_epochs 1 \
    --max_train_batches 20 \
    --max_val_batches 10 \
    --num_workers 2
  ```
- **Observed Launcher Failure:**
  The forward/backward passes, metric computations, and validation loop executed properly (Train Loss: `2.8243`, Val Macro F1: `0.5471`), but at the conclusion of Epoch 1, the program crashed during post-loop completion handling at line 715:
  ```text
  [FATAL ERROR] Training run failed with exception: 'RunManifest' object has no attribute 'reproducibility'
  Traceback (most recent call last):
    File "/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/train_source_model.py", line 715, in train_source_model
      manifest.reproducibility["completion_status"] = completed_status
  AttributeError: 'RunManifest' object has no attribute 'reproducibility'
  ```
- **Code Correction Applied:**
  In `tools/train_source_model.py`, replaced the direct dictionary access `manifest.reproducibility["completion_status"] = completed_status` with the formal method call `manifest.mark_completed(completed_status)`, and cleaned up obsolete `manifest.reproducibility` indexing in signal handling and exception blocks.
- **Directory Reset & Deletion Command:**
  Because the crash occurred midway through post-loop manifest generation, `outputs/smoke_model_a_fp32` was left with partially written and inconsistent metadata. To ensure that both Model A and Model B bounded smoke tests ran cleanly from scratch through the corrected launcher with identical, pristine directory structures and uncompromised provenance, both scratch directories were wiped using the exact command:
  ```bash
  rm -rf outputs/smoke_model_a_fp32 outputs/smoke_model_b_fp32
  ```
- **Honest Disclosure & Protocol Deviation Notice:**
  - **Development vs. Canonical Experiments:** These preliminary runs were development and debugging smoke tests designed to verify safeguard instrumentation and launcher plumbing, not formal canonical training experiments.
  - **Protocol Deviation:** Deleting these development scratch directories rather than preserving them as historical artifacts is a documented protocol deviation. Not every preliminary development attempt was preserved on disk.
  - **No Attempt Re-creation:** In accordance with audit standards, the deleted preliminary development attempts are not retroactively recreated or rerun.
  - **Integrity of Final Results:** The final smoke artifacts reported in Sections 5.2, 5.3, and 6 below were generated completely from scratch following that reset and were **not** manually edited.
  - **Data Boundary Compliance:** During both the initial failed smoke run and the final clean smoke runs, **zero protected events were accessed** (0 quarantined TTA-validation events and 0 sealed-test events).

### 5.2 Model A Clean FP32 Smoke Run (`outputs/smoke_model_a_fp32/`)
- **Device:** `cuda:1` (RTX 3090)
- **Phase 1 (Epoch 1):** 20 training microbatches (160 samples), 10 validation microbatches (80 samples).
  - Train Loss: `2.8225` (CE: `5.1068`, Dice: `0.5382`), Peak VRAM: `4914.9 MiB`.
  - Inner-Val Macro F1: `0.5413` (Macro Recall: `0.7548`, Worst F1: `0.1402` on Event 518).
  - Status: `[★ NEW BEST] [Collapse: 0/3]`.
- **Phase 2 (Resumption to Epoch 2):** Resumed from `last_checkpoint.pt` at epoch 1, step 5. Executed 10 training batches, 5 validation batches.
  - Train Loss: `0.8025` (CE: `1.3345`, Dice: `0.2705`), Peak VRAM: `4960.1 MiB`.
  - Inner-Val Macro F1: `0.5476` (Macro Recall: `0.6754`, Worst F1: `0.0509` on Event 518).
  - Status: `[★ NEW BEST] [Collapse: 0/3]`.
- **Numerical Audit:** Audited all `12,042,115` floating-point elements in `best_checkpoint.pt` and `last_checkpoint.pt`: **0 nonfinite elements**.
- **Data Boundary Verification (`access_trace.json`):**
  - Quarantined TTA-val events accessed: `0` (`no_tta_val_accessed: true`)
  - Sealed-test events accessed: `0` (`no_sealed_test_accessed: true`)
  - Train/Val event intersection: `empty` (`train_val_events_disjoint: true`)

### 5.3 Model B Clean FP32 Smoke Run (`outputs/smoke_model_b_fp32/`)
- **Device:** `cuda:2` (RTX 3090)
- **Slope Statistics Verification:** Verified hash `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`.
- **Phase 1 (Epoch 1):** 20 training microbatches (160 samples), 10 validation microbatches (80 samples).
  - Train Loss: `1.8895` (CE: `3.2717`, Dice: `0.5072`), Peak VRAM: `4923.3 MiB`.
  - Inner-Val Macro F1: `0.5412` (Macro Recall: `0.7580`, Worst F1: `0.1558` on Event 518).
  - Status: `[★ NEW BEST] [Collapse: 0/3]`.
- **Phase 2 (Resumption to Epoch 2):** Resumed from `last_checkpoint.pt` at epoch 1, step 5. Executed 10 training batches, 5 validation batches.
  - Train Loss: `0.7361` (CE: `1.1403`, Dice: `0.3318`), Peak VRAM: `4964.5 MiB`.
  - Inner-Val Macro F1: `0.5397` (Macro Recall: `0.5227`, Worst F1: `0.0237` on Event 518).
  - Status: `[Patience: 1/20] [Collapse: 0/3] (Best Epoch: 1)`.
- **Numerical Audit:** Audited all `12,042,403` floating-point elements in `best_checkpoint.pt` and `last_checkpoint.pt`: **0 nonfinite elements**.
- **Data Boundary Verification (`access_trace.json`):**
  - Quarantined TTA-val events accessed: `0` (`no_tta_val_accessed: true`)
  - Sealed-test events accessed: `0` (`no_sealed_test_accessed: true`)
  - Train/Val event intersection: `empty` (`train_val_events_disjoint: true`)

---

## 6. Smoke Artifact Hashes

### Model A Smoke Artifacts (`outputs/smoke_model_a_fp32/`)
| File | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `access_trace.json` | 848 | `dc985036ad0a393c6c92370bcc8561fdb30b1f56d408fcbd7781d1c52ede4ffe` |
| `best_checkpoint.pt` | 144,674,644 | `de3e3cf8eaf5059e541870ad2ef418849e75bd37c111021c443f9ca42607999d` |
| `last_checkpoint.pt` | 144,674,644 | `4a63eb0370f69d7ffe81c749ba8b1e37804815ba7a29997eb0f8ba96dee442e8` |
| `checkpoint_hashes.json` | 328 | `d0b41b018a8fc335be5342c7014b31ff1262afffc1581774d80a83036f510335` |
| `run_manifest.json` | 3,332 | `0afd34cf6d8880935612c018ce25427c2385a78cd138d34e456c403170ee5391` |
| `train_history.csv` | 259 | `3a93ae04a378167f1af2feea0f442a0efc8ae9449dc87d09fa1737d2839ae6e6` |
| `validation_history.csv` | 545 | `011fe5a4bbf164290d82831fca7c45aeb1229396b40c9996430c1de41469ccdf` |
| `event_validation_metrics.csv` | 1,010 | `951d698db498e296f74ff88c526e332a6d1025dfa320f5d9ad4b818e9e2a63ee` |
| `train.log` | 4,226 | `f41c23707fe3ed5fb0122b710e2d057cf6a9ba81a26b45967645477cab8b4e1d` |

### Model B Smoke Artifacts (`outputs/smoke_model_b_fp32/`)
| File | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `access_trace.json` | 902 | `19658a7a568d115f40987b8333a29b3795bfbe7b189f5cd47e616b0b11babfba` |
| `best_checkpoint.pt` | 144,676,500 | `4b7c031f6e5f7fca50c4295d5d3c1a1a9c0b567619b2ffc6b1dcdb76b2f0e0f8` |
| `last_checkpoint.pt` | 144,678,484 | `77a386eae5d59bd1cd4576465e5295e1b90e7f1cb2b3587d323f9557c610baf8` |
| `checkpoint_hashes.json` | 328 | `f0cfe0ee5c0ba7e794d3f3e358ea2dc2f62a835195c728ea41d057a8c2dfc3fe` |
| `run_manifest.json` | 3,410 | `5813ee110e0be331e9f1a36fb9f408d355502fc283ecb5a830418e593277cf98` |
| `train_history.csv` | 259 | `11c6a0cee4357697ed3301b837ba86b67ec68dbe18dd8e06aa20bd7c2021a7a1` |
| `validation_history.csv` | 545 | `d1fa561787a96e8abcb97c4c06b3f12c1c4b0b14b74081654c48d3ed0766c2f9` |
| `event_validation_metrics.csv` | 1,008 | `298edb2eb1fac1445ed9236abd462c960a62e4df89689e548bfe6ca416681f91` |
| `train.log` | 4,731 | `d1c9bd9b1a0f75b9317c45de93480a304be7b20f0be81776eeeb2d9b45f0c716` |

---

## 7. Gate 3 Readiness & Proposed Canonical Launch Commands

With Gate 1 (unit testing) and Gate 2 (bounded smoke testing) fully passed, the production environment is verified and ready for formal execution. 

### Shared-Server-Safe Device Isolation Pattern
On multi-GPU shared systems (`gpu22`), directly targeting physical devices (e.g., `--device cuda:1` or `--device cuda:2`) poses high operational risk of GPU collisions with other users' concurrent processes. Furthermore, GPUs must **never be preassigned in advance**; physical GPU availability, existing GPU processes, and free VRAM must be verified via `nvidia-smi` immediately prior to launch.

All future canonical recovery launches must use the shared-server-safe device isolation pattern:
```bash
CUDA_VISIBLE_DEVICES=<verified-idle-physical-GPU> ... --device cuda:0
```
**Mechanism:** Setting `CUDA_VISIBLE_DEVICES=<gpu_id>` isolates the process environment so PyTorch detects only that single physical GPU and maps it to internal logical index `cuda:0`. Passing `--device cuda:0` ensures PyTorch allocates memory exclusively on the assigned physical GPU and prevents accidental memory claims or crosstalk on other devices.

### Proposed Future Canonical Launch Commands
When authorized, canonical recovery runs will be launched in dedicated tmux sessions using physical GPUs verified idle immediately prior to launch:

```bash
# Model A Canonical Recovery (Seed 0, FP32, using verified-idle physical GPU, e.g., GPU 1)
CUDA_VISIBLE_DEVICES=1 tmux new-session -d -s cg_source_model_a_v3_fp32 \
  "/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py \
    --config configs/source/snunet_vvvh_v2_fp32.json \
    --output_dir outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32 \
    --run_id source_snunet_vvvh_seed0_v3_fp32 \
    --device cuda:0 \
    --seed 0 \
    --num_workers 2"

# Model B Canonical Recovery (Seed 0, FP32, using verified-idle physical GPU, e.g., GPU 2)
CUDA_VISIBLE_DEVICES=2 tmux new-session -d -s cg_source_model_b_v3_fp32 \
  "/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py \
    --config configs/source/snunet_vvvh_slope_v2_fp32.json \
    --output_dir outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32 \
    --run_id source_snunet_vvvh_slope_seed0_v3_fp32 \
    --device cuda:0 \
    --seed 0 \
    --num_workers 2"
```

> [!IMPORTANT]
> **No full canonical training has been launched.** In accordance with the prompt constraints, full canonical runs remain unstarted pending commit checkpointing and user launch authorization. The GPU indices in the examples above (`CUDA_VISIBLE_DEVICES=1` and `CUDA_VISIBLE_DEVICES=2`) are illustrative; physical GPU assignments will be chosen strictly upon immediate pre-launch audit of `nvidia-smi`.

