# Week 2 Canonical v3 Source-Training Launch Failure & Fix Report

## 1. Executive Summary & Incident Classification

During the execution of the canonical Week 2 FP32 seed-0 source training recovery runs (launched from committed HEAD `391235b`), both Model A and Model B crashed at the conclusion of Epoch 1 with an identical fatal exception:

```text
[FATAL ERROR] Training run failed with exception: name 'split_hash' is not defined
Traceback (most recent call last):
  File "tools/train_source_model.py", line 745, in train_source_model
    "split_hash": split_hash,
NameError: name 'split_hash' is not defined
```

### Critical Incident Finding: Software/Provenance Recording Defect, NOT Numerical Instability
1. **Full Training Epoch Succeeded Without Numerical Defect:**
   Prior to the crash, both models executed all **2,398 training batches** (19,178 samples) of Epoch 1 cleanly and stably under FP32. Every 50-step progress record was finite:
   - **Model A Final Train Step (Step 600, Batch 2398/2398):** Total Loss `0.3073` (CE: `0.4249`, Soft Dice: `0.1897`, LR: `0.001000`).
   - **Model B Final Train Step (Step 600, Batch 2398/2398):** Total Loss `0.1751` (CE: `0.1851`, Soft Dice: `0.1650`, LR: `0.001000`).
   - Peak VRAM remained bounded at ~6,044 MiB on RTX 3090.
2. **Inner Validation Completed:**
   The validation loop across all 2,284 inner-validation samples (5 events) executed fully and without numerical exception:
   - Predictive collapse detectors evaluated successfully without triggering.
   - Tie-breaking checkpoint selection logic evaluated successfully.
3. **Point of Failure:**
   The failure occurred solely in post-validation checkpoint metadata assembly at line 745 of `tools/train_source_model.py` when assembling `extra_meta` for `trainer.save_checkpoint`. Because Python raised `NameError`, checkpoint writing, CSV history row logging, and trace updates were aborted by the unhandled exception.
4. **Resolution:**
   The variable scope defect was resolved by explicitly extracting, asserting, and propagating verified provenance variables (`split_hash`, `slope_stats_hash`, `protocol_hash`, `lock_hash`) from the immutable lock validation result directly into checkpoint metadata, `RunManifest`, and exception logging.

---

## 2. Preservation of Failed Attempt Artifacts

Both failed attempt output directories have been preserved in-place without modification, deletion, truncation, or resumption:
- `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32/`
- `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32/`

### Preserved Artifact Hashes & Metrics

#### Model A (`outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32/`)
| File | Size (Bytes) | SHA-256 Checksum | Description / State |
| :--- | :--- | :--- | :--- |
| `train.log` | 3,140 | `4419278313918c085a84004c3f53966252edee0eae80c0a596aaf2be1596387d` | Complete stdout/stderr log up to Batch 2398/2398 and `NameError` traceback |
| `run_manifest.json` | 2,922 | `1e43623a0d9e59d4d58ebd162c4e4ac2ab5fc0a379588f719ca307c11d2b837a` | Manifest in `INITIALIZED` state (save bypassed during unhandled exception) |
| `train_history.csv` | 138 | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | Header only (epoch row write aborted by exception) |
| `validation_history.csv` | 256 | `3114189633591773e089d3849dc57ef5d7f818eb850b48a2c1dc6d8a017bda30` | Header only (epoch row write aborted by exception) |
| `event_validation_metrics.csv` | 112 | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | Header only (epoch row write aborted by exception) |

- **Final Completed Training Batch:** Batch 2,398 / 2,398 (Step 600) of Epoch 1.
- **Validation Execution Status:** Executed fully in memory; aborted during checkpoint serialization.
- **Checkpoints Written:** None (`best_checkpoint.pt` and `last_checkpoint.pt` were not reached).
- **Failure Manifest:** None (unhandled exception block crashed attempting `manifest.extra_metadata["error"] = str(e)`).
- **Accessed Training Events:** All 22 approved source-inner events: `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`.
- **Accessed Validation Events:** All 5 approved source-inner events: `[267, 275, 497, 518, 1111005]`.
- **Quarantined TTA Validation Events Accessed:** **0** (strictly disjoint).
- **Sealed Test Events Accessed:** **0** (strictly disjoint).

#### Model B (`outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32/`)
| File | Size (Bytes) | SHA-256 Checksum | Description / State |
| :--- | :--- | :--- | :--- |
| `train.log` | 3,387 | `cf26f176f6519026bff83acfd35ebb70fab92041f71360b0113197a326489a04` | Complete stdout/stderr log up to Batch 2398/2398 and `NameError` traceback |
| `run_manifest.json` | 2,994 | `4eb720cbc8731cea3d44746a9101427748288484145a744775ea790ed6b6d294` | Manifest in `INITIALIZED` state (save bypassed during unhandled exception) |
| `train_history.csv` | 138 | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | Header only (epoch row write aborted by exception) |
| `validation_history.csv` | 256 | `3114189633591773e089d3849dc57ef5d7f818eb850b48a2c1dc6d8a017bda30` | Header only (epoch row write aborted by exception) |
| `event_validation_metrics.csv` | 112 | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | Header only (epoch row write aborted by exception) |

- **Final Completed Training Batch:** Batch 2,398 / 2,398 (Step 600) of Epoch 1.
- **Validation Execution Status:** Executed fully in memory; aborted during checkpoint serialization.
- **Checkpoints Written:** None (`best_checkpoint.pt` and `last_checkpoint.pt` were not reached).
- **Failure Manifest:** None.
- **Accessed Training Events:** All 22 approved source-inner events: `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`.
- **Accessed Validation Events:** All 5 approved source-inner events: `[267, 275, 497, 518, 1111005]`.
- **Quarantined TTA Validation Events Accessed:** **0** (strictly disjoint).
- **Sealed Test Events Accessed:** **0** (strictly disjoint).

---

## 3. Root Cause Analysis

### Where the Provenance Variables are Produced
In `tools/train_source_model.py`, `validate_provenance_and_config` validates all configuration files, protocols, splits, and slope statistics against `configs/source/source_training_v2_lock.json`. It returns a dictionary:
```python
return {
    "status": "VALIDATED",
    "model_key": model_key,
    "config_hash": actual_config_hash,
    "protocol_hash": actual_proto_hash,
    "split_hash": actual_split_hash,
    "slope_stats_hash": actual_slope_hash,
    "precision": "fp32",
    "lock_path": str(lock_path),
    "config": config,
    "lock": lock,
}
```

### Why `split_hash` Was Unavailable at Epoch Finalization
In the original implementation (commit `5ca642a`), `split_hash` and `slope_stats_hash` were local variables computed directly inside `train_source_model`:
```python
split_hash = compute_file_sha256(split_path)
```
When circular dependency resolution was implemented, hash verification was extracted into `validate_provenance_and_config`. In `train_source_model`:
```python
provenance = validate_provenance_and_config(...)
config = provenance["config"]
config_hash = provenance["config_hash"]
model_type = provenance["model_key"]
```
However, `split_hash = provenance["split_hash"]` was inadvertently omitted from local variable unpacking. When execution reached line 745 (`"split_hash": split_hash`), Python encountered an unbound variable and raised `NameError`.

### Why Earlier Tests Missed the Defect
1. **Unit Tests Tested Components in Isolation:**
   `tests/test_fp32_safeguards.py` verified `validate_provenance_and_config(...)` and tested `SourceTrainer` directly with mock data (`trainer.train_steps(...)`, `trainer.save_checkpoint(...)`). No unit test ran the outer `train_source_model` orchestrator through an epoch finalization.
2. **Circular Dependency Task Prohibited Smoke Reruns:**
   During the previous task ("Resolve Protocol-v2/config hash circular dependency only. Do not launch canonical training or rerun the full smoke tests"), the instruction explicitly forbade rerunning the full smoke tests. Therefore, only `--validate_only` (which terminates before training) and unit tests were run.
3. **Secondary Failure in Exception Handler:**
   When `NameError` was caught at line 956, the handler attempted `manifest.extra_metadata["error"] = str(e)`. Because `RunManifest` lacked an `extra_metadata` dictionary attribute, this raised `AttributeError: 'RunManifest' object has no attribute 'extra_metadata'`, causing the exception handler's `except Exception: pass` to swallow the manifest save and leave `run_manifest.json` in the `INITIALIZED` state without writing `failure_manifest.json`.

---

## 4. Code Fixes Implemented

### 1. `tools/train_source_model.py`
1. **Explicit Provenance Unpacking and Strict Early Assertion:**
   Immediately following `provenance = validate_provenance_and_config(...)`, all verified hashes are unpacked and validated:
   ```python
   config = provenance.get("config")
   config_hash = provenance.get("config_hash")
   model_type = provenance.get("model_key")
   protocol_hash = provenance.get("protocol_hash")
   split_hash = provenance.get("split_hash")
   slope_stats_hash = provenance.get("slope_stats_hash")
   lock_path = provenance.get("lock_path")

   if not config or not isinstance(config, dict):
       raise RuntimeError("Verified config dictionary missing from provenance validation result")
   if not config_hash or not isinstance(config_hash, str) or len(config_hash) != 64:
       raise RuntimeError(f"Verified config_hash missing or invalid: {config_hash}")
   if not protocol_hash or not isinstance(protocol_hash, str) or len(protocol_hash) != 64:
       raise RuntimeError(f"Verified protocol_hash missing or invalid: {protocol_hash}")
   if not split_hash or not isinstance(split_hash, str) or len(split_hash) != 64:
       raise RuntimeError(f"Verified split_hash missing or invalid: {split_hash}")
   if not lock_path or not Path(lock_path).exists():
       raise RuntimeError(f"Verified lock_path missing or invalid: {lock_path}")
   lock_hash = compute_file_sha256(lock_path)
   if not lock_hash or len(lock_hash) != 64:
       raise RuntimeError(f"Failed to compute SHA-256 for lock manifest: {lock_path}")

   in_channels = int(config["architecture"]["in_channels_per_branch"])
   include_slope = (in_channels == 3)
   if include_slope:
       if not slope_stats_hash or not isinstance(slope_stats_hash, str) or len(slope_stats_hash) != 64:
           raise RuntimeError(f"Verified slope_stats_hash missing or invalid for Model B: {slope_stats_hash}")
   ```
2. **Propagate Hashes to `RunManifest`:**
   Passed `protocol_hash`, `lock_hash`, `split_hash`, and `slope_stats_hash` to `RunManifest`.
3. **Propagate Hashes to Checkpoint `extra_metadata`:**
   Included all verified hashes (`protocol_hash`, `config_hash`, `lock_hash`, `split_hash`, and `slope_stats_hash` for Model B) in `extra_meta`.
4. **Update Manifest Checkpoints on Save:**
   At epoch end and run completion, called `manifest.update_checkpoint(...)` when checkpoints exist to ensure valid SHA-256 sums are recorded.
5. **Robust Exception and Failure Manifest Generation:**
   Fixed `except Exception as e` to safely mark `manifest.mark_completed("FAILED")`, guard `extra_metadata`, save `run_manifest.json`, and write `failure_manifest.json`.

### 2. `src/evaluation/run_manifest.py`
1. Added `protocol_hash`, `lock_hash`, `split_hash`, `slope_stats_hash`, and `extra_metadata` optional arguments to `RunManifest.__init__`.
2. Added `update_checkpoint(self, checkpoint_path)` method.
3. Included `protocol_sha256`, `lock_sha256`, `split_sha256`, `slope_stats_sha256`, and `extra_metadata` in `reproducibility` serialization.

---

## 5. Regression Verification & Unit Test Suite

Four new dedicated regression unit tests were added to `tests/test_fp32_safeguards.py`:

| Test Name | Test Objective | Result |
| :--- | :--- | :--- |
| `test_one_completed_epoch_finalization_model_a` | Verifies Model A completes 1 epoch through production finalization without `NameError`, verifies checkpoint `extra_metadata["split_hash"]`, and validates manifest output | **PASS** |
| `test_one_completed_epoch_finalization_model_b` | Verifies Model B completes 1 epoch through production finalization without `NameError`, verifies checkpoint `extra_metadata["slope_stats_hash"]`, and validates manifest output | **PASS** |
| `test_missing_or_corrupted_provenance_fails_before_training` | Verifies undefined or missing provenance values fail fast with `RuntimeError` before training starts | **PASS** |
| `test_run_failure_recorded_accurately_on_unhandled_exception` | Verifies unhandled exception records `FAILED` in `run_manifest.json` and writes `failure_manifest.json` | **PASS** |

### Full Test Suite Results
- `tests/test_fp32_safeguards.py`: **16/16 tests passing (100% PASS in 32.368s)**.
- Complete discovery test suite (`python -m unittest discover -s tests`): **101/101 tests passing (100% PASS in 42.601s)**.
- Pre-execution launcher validation (`--validate_only`): **100% PASSED** for both Model A and Model B.

---

## 6. Bounded End-to-End Regression Micro-Smokes

Sequential single-epoch micro-smokes were executed on physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, `--device cuda:0`) using real source-inner data (`max_train_batches=1`, `max_val_batches=1`, seed 0, FP32):

### Micro-Smoke Results Summary
| Run | Model | Train Loss | Val Macro F1 | Epoch Finalization | Status | Checkpoints Written |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model A | `source_snunet_vvvh_seed0_v3_fp32` | `6.3440` | `0.1316` | **PASSED** (0 `NameError`) | `COMPLETED_MAX_EPOCH` | `best_checkpoint.pt`, `last_checkpoint.pt` |
| Model B | `source_snunet_vvvh_slope_seed0_v3_fp32` | `2.9547` | `0.1348` | **PASSED** (0 `NameError`) | `COMPLETED_MAX_EPOCH` | `best_checkpoint.pt`, `last_checkpoint.pt` |

### Micro-Smoke Artifact Checksums

#### Model A (`outputs/week2_launcher_fix_smoke/model_a/`)
- `run_manifest.json`: `da07c87710b4b6214a97dee00293be314599d650f296cdb549a3535e2ffd1715` (records `split_sha256: 7afd9768da9f...`, `completion_status: COMPLETED_MAX_EPOCH`)
- `best_checkpoint.pt`: `8796939a45c09a7d592d176358261d9ef26a86405a8c1f6530c53cdf0e571d73` (contains verified `split_hash`)
- `last_checkpoint.pt`: `08a85f21da0938a38ed906c39f3c2d87ffd81653f4c874a5269f9e9b67dd3a2b`
- `access_trace.json`: `1c60d904b5d3698a0cdfee0206b84796b90975c0077874b642d360525bfa7fdc`
- `train.log`: `325f9fb781e8f9d426f8cfa00c544dee75fc3dad4314d87eaa3ebf60ff5f3835`

#### Model B (`outputs/week2_launcher_fix_smoke/model_b/`)
- `run_manifest.json`: `a834f36160f2832b8b86f28b16d2f06b4792b898ee27228ca2b1f3e801525854` (records `split_sha256: 7afd9768da9f...`, `slope_stats_sha256: bc00249a045f...`, `completion_status: COMPLETED_MAX_EPOCH`)
- `best_checkpoint.pt`: `02fde00283435b1f8606d77e4e7b671d1d9f66aa21ecf48fa5b41f097d196510` (contains verified `split_hash` and `slope_stats_hash`)
- `last_checkpoint.pt`: `14b56f066fab739d38bf4183379618323c70a451ce9d0f8d2d40aa0881615a37`
- `access_trace.json`: `50d43b598765e6b1cd764483729467d2aef79a2b805c95bd90842f20a6bc52ab`
- `train.log`: `aae16f3b80037218344e1cd976f9b376b033909721bd52a1d8776747d3256838`

### Data Boundary Compliance
During both regression micro-smokes:
- Quarantined TTA-validation events accessed: **0** (`quarantined_tta_event_accessed: false`)
- Sealed-test events accessed: **0** (`sealed_event_accessed: false`)

---

## 7. Modified Files and Working Tree State

- **Modified Files:**
  1. `src/evaluation/run_manifest.py`: Added provenance hash and `extra_metadata` support; added `update_checkpoint`.
  2. `tools/train_source_model.py`: Explicit provenance unpacking and assertion; fixed `extra_meta` scope; safe exception and failure manifest writing.
  3. `tests/test_fp32_safeguards.py`: Added 4 dedicated regression tests.
- **Git Status (`git status --short`):**
  ```text
   M src/evaluation/run_manifest.py
   M tests/test_fp32_safeguards.py
   M tools/train_source_model.py
  ?? docs/week2_v3_launch_failure_and_fix.md
  ?? outputs/week2_launcher_fix_smoke/
  ```
- **Git Diff Summary (`git diff --stat`):**
  ```text
   src/evaluation/run_manifest.py |  69 ++++++++++----
   tests/test_fp32_safeguards.py  | 210 +++++++++++++++++++++++++++++++++++++++++
   tools/train_source_model.py    |  66 ++++++++++++-
   3 files changed, 321 insertions(+), 24 deletions(-)
  ```
  *(No files staged or committed).*

---

## 8. Recommended Full Canonical Retry Specification

For the formal full Week 2 seed-0 recovery runs, the protocol-approved canonical run IDs remain unchanged, but dedicated fresh output directories must be used:

### Model A (Radar Only `[VV, VH]`)
- **Run ID:** `source_snunet_vvvh_seed0_v3_fp32` (unchanged)
- **Config:** `configs/source/snunet_vvvh_v2_fp32.json`
- **Recommended Output Directory:** `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/`
- **Tmux Session:** `cg_source_a_s0_v3_fp32_att2`
- **Start Condition:** Start from Epoch 1 from scratch; do NOT resume from failed attempt.

### Model B (Radar + Slope `[VV, VH, slope]`)
- **Run ID:** `source_snunet_vvvh_slope_seed0_v3_fp32` (unchanged)
- **Config:** `configs/source/snunet_vvvh_slope_v2_fp32.json`
- **Recommended Output Directory:** `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2/`
- **Tmux Session:** `cg_source_b_s0_v3_fp32_att2`
- **Start Condition:** Start from Epoch 1 from scratch; do NOT resume from failed attempt.

