# ChangeGuard Week 11C Smoke Infrastructure Corrections Note (v2)

**Date:** 2026-10-18  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Review:** [`docs/reviews/week11c_smoke_infrastructure_validation_v2.md`](reviews/week11c_smoke_infrastructure_validation_v2.md)  
**Status:** Remediated & Verified on CPU synthetic/mock infrastructure.  
**Execution Boundary:** Zero real SAR rasters or GT annotations read; zero real checkpoints loaded in synthetic mode; zero model/smoke experiments launched. Draft configuration on disk remains strictly unauthorized (`real_data_execution_authorized: false`, `gpu_execution_authorized: false`). Note on execution scope: An earlier terminal probe initialized CUDA and transferred a 2×2 synthetic tensor to `cuda:0`; no model forwards, backwards, real rasters, GT annotations, or smoke experiments were executed.

---

## 1. Executive Summary & Review Mapping

In [`docs/reviews/week11c_smoke_infrastructure_validation_v2.md`](reviews/week11c_smoke_infrastructure_validation_v2.md), the independent reviewer confirmed that W11C-R01, W11C-R03, W11C-R05, W11C-R06, and W11C-R07 passed. Two remaining technical code blockers were identified:
1. **W11C-R02 (Lifecycle Exception Safety & Asset Provenance):** Adaptation-phase exceptions bypassed terminal failure recording; per-batch durable progress was absent on disk; checkpoint/manifest SHA-256 digests were omitted from manifests.
2. **W11C-R04 (Fail-Closed Device Syntax Enforcement):** Real execution accepted unsupported device strings (e.g. `'not-a-device'`, `'mps'`) without strict syntax checking prior to device/model work.

Additionally, the delivery record discrepancy in `v1` regarding an earlier CUDA tensor transfer probe was corrected. This note documents the comprehensive implementation and CPU synthetic verification of these remaining items.

---

## 2. Technical Remediation Details

### W11C-R02 — Phase-Aware Lifecycle, Per-Batch Durable Progress, and Digest Provenance
1. **Phase-Aware Lifecycle Architecture:**
   - In [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py), the entire execution workflow is wrapped in a top-level phase-aware exception handler with explicit phase transitions:
     - `PREFLIGHT`: Configuration, authorization, and metadata audit.
     - `INITIALIZATION`: Model loading, adapter setup, dataset resolution, and writing `initial_manifest.json`.
     - `ADAPTATION`: Sequential processing of Event 520 batches with per-batch durable state persistence.
     - `PERSISTENCE`: Atomic persistence of ID-bound `frozen_predictions.npz` and adaptation diagnostic records (`adaptation_step_records.json` and `.csv`) strictly **prior** to any ground-truth access.
     - `OFFLINE_SCORING`: Ground-truth evaluation, double-masking metrics computation, and validation assertions.
     - `COMPLETION`: Final manifest writing and creation of `completion_status.json`.

2. **Per-Batch Durable Progress Tracking:**
   - At the completion of every adaptation batch, [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py) writes/updates `adaptation_progress.json` atomically:
     ```json
     {
       "status": "RUNNING_ADAPTATION",
       "batches_completed": 1,
       "total_expected_batches": 4,
       "batch_sizes_completed": [16],
       "samples_completed": 16,
       "completed_sample_ids": ["s520_0000", ...],
       "latest_action": "update_applied",
       "updates_applied": 1,
       "updates_skipped": 0,
       "updates_failed": 0,
       "total_steps": 1,
       "last_updated_utc": "2026-10-18T..."
     }
     ```
   - If an exception, crash, or OOM occurs during adaptation, `adaptation_progress.json` records exactly what batches/samples completed before the failure.

3. **Phase-Aware Terminal Failure Recording:**
   - Any unhandled exception during any phase triggers the handler, which persists `execution_failure.json`:
     ```json
     {
       "status": "FAILED",
       "phase": current_phase,
       "error_type": type(exc).__name__,
       "error_message": str(exc),
       "traceback": traceback.format_exc(),
       "last_durable_progress": latest_progress_data,
       "timestamp_utc": "..."
     }
     ```
   - Partial or failed runs are **never** marked with `completion_status.json`.

4. **Asset Digest Propagation:**
   - In real execution, `initial_manifest.json` and `smoke_manifest.json` capture:
     - `checkpoint_expected_sha256`: Expected SHA-256 from configuration.
     - `checkpoint_actual_sha256`: Computed SHA-256 of the real source checkpoint.
     - `manifest_sha256`: Computed SHA-256 of the KuroSiwo test pickle manifest.
   - In synthetic mock mode, these fields are explicitly populated with `"MOCKED_NOT_APPLICABLE"` to preserve schema consistency while guaranteeing zero physical asset access.

---

### W11C-R04 — Fail-Closed Device Syntax & Authorization Matrix
1. **Strict Device Syntax Regex:**
   - Real execution strictly requires matching regex `^cuda:(0|[1-9]\d*)$` via `CUDA_DEVICE_REGEX` before any asset, checkpoint, or device instantiation:
     ```python
     if not CUDA_DEVICE_REGEX.match(self.device_str):
         raise ValueError(
             f"Real smoke execution strictly requires device format 'cuda:<non-negative integer>' (e.g. 'cuda:0'), "
             f"got '{self.device_str}'. Bare 'cuda', 'cpu', 'mps', negative indices and arbitrary names are rejected."
         )
     ```
   - Synthetic mock execution strictly requires `device == "cpu"`, rejecting any other device string with `PermissionError`.
2. **Fail-Closed Verification:**
   - The device string is checked without invoking `torch.cuda` functions, preventing unintended CUDA lazy initialization during device syntax rejection.

---

## 3. Test Suite Expansion & Verification Results

The regression test suite [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py) was expanded from 7 to 11 tests:
- `test_r02_adaptation_failure_preserves_durable_progress`: Injects an adaptation crash on the 2nd batch; verifies `adaptation_progress.json` preserves `batches_completed: 1`, `samples_completed: 16`, and `execution_failure.json` records `phase: "ADAPTATION"`.
- `test_r02_persistence_failure_records_phase`: Injects a disk write error during prediction saving; verifies `execution_failure.json` records `phase: "PERSISTENCE"`.
- `test_r04_device_syntax_validation`: Validates regex rejection of `["not-a-device", "mps", "cpu", "cuda", "cuda:", "cuda:abc", "cuda:-1", "cuda: 0", "gpu:0"]` with `ValueError`, and acceptance of valid formats `["cuda:0", "cuda:1"]`.
- `test_manifest_digest_propagation`: Validates that `checkpoint_expected_sha256`, `checkpoint_actual_sha256`, and `manifest_sha256` keys are present in both initial and completion manifests.

### Verification Commands & Results

All tests executed with Python `/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python`:

```bash
# 1. Expanded real-data smoke infrastructure suite (11 tests)
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python -m unittest tests/test_week11_real_data_smoke.py
# Result: Ran 11 tests in 14.901s — OK

# 2. Existing CPU evidence consistency suite (12 tests)
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python -m unittest tests/test_week11_evidence_consistency.py
# Result: Ran 12 tests in 9.766s — OK

# 3. Combined execution of all 23 Week 11 unit and orchestration tests
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python -m unittest tests/test_week11_evidence_consistency.py tests/test_week11_real_data_smoke.py
# Result: Ran 23 tests in 14.971s — OK
```

---

## 4. Current File Hashes

| File | Status | SHA-256 | Description |
|---|---|---|---|
| [`src/changeguard/adapter.py`](../src/changeguard/adapter.py) | Stable | `f1b5d6125463f384407b83ff5ab352420026655ff9fd2c8c198c775dfb8a4c5c` | Gated CUDA RNG preservation on `(self.device.type == "cuda" and torch.cuda.is_available())`. |
| [`configs/changeguard/week11_smoke_event520_draft.json`](../configs/changeguard/week11_smoke_event520_draft.json) | Stable | `4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55` | Canonical paths; on-disk draft remains unauthorized (`false`/`false`). |
| [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py) | **Remediated** | `8a89e5aab9c26b2454bc9c34aa58e1b5dc50bb8b9b221845aff07c91d158ce4c` | Phase-aware lifecycle, per-batch durable progress, regex device syntax check, asset digest propagation. |
| [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py) | **Expanded** | `3dfa5469a61121d22ea816913209c1de1b06ae57d181bcc1910eb7282f0de48d` | 11 focused regression tests covering W11C-R01 to W11C-R07, failure recovery, syntax rejections, and digests. |
| [`docs/week11c_infrastructure_corrections_v1.md`](week11c_infrastructure_corrections_v1.md) | Corrected | `5e2ee282ee8376684fa0c8d037d835620efa331d77b874713dea9a9bb230a116` | Historical v1 note updated with honest CUDA probe disclosure. |

---

## 5. Integrity Statements

1. **No Real-Data Execution:** Zero real SAR images or GT annotations have been accessed or scored; zero model forwards/backwards or smoke experiments have been executed.
2. **Draft Configuration Integrity:** [`configs/changeguard/week11_smoke_event520_draft.json`](../configs/changeguard/week11_smoke_event520_draft.json) remains in draft status (`status: "DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED"`, `real_data_execution_authorized: false`, `gpu_execution_authorized: false`).
3. **Execution Scope Disclosure:** No Week 11C smoke/model or real-data experiment was launched. A separate earlier CUDA availability/transfer probe ran successfully (exit 0), initializing CUDA and moving a 2×2 synthetic tensor to `cuda:0`; therefore zero-GPU-operation/strictly-CPU-only execution cannot be claimed across the session.
4. **Prerequisites for Launch:** Execution of the real-data smoke requires passing the implementer-owned self-audit, freezing a scoped Git checkpoint, creating an explicit user-approved configuration (`APPROVED_FOR_REAL_DATA_SMOKE`), and verifying all runtime preflight prerequisites.
