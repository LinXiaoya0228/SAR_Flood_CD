# ChangeGuard Week 11C Smoke Infrastructure Corrections Note (v1)

**Date:** 2026-10-18  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Review:** [`docs/reviews/week11c_smoke_infrastructure_validation.md`](reviews/week11c_smoke_infrastructure_validation.md)  
**Status:** Remediated & Verified on CPU synthetic/mock infrastructure.  
**Execution Boundary:** Zero real SAR rasters or GT annotations read; zero real checkpoints hashed or loaded in synthetic mode; zero experiments launched. Draft configuration on disk remains strictly unauthorized (`real_data_execution_authorized: false`, `gpu_execution_authorized: false`). Note on GPU usage: No Week 11C smoke/model or real-data experiment was launched. A separate earlier CUDA availability/transfer probe ran successfully (exit 0), initializing CUDA and moving a 2×2 synthetic tensor to `cuda:0`; therefore zero-GPU-operation/strictly-CPU-only execution cannot be claimed across the session.

---

## 1. Technical Remediation Mapping (W11C-R01 to W11C-R07)

### W11C-R01 — Dataset & Manifest Path Resolution & Preflight Metadata Audit
- **Root Cause:** In the initial draft config, dataset root and manifest path were specified relative to `ChangeGuard/` (`datasets/KuroSiwo` and `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz`), which were absent under `REPO_ROOT` and only existed in the parent workspace (`REPO_ROOT.parent`, i.e. `/usr1/home/s126mdg22_06/FYP_SARCD`). Furthermore, preflight did not validate required path metadata before model/device access.
- **Remediation:**
  1. Updated [`configs/changeguard/week11_smoke_event520_draft.json`](../configs/changeguard/week11_smoke_event520_draft.json) to reference the canonical parent locations: `"../datasets/KuroSiwo"` and `"../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"`.
  2. In [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py), added `_resolve_path()` supporting absolute paths, `REPO_ROOT`-relative, and parent-relative paths.
  3. Added explicit preflight validation: in real execution mode, `validate_preflight()` verifies `self.dataset_root.is_dir()` and `self.manifest_path.is_file()` before any device or model instantiation, failing closed with `FileNotFoundError` if either is missing, without opening or reading raster/manifest contents. In synthetic mock mode, path checking is bypassed and marked `"MOCKED_SYNTHETIC"`.

### W11C-R02 — Durable Pre-GT Prediction/Action Persistence & Crash Recovery
- **Root Cause:** In the initial runner, primary pre-update predictions and adaptation diagnostics were stored only in memory (`frozen_predictions` and `step_records`) until after offline ground-truth scoring. A crash or injected failure during offline evaluation left an empty output directory with zero recoverable predictions or run records.
- **Remediation:**
  1. **Initial Manifest:** Runner writes `initial_manifest.json` immediately upon run start containing git commit, working-tree cleanliness, reviewed code hashes, config SHA-256, checkpoint paths/hashes, event identity, and start timestamp.
  2. **Durable Pre-GT Boundary:** Immediately after the adaptation loop finishes and **strictly before any ground truth or offline dataset access**:
     - Persists compressed predictions: `frozen_predictions.npz` (contains 50 predicted class maps keyed by sample ID).
     - Persists adaptation diagnostics: `adaptation_step_records.json` and `adaptation_step_records.csv`.
     - Persists adaptation progress: `adaptation_progress.json` with status `"ADAPTATION_COMPLETED_PENDING_OFFLINE_SCORING"`.
  3. **Failure Recovery:** Wrapped offline scoring in `try...except Exception as exc:`. On offline failure, writes `execution_failure.json` detailing the exception, error traceback, and failure timestamp. All pre-GT adaptation artifacts and predictions remain intact on disk.
  4. **Completion:** On successful offline scoring, writes `completion_status.json` (`status: "COMPLETED"`) alongside the final `smoke_manifest.json` and metrics summaries.

### W11C-R03 — True Negatives Restricted to Offline Validity Mask
- **Root Cause:** In the initial runner, line 515 calculated true negatives as `tn = int(np.sum(~p_flood & ~g_flood))` without intersecting `valid = (valid_mask == 1) & (gt != 3)`. Because invalid pixels (both sensory nodata `valid_mask == 0` and unannotated label 3) have neither predicted flood nor GT flood, `~p_flood & ~g_flood` evaluated to True for all invalid pixels, improperly counting non-valid pixels as True Negatives and causing confusion-cell sums to exceed valid pixel counts.
- **Remediation:**
  1. Updated TN calculation in [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py) to:
     $$\text{tn} = \sum (\neg p_{\text{flood}} \land \neg g_{\text{flood}} \land \text{valid})$$
  2. Added strict per-sample, per-step, and per-event assertions:
     ```python
     assert (tp + fp + fn + tn) == valid_pixels
     ```
  3. Verified via regression test `test_r03_confusion_matrix_validity_and_true_negatives` that 50 synthetic samples with sensory-invalid pixels and label-3 pixels evaluate valid pixels as exactly 1,100, TN as 0, and confusion sum identically equals 1,100 (never 1,200).

### W11C-R04 — Stage, Approval, Boolean, and Device Combination Matrix
- **Root Cause:** The runner only checked boolean types of authorization flags but did not validate `status`, `approval_status`, or contradictory authorization combinations.
- **Remediation:**
  1. Defined strict whitelists:
     - Synthetic statuses: `SUPPORTED_SYNTHETIC_STATUSES = {"DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED", "APPROVED_FOR_CPU_SYNTHETIC_SMOKE_VERIFICATION"}`.
     - Real execution statuses: `SUPPORTED_REAL_STATUSES = {"APPROVED_FOR_REAL_DATA_SMOKE"}`.
  2. Enforced fail-closed validation in `_enforce_authorizations()` and `validate_preflight()`:
     - In real execution mode: reject any status starting with `DRAFT_` (specifically `DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED`) with `PermissionError`. Boolean `true` cannot override draft status.
     - Reject missing, null, or unsupported status with `ValueError`.
     - Reject `approval_status != "APPROVED_FOR_EXECUTION"` with `PermissionError`.
     - In synthetic mock mode: reject any device other than `"cpu"` with `PermissionError`.

### W11C-R05 — Strict Bounded Smoke Invariants Enforcement
- **Root Cause:** Preflight did not validate constructor parameters against the declared bounded smoke identity (seed 0, Event 520, order_a, head 50, batch 16, batches `[16, 16, 16, 2]`), allowing unbounded (`max_samples <= 0`), oversized, or reversed configurations to execute under the smoke identifier.
- **Remediation:**
  1. In `validate_preflight()`, strictly enforce:
     - `checkpoint_seed == 0`
     - `target_events == [520]`
     - `stream_order == "order_a"`
     - `reset_mode == "reset_per_event"`
     - `batch_size == 16`
     - `max_samples == 50` (hard ceiling; reject `<= 0`, `!= 50`)
     - `sample_selection_rule == "deterministic_sorted_sample_id_ascending_head_50"`
     - Model invariants: `SNUNet_ECAM`, 2 in_channels (`["VV", "VH"]`), 3 classes, base_channel 32, FP32, canonical preevent `pre_event_2`, frozen SAR normalization constants `(0.0953, 0.0264)`, `(0.0427, 0.0215)`, clamp 0.15.
  2. In `_resolve_samples_for_event()`: sort strictly ascending by alphanumeric `sample_id`, enforce positive hard ceiling of exactly 50 samples, and verify uniqueness.
  3. In batch loop: verify executed batch sizes strictly equal `[16, 16, 16, 2]`.

### W11C-R06 — Pure Synthetic Mode Asset Independence
- **Root Cause:** In synthetic mock mode, preflight previously called `compute_file_sha256()` on the configured checkpoint path, causing tests and synthetic runs to read physical checkpoint bytes from disk.
- **Remediation:**
  1. Gated checkpoint path resolution and hashing: when `self.synthetic_mock` is True, physical checkpoint hashing and file existence checks are skipped entirely.
  2. Preflight metadata in synthetic mode honestly records `checkpoint_path = "MOCKED_SYNTHETIC"` and `checkpoint_sha256 = "MOCKED_NOT_APPLICABLE"`.
  3. Retained strict real checkpoint existence and SHA-256 verification (`247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`) for real execution mode.
  4. Verified via `test_r06_synthetic_mode_zero_asset_reads` that zero checkpoint hashing calls occur during synthetic mock runs.

### W11C-R07 — CUDA RNG Device Context Guard
- **Root Cause:** In [`src/changeguard/adapter.py`](../src/changeguard/adapter.py), CUDA RNG state was saved and restored whenever `torch.cuda.is_available()` was True, regardless of whether the adapter was operating on CPU. In PyTorch, calling `torch.cuda.get_rng_state_all()` invokes `_lazy_init()` and eagerly initializes CUDA on GPU-visible hosts.
- **Remediation:**
  1. Gated CUDA RNG preservation in both probe locations on an intended CUDA device context:
     ```python
     is_cuda_context = (self.device.type == "cuda" and torch.cuda.is_available())
     src_cuda_rng_state = torch.cuda.get_rng_state_all() if is_cuda_context else None
     ```
     and in `finally:`:
     ```python
     if src_cuda_rng_state is not None:
         torch.cuda.set_rng_state_all(src_cuda_rng_state)
     ```
  2. Applied symmetrically to both the `source_candidate_disagreement` forward and `_evaluate_null_probes_state_neutral`.
  3. Verified via `test_r07_cuda_rng_device_context_guard` that when running on CPU with CUDA available, zero CUDA RNG APIs are invoked. When running in a mocked CUDA device context, RNG state is captured and restored exception-safely.

---

## 2. Test Execution & Verification Results

All tests executed with Python `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`:

```bash
# 1. Real-data smoke infrastructure suite (7 tests resolving W11C-R01 through W11C-R07)
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_week11_real_data_smoke.py
# Result: Ran 7 tests in 16.538s — OK

# 2. Existing CPU evidence consistency suite (12 tests)
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_week11_evidence_consistency.py
# Result: Ran 12 tests in 7.732s — OK

# 3. Combined execution of all 19 Week 11 unit and orchestration tests
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_week11_evidence_consistency.py tests/test_week11_real_data_smoke.py
# Result: Ran 19 tests in 10.732s — OK
```

---

## 3. Modified Files & SHA-256 Hashes

| File | Status | SHA-256 | Description |
|---|---|---|---|
| [`src/changeguard/adapter.py`](../src/changeguard/adapter.py) | **Modified** | `f1b5d6125463f384407b83ff5ab352420026655ff9fd2c8c198c775dfb8a4c5c` | Gated CUDA RNG preservation on `(self.device.type == "cuda" and torch.cuda.is_available())`. |
| [`configs/changeguard/week11_smoke_event520_draft.json`](../configs/changeguard/week11_smoke_event520_draft.json) | **Modified** | `4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55` | Corrected dataset paths to canonical parent locations `../datasets/KuroSiwo` and `../third_party/KuroSiwo/...`. On-disk draft remains unauthorized. |
| [`tools/run_week11_real_data_smoke.py`](../tools/run_week11_real_data_smoke.py) | **Modified** | `8de61c1b18c5dadf6d87e2d108348973d64b3678933d72b66b72932fa752cfb1` | Hardened preflight path metadata checks, durable pre-GT persistence, TN valid mask restriction, stage/approval enforcement, bounded smoke invariants, and synthetic asset independence. |
| [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py) | **Modified** | `487f09e2934920d70c9e385ffd2beb49f8a358b05a4ae40feaf41721d451e44e` | Added targeted regressions for W11C-R01 through W11C-R07. |

---

## 4. Integrity Statements & Unresolved Issues

1. **No Real-Data Execution:** No real SAR images or GT labels were accessed; no checkpoint weights were loaded; no smoke or model experiment was launched. A separate earlier CUDA availability/transfer probe ran successfully (exit 0), initializing CUDA and moving a 2×2 synthetic tensor to `cuda:0`; therefore zero-GPU-operation/strictly-CPU-only execution cannot be claimed across the session.
2. **Draft Unauthorized:** Configuration on disk remains `DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED`, `real_data_execution_authorized: false`, and `gpu_execution_authorized: false`.
3. **No Independent Acceptance Claim:** This document reports implementer remediation and CPU synthetic verification; independent review and authorization remain required before any real smoke launch.
4. **Historical Note:** Replaced by `docs/week11c_infrastructure_corrections_v2.md` following independent review `docs/reviews/week11c_smoke_infrastructure_validation_v2.md` which identified remaining code blockers in R02 (adaptation lifecycle exception handling & asset digest propagation) and R04 (unsupported device syntax enforcement).

