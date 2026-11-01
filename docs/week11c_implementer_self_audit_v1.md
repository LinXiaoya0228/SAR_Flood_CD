# ChangeGuard Week 11C Smoke Infrastructure Implementer Self-Audit (v1)

**Date:** 2026-10-18  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22`  
**Governing Reviews:**  
- [`docs/reviews/week11c_smoke_infrastructure_validation.md`](reviews/week11c_smoke_infrastructure_validation.md)  
- [`docs/reviews/week11c_smoke_infrastructure_validation_v2.md`](reviews/week11c_smoke_infrastructure_validation_v2.md)  
**Governing Corrections Notes:**  
- [`docs/week11c_infrastructure_corrections_v1.md`](week11c_infrastructure_corrections_v1.md)  
- [`docs/week11c_infrastructure_corrections_v2.md`](week11c_infrastructure_corrections_v2.md)  
**Audit Status:** **PASS** (Implementer-owned fail-closed audit complete; all preconditions met).  
**Execution Boundary:** Zero real SAR rasters or GT annotations read; zero checkpoint weights loaded in synthetic mode; zero model/smoke experiments launched. Draft configuration on disk remains strictly unauthorized (`real_data_execution_authorized: false`, `gpu_execution_authorized: false`). Note on execution scope: An earlier terminal probe initialized CUDA and transferred a 2×2 synthetic tensor to `cuda:0`; no model forwards, backwards, real rasters, GT annotations, or smoke experiments were executed.

---

## 1. Audit Scope & Verification Objectives

This implementer-owned self-audit verifies that all code blockers identified in [`docs/reviews/week11c_smoke_infrastructure_validation_v2.md`](reviews/week11c_smoke_infrastructure_validation_v2.md) are fully remediated and fail closed before any real-data or GPU authorization is granted:
1. **W11C-R02 Adaptation-Failure Regression:** The synthetic mock runner must catch an injected adaptation exception, write `execution_failure.json` detailing the failing phase (`ADAPTATION`), error message, and traceback, and leave `adaptation_progress.json` reflecting the exact completed batches, sample counts, and sample IDs.
2. **W11C-R02 Persistence-Failure Regression:** The synthetic mock runner must catch an injected persistence failure (e.g. disk write error saving `frozen_predictions.npz`), write `execution_failure.json` with phase `PERSISTENCE`, and never mark the run complete (`completion_status.json` must be absent).
3. **W11C-R04 Device-Rejection Probes:** Verify that `not-a-device`, `mps`, `cpu` (when real execution authorized), `cuda`, `cuda:`, `cuda:abc`, and `cuda:-1` all fail closed before any model, checkpoint, or device work.
4. **All R01–R07 Regressions:** Verify that all 23 existing unit and orchestration tests pass without errors.
5. **Pre- and Post-Audit Hash Invariance:** Verify that all relevant code, test, config, and review files remain bit-identical before and after the audit pass.

---

## 2. File Stability & Hash Invariance Audit

All relevant files were hashed immediately prior to and following the test execution pass:

| File Path | Pre-Test SHA-256 | Post-Test SHA-256 | Status |
|---|---|---|---|
| `tools/run_week11_real_data_smoke.py` | `8a89e5aab9c26b2454bc9c34aa58e1b5dc50bb8b9b221845aff07c91d158ce4c` | `8a89e5aab9c26b2454bc9c34aa58e1b5dc50bb8b9b221845aff07c91d158ce4c` | **Unchanged** |
| `tools/run_week11_cpu_evidence.py` | `ce6ac897af1278807cfde47c621a89744b6e0eb489082d9683987e59aec4beb3` | `ce6ac897af1278807cfde47c621a89744b6e0eb489082d9683987e59aec4beb3` | **Unchanged** |
| `src/changeguard/adapter.py` | `f1b5d6125463f384407b83ff5ab352420026655ff9fd2c8c198c775dfb8a4c5c` | `f1b5d6125463f384407b83ff5ab352420026655ff9fd2c8c198c775dfb8a4c5c` | **Unchanged** |
| `tests/test_week11_real_data_smoke.py` | `3dfa5469a61121d22ea816913209c1de1b06ae57d181bcc1910eb7282f0de48d` | `3dfa5469a61121d22ea816913209c1de1b06ae57d181bcc1910eb7282f0de48d` | **Unchanged** |
| `tests/test_week11_evidence_consistency.py` | `020ad3ecf71b97c9876e92ec6214571257bbbccbd5e7631b55b6e626116235ed` | `020ad3ecf71b97c9876e92ec6214571257bbbccbd5e7631b55b6e626116235ed` | **Unchanged** |
| `configs/changeguard/week11_smoke_event520_draft.json` | `4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55` | `4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55` | **Unchanged** |
| `docs/reviews/week11c_smoke_infrastructure_validation_v2.md` | `1ecbaca047cdd5bf3247a14729c305ec1dc0c4680dbbd6889ddcabdaa38de038` | `1ecbaca047cdd5bf3247a14729c305ec1dc0c4680dbbd6889ddcabdaa38de038` | **Unchanged** |
| `docs/week11c_infrastructure_corrections_v1.md` | `5e2ee282ee8376684fa0c8d037d835620efa331d77b874713dea9a9bb230a116` | `5e2ee282ee8376684fa0c8d037d835620efa331d77b874713dea9a9bb230a116` | **Unchanged** |
| `docs/week11c_infrastructure_corrections_v2.md` | `1d0cbed974fbdd31996d8b9dc52a8b7a9cc3765e050568088b25419bd3749a68` | `1d0cbed974fbdd31996d8b9dc52a8b7a9cc3765e050568088b25419bd3749a68` | **Unchanged** |

Zero files were mutated or corrupted during the audit execution pass.

---

## 3. Detailed Audit Findings by Remediation Area

### 3.1 W11C-R02 Adaptation-Failure Regression
- **Test:** `test_r02_adaptation_failure_preserves_durable_progress` in [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py)
- **Mechanism:** Injected an intentional `RuntimeError("Simulated crash during 2nd adaptation batch")` on batch 2 of the 50-sample synthetic stream.
- **Observations:**
  - `adaptation_progress.json` exists on disk with `batches_completed: 1`, `samples_completed: 16`, `len(completed_sample_ids): 16`.
  - `execution_failure.json` exists on disk recording:
    - `"phase": "ADAPTATION"`
    - `"error_type": "RuntimeError"`
    - `"error_message": "Simulated crash during 2nd adaptation batch"`
    - Traceback accurately identifying the point of failure.
    - `"last_durable_progress"` snapshot matching the 1 completed batch.
  - `completion_status.json` was **not** created.
- **Verdict:** **PASS**

### 3.2 W11C-R02 Persistence-Failure Regression
- **Test:** `test_r02_persistence_failure_records_phase` in [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py)
- **Mechanism:** Injected an `IOError("Simulated disk write error")` during `np.savez_compressed` in the persistence phase.
- **Observations:**
  - `execution_failure.json` exists on disk recording `"phase": "PERSISTENCE"`, `"error_type": "IOError"`, and the simulated error traceback.
  - `completion_status.json` was **not** created.
- **Verdict:** **PASS**

### 3.3 W11C-R04 Device-Rejection Probes
- **Test:** `test_r04_device_syntax_validation` in [`tests/test_week11_real_data_smoke.py`](../tests/test_week11_real_data_smoke.py)
- **Mechanism:** Tested each invalid device string against the real execution runner (`synthetic_mock=False`):
  - `not-a-device`: Rejected with `ValueError` matching `^cuda:(0|[1-9]\d*)$` error message.
  - `mps`: Rejected with `ValueError`.
  - `cpu`: Rejected with `ValueError` (real execution requires CUDA).
  - `cuda`: Rejected with `ValueError` (bare `cuda` without index rejected).
  - `cuda:`: Rejected with `ValueError`.
  - `cuda:abc`: Rejected with `ValueError`.
  - `cuda:-1`: Rejected with `ValueError`.
  - `cuda: 0`: Rejected with `ValueError` (whitespace rejected).
  - `gpu:0`: Rejected with `ValueError`.
- **Positive Control:** Tested `cuda:0` and `cuda:1` under authorized configuration; passed device syntax validation.
- **Verdict:** **PASS**

### 3.4 Digest Propagation & Offline Scoring Invariants
- **Tests:** `test_manifest_digest_propagation` and `test_r03_confusion_matrix_validity_and_true_negatives`.
- **Observations:**
  - Initial and completion manifests consistently record `checkpoint_expected_sha256`, `checkpoint_actual_sha256`, and `manifest_sha256`.
  - Confusion matrix calculations strictly intersect with `valid = (valid_mask == 1) & (gt != 3)`; confusion cell sum identically equals valid pixels.
- **Verdict:** **PASS**

---

## 4. Full Test Suite Execution Summary

Executed with Python `/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python`:

```text
Ran 23 tests in 15.369s

OK
```

All 23 focused regression tests passed with zero failures and zero errors.

---

## 5. Audit Conclusion & Phase Transition

The implementer-owned self-audit confirms that all technical blockers and integrity preconditions are resolved:
1. `tools/run_week11_real_data_smoke.py` is robust, phase-aware, durable, and fail-closed.
2. `tests/test_week11_real_data_smoke.py` rigorously validates every boundary condition.
3. The on-disk draft configuration remains unauthorized (`false`/`false`).
4. File hash invariance confirms that no unauthorized writes or modifications occurred during audit.

Phase 2 is formally complete. Proceeding to Phase 3 (Approved Configuration Creation).
