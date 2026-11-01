# Week 11A CPU Implementation Corrections Note (v1)

**Date:** 2026-10-18  
**Implementer Verification Status:** COMPLETED (Implementer Verified — Pending Independent Review)  
**Execution Scope:** Strictly CPU synthetic implementation only; zero real-data access, zero GPU utilization, zero test set unsealing.

---

## 1. Summary of Corrected Review Findings

This document records the implementer-side corrections addressing the independent Week 11B review findings (`docs/reviews/week11a_cpu_validation.md`). Prior review artifacts, logs, and evidence under `outputs/week11_review_evidence/week11b_independent_v1/` remain preserved and unmodified.

| Finding ID | Finding Title | Changed File(s) | Regression Test / Evidence |
| :--- | :--- | :--- | :--- |
| **W11-R01** | Probe isolation & forward state preservation | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_05_r01_state_neutral_probes_and_exception_safety` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R02** | Guard against non-finite updates & parameter rollback | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_06_r02_numerical_failures_guarded_and_rolled_back` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R03** | Timing clarification (PRE vs POST update indicators) | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py), [`configs/changeguard/week11_evidence_consistency_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week11_evidence_consistency_v1.json), [`docs/protocols/week11_evidence_consistency_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/protocols/week11_evidence_consistency_protocol_v1.md) | `test_07_r03_diagnostic_timing_clarification` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R04** | Persisted sample/event identity and batch mapping | [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `test_r04_persisted_runner_schema_and_batch_mapping` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R05** | Strict boolean authorization & CPU stage boundaries | [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `test_r05_strict_boolean_authorization_enforcement` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R06** | Source model independence & storage aliasing elimination | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_08_r06_source_model_storage_independence` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-N01** | Sensory mask shape validation & singleton normalization | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_09_n01_mask_shape_validation_and_normalization` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-N02** | Skipped loss representation as `None` | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_04_empty_selection_and_zero_valid_skip` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |

---

## 2. Technical Details of Implemented Corrections

### 2.1 W11-R01: State-Neutral Probe Isolation & Exception Safety
- **Root Cause:** Probing forwards and reference forwards saved RNG state but did not restore model registered buffers, and lacked `try...finally` protection if a forward pass raised an exception.
- **Fix:** Both `_evaluate_null_probes_state_neutral()` and the source reference evaluation in `adapt_and_predict()` capture buffer snapshots (`named_buffers()`) and RNG state before execution, restoring them within an unconditional `finally:` block.
- **Verification:** Exercised against `StatefulMockModel`, proving that forward counters and RNG are restored on both successful and exception-raising forward passes.

### 2.2 W11-R02: Numerical Boundaries & Candidate Rollback
- **Root Cause:** When selected loss or backward gradients overflowed to `inf` or `nan`, the adapter previously applied the corrupting step and reported `update_applied = True`.
- **Fix:** Three guarded checkpoints were added:
  1. Finite loss check before `loss.backward()`.
  2. Finite gradient check across all trainable parameters before `self.optimizer.step()`.
  3. Finite parameter check across all candidate parameters after stepping; any non-finite parameter triggers an immediate restoration of pre-step parameters and optimizer state dict.
  On any numerical failure, `update_applied = False`, `action = "failure"`, `loss = None`, `failure_reason` is populated, and `self.num_updates_failed` increments.
- **Verification:** Exercised using `LargeFiniteLogitsModel` (loss overflow) and backward hook gradient fault injection (infinite gradient), verifying that candidate parameters remain finite and bit-identical.

### 2.3 W11-R03: Timing Clarification
- **Root Cause:** Disagreement rate formula in the protocol used pre-update weak predictions $p^w$, while a downstream schema section labeled diagnostic fields as post-update.
- **Fix:** Clarified in the protocol, config JSON, implementation, and runner outputs:
  - **PRE_UPDATE:** Primary batch prediction, sensory-valid normalized entropy, max confidence, and source/candidate disagreement rate (`source_disagreement_rate_pre_update`).
  - **POST_UPDATE:** Null evidence probes (`(pre2, pre2)` and `(post, post)`).
  - Explicit `timing_contexts` metadata block added to `risk_evidence`.
- **Verification:** Regression test `test_07_r03_diagnostic_timing_clarification` validates timing labels and consistency.

### 2.4 W11-R04: Persisted Sample/Event Identity & Context
- **Root Cause:** Runner previously dropped sample IDs and context metadata during CSV/JSON serialization.
- **Fix:** `tools/run_week11_cpu_evidence.py` now persists:
  - Actual pipe-separated sample IDs and synthetic event ID in CSV rows.
  - Full `batch_to_samples` mapping dictionary and `all_sample_ids` list in `synthetic_run_summary.json`.
  - Honest synthetic provenance metadata (`checkpoint_seed: "not_applicable_synthetic"`, `stream_order: "deterministic_synthetic_stream"`, `model_architecture: "MockSNUNet_ECAM_proxy"`).
  - Explicit candidate and source BN contexts (`candidate_bn_context`, `source_bn_context`).
  - Prediction flood pixel counts, offline confusion counts (`tp`, `fp`, `fn`, `tn`), and action categories (`update`, `skip`, `failure`).
- **Verification:** Regression test `test_r04_persisted_runner_schema_and_batch_mapping` executes runner pipeline to a temporary directory and validates schema completeness.

### 2.5 W11-R05: Fail-Closed CPU Authorization & Type Enforcement
- **Root Cause:** `bool(auth.get(...))` treated string `"false"` as Python `True`, allowing malformed configurations to bypass CPU preflight checks.
- **Fix:** `_enforce_authorizations()` in `tools/run_week11_cpu_evidence.py` now enforces `isinstance(flag, bool)`. Strings, integers, or missing flags raise `TypeError`. Furthermore, under `APPROVED_FOR_CPU_IMPLEMENTATION`, any non-CPU device request or attempt to set authorization flags to true raises `PermissionError` before model or device access.
- **Verification:** Regression test `test_r05_strict_boolean_authorization_enforcement` verifies rejection of string flags and unauthorized device requests.

### 2.6 W11-R06: Source Model Independence & Storage Aliasing
- **Root Cause:** Passing `source_model = model` aliased the candidate and source objects, causing candidate BN reconfiguration to mutate the source reference model.
- **Fix:** `EvidenceConsistencyAdapter.__init__()` always performs an independent deepcopy of `source_model` (or `model`), places it in `eval()` mode, freezes its parameters, and verifies via data pointers that zero parameter or buffer storage is shared with the candidate model.
- **Verification:** Regression test `test_08_r06_source_model_storage_independence` tests `Adapter(model, source_model=model)`, proving complete isolation.

### 2.7 W11-N01 & W11-N02: Mask Shape Validation & Skipped Loss Representation
- **N01 Fix:** 4D sensory valid masks with singleton channel `(B, 1, H, W)` are explicitly squeezed to `(B, H, W)`. Any other dimensionality or spatial dimension mismatch raises `ValueError`.
- **N02 Fix:** Skipped updates consistently record `loss: None` in metadata and `"null"` in serialized CSVs, distinguishing unperformed updates from computed zero KL divergence.

---

## 3. Test Commands & Implementer Verification Results

The test suite was run using the approved Python environment:
`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`

### 3.1 Focused Week 11A Unit Test Suite
**Command:**
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_week11_evidence_consistency.py -v
```
**Output:**
```text
test_r04_persisted_runner_schema_and_batch_mapping (tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration)
R04: Saved runner output contains sample IDs, event ID, timing, and batch-to-sample mapping. ... ok
test_r05_strict_boolean_authorization_enforcement (tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration)
R05: Strict boolean validation rejects strings, numbers, and unsupported devices. ... ok
test_01_trainable_scope_and_finite_gradients (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test 1: Trainable scope restricted to BN affine parameters; gradients finite. ... ok
test_02_detached_weak_target_and_one_way_kl (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test 2: Detached weak target, correct one-way KL, confidence thresholding. ... ok
test_03_horizontal_flip_spatial_alignment (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test 3: Horizontal flip alignment on xA, xB, valid_mask and spatial inversion. ... ok
test_04_empty_selection_and_zero_valid_skip (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test 4: Empty selection / zero-valid update skipping preserves parameters and returns loss=None (N02). ... ok
test_05_r01_state_neutral_probes_and_exception_safety (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R01: Probe isolation restores buffers and RNG, including on forward exceptions. ... ok
test_06_r02_numerical_failures_guarded_and_rolled_back (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R02: Non-finite loss/gradients/parameters are caught, rolled back, and reported as failure. ... ok
test_07_r03_diagnostic_timing_clarification (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R03: Explicit labeling of PRE_UPDATE vs POST_UPDATE diagnostic indicators. ... ok
test_08_r06_source_model_storage_independence (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R06: Whole-object and shared storage aliasing is safely eliminated. ... ok
test_09_n01_mask_shape_validation_and_normalization (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
N01: Validate and normalize sensory mask shape without silent broadcasting. ... ok
test_10_reset_to_source (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test reset_to_source restores model state and optimizer bit-identically. ... ok

----------------------------------------------------------------------
Ran 12 tests in 1.510s

OK
```

### 3.2 Runner Preflight & Fail-Closed Validation
- Preflight validation passed:
  ```bash
  /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_week11_cpu_evidence.py --validate_only
  # Exit code 0, PREFLIGHT OK
  ```
- Unauthorized execution barrier verified:
  ```bash
  /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_week11_cpu_evidence.py
  # PermissionError: REAL DATA EXECUTION NOT AUTHORIZED: 'real_data_execution_authorized' is false in week11_evidence_consistency_v1.json.
  ```

---

## 4. Modified Files & SHA-256 Hashes

| File | SHA-256 Digest | Status |
| :--- | :--- | :--- |
| [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `10e986a1f7fad6d3902f18cc474dbb81e102432718d6898443c92b90451a3c4b` | Corrected (R01, R02, R03, R06, N01, N02) |
| [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `eb1ff51c8149dac7c4acb4385a1ef024796e1bbe2e4d83b951e77d7fb500e398` | Corrected (R04, R05) |
| [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) | `e5abb70ede964f006e03da9b059ad47d892a121a82c938d994f769a5c92f4f08` | Expanded with 12 focused regressions |
| [`configs/changeguard/week11_evidence_consistency_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week11_evidence_consistency_v1.json) | `21b3f40f4b836389e1f1114711b53a4e51ab2d018001721b79f350119e1d127f` | Updated (R03 timing synchronization) |
| [`docs/protocols/week11_evidence_consistency_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/protocols/week11_evidence_consistency_protocol_v1.md) | `84aab300c10c74aa2548ba75de2cce57e3928a0f9dfa07cd1efb687d11c03faf` | Synchronized protocol specification |

---

## 5. Unresolved Issues & Boundaries

1. **Independent Review Status:** All findings above have been verified by implementer unit tests. They remain subject to formal independent review and acceptance.
2. **Real-Data & GPU Work:** Strictly zero real Kuro Siwo SAR rasters, ground-truth annotations, source checkpoints, or Fisher matrices were loaded. Zero GPU operations occurred.
3. **Execution Infrastructure Readiness:** Real-data loading and stream execution infrastructure (`run_real_stream()`) remain unimplemented by design pending future authorization.

