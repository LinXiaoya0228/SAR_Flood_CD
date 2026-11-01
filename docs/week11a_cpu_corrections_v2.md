# Week 11A CPU Implementation Corrections Note (v2)

**Date:** 2026-10-18  
**Implementer Verification Status:** COMPLETED (Implementer Verified — Pending Independent Review v3)  
**Execution Scope:** Strictly CPU synthetic implementation only; zero real-data access, zero GPU utilization, zero test set unsealing.

---

## 1. Summary of Corrected Review Findings (v2)

This document details the targeted corrections resolving the three remaining findings identified in the Week 11B targeted re-verification report ([`docs/reviews/week11a_cpu_validation_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/reviews/week11a_cpu_validation_v2.md)): **R02**, **R05**, and **N02**.

All previously accepted findings (**R01**, **R03**, **R04**, **R06**, **N01**) and the initial correction note ([`docs/week11a_cpu_corrections_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week11a_cpu_corrections_v1.md)) remain preserved intact without reopening or modification.

| Finding ID | Finding Title | Changed File(s) | Regression Test / Evidence |
| :--- | :--- | :--- | :--- |
| **W11-R02** | Post-step optimizer-state finiteness check & momentum rollback | [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `test_06_r02_numerical_failures_guarded_and_rolled_back` (Case C & Case D) in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-R05** | Strict status validation (`APPROVED_FOR_CPU_IMPLEMENTATION` only) | [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `test_r05_strict_boolean_and_stage_enforcement` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |
| **W11-N02** | Python `None` preservation in JSON records (actual JSON `null`) | [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `test_r04_and_n02_persisted_json_null_and_csv_text_null` in [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) |

---

## 2. Technical Details of Implemented Corrections

### 2.1 W11-R02: Optimizer-State Finiteness & Momentum Rollback
- **Review Finding:** After `optimizer.step()`, candidate parameters were inspected for non-finite values, but `self.optimizer.state_dict()` was not. Consequently, optimizer-only corruption (e.g. `inf`/`NaN` in `momentum_buffer`) survived undetected and was logged as `update_applied = True`.
- **Correction:**
  - Added recursive helper method `EvidenceConsistencyAdapter._is_state_dict_finite(obj)` inspecting all nested tensors and scalar floats in the optimizer state dict.
  - After `self.optimizer.step()`, the adapter inspects both candidate parameters and `self.optimizer.state_dict()`.
  - If either parameters or optimizer state contains non-finite values:
    1. Pre-step candidate parameters are restored bit-identically.
    2. Pre-step optimizer state dict (including pre-populated, finite momentum buffers) is reloaded.
    3. `self.optimizer.zero_grad()` is invoked.
    4. Counters are updated honestly: `self.num_updates_failed += 1`, while `self.num_updates_applied` is **not** incremented.
    5. The step returns `update_applied = False`, `action = "failure"`, `loss = None`, and detailed `failure_reason`.
- **Regression Evidence:** Exercised with a fault-injected `corrupt_momentum` wrapper where parameters remain finite while momentum becomes `inf`. Verified that `action == "failure"`, `update_applied == False`, `loss is None`, pre-step momentum is restored, `num_updates_applied` is preserved at 1, `num_updates_failed` increments to 1, followed by a successful positive-control update reaching `num_updates_applied = 2`.

### 2.2 W11-R05: Strict CPU Status Validation Without Permissive Fallback
- **Review Finding:** The CPU runner applied CPU-stage restrictions only when `status == "APPROVED_FOR_CPU_IMPLEMENTATION"`. An unknown or `null` status fell through to an `else:` branch that permitted non-CPU execution if `gpu_execution_authorized` was set to `True`.
- **Correction:**
  - Removed the permissive `else:` fallback in `Week11CPURunner._enforce_authorizations()`.
  - Added strict validation in both `__init__()` and `validate_preflight()`:
    ```python
    status = self.cfg.get("status")
    if status != "APPROVED_FOR_CPU_IMPLEMENTATION":
        raise ValueError(
            f"Unsupported configuration status '{status}'. "
            "This CPU runner strictly requires status == 'APPROVED_FOR_CPU_IMPLEMENTATION'."
        )
    ```
  - Rejects missing, `None`, unknown, or unsupported status values immediately with `ValueError` before any model or device access occurs.
- **Regression Evidence:** Regression test `test_r05_strict_boolean_and_stage_enforcement` tests configurations with `status: null`, `status: "UNSUPPORTED_STAGE"`, and `status: "SOME_FUTURE_STAGE"`, verifying immediate `ValueError` rejection in both the constructor and preflight without GPU initialization.

### 2.3 W11-N02: Preserved Python `None` for JSON `null`
- **Review Finding:** The runner serialization in `tools/run_week11_cpu_evidence.py` previously converted `None` to the string `"null"` within the shared record dict before writing `synthetic_run_summary.json`. Deserializing the JSON output yielded Python string `str` instead of `None`.
- **Correction:**
  - Preserved Python `None` directly in `record["loss"]` within `step_records` and `summary`.
  - Standard `json.dump()` serializes Python `None` as native JSON `null`.
  - Created a separate `csv_records` list specifically for `synthetic_step_metrics.csv` that maps `None` to textual `"null"`.
- **Regression Evidence:** Regression test `test_r04_and_n02_persisted_json_null_and_csv_text_null` runs the synthetic pipeline, loads the saved `synthetic_run_summary.json` via `json.loads()`, and asserts `rec["loss"] is None` for all skipped records, while confirming textual `"null"` in `synthetic_step_metrics.csv`.

---

## 3. Test Commands & Implementer Verification Results

**Test Command:**
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_week11_evidence_consistency.py -v
```

**Execution Output:**
```text
test_r04_and_n02_persisted_json_null_and_csv_text_null (tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration)
R04 & N02: Saved JSON summary has Python None (JSON null) while CSV has textual 'null'. ... ok
test_r05_strict_boolean_and_stage_enforcement (tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration)
R05: Strict boolean validation and enforcement of APPROVED_FOR_CPU_IMPLEMENTATION status. ... ok
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
R02: Non-finite loss/gradients/parameters/optimizer momentum are caught, rolled back, and reported as failure. ... ok
test_07_r03_diagnostic_timing_clarification (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R03: Explicit labeling of PRE_UPDATE vs POST_UPDATE diagnostic indicators. ... ok
test_08_r06_source_model_storage_independence (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
R06: Whole-object and shared storage aliasing is safely eliminated. ... ok
test_09_n01_mask_shape_validation_and_normalization (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
N01: Validate and normalize sensory mask shape without silent broadcasting. ... ok
test_10_reset_to_source (tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency)
Test reset_to_source restores model state and optimizer bit-identically. ... ok

----------------------------------------------------------------------
Ran 12 tests in 8.219s

OK
```

**CLI Preflight & Fail-Closed Validation:**
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_week11_cpu_evidence.py --validate_only
# Exit code 0, [PREFLIGHT OK]
```

---

## 4. Modified Files & Exact SHA-256 Hashes

| Path | SHA-256 Digest | Status |
| :--- | :--- | :--- |
| [`src/changeguard/adapter.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/changeguard/adapter.py) | `4a410e382c8691e2fe8ad947614b53d0b6856c30fc6a3b0858e1f6e863032879` | Modified (R02 post-step optimizer check) |
| [`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py) | `ce6ac897af1278807cfde47c621a89744b6e0eb489082d9683987e59aec4beb3` | Modified (R05 status validation, N02 JSON null) |
| [`tests/test_week11_evidence_consistency.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_week11_evidence_consistency.py) | `020ad3ecf71b97c9876e92ec6214571257bbbccbd5e7631b55b6e626116235ed` | Modified (Regressions for R02, R05, N02) |
| [`docs/week11a_cpu_corrections_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week11a_cpu_corrections_v2.md) | *Current File* | New |
| [`docs/week11a_cpu_corrections_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week11a_cpu_corrections_v1.md) | `b7ac8d01ff69da1ab37fe0054e5032e9b470bb5ce823eb63e4a400f45b7a7b9a` | Preserved |
| [`configs/changeguard/week11_evidence_consistency_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week11_evidence_consistency_v1.json) | `21b3f40f4b836389e1f1114711b53a4e51ab2d018001721b79f350119e1d127f` | Preserved from v1 |
| [`docs/protocols/week11_evidence_consistency_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/protocols/week11_evidence_consistency_protocol_v1.md) | `84aab300c10c74aa2548ba75de2cce57e3928a0f9dfa07cd1efb687d11c03faf` | Preserved from v1 |

---

## 5. Integrity Declarations & Status

1. **Implementer Verification vs. Independent Acceptance:** All 12 test assertions pass under implementer execution. This work is verified on the implementer side and remains subject to independent review; it is **not** labeled independently approved.
2. **Real-Data and GPU Confirmation:** Strictly zero real SAR rasters, ground-truth annotations, source checkpoints, or Fisher matrices were opened or loaded. Zero GPU operations occurred. The 10 sealed test events remain fully quarantined.
3. **Git Cleanliness:** `git diff` and `git diff --staged` are completely empty. No changes have been staged or committed.

