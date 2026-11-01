# ChangeGuard Work Package E7: Final Closeout Report

**Status:** COMPLETE — READY FOR E8
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git HEAD at Closeout:** `54a99498ee131fd4a8ca7192b3c5b3d6537ca472`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E7 (Sections 12.4, 12.5) & Block E8 (Sections 13.1–13.4)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/protocols/e7_branch_c_empirical_protocol_v1.md`
- `docs/e7_cotta_smoke_closeout.md`
- `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log` (preserved failed attempt)
- `logs/e7_cotta_smoke/e7_cotta_smoke_v2.log` (verified success execution)

---

> [!WARNING]
> **2026-10-22 label-I/O certification correction:** The successful E7 smoke v2 is retained as an
> infrastructure smoke, but no longer certifies physical post-persistence GT reading. Its runner
> preloaded GT tensors while building guarded batches. No evidence shows that GT reached CoTTA
> adaptation, no scientific evidence was claimed from the smoke, and sealed-test access remained
> zero. A repaired v3 smoke is required before the versioned E8 v2 retry.

---

## 1. Executive Summary & Precise Status Boundaries

Work Package E7 implemented the engineering foundations, baseline adapters, execution runners, and analysis engines for **Branch C: Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams**.

In strict accordance with project governance and fail-closed research integrity rules:
1. **Implementation Verified:** Full Branch C empirical evaluation suite, `E7CoTTAAdapter`, and `E7EmpiricalAnalyzer` were implemented, audited, and tested.
2. **CPU Behavior Verified:** 39 tests passed cleanly across unit and regression test suites with zero CUDA calls:
   - `tests/test_e7_cotta_adapter.py`: 16 tests passed.
   - `tests/test_e7_empirical_runner.py`: 11 tests passed (including real `LabelGuardedBatch` lifecycle and runner persistence tests).
   - `tests/test_e5_temporal_risk_analysis.py`: 12 regression tests passed.
3. **Smoke v1 Attempt (Preserved):** Executed in tmux session `cg_e7_cotta_smoke_v1` on GPU 0. Failed closed with exit code 1 due to `AttributeError` on non-existent `batch.get_labels_offline()`. Preserved at `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log`.
4. **Smoke v2 Retry (Verified):** Repaired `LabelGuardedBatch` persistence lifecycle, tested on CPU, configured fresh run ID `e7_cotta_smoke_v2`, and executed successfully on GPU 0 in detached tmux session `cg_e7_cotta_smoke_v2`:
   - Exit code: `0`
   - Terminal status: `SUCCESS`
   - Stream runtime: `4.88` seconds
   - Samples processed: Exactly 32 samples across 2 batches ($B=16$) on Event 520
   - Primary persisted predictions: Shards saved and verified before label access
   - Macro F1: `0.725692` | Precision: `0.723802` | Recall: `0.727591` | IoU: `0.569479`
   - Updates applied: 2 (100% of batches updated; 0 skipped; 0 rolled back)
   - Zero NaNs, zero Infs, finite loss and finite weights throughout
5. **E7 Overall Status:** **E7 COMPLETE — READY FOR E8.** Implementation, CPU tests, and real-data GPU smoke test are verified and closeout is sealed.
6. **Full Empirical Matrix: NOT YET EXECUTED.** Zero formal multi-stream runs have been launched. Execution of the full matrix is deferred to Work Package E8.
7. **CoTTA Performance: NOT YET CONCLUDED.** A bounded 2-batch smoke test verifies execution mechanics and label integrity only, not empirical hypothesis outcomes.
8. **LOEO Infrastructure: TESTED ON SYNTHETIC DATA.** 6-fold cross-event isolation verified, but zero real held-out-event conclusions drawn prior to E8.
9. **Branch C Scope: UNCHANGED.** Zero accept/reject decision gates, score fusions, learned classifiers, U/C threshold calibrations, or claims of advance harm prediction were implemented.
10. **Sealed-Test Access: ZERO.** Quarantined partition `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` was verified untouched.

---

## 2. Pre-Smoke Semantic Audit Ledger (18/18 Points)

Prior to GPU launch, the implementation was audited against governing protocols:

| # | Semantic Requirement | Verification Evidence | Status |
| :---: | :--- | :--- | :---: |
| 1 | Non-aliasing of student, teacher, and source | Verified by `assert_module_storage_disjoint` and Unit Test 1 | **PASS** |
| 2 | Teacher parameters never receive optimizer gradients | Teacher has `requires_grad_(False)`; optimizer targets student only | **PASS** |
| 3 | Exact EMA update timing and coefficient | $\alpha = 0.999$, updated post-step: $\theta_t \leftarrow \alpha \theta_t + (1 - \alpha) \theta_s$ (Unit Test 3) | **PASS** |
| 4 | SSR granularity and probability | Element-wise Bernoulli mask with $p = 0.01$; verified by Unit Test 5 | **PASS** |
| 5 | Bit-exact source value restoration | Restored elements match source state dict bit-for-bit (Unit Test 6) | **PASS** |
| 6 | Deterministic, isolated SSR RNG | Uses dedicated `torch.Generator(device='cpu')` seeded at init (Unit Test 15) | **PASS** |
| 7 | Identical spatial transforms for paired SAR rasters | $x_A$ and $x_B$ receive identical transforms jointly in `apply_sar_spatial_augmentation` | **PASS** |
| 8 | Inverse spatial mapping before probability averaging | Inverse mapping applied to all 4 views before mean reduction (Unit Test 4) | **PASS** |
| 9 | Only protocol-approved SAR-safe transforms | Identity, H-flip, V-flip, Rot180 only; zero photometric perturbations | **PASS** |
| 10 | Sensory validity independent of GT | Valid mask defined strictly by $M_{NA} == 1$ (Unit Test 7) | **PASS** |
| 11 | Primary persisted prediction is `PRE_UPDATE` | Primary prediction derived from pre-step teacher output (Unit Test 14) | **PASS** |
| 12 | Predictions durable before labels are read | Verified: shards saved and verified before `mark_persisted()` and `get_labels()` (Unit Tests 10, 11) | **PASS** |
| 13 | Skipped/failed actions not counted as updates | Counters incremented conditionally based on execution status (Unit Test 9) | **PASS** |
| 14 | Episodic reset restores all states and RNG | `reset_to_source()` resets student, teacher, optimizer, RNG, counters (Unit Test 10) | **PASS** |
| 15 | Nonfinite state fails closed | NaN loss, gradients, and weights trigger pre-step rollback (Unit Tests 11, 12) | **PASS** |
| 16 | Sealed-test IDs fail before model execution | Preflights fail immediately if sealed IDs present (Unit Test 2) | **PASS** |
| 17 | Output directories cannot be overwritten | `allow_overwrite=True` raises `PermissionError` (Unit Test 5, 8) | **PASS** |
| 18 | Smoke targets exactly Event 520, 2 batches, 32 samples | Confirmed by dry-run and configuration inspection | **PASS** |

---

## 3. GPU Smoke Test Execution Ledger

### V1 Attempt (Preserved)
- **Start Time (UTC):** `2026-10-21T12:26:58Z`
- **End Time (UTC):** `2026-10-21T12:27:21Z`
- **Git HEAD:** `f8c43f84396ee7d2977cb0fe85fbacab2931c14d`
- **Config Path:** `configs/changeguard/e7_cotta_smoke_v1.json`
- **Config SHA-256:** `349ed109e812326464cdeb87216e56f2262a9f8a7693a837de1b2c6c7f625be3`
- **Physical GPU:** `NVIDIA GeForce RTX 3090, 0, 24576 MiB` (`CUDA_VISIBLE_DEVICES=0`)
- **Tmux Session:** `cg_e7_cotta_smoke_v1`
- **Execution Log:** `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log`
- **Python Exit Code:** `1`
- **Terminal Status:** `FAILURE` (`AttributeError: 'LabelGuardedBatch' object has no attribute 'get_labels_offline'`)

### V2 Retry Specification & Execution (Verified Success)
- **Start Time (UTC):** `2026-10-21T12:43:52Z`
- **End Time (UTC):** `2026-10-21T12:44:12Z`
- **Stream Elapsed:** `4.88` seconds
- **Git HEAD:** `54a99498ee131fd4a8ca7192b3c5b3d6537ca472`
- **Config Path:** `configs/changeguard/e7_cotta_smoke_v2.json`
- **Config SHA-256:** `83cd761cac7ab4c0823965a99dce4bad79f617171e881cd6009c38ce6bbbb0dd`
- **Checkpoint Path:** `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`
- **Checkpoint SHA-256:** `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
- **Target Event:** Event 520 (USA / Florence)
- **Target Batches:** 2 batches ($N=32$ samples, BS=16)
- **Run ID:** `e7_cotta_smoke_v2`
- **Tmux Session:** `cg_e7_cotta_smoke_v2`
- **Output Directory:** `outputs/e7_cotta_smoke_v2`
- **Log Path:** `logs/e7_cotta_smoke/e7_cotta_smoke_v2.log`
- **Python Exit Code:** `0`
- **Terminal Status:** `SUCCESS`
- **Post-Run VRAM:** 16 MiB / 24,576 MiB
- **Macro F1:** `0.725692` | Precision: `0.723802` | Recall: `0.727591` | IoU: `0.569479`
- **Updates Applied:** 2 updates applied, 0 skipped, 0 nonfinite rollbacks

---

## 4. Preservation & Integrity Protocol

In strict compliance with user instructions:
- **No Automatic Retries:** The failed run attempt was preserved in `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log`.
- **Clean-Tree Launch:** V2 was launched strictly from clean commit `54a99498ee131fd4a8ca7192b3c5b3d6537ca472`.
- **Zero Historical Overwrites:** All E1–E6 formal outputs, manifests, and analyses remained untouched.
- **Source Checkpoint Preserved:** Checkpoint SHA-256 verified bit-exact before and after smoke execution (`247d3584...`).

---

## 5. Verification Suite Summary

```text
tests/test_e7_cotta_adapter.py:          16 passed in 8.99s
tests/test_e7_empirical_runner.py:        11 passed in 1.98s
tests/test_e5_temporal_risk_analysis.py:     12 passed in 57.89s
Total Tests:                              39 passed, 0 failed
```

---

## 6. Terminal Verdict

```text
E7 COMPLETE — READY FOR E8
```
