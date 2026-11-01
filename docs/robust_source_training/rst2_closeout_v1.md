# RST2 Closeout Report: Event-Disjoint Training and Evaluation Infrastructure

Date: 2026-10-26
Status: COMPLETED
Route: Robust Source Training (RST)
Authoritative Proposal: `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
Prelaunch Commit: `edf6b9e58f47785a867f107491398911fa401430`

---

## 1. Executive Summary

Work package **RST2 (Event-disjoint training and evaluation infrastructure)** is successfully completed.

All infrastructure requirements defined in `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md` and `docs/robust_source_training/rst2_event_disjoint_infrastructure_protocol_v1.md` have been implemented, unit tested across 32 criteria (22 test functions, 100% pass rate), and verified through an authorized end-to-end real-data infrastructure smoke run on GPU (`rst2_r0_smoke_snunet_foldc_s0_v1`).

The infrastructure enforces strict event-disjointness, fail-closed contract validations, quarantine isolation of the 16 forbidden historical target events, deterministic batch loading, bit-identical checkpoint-resume verification, strictly independent event-wise evaluation with zero-denominator safety, and a two-phase immutable provenance manifest lifecycle.

---

## 2. Infrastructure Deliverables

All deliverables were implemented as modular, additive components under the dedicated `robust_source_training` namespace:

| Component | Path | Purpose |
| :--- | :--- | :--- |
| **Event Contracts** | `src/robust_source_training/contracts.py` | Fail-closed validation of event allowlists, mutual exclusivity, fold-role matching, and forbidden event quarantine. |
| **Dataset & Loader** | `src/robust_source_training/dataset.py` | `RSTR0SourceDataset` with strict allowlist filtering, double-masking (`label[valid_mask == 0] = 3`), linear amplitude pairing, and deterministic dataloader. |
| **Independent Evaluator** | `src/robust_source_training/evaluation.py` | Event-by-event isolated evaluation, valid-pixel confusion counting, rate calculations with zero-denominator fallback, macro and pooled metrics. |
| **Checkpoint & State** | `src/robust_source_training/checkpoint.py` | Atomic serialization with full metadata and complete RNG states (Python, NumPy, PyTorch CPU & CUDA); bit-level state verification (`assert_states_equal`). |
| **Manifest Lifecycle** | `src/robust_source_training/manifest.py` | Two-phase manifest lifecycle: immutable `preflight_manifest.json` before compute, `final_manifest.json` referencing preflight hash upon completion, and atomic `terminal_status.json`. |
| **Execution Engine** | `src/robust_source_training/runner.py` | Unified execution engine managing training loop, gradient accumulation, validation, checkpointing, resume verification, evaluation, and manifests. |
| **Unified Runner Tool** | `tools/robust_source_training/run_rst2_infrastructure.py` | CLI entrypoint supporting `--validate_only`, `--dry_run`, `--execute`, and `--resume`. |
| **Smoke Configuration** | `configs/robust_source_training/rst2_r0_infrastructure_smoke_v1.json` | Frozen configuration for the authorized matched-R0 infrastructure smoke test. |
| **Unit Test Suite** | `tests/test_rst2_infrastructure.py` | 22 comprehensive test functions validating all 32 required infrastructure invariants. |

---

## 3. Unit Test Verification (32 Criteria)

The test suite in `tests/test_rst2_infrastructure.py` was executed with the project's Python environment (`/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python`). All 22 test functions passed without errors:

```text
Running 22 infrastructure tests...
  [PASS] test_1_train_eval_overlap_rejection
  [PASS] test_2_forbidden_event_rejection
  [PASS] test_3_unknown_event_rejection
  [PASS] test_4_missing_allowlist_rejection
  [PASS] test_5_duplicate_event_rejection
  [PASS] test_6_fold_role_mismatch_rejection
  [PASS] test_7_8_no_partition_cross_contamination
  [PASS] test_9_10_label_mapping_and_valid_masking
  [PASS] test_11_input_channel_pairing
  [PASS] test_12_13_augmentation_disabled_and_uniform_sampling
  [PASS] test_14_fixed_seed_sample_order_determinism
  [PASS] test_15_worker_initialization_reproducibility
  [PASS] test_16_event_wise_confusion_counts
  [PASS] test_17_18_19_macro_and_pooled_metrics
  [PASS] test_20_zero_denominator_behavior
  [PASS] test_21_22_manifest_lifecycle
  [PASS] test_23_24_25_26_checkpoint_reload_and_states
  [PASS] test_27_28_resume_mismatch_rejection
  [PASS] test_29_output_root_collision_rejection
  [PASS] test_30_durable_failure_status
  [PASS] test_31_zero_forbidden_event_access
  [PASS] test_32_no_historical_file_mutation

Test Summary: 22 passed, 0 failed out of 22 tests.
```

---

## 4. Real-Data Infrastructure Smoke Run Execution

The real-data smoke run was executed under strict infrastructure bounds and explicit authorization:

- **Run ID**: `rst2_r0_smoke_snunet_foldc_s0_v1`
- **Execution Class**: `INFRASTRUCTURE_SMOKE` (explicit non-scientific evidence)
- **Hardware Device**: `cuda:0` (NVIDIA GeForce RTX 3090, 24 GB VRAM)
- **Training Events**: `[273, 518]` (Group A 112 patches, Group B median 306 patches; 64 samples observed in smoke)
- **Evaluation Events**: `[130]` (Group C discovery target, 32 samples evaluated in smoke)
- **Smoke Bounds**: 1 epoch, 8 microbatches (effective batch size 32, micro-batch 8, grad-accum 4), 2 optimizer steps, 2 eval batches.
- **Observed Steps & Loss**:
  - Step 1 (microbatch 4): Loss = 8.9180 (CE = 16.9369, Dice = 0.8990)
  - Step 2 (microbatch 8): Loss = 1.0199 (CE = 1.4988, Dice = 0.5411)
- **Wall-clock Runtime**: 9.64 seconds
- **Peak VRAM**: 4910.5 MB
- **Checkpoint-Resume Verification**: The final model checkpoint was saved atomically, reloaded into a completely separate freshly initialized model, and verified bit-for-bit identical via `assert_states_equal` (verifying all parameters and buffers).
- **Independent Event Evaluation**:
  - Event 130: 32 patches evaluated (1,533,504 valid pixels).
  - Confusion counts: TP = 10,526, FP = 611, FN = 1,000,722, TN = 521,645.
  - Event Metrics: Precision = 0.945138, Recall = 0.010409, F1 = 0.020591, IoU = 0.010403.
- **Terminal Status**: `COMPLETED_INFRASTRUCTURE_SMOKE` (exit code 0).

---

## 5. Artifact Provenance and Hash Ledger

| File Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `configs/robust_source_training/rst2_r0_infrastructure_smoke_v1.json` | `be603940fa5dfd6b02c9d7aa25017595e62f3b983d43eb75d78a917668b672bc` | Authorized frozen smoke configuration |
| `docs/robust_source_training/rst2_event_disjoint_infrastructure_protocol_v1.md` | `f1562d49104679b5dd2aa6b2a69e6ace34b960fcbba9de16367845dbdd81392c` | Frozen RST2 infrastructure protocol |
| `src/robust_source_training/contracts.py` | `a48376c7230b2d0b1584e331d99073956ede2fd619d570b26e71f3110387937a` | Event contracts validation module |
| `src/robust_source_training/dataset.py` | `a44f59e45b376ee1f7f11d0eb44fc32f356565777eca0e5c124be63d4ba1242f` | R0 dataset and deterministic loader |
| `src/robust_source_training/evaluation.py` | `f0d23756888b6cca9d78e11dd71494fc9622dfaea8fd6a909aac0b7a71032fd9` | Independent event evaluator |
| `src/robust_source_training/checkpoint.py` | `f46125e0a308f742367846626c3a8e265dc29248d1bc90158c04fe7770b7c94d` | Checkpoint and resume verification |
| `src/robust_source_training/manifest.py` | `768bbe8353a1e77f8b32e780d69b7fb314f321c2762e2f0c389d32925bc17d77` | Two-phase manifest and status lifecycle |
| `src/robust_source_training/runner.py` | `d876adbb315d9ae99304213658a0c6a47188c7f6e96c36118ab069cd2469d6cc` | Unified RST execution engine |
| `tools/robust_source_training/run_rst2_infrastructure.py` | `7e04c284b5e698486eb68409cc7c9177529e4c278777ea4af5b9fd7cc0bf3884` | CLI runner entrypoint |
| `tests/test_rst2_infrastructure.py` | `97442974ed624911afca042ef9daa7a147d4d5d9abbb02bf53f95f0694609d10` | Unit test suite |
| `outputs/robust_source_training/rst2/rst2_r0_smoke_snunet_foldc_s0_v1/preflight_manifest.json` | `707b7c9daccc0c7d6b27916b9068f9fde80aab5cd2ca706682099fab5ec85aad` | Immutable preflight run manifest |
| `outputs/robust_source_training/rst2/rst2_r0_smoke_snunet_foldc_s0_v1/final_manifest.json` | `bafb42c54731592a5560d22930d147c1011f000dfab66903a03812e70c89cc1b` | Completed final run manifest |
| `outputs/robust_source_training/rst2/rst2_r0_smoke_snunet_foldc_s0_v1/terminal_status.json` | `a872de01dbdf1d3a33248bb37d2212efd7598064a224ca86097494c0e282b53c` | Atomic run terminal status |
| `outputs/robust_source_training/rst2/rst2_r0_smoke_snunet_foldc_s0_v1/event_metrics.json` | `b598433a5d3589c8863fbe9fcb74b107351f13564f08962efd67a471558f2e04` | Event-wise metrics output |
| `docs/robust_source_training/rst2_failed_attempt_preservation_addendum_v1.md` | Disclosed | Failed-attempt preservation disclosure addendum |

---

## 6. Research Integrity and Historical Boundary Compliance

1. **Quarantine Adherence**: Zero access was attempted or performed on any of the 16 forbidden target events:
   `[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`.
2. **Historical E11 File Invariance**: The two historical dirty files remain completely untouched and byte-identical:
   - `docs/e11_final_scientific_conclusions.md` SHA-256: `26832a93be8efd85ada522c39e9334a678c6b3860f3379efe479567d879b3f36`
   - `docs/e11_final_test_closeout.md` SHA-256: `2e0d460ce088a8a6883ed5cdfadb61da01d9baaa3bf2b28249efcd6bcfd87ff8`
3. **No Premature Implementation**: Components M1 (event-balanced sampling), M2 (flood-aware loss), and M3 (paired SAR augmentation) remain completely unimplemented and deferred to RST3 as strictly required.
4. **Attempt Preservation Disclosure**: As detailed in [`docs/robust_source_training/rst2_failed_attempt_preservation_addendum_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/robust_source_training/rst2_failed_attempt_preservation_addendum_v1.md), two preliminary diagnostic attempts were deleted during debugging prior to the final successful smoke. The technical validity of the infrastructure code, tests, and final successful smoke run is unaffected, but attempt preservation was imperfect in RST2 and strict versioned preservation is mandated for RST3 onward.

---

## 7. Next Stage Handoff

RST2 is completed and fully validated. The infrastructure is frozen, verified, and ready for:

> **RST3 - Implement the three method components (M1 event-balanced sampling, M2 flood-aware loss, M3 paired SAR augmentation)**
