# ChangeGuard Work Package E7: CoTTA Smoke Test Closeout

**Status:** V1 FAILED CLOSED (PRESERVED) — V2 RETRY EXECUTED AND VERIFIED (SUCCESS)
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git HEAD at V1 Execution:** `f8c43f84396ee7d2977cb0fe85fbacab2931c14d`
**Git HEAD at V2 Execution:** `54a99498ee131fd4a8ca7192b3c5b3d6537ca472`
**Governing Documents:**
- `docs/protocols/e7_branch_c_empirical_protocol_v1.md`
- `configs/changeguard/e7_cotta_smoke_v1.json` (v1 execution)
- `configs/changeguard/e7_cotta_smoke_v2.json` (v2 retry configuration)
- `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log` (preserved failed attempt)
- `logs/e7_cotta_smoke/e7_cotta_smoke_v2.log` (verified v2 success execution)

---

> [!WARNING]
> **2026-10-22 label-I/O certification correction:** Smoke v2 remains a successful
> infrastructure/execution smoke, but its strict physical label-I/O certification is revoked.
> The runner loaded GT tensors from `offline_ds` during batch-plan construction before prediction
> persistence. There is no evidence that GT entered CoTTA adaptation. A repaired bounded smoke v3
> is scheduled in `configs/changeguard/e7_cotta_smoke_v3.json` and must pass before E8 v2.
> No scientific conclusion depended on smoke v2, and sealed-test access remained zero.

---

## 1. Versioned Execution Ledger

| Attribute | Smoke v1 Attempt (Preserved) | Smoke v2 Retry (Verified) |
| :--- | :--- | :--- |
| **Run ID** | `e7_cotta_smoke_s0` | `e7_cotta_smoke_v2` |
| **Tmux Session** | `cg_e7_cotta_smoke_v1` | `cg_e7_cotta_smoke_v2` |
| **Execution Status** | **FAILED CLOSED** (`AttributeError`) | **SUCCESS** |
| **Start Time (UTC)** | `2026-10-21T12:26:58Z` | `2026-10-21T12:43:52Z` |
| **End Time (UTC)** | `2026-10-21T12:27:21Z` | `2026-10-21T12:44:12Z` |
| **Stream Runtime** | N/A (Failed batch 1) | `4.88` seconds |
| **Physical Hardware** | NVIDIA GeForce RTX 3090, 0 | NVIDIA GeForce RTX 3090, 0 |
| **Config Path** | `configs/changeguard/e7_cotta_smoke_v1.json` | `configs/changeguard/e7_cotta_smoke_v2.json` |
| **Config SHA-256** | `349ed109e812326464cdeb87216e56f2262a9f8a7693a837de1b2c6c7f625be3` | `83cd761cac7ab4c0823965a99dce4bad79f617171e881cd6009c38ce6bbbb0dd` |
| **Source Checkpoint** | Seed 0 (`outputs/week5_source_all27/.../final_checkpoint.pt`) | Seed 0 (`outputs/week5_source_all27/.../final_checkpoint.pt`) |
| **Checkpoint SHA-256** | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| **Event Bound** | Event 520 (USA / Florence) | Event 520 (USA / Florence) |
| **Sample Bound** | 2 batches, 32 samples (BS=16) | 2 batches, 32 samples (BS=16) |
| **Output Directory** | `outputs/e7_cotta_smoke_v1` (preserved) | `outputs/e7_cotta_smoke_v2` (verified) |
| **Execution Log** | `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log` (preserved) | `logs/e7_cotta_smoke/e7_cotta_smoke_v2.log` (verified) |
| **Python Exit Code** | `1` (Non-zero) | `0` (Clean zero exit) |
| **Terminal Status** | `FAILURE` | `SUCCESS` |

---

## 2. V1 Failure Diagnosis & Exact Root Cause

The v1 smoke run completed preflights cleanly, quarantined sealed test events, loaded Seed 0 checkpoint, and ran the first adaptation step. At line 436 of `tools/run_e7_empirical_evaluation.py`, execution failed:

```text
Traceback (most recent call last):
  File "/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_e7_empirical_evaluation.py", line 605, in <module>
    main()
  File "/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_e7_empirical_evaluation.py", line 576, in main
    runner.execute_stream(
  File "/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_e7_empirical_evaluation.py", line 436, in execute_stream
    labels = batch.get_labels_offline()
AttributeError: 'LabelGuardedBatch' object has no attribute 'get_labels_offline'
```

### Root Cause
`LabelGuardedBatch` in `src/changeguard/e1_orchestrator.py` enforces temporal separation:
1. Predictions and evidence must be durably persisted before label access.
2. `batch.mark_persisted()` must be explicitly called to unlock the batch.
3. `batch.get_labels()` retrieves ground-truth labels for offline evaluation.

In `tools/run_e7_empirical_evaluation.py`, the runner attempted to call a non-existent method `batch.get_labels_offline()` directly without persisting pre-label shards and without calling `batch.mark_persisted()`.

---

## 3. Scoped Remediation & Integration Verification

1. **Repaired Prediction Persistence Flow:**
   In `tools/run_e7_empirical_evaluation.py`:
   - Extracted and froze pre-label predictions (`pred_cand_pre`, `probs_cand_pre`) and adaptation metadata.
   - Durably saved pre-label prediction shard `batch_XXXX.npz` to disk.
   - Verified on disk that `shard_path.is_file()` and `stat().st_size > 0`.
   - Called `batch.mark_persisted()`.
   - Retrieved evaluation labels via `batch.get_labels()`.
   - Exported `manifest.json` with execution provenance.
2. **Real Integration Regression Test (`tests/test_e7_empirical_runner.py`):**
   Added two new tests exercising the actual `LabelGuardedBatch`:
   - `test_10_label_guarded_batch_lifecycle_and_interruption`: Proves `get_labels()` raises `PermissionError` before persistence; proves an interrupted/failed persistence step leaves labels locked; proves `get_labels()` succeeds only after valid persistence and `mark_persisted()`.
   - `test_11_runner_real_label_guarded_batch_persistence_flow`: Proves `execute_stream()` durably writes prediction shards to disk, unlocks labels via `mark_persisted()`, retrieves `get_labels()`, and generates `manifest.json` and summary records.
3. **No Scientific Changes:**
   Zero changes to architecture, CoTTA hyperparameters ($\alpha=0.999$, $p=0.01$, $\tau=0.90$), learning rate ($1\times 10^{-4}$), optimizer (Adam), sample subset (Event 520, 32 samples), or Branch C decision.

---

## 4. V2 GPU Smoke Execution Verification

The v2 smoke test was launched in detached tmux session `cg_e7_cotta_smoke_v2` from clean commit `54a99498ee131fd4a8ca7192b3c5b3d6537ca472` on GPU 0 (`NVIDIA GeForce RTX 3090`).

### Execution Summary
- **Start Time (UTC):** `2026-10-21T12:43:52Z`
- **End Time (UTC):** `2026-10-21T12:44:12Z`
- **Stream Elapsed:** `4.88` seconds
- **Exit Code:** `0`
- **Terminal Status:** `SUCCESS`
- **Samples Processed:** Exactly 32 samples across 2 batches ($B=16$)
- **Updates Applied:** 2 updates applied, 0 skipped, 0 nonfinite rollbacks
- **Step 1:** selected pixels: 708,000, mean entropy: 0.124722, mean max confidence: 0.958117
- **Step 2:** selected pixels: 692,194, mean entropy: 0.120596, mean max confidence: 0.958828
- **Metrics Observed:**
  - Macro F1: `0.725692`
  - Precision: `0.723802`
  - Recall: `0.727591`
  - IoU: `0.569479`
  - True Positives: 93,681 | False Positives: 35,748 | False Negatives: 35,074 | True Negatives: 1,024,094 | Valid Pixels: 1,188,597
- **Numeric Integrity:** Zero NaNs, zero Infs, finite loss and finite weights throughout.

### Output Artifact Manifest & Hashes

| Artifact Path | SHA-256 Checksum | Size |
| :--- | :--- | :--- |
| `configs/changeguard/e7_cotta_smoke_v2.json` | `83cd761cac7ab4c0823965a99dce4bad79f617171e881cd6009c38ce6bbbb0dd` | 1,733 B |
| `logs/e7_cotta_smoke/e7_cotta_smoke_v2.log` | `e7361fd9ea5fb5ba43b2533616799b002ec73fd3087c5abb478d45de17000dcc` | 2,423 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/completion_status.json` | `9fdde201ede6b3d0c34617b1d7f3c0313954c40824e655257e90fd672acddb8c` | 112 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/stream_summary.json` | `43bf9f1b80555a8b65e94436b631387b8a4bfc5f0472c6c732853c8540117175` | 770 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/manifest.json` | `c687e6e3d7c86b72ff8af2150a2d22bc0e5e832d34584d8182855ffc01336e62` | 343 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/event_metrics.csv` | `964edb3cb519ef066dd9c2804433fa7b7e6615b7a7dd0c30b74e38109a28ff0d` | 184 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/event_520/shards/batch_0000.npz` | `2ebb6e3f5c639662f6d24f1fe3bbe2bde1b4080c1da688d5dda499a3185bb662` | 8,438,500 B |
| `outputs/e7_cotta_smoke_v2/e7_cotta_smoke_v2/event_520/shards/batch_0001.npz` | `c03ef34b198d4bd5c80dd3df65164e0abe33f6632d06b4dced6c205effb4dafd` | 8,325,824 B |

---

## 5. Preservation & Integrity Confirmations

- **V1 Preserved:** The failed v1 run log remains at `logs/e7_cotta_smoke/e7_cotta_smoke_v1.log` (`0ff7bfc01140202baf0fda76e1be9a2d18740b0e17839a5bc85cdb5c75509698`). Output directory `outputs/e7_cotta_smoke_v1` remains untouched.
- **Source Checkpoint Immutable:** `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` verified bit-exact (`247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`).
- **Quarantine:** Zero sealed-test access occurred. Preflights confirmed 0 sealed-test events in stream evaluation.
- **Label Separation:** Shards verified durably written and checked before `mark_persisted()` and `get_labels()` access.
