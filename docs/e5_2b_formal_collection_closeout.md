# ChangeGuard Work Package E5.2B: Formal Temporal-Risk Collection Closeout

> [!IMPORTANT]
> **Formal Collection Status: COMPLETED & VERIFIED**
> - **Run Identifier**: `e5_temporal_risk_s0_oa_v1`
> - **Execution Commit**: `7ae4535361e04a91728b9105f3cb03f9444cb2be` (`feat(e5): freeze bounded-memory formal risk collection`)
> - **Worktree State**: `CLEAN` (`git_dirty: false` recorded in initial and final manifests)
> - **Status**: `COMPLETED_RUN` (atomic `completion_status.json`)
> - **Coverage**: Exactly 6 declared development events, 2,826 unique samples, 179 authoritative batch shards.
> - **Zero Sealed-Test Access**: Strictly enforced; sealed-test events quarantined with zero access.

---

## 1. Executive Summary & Verification Verdict

Work Package E5.2B executed the formal, end-to-end evidence collection for the primary seed-0 / order-A cell under the governing protocol [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) and contract [`docs/e5_preflight_and_contracts.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_preflight_and_contracts.md).

- **Source Checkpoint**: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`
- **Source Checkpoint SHA-256**: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` (Verified bit-exact)
- **Candidate Platform**: Internal `candidate_e2_conf90` ($\text{lr}=10^{-4}$, Adam, BN affine only, reset per event, `PRE_UPDATE` timing)
- **Physical Device**: Physical GPU 0 (`NVIDIA GeForce RTX 3090`, Bus ID `0000:19:00.0`, isolated via `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`)
- **Execution Wall Clock**: 1,053.37 seconds (17.56 minutes)
- **Peak GPU VRAM Allocation**: 18,311.79 MB (17.88 GB / 24,576 MB capacity, 74.5% peak utilization)
- **Disk Storage Footprint**: 6.2 GB (179 compressed batch shards and indexes; zero convenience aggregate)
- **Zero Sealed-Test Access**: Quarantined events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` strictly unaccessed
- **Historical Integrity**: Blocks E1–E4 outputs and `smoke_v1`–`smoke_v4` artifacts remain completely untouched

---

## 2. Event Partition & Sample Accounting

All six declared development events were evaluated in strict Order A sequence with exact sample counts, zero duplicate IDs, and zero missing items:

| Event ID | Geographic Domain | Expected Samples | Completed Samples | Completed Batches | Sample Sequence Match | `evidence_index.json` Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **279** | Spain / Ebro | 2,381 | 2,381 | 149 | Bit-exact match | Bit-exact match (149 shards) |
| **437** | UK / Severn | 169 | 169 | 11 | Bit-exact match | Bit-exact match (11 shards) |
| **520** | USA / Florence | 92 | 92 | 6 | Bit-exact match | Bit-exact match (6 shards) |
| **559** | USA / Midwest | 42 | 42 | 3 | Bit-exact match | Bit-exact match (3 shards) |
| **1111003** | Bangladesh / Ganges | 19 | 19 | 2 | Bit-exact match | Bit-exact match (2 shards) |
| **1111008** | Myanmar / Sagaing | 123 | 123 | 8 | Bit-exact match | Bit-exact match (8 shards) |
| **Total** | **6 Development Events** | **2,826** | **2,826** | **179** | **100.0% Bit-exact** | **179 Shards Verified** |

All 2,826 sample IDs across all 179 shards match the ordered sample lists recorded in [`initial_manifest.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/formal_s0_oa_v1/initial_manifest.json) bit-exact.

---

## 3. Authoritative Evidence Shards & Memory-Bounded Finalization

1. **Shard Primacy Enforced**:
   - Every batch flushed an atomic shard (`batch_000000.npz` through `batch_000148.npz` for Event 279, etc.) to disk.
   - Each shard contains 24 raw arrays per sample (384 arrays per 16-sample batch):
     - Sensory mask (`_sensory_mask`)
     - Model predictions across all 3 pair types for 3 contexts (`source`, `bn_control`, `candidate`)
     - Float16 probability tensors across all 3 pair types and strong augmentations
     - Step action, update status, and context references
   - Every single on-disk shard SHA-256 hash was recomputed and matched against the corresponding `evidence_index.json` entry with zero discrepancies.
2. **Convenience Aggregate Omitted for Bounded Memory**:
   - As established in E5.2A, `produce_convenience_aggregate` was explicitly configured as `false` (`aggregate_cache_policy: "disabled_for_bounded_memory"`).
   - Event 279 avoided loading >57,000 arrays into system RAM, executing with zero OOM events.
   - No `frozen_predictions.npz` files were written in any event directory.
3. **No Failure Artifacts**:
   - `failure.json` was not created in the run root or in any event directory.

---

## 4. Dual Timing & Pre-Label Sealing Invariants

1. **Pre-Label Sealing**:
   - In all 179 batches across all 6 events, evidence shards were physically flushed, closed, and confirmed on disk before `batch.mark_persisted()` and `batch.get_labels()` unlocked ground truth.
2. **Timing Disambiguation & Monotonicity**:
   - `prelabel_evidence_elapsed_ms`: Measured strictly from batch arrival through model forward passes, temporal probes, and shard generation immediately prior to persistence.
   - `batch_total_elapsed_ms`: Measured from batch arrival through shard write, candidate adaptation update, and offline evaluation.
   - **Timing Invariant**: In all 179 step records, `batch_total_elapsed_ms >= prelabel_evidence_elapsed_ms > 0` strictly holds.
   - All timing and telemetry values in [`e5_step_risk_summary.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/formal_s0_oa_v1/e5_step_risk_summary.csv) (179 rows) are strictly finite and non-negative.

---

## 5. Offline Confusion Matrix Conservation

For all six events, ground-truth evaluation executed strictly post-persistence:

| Event ID | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Valid Pixels ($N_{\text{valid}}$) | Conservation Check ($\Sigma = N_{\text{valid}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **279** | 621,289 | 2,363,739 | 247,408 | 112,966,811 | 116,199,247 | Verified Bit-Exact |
| **437** | 49,014 | 41,318 | 14,988 | 7,754,110 | 7,859,430 | Verified Bit-Exact |
| **520** | 227,160 | 92,858 | 102,083 | 3,033,147 | 3,455,248 | Verified Bit-Exact |
| **559** | 29,037 | 4,178 | 21,635 | 1,788,670 | 1,843,520 | Verified Bit-Exact |
| **1111003** | 3,149 | 13,652 | 3,812 | 794,877 | 815,490 | Verified Bit-Exact |
| **1111008** | 396,239 | 58,044 | 84,062 | 5,147,800 | 5,686,145 | Verified Bit-Exact |

*Note: In accordance with E5.2C1 instructions, risk-performance relationship interpretations and comparative conclusions are reserved for the formal analysis report.*

---

## 6. Formal Run Artifact Inventory & SHA-256 Checksums

### 6.1 Top-Level Artifacts
| Artifact Path | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/initial_manifest.json` | 242,275 | `06b90118b4e0640b1d6b760535c3b435f83cbe1c3ae255f828e53e2642008a33` | Initial preflight manifest with contracts |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/initial_provenance.json` | 382 | `e1206d4295ecbc95320f31869ecb1784cf65fe7ca8a284771c1b925936afa11b` | Checkpoint and environment provenance |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/cache_schema.json` | 3,442 | `a9206c45579bf24dc1816f677fd025a220014dcf034003ddbd5080bbf0fb845d` | Shard array caching contract schema |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/completion_status.json` | 303 | `35b7cc392995193139f6bd6b9653582225d7d0dedb72c1758ddb2c4e2c562321` | Terminal status (`COMPLETED_RUN`) |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/condition_summary.json` | 148 | `290199c52cd232805ae975fe18dfc28544e10bf744fd98ad572a4c6fea66cc8c` | Macro condition summary |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/run_manifest.json` | 3,700 | `da4a73d65e1b6a78975be24ac09081878e13399b236ad6604ae5a77e29adc7a9` | Final execution run manifest |
| `outputs/e5_temporal_evidence/formal_s0_oa_v1/e5_step_risk_summary.csv` | 149,583 | `5a8f53f4e0713ad722b2c0e6aad4fd6cf0180756060b43fdf474d8476681aebc` | 179-step tabular risk summary |
| `logs/e5_temporal_evidence/formal_s0_oa_v1.log` | 907 | `0135cacb880e940c1f9ef37878d1ac8d750a598bdab02db695ee9a9b27764e76` | Execution log output |

### 6.2 Per-Event Metadata Artifacts
| Event ID | `event_metrics.json` SHA-256 | `adaptation_progress.json` SHA-256 | `evidence_index.json` SHA-256 | `step_records.json` SHA-256 | Shard Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **279** | `4bc6af34...` | `0762195a...` | `da0e1208...` | `9155f593...` | 149 |
| **437** | `f91794b7...` | `c4561277...` | `6961cb95...` | `4be5fa95...` | 11 |
| **520** | `92f807f8...` | `4408b780...` | `bb9d2810...` | `93849ed0...` | 6 |
| **559** | `908ec870...` | `c303c1d8...` | `0192e705...` | `e45dfd4e...` | 3 |
| **1111003** | `c54a0f2e...` | `fc130399...` | `9f3ed6d6...` | `db254709...` | 2 |
| **1111008** | `8f536ba0...` | `aabb40f9...` | `1ba7144e...` | `839a2dd0...` | 8 |

---

## 7. Integrity Verdict

```text
================================================================================
FINAL VERDICT: E5.2B FORMAL COLLECTION PASS — INTEGRITY VERIFIED
================================================================================
```

