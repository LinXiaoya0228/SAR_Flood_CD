# ChangeGuard Work Package E5.1B: Implementation Self-Audit & Real-Data GPU Smoke Closeout

> [!IMPORTANT]
> **Quadri-Execution Provenance Summary**:
> - **`smoke_v1`**: Preliminary, noncanonical, dirty-worktree infrastructure smoke (Base commit `3f30603d`, uncommitted runner modifications at execution). Retained as preliminary infrastructure evidence; not canonical acceptance.
> - **`smoke_v2`**: Canonical clean-checkpoint failed timing attempt (Commit `62b5ebe02890427b97483854a9d980b315d35691`, `fix(e5): enforce durable pre-label evidence persistence`). Failed fail-closed via `UnboundLocalError: local variable 'b_time_ms' referenced before assignment`.
> - **`smoke_v3`**: Canonical clean-checkpoint failed finalization attempt (Commit `a94704c70f626ce272033d1701b1abc23d1ee302`, `fix(e5): initialize pre-label shard timing before persistence`). Failed fail-closed at event completion via `NameError: name 'event_samples' is not defined in tools/run_e5_risk_scoring.py at line 855`.
> - **`smoke_v4`**: Canonical clean-checkpoint accepted run (Commit `8ca4c28b04be28451f6649f1536570a9b0708c49`, `fix(e5): complete event finalization and lifecycle coverage`). Executed cleanly to completion in 18.11s; verified bit-exact against all scientific and integrity contracts.

---

## 1. Executive Summary

- **Work Package**: E5.1B (Bounded Real-Data GPU Smoke & Runner Self-Audit)
- **Target Git Commit**: `8ca4c28b04be28451f6649f1536570a9b0708c49` (`fix(e5): complete event finalization and lifecycle coverage`)
- **Execution Worktree State**: `CLEAN` (`git_dirty: false` recorded in initial manifest)
- **Source Checkpoint Path**: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`
- **Source Checkpoint SHA-256**: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` (Bit-exact verified)
- **Execution Target & Isolation**: Physical GPU 0 (`NVIDIA GeForce RTX 3090`, Bus ID `0000:19:00.0`, isolated via `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`)
- **Scope Boundary**: Bounded strictly to Event 520 (first 32 deterministic samples across 2 batches of 16); zero sealed-test data accessed; no full 6-event scientific collection launched.
- **Canonical Execution Verdict**: `E5.1 CANONICAL PASS — READY FOR FINAL CLOSEOUT CHECKPOINT`

---

## 2. Scientific Inputs & Preflight Integrity Verification

Prior to launching `smoke_v4`, all 37 unit and integration tests in `tests/test_e5_risk_contracts.py` passed cleanly (11.685s).
All six mandatory scientific inputs were verified against Git HEAD `8ca4c28b04be28451f6649f1536570a9b0708c49`:

| Scientific Input File | SHA-256 Checksum | Clean Status |
| :--- | :--- | :--- |
| `configs/changeguard/e5_temporal_risk_v1.json` | `80606f0018a773a82cb66266a4b79db3ca88dc2c3cc1f3832f5708b57b270d23` | Clean |
| `docs/e5_preflight_and_contracts.md` | `531cbefab7de62178c6751db4e5b243cbb4d0f220a4d9b4633414d4b9586ae91` | Clean |
| `src/changeguard/e5_risk_probes.py` | `9d4a47e2895f641785cf9c2ddc727a8bbbde198ca5a63b195728b0c02d54946b` | Clean |
| `src/changeguard/recorder.py` | `c1d5861f389d350f16278d01d9c654abb34289567efc596b7b539a568156d39d` | Clean |
| `src/data/evaluation_dataset.py` | `704d86ec67aa58d0084f8e5efead5ca1370c9b403f7982e502db9cc0c0d45567` | Clean |
| `tools/run_e5_risk_scoring.py` | `5e5a5e8680a80d245f7e3f7cd38efba39cbe25730bb89dac9ce9a47a928e0c3b` | Clean |

---

## 3. Comparative Record: smoke_v1 vs smoke_v2 vs smoke_v3 vs smoke_v4

| Dimension | `smoke_v1` | `smoke_v2` | `smoke_v3` | `smoke_v4` |
| :--- | :--- | :--- | :--- | :--- |
| **Git Commit** | `3f30603d` | `62b5ebe0` | `a94704c7` | `8ca4c28b` |
| **Worktree State** | `DIRTY` (uncommitted runner) | `CLEAN` | `CLEAN` (`git_dirty: false`) | `CLEAN` (`git_dirty: false`) |
| **Target Directory** | `outputs/.../smoke_v1/` | `outputs/.../smoke_v2/` | `outputs/.../smoke_v3/` | `outputs/.../smoke_v4/` |
| **Log Path** | `logs/.../smoke_v1.log` | `logs/.../smoke_v2.log` | `logs/.../smoke_v3.log` | `logs/.../smoke_v4.log` |
| **Tmux Session** | `cg_e5_1b_smoke_v1` | `cg_e5_1b_smoke_v2` | `cg_e5_1b_smoke_v3` | `cg_e5_1b_smoke_v4` |
| **Physical Isolation** | GPU 0 (`RTX 3090`) | GPU 0 (`RTX 3090`) | GPU 0 (`RTX 3090`) | GPU 0 (`RTX 3090`) |
| **Contexts Handled** | 2 contexts | 3 contexts | 3 contexts (`source`, `bn_control`, `candidate`) | 3 contexts (`source`, `bn_control`, `candidate`) |
| **Persistence Model** | Monolithic `frozen_predictions.npz` | Batch-atomic shards in `evidence_shards/` | Batch-atomic shards in `evidence_shards/` | Batch-atomic shards in `evidence_shards/` |
| **Batch Shards Produced** | 0 (monolithic) | 0 (crashed before shard 0) | 2 (`batch_000000.npz`, `batch_000001.npz`) | 2 (`batch_000000.npz`, `batch_000001.npz`) |
| **Pre-label Timing** | Not isolated | Unbound variable crash | Durably isolated (`2811.89ms`, `2391.02ms`) | Durably isolated (`2738.33ms`, `2389.55ms`) |
| **Batch Total Timing** | Total run only | Unbound variable crash | Durably recorded (`6124.93ms`, `5407.31ms`) | Durably recorded (`5979.00ms`, `5450.06ms`) |
| **Event Finalization** | Ad hoc | Not reached | Crashed (`NameError: event_samples`) | Verified contract gate (`COMPLETED_RUN`) |
| **Execution Result** | Completed (`COMPLETED_SMOKE`) | Failed (`UnboundLocalError: b_time_ms`) | Failed (`NameError: event_samples`) | Complete Pass (`COMPLETED_RUN`, exit 0) |
| **Scientific Role** | Preliminary infrastructure | Clean failure record | Clean failure record | Canonical acceptance run |

---

## 4. smoke_v4 Canonical Execution & Scientific Verification

### 4.1 Launch Command & Runtime Environment
```bash
mkdir -p logs/e5_temporal_evidence
tmux new-session -d -s cg_e5_1b_smoke_v4 \
  "cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard && \
   CUDA_VISIBLE_DEVICES=0 \
   PYTHONPATH=. \
   /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python \
   tools/run_e5_risk_scoring.py --smoke --device cuda \
   > logs/e5_temporal_evidence/smoke_v4.log 2>&1"
```

- **Runtime**: Python 3.10.20, PyTorch 2.13.0+cu130, CUDA 13.0, Driver 580.173.02.
- **Duration**: Completed in 18.11s with exit code 0.

### 4.2 Verified Scientific Behaviors & Acceptance Criteria

1. **Deterministic Sample Order & Contract Gates**:
   - Initial manifest recorded 32 deterministic sample IDs across 2 batches of 16 for Event 520 (`deterministic_sample_ids_hash: 26d91080...`).
   - Immutable `event_contracts` verified pre-execution and passed through to event finalization without mutation.
   - All 32 samples processed in exact order with zero duplicate IDs.

2. **Authoritative Evidence Sharding & Pre-Label Sealing**:
   - Two batch shards were durably written and closed on disk before `batch.mark_persisted()` and before `batch.get_labels()` unlocked ground truth:
     - `batch_000000.npz`: 16 samples, 384 raw arrays (24 per sample), 39,443,887 bytes, SHA-256 `0a91844b60b9f12db374101297793b45213cb1045202187bd12f71d92b3455e3`.
     - `batch_000001.npz`: 16 samples, 384 raw arrays (24 per sample), 38,501,215 bytes, SHA-256 `523fe4c5b4e22725a2b7a9fe90ad23e634946c67ee67beafd8d80b45e3d21e67`.
   - Shards contain full sensory masks, predictions, probabilities (float16), null pairs (pre2 and post), and context baselines for `source`, `bn_control`, and `candidate`.
   - On-disk shard hashes match `evidence_index.json` bit-exact.

3. **Temporal Timing Disambiguation & Positive Monotonic Invariants**:
   - Step 0:
     - `prelabel_evidence_elapsed_ms = 2738.33 ms`
     - `batch_total_elapsed_ms = 5979.00 ms`
   - Step 1:
     - `prelabel_evidence_elapsed_ms = 2389.55 ms`
     - `batch_total_elapsed_ms = 5450.06 ms`
   - Invariant holds strictly: `batch_total_elapsed_ms >= prelabel_evidence_elapsed_ms > 0` for all steps.

4. **State-Neutrality & Declared Candidate Adaptation**:
   - **Source Model**: Parameters frozen throughout execution; parameter hash invariant.
   - **Matched BN Control Model**: Updated running statistics on batch size 16 without affine parameter updates; BN weight displacement = `0.0`, BN bias displacement = `0.0`.
   - **Candidate Model**:
     - Step 0 (pre-update): BN weight displacement = `0.0`, BN bias displacement = `0.0`.
     - Step 1 (post-update): BN weight displacement = `0.005294`, BN bias displacement = `0.005215`.
     - Training dynamics: Step 0 loss `0.01353`, grad norm `0.01918`; Step 1 loss `0.01240`, grad norm `0.02024`.

5. **Shard Metric Recomputation Fidelity**:
   - Recomputation of $s_{\text{null}}$ directly from raw shard float16 arrays matches online recorded metrics within $1.33 \times 10^{-6}$ (exact float16 roundoff tolerance):
     - Step 0: online `0.00293674`, recomputed `0.00293541`, diff `0.00000133`.
     - Step 1: online `0.00238781`, recomputed `0.00238800`, diff `0.00000019`.

6. **Offline Confusion Matrix Conservation & Precision/Recall/F1 Arithmetic**:
   - Event 520 evaluation metrics:
     - $\text{TP} = 92,833$
     - $\text{FP} = 35,864$
     - $\text{FN} = 35,922$
     - $\text{TN} = 1,023,978$
     - $\text{Valid Pixels} = \text{TP} + \text{FP} + \text{FN} + \text{TN} = 1,188,597$ (exact conservation invariant satisfied).
   - Flood Metrics:
     - $\text{Precision} = \frac{92833}{92833 + 35864} = 0.721330$
     - $\text{Recall} = \frac{92833}{92833 + 35922} = 0.721005$
     - $\text{Flood F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.721167$
     - Macro Flood F1 matches run manifest bit-exact (`0.7211674409210261`).

7. **Resource Footprint & Peak Memory**:
   - Peak allocated VRAM: `18,309.32 MB` (74.5% of 24 GB RTX 3090 capacity).
   - Storage consumed by `smoke_v4`: `155.90 MB` total (77.95 MB in evidence shards, 77.95 MB in final frozen predictions).
   - Zero failure record: `failure.json` does not exist in `smoke_v4`.
   - Atomic completion: `completion_status.json` durably written with `status: "COMPLETED_RUN"`.

---

## 5. Artifact Inventory & Checksums

### 5.1 `smoke_v1` Artifacts (Preliminary Dirty-Worktree Smoke, Preserved Untouched)
| Artifact Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `outputs/e5_temporal_evidence/smoke_v1/initial_manifest.json` | 1,173 | `05b2d4dfcfebb7a7389583c4210b0c8ed58a792f9bb46dadd230b15bd46e2d8e` |
| `outputs/e5_temporal_evidence/smoke_v1/initial_provenance.json` | 369 | `e0947c25d6d0fb1c6d40cccde5c6c7576079e5708f39dee0f8924e3a4c3f13c2` |
| `outputs/e5_temporal_evidence/smoke_v1/cache_schema.json` | 1,407 | `23a60f951c020b14fa4736f5f8eae80026ecb20063a2c69262b0a8a3cd132a68` |
| `outputs/e5_temporal_evidence/smoke_v1/run_manifest.json` | 1,569 | `98c0ae1403db13dac80ff18c44e2356b18bca835b20c44876e86374a63e94351` |
| `outputs/e5_temporal_evidence/smoke_v1/e5_step_risk_summary.csv` | 973 | `6c2f347fb05a235b534c0a33b8904a6e4b1c7f53124773e392698598981196f4` |
| `outputs/e5_temporal_evidence/smoke_v1/event_520/step_records.json` | 4,661 | `2082c0139c016fdc85d20d948f9a1da694c1e89d2d0dc439600edb559d1d5444` |
| `outputs/e5_temporal_evidence/smoke_v1/event_520/event_metrics.json` | 223 | `74a692660b7fc5d96ce712e798823fa31a6ab6c02e8e99fd76a10cfaadfd1d66` |
| `outputs/e5_temporal_evidence/smoke_v1/event_520/adaptation_progress.json` | 1,409 | `55cf607fc2ca3fe41c4ca59f0fff2d3700896415a067dacceea68ac1aa74e10c` |
| `outputs/e5_temporal_evidence/smoke_v1/event_520/frozen_predictions.npz` | 5,764,854 | `2343fb2af2c7d2e6a8720ad82c2fb728d7aa2c16b1e08372451a5c2b706a75b2` |
| `logs/e5_temporal_evidence/smoke_v1.log` | 268 | `c4482e557c306942b82ce2d97e5f5279b3a5650760339b49eda7393225325ee3` |

### 5.2 `smoke_v2` Artifacts (Canonical Timing Failure, Preserved Untouched)
| Artifact Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `outputs/e5_temporal_evidence/smoke_v2/initial_manifest.json` | 4,562 | `924bfe4110991c1356ab4a03037d633d55d41509628d2a3369b52bf799b3e879` |
| `outputs/e5_temporal_evidence/smoke_v2/initial_provenance.json` | 369 | `e429ab8e3693ee203d1c3c2cfeae4c9be030f8e67d0b0db52003b726d982f2a0` |
| `outputs/e5_temporal_evidence/smoke_v2/cache_schema.json` | 2,948 | `033b08a45c7992cf720025ba2a0ca0326f370e1a79a3865970223617dd29a0b3` |
| `outputs/e5_temporal_evidence/smoke_v2/event_520/adaptation_progress.json` | 128 | `bf09c4e904f995b9c3f30c2436b23dcdd0694e623ad631fc0dbd553fddf2da6c` |
| `outputs/e5_temporal_evidence/smoke_v2/event_520/evidence_index.json` | 180 | `873dc7c6c8592b1745e3edc8261586270e4913205032c239995e0174f75ae55a` |
| `outputs/e5_temporal_evidence/smoke_v2/event_520/failure.json` | 240 | `05c89adad332dcfaf3328db10c9cb1c8da2675bdf446f05a815c0feac1e0bca3` |
| `logs/e5_temporal_evidence/smoke_v2.log` | 436 | `1bdbe2f2e623df8bea156b2fca4c81f030f092b1ad80bd6deb7fb41de6536e7d` |

### 5.3 `smoke_v3` Artifacts (Canonical Finalization Failure, Preserved Untouched)
| Artifact Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `outputs/e5_temporal_evidence/smoke_v3/initial_manifest.json` | 4,563 | `c308878923cd83bc19327c82ad28059f1337bf9b0a293f9d716fc577785dea73` |
| `outputs/e5_temporal_evidence/smoke_v3/initial_provenance.json` | 369 | `75bdd55679e3e4e187f7cc7350b3ade3fa8456c34a2e90bee401a41dbed4fa12` |
| `outputs/e5_temporal_evidence/smoke_v3/cache_schema.json` | 3,442 | `a9206c45579bf24dc1816f677fd025a220014dcf034003ddbd5080bbf0fb845d` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/adaptation_progress.json` | 1,411 | `5c43067f5a20e89bb4a604c579c20be141ce9d5ea2438e573d8223d203a081e4` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/evidence_index.json` | 149,349 | `333d5752837b51e09c6fdaa1706363c39e99ad47d63924ef48a057f13b20ef22` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/failure.json` | 220 | `9aadc6f2464d1d6c26e3a4d9a80d7a11710041e35fc8b0e97d11771a8e2be444` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/step_records.json` | 7,545 | `44ab66e798d38faa7905dfe7aaa9fa7b03d3bbdb4d41ab0d123f58b3ceec8c2b` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/evidence_shards/batch_000000.npz` | 39,443,887 | `0a91844b60b9f12db374101297793b45213cb1045202187bd12f71d92b3455e3` |
| `outputs/e5_temporal_evidence/smoke_v3/event_520/evidence_shards/batch_000001.npz` | 38,501,215 | `523fe4c5b4e22725a2b7a9fe90ad23e634946c67ee67beafd8d80b45e3d21e67` |
| `logs/e5_temporal_evidence/smoke_v3.log` | 448 | `2e628a6156e95918203fb90837d2628939ac18c2bebd9e9f0124bed533a91567` |

### 5.4 `smoke_v4` Artifacts (Canonical Accepted Run Artifacts)
| Artifact Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `outputs/e5_temporal_evidence/smoke_v4/initial_manifest.json` | 6,213 | `11eea21eb19230f3ff1e7bd39f75817b3503234f04cb0d0bc0e7b8e5d830a9ce` |
| `outputs/e5_temporal_evidence/smoke_v4/initial_provenance.json` | 369 | `e7e5edfdc28136a863dc51fd44a650b25341a0fed574440b1ee51f31708161b0` |
| `outputs/e5_temporal_evidence/smoke_v4/cache_schema.json` | 3,442 | `a9206c45579bf24dc1816f677fd025a220014dcf034003ddbd5080bbf0fb845d` |
| `outputs/e5_temporal_evidence/smoke_v4/condition_summary.json` | 146 | `db7562574642a848c1bd59415a4e8bfa657c7751515dd21dd68da6dbfde85f5b` |
| `outputs/e5_temporal_evidence/smoke_v4/run_manifest.json` | 2,274 | `947eb175217b4cdf7d836fc9b217d77ceedcbb7b41ecc7c532c3b05602f1c811` |
| `outputs/e5_temporal_evidence/smoke_v4/completion_status.json` | 297 | `18b713580f0718523c54f75ec25e21d099b0c5aef89ae84612e546fa98d1a01e` |
| `outputs/e5_temporal_evidence/smoke_v4/e5_step_risk_summary.csv` | 2,572 | `216d4f3f211fdf8ab6a89801b51eac6a80541ef6e97f4d6936e13ecad38dfa03` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/event_metrics.json` | 223 | `74a692660b7fc5d96ce712e798823fa31a6ab6c02e8e99fd76a10cfaadfd1d66` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/adaptation_progress.json` | 1,409 | `fe3a9ff173ef70fb2c236b86247644539f27b34b049a019b2384b9e74b7ebb06` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/evidence_index.json` | 149,349 | `953692a6419b7a83cc7c34c431a810c5bc435c4f9e23cd7ff631aa20940746ad` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/step_records.json` | 7,543 | `96a1f559a5eaaa569576969a85ee8b112edea908cda9621c80759cebfb38300f` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/evidence_shards/batch_000000.npz` | 39,443,887 | `0a91844b60b9f12db374101297793b45213cb1045202187bd12f71d92b3455e3` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/evidence_shards/batch_000001.npz` | 38,501,215 | `523fe4c5b4e22725a2b7a9fe90ad23e634946c67ee67beafd8d80b45e3d21e67` |
| `outputs/e5_temporal_evidence/smoke_v4/event_520/frozen_predictions.npz` | 77,945,080 | `48d4e12c0624b7b34b8ea3a47f1fc882e900e205377c7c3e60a0b8e47b6d02d9` |
| `logs/e5_temporal_evidence/smoke_v4.log` | 268 | `3b2e53c60f4e61cd47f8eb0660867fd1d28f2d7bea1a00d7ec6d762cdddfa17b` |

---

## 6. Closeout Verdict & Next Scientific Transition Gate

Every scientific, architectural, temporal, and governance contract specified in `docs/MASTER_EXECUTION_PLAN_V3.md` and `docs/e5_preflight_and_contracts.md` has been verified end-to-end against physical real-data GPU execution without compromise.

- **Zero Sealed-Test Access**: Verified.
- **Fail-Closed Gatekeeper Provenance**: Verified across 4 independent executions.
- **Pre-Label Shard Durability**: Verified.
- **Timing Monotonicity & Disambiguation**: Verified.
- **State Neutrality & Invariance**: Verified.
- **Recomputation & Conservation Invariants**: Verified.

```text
================================================================================
FINAL VERDICT: E5.1 CANONICAL PASS — READY FOR FINAL CLOSEOUT CHECKPOINT
================================================================================
```
