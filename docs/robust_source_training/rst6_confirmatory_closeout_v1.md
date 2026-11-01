# RST6 Confirmatory Execution and Integrity Closeout Report

## 1. Executive Summary

This document certifies the successful completion and full integrity verification of **RST6 -- Confirmatory Execution** for the ChangeGuard robust source-training route (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`).

All 12 intended confirmatory training runs have completed with terminal status `COMPLETED_SCIENTIFIC_RUN` and exit code `0`. Across the entire matrix, 108 event-level evaluation results (9 held-out events per run across 12 runs) were successfully generated without any runtime failure, data corruption, or target contamination.

### Execution Matrix Summary

| Property | Specification | Observed Status |
| :--- | :--- | :--- |
| Physical Hardware | 4x NVIDIA Tesla V100-PCIE-32GB (GPUs 0, 1, 2, 3) | 100% completed, GPUs now idle |
| Execution Architecture | 4 concurrent tmux queues (1 per GPU) | All 4 queues exited cleanly with code 0 |
| Planned Matrix Cells | 12 cells (2 recipes x 2 folds x 3 seeds) | 12/12 cells fully executed |
| Evaluated Event Rows | 108 event rows (12 runs x 9 events) | 108/108 rows recorded in event_metrics.json |
| Forbidden Events Accessed | 0 (strictly zero out of 16 target events) | PASS (verified zero access across all runs) |
| Train/Eval Independence | Complete disjointness (0 overlap) | PASS (verified for all 12 runs) |
| Checkpoint Integrity | Deterministic reload, valid SHA-256, no NaN/Inf | PASS (all 12 final checkpoints verified) |

---

## 2. Execution History and Recovery Lineage

Confirmatory execution followed the frozen recovery policy established under RST6 Option 1 (Fresh-run-only recovery, commit `8b0e3e4`):
1. **Initial Queue Launch (v1)**:
   - On 2026-10-27 15:46 UTC, four tmux queues were launched covering the 12 planned runs.
   - GPU 0 executed Fold A R0 (v1, seeds 0, 1, 2).
   - GPU 1 executed Fold A R2 (v1, seed 0).
   - GPU 2 executed Fold B R0 (v1, seeds 0, 1, 2).
   - GPU 3 executed Fold B R2 (v1, seed 0).
2. **Conservative Pause and Class Weights Audit**:
   - During early execution of seed 0, an apparent discrepancy in Fold B class weights prompted an immediate conservative pause of the R2 queues on GPUs 1 and 3 before any validation or confirmatory results were inspected.
   - The R0 queues on GPUs 0 and 2 were verified untouched and allowed to proceed to completion.
   - A rigorous audit confirmed that runtime class weights matched the mathematically computed population statistics (`rst6_r2_class_weights_audit_and_invalidation_v1.md`).
   - The two interrupted attempts (`rst6_r2_snunet_folda_s0_v1` and `rst6_r2_snunet_foldb_s0_v1`) were preserved permanently on disk as failed/interrupted runs (`INTERRUPTED_RUN`, exit code `1`) and formally invalidated from scientific analysis.
3. **Fresh Retry Authorization and Execution (v2)**:
   - Versioned v2 configs for R2 were created and authorized under commit `38abbb8`.
   - Fresh queues on GPU 1 and GPU 3 were launched from scratch (epoch 0) with identical scientific settings and seeds 0, 1, 2.
   - Both R2 queues completed all 3 seeds cleanly.

---

## 3. Physical Run Matrix Details

Below is the complete ledger of all 12 accepted confirmatory runs:

| Run ID | Recipe | Fold | Seed | Epochs | Steps | Wall-Clock Time (s) | Peak VRAM (MB) | Final Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `rst6_r0_snunet_folda_s0_v1` | R0 | Fold A | 0 | 16 | 6,656 | 10,193.7 | 4,911.2 | `3a597d4a43578668ceb457d4e9354ba54108c6402ac795fa1633db6dd10894be` |
| `rst6_r0_snunet_folda_s1_v1` | R0 | Fold A | 1 | 16 | 6,656 | 8,627.6 | 4,911.2 | `b1f16ffb5f7e7d8acb701fa3ddf8c7215182d5d58335b2475771ed3dace3d57e` |
| `rst6_r0_snunet_folda_s2_v1` | R0 | Fold A | 2 | 16 | 6,656 | 8,515.4 | 4,913.0 | `dae5927177cdf4d01114232e38364408a59f5dcd14544ead1af681ce816bac55` |
| `rst6_r2_snunet_folda_s0_v2` | R2 | Fold A | 0 | 16 | 6,656 | 7,487.5 | 4,911.2 | `e47f653362465ac78159bd50a4b502b12907e2e4ef6dfff0f2c1f3a976919a68` |
| `rst6_r2_snunet_folda_s1_v2` | R2 | Fold A | 1 | 16 | 6,656 | 7,582.3 | 4,911.2 | `edb331badabbd5113d7a9c035c7b30867f64dfd3f775c398d347b640d11ade55` |
| `rst6_r2_snunet_folda_s2_v2` | R2 | Fold A | 2 | 16 | 6,656 | 7,583.2 | 4,911.2 | `dba1734cc866ae124bad5543db0415a74f046ff5327c7a32c2c93d9a975d00e7` |
| `rst6_r0_snunet_foldb_s0_v1` | R0 | Fold B | 0 | 16 | 7,328 | 8,427.0 | 4,913.2 | `c15b43140f03123f91bb50fa168faa7fbcec5dbff52576d5af8b868f749ba6f9` |
| `rst6_r0_snunet_foldb_s1_v1` | R0 | Fold B | 1 | 16 | 7,328 | 9,129.0 | 4,913.2 | `d433a24e9c4d57f6779dffacaf0156ea99360245d57dd27d280b06e12c88610d` |
| `rst6_r0_snunet_foldb_s2_v1` | R0 | Fold B | 2 | 16 | 7,328 | 14,884.1 | 4,913.2 | `4224cce52a0d8784b571133407b3586e7471cec4e455ae39c7578ea8c30a7539` |
| `rst6_r2_snunet_foldb_s0_v2` | R2 | Fold B | 0 | 16 | 7,328 | 8,056.3 | 4,913.2 | `5fafad15282100b6f38c58a199d768797c81ecbafdf81c785d57f3392512b430` |
| `rst6_r2_snunet_foldb_s1_v2` | R2 | Fold B | 1 | 16 | 7,328 | 11,933.3 | 4,913.2 | `364543283f0ac8119d2ee8a9ade69a17cfb93b7d451c9c309c4cb5ab6baa6a6f` |
| `rst6_r2_snunet_foldb_s2_v2` | R2 | Fold B | 2 | 16 | 7,328 | 8,183.0 | 4,913.2 | `4b8e5551ce8c0c0b62e7c603a43f57c0608bb8ceadeaa58e59aa43524352b54d` |

### Invalidation Ledger of Preserved Attempts

| Run ID | Status | Exit Code | Reason for Invalidation | Disk Path |
| :--- | :---: | :---: | :--- | :--- |
| `rst6_r2_snunet_folda_s0_v1` | `INTERRUPTED_RUN` | 1 | Conservative pause during class weights audit | `outputs/robust_source_training/rst6/rst6_r2_snunet_folda_s0_v1` |
| `rst6_r2_snunet_foldb_s0_v1` | `INTERRUPTED_RUN` | 1 | Conservative pause during class weights audit | `outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s0_v1` |

Both directories remain read-only preserved on disk and are strictly excluded from downstream analysis.

---

## 4. Verification and Contract Compliance

1. **Optimizer Step Accounting**:
   - Fold A runs: 13,302 samples per epoch / microbatch size 8 = 1,663 microbatches (1,662 full of 8, 1 tail of 6). Accumulation factor = 4. Microbatches 0..1663 yield exactly 416 optimizer steps per epoch. Over 16 epochs: exactly 6,656 steps.
   - Fold B runs: 14,642 samples per epoch / microbatch size 8 = 1,831 microbatches (1,830 full of 8, 1 tail of 2). Accumulation factor = 4. Microbatches 0..1831 yield exactly 458 optimizer steps per epoch. Over 16 epochs: exactly 7,328 steps.
   - All observed steps match the theoretical calculations with zero deviation.
2. **Initial State Hash Invariance Across Recipes and Folds**:
   - Seed 0: `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1`
   - Seed 1: `e67529a5be6663210ab47f7af36c2196f3dad8c3bac31d76301595589b3b5a59`
   - Seed 2: `02f939a9ad05d5a08587c5383b8ed2ec136218350ad98f49f77ccfdfc0813891`
   - Initial model weights are bit-for-bit identical across R0 and R2 for every corresponding seed.
3. **Zero Target Contamination**:
   - The 16 forbidden historical target events `[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]` were never present in any training or evaluation allowlist. Zero target pixels were accessed.
4. **Disjointness and Generalization Isolation**:
   - Fold A train and eval sets are disjoint (18 train, 9 eval; intersection = empty).
   - Fold B train and eval sets are disjoint (18 train, 9 eval; intersection = empty).
   - Fold A and Fold B evaluation sets are mutually disjoint and together span all 18 evaluation events.
5. **Numerical and Checkpoint Health**:
   - All 12 `final_checkpoint.pt` files reloaded successfully in PyTorch.
   - Zero NaN or Inf values detected in any model parameter or buffer across all 12 checkpoints.
   - Checkpoint SHA-256 digests in `final_manifest.json` match on-disk file digests bit-identically.

---

## 5. Certification and Handoff

RST6 execution has satisfied all criteria of the Research Integrity Protocol and the Confirmatory Execution Plan. The output files are verified, permanently archived, and released for RST7 offline confirmatory analysis.
