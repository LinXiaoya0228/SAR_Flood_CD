# ChangeGuard Work Package E11: Final Test Rehearsal Closeout

**Document Identifier:** `docs/e11_final_test_rehearsal_closeout.md`  
**Execution Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E11 (Section 16.3)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `REHEARSAL_SUCCESSFUL — GATE_T_REHEARSAL_VERIFIED`  

---

## 1. Executive Summary

In preparation for Gate T and eventual sealed-test release, the complete execution and analysis pipeline for Work Package E11 was rehearsed on permitted development events (`[520, 559]`). 

The rehearsal validated:
1. **Config & Checkpoint Cryptographic Provenance:** 100% bit-exact SHA-256 match across all 6 source checkpoints (3 SNUNet-ECAM + 3 SiamUnet_diff).
2. **Fail-Closed Execution Gate:** Verified that attempting to run `--mode execute_final` raises a hard `RuntimeError`, blocking sealed test execution prior to Gate T authorization.
3. **Stream Execution Accounting:** Executed all 21 physical streams defined in `configs/changeguard/e11_final_test_ledger_v1.json` (12 SNUNet-ECAM streams, 9 SiamUnet_diff streams).
4. **Research Integrity Barriers:** Verified durable disk persistence of primary prediction shards (`.npz`) prior to unlocking ground-truth labels (`mark_persisted()`).
5. **Resumption & No-Overwrite Safety:** Re-running verified that completed streams are detected and skipped without data overwriting.
6. **Consolidated Statistical Analysis:** Verified paired event-cluster bootstrap (10,000 resamples), 95% confidence intervals, and markdown table generation via `tools/analyze_e11_final_test.py`.

---

## 2. Cryptographic Checksum Manifest

All six source checkpoints were verified prior to stream execution:

| Architecture | Seed | Checkpoint Path | Verified SHA-256 Checksum | Provenance Status |
| :--- | :---: | :--- | :--- | :---: |
| **SNUNet_ECAM** | 0 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | **MATCH** |
| **SNUNet_ECAM** | 1 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` | **MATCH** |
| **SNUNet_ECAM** | 2 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` | **MATCH** |
| **SiamUnet_diff** | 0 | `outputs/e10_second_backbone/all27_seed0/final_checkpoint.pt` | `8e23da9a4d5f1ad01889075d6c98a7197ce295f77ea8d377f1ef26ac814b1532` | **MATCH** |
| **SiamUnet_diff** | 1 | `outputs/e10_second_backbone/all27_seed1/final_checkpoint.pt` | `a92f52153fda9e966d8f8e5ffe5d92c36783d1323d44a0c855d6464ae63e60ea` | **MATCH** |
| **SiamUnet_diff** | 2 | `outputs/e10_second_backbone/all27_seed2/final_checkpoint.pt` | `d96bb4e8d74b6e7d0163129d8990769f5e1650ed9d744cafa2a291638b251aab` | **MATCH** |

---

## 3. Rehearsal Execution Telemetry (Permitted Fixture Split)

Rehearsal was executed across all 21 streams on development events `[520, 559]`:

| Stream Index | Stream Identifier | Architecture | Method | Seed | Event 520 F1 | Event 559 F1 | Macro F1 | Rehearsal Runtime |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `final_snunet_source_only_seed0` | SNUNet_ECAM | `source_only` | 0 | 0.7200 | 0.7444 | 0.7322 | 2.1s |
| 2 | `final_snunet_bn_no_update_seed0` | SNUNet_ECAM | `bn_no_update` | 0 | 0.7029 | 0.7650 | 0.7339 | 2.3s |
| 3 | `final_snunet_tent_seed0` | SNUNet_ECAM | `tent` | 0 | 0.7029 | 0.7650 | 0.7339 | 3.3s |
| 4 | `final_snunet_cotta_seed0` | SNUNet_ECAM | `cotta` | 0 | 0.7074 | 0.7570 | 0.7322 | 5.0s |
| 5 | `final_snunet_source_only_seed1` | SNUNet_ECAM | `source_only` | 1 | 0.6985 | 0.7258 | 0.7121 | 1.9s |
| 6 | `final_snunet_bn_no_update_seed1` | SNUNet_ECAM | `bn_no_update` | 1 | 0.7231 | 0.6949 | 0.7090 | 2.3s |
| 7 | `final_snunet_tent_seed1` | SNUNet_ECAM | `tent` | 1 | 0.7231 | 0.6949 | 0.7090 | 3.1s |
| 8 | `final_snunet_cotta_seed1` | SNUNet_ECAM | `cotta` | 1 | 0.7262 | 0.6964 | 0.7113 | 4.9s |
| 9 | `final_snunet_source_only_seed2` | SNUNet_ECAM | `source_only` | 2 | 0.7437 | 0.7453 | 0.7445 | 1.9s |
| 10 | `final_snunet_bn_no_update_seed2` | SNUNet_ECAM | `bn_no_update` | 2 | 0.7348 | 0.7032 | 0.7190 | 2.5s |
| 11 | `final_snunet_tent_seed2` | SNUNet_ECAM | `tent` | 2 | 0.7348 | 0.7032 | 0.7190 | 3.2s |
| 12 | `final_snunet_cotta_seed2` | SNUNet_ECAM | `cotta` | 2 | 0.7394 | 0.7041 | 0.7217 | 5.0s |
| 13 | `final_siam_diff_source_only_seed0` | SiamUnet_diff | `source_only` | 0 | 0.6898 | 0.6458 | 0.6678 | 1.3s |
| 14 | `final_siam_diff_bn_no_update_seed0` | SiamUnet_diff | `bn_no_update` | 0 | 0.6337 | 0.6003 | 0.6170 | 1.3s |
| 15 | `final_siam_diff_tent_seed0` | SiamUnet_diff | `tent` | 0 | 0.6337 | 0.6003 | 0.6170 | 1.5s |
| 16 | `final_siam_diff_source_only_seed1` | SiamUnet_diff | `source_only` | 1 | 0.6364 | 0.6593 | 0.6478 | 1.3s |
| 17 | `final_siam_diff_bn_no_update_seed1` | SiamUnet_diff | `bn_no_update` | 1 | 0.6644 | 0.6448 | 0.6546 | 1.4s |
| 18 | `final_siam_diff_tent_seed1` | SiamUnet_diff | `tent` | 1 | 0.6644 | 0.6448 | 0.6546 | 1.5s |
| 19 | `final_siam_diff_source_only_seed2` | SiamUnet_diff | `source_only` | 2 | 0.6525 | 0.6177 | 0.6351 | 1.3s |
| 20 | `final_siam_diff_bn_no_update_seed2` | SiamUnet_diff | `bn_no_update` | 2 | 0.6515 | 0.6778 | 0.6647 | 1.4s |
| 21 | `final_siam_diff_tent_seed2` | SiamUnet_diff | `tent` | 2 | 0.6515 | 0.6778 | 0.6647 | 1.5s |

---

## 4. Final Evaluation Specification & Production Packaging

When authorized by user and supervisor, the final evaluation will execute with the following exact parameters:

### 4.1 Expected Counts & Directory Tree
- **Physical Streams:** Exactly 21 streams (12 SNUNet + 9 SiamUnet).
- **Target Events:** Exactly 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).
- **Total Expected Event Rows:** Exactly **210 rows**.
- **Output Directory Structure:**
  ```text
  outputs/e11_final_test/
  ├── final_snunet_source_only_seed0/
  │   ├── event_205/shards/*.npz
  │   ├── ...
  │   ├── event_1111013/shards/*.npz
  │   └── stream_summary.json
  ├── ... [21 stream directories]
  ├── e11_consolidated_summary.json
  └── e11_final_tables.md
  ```

### 4.2 Resource and Runtime Estimates
- **Hardware Distribution:** 3 NVIDIA RTX 3090 GPUs running concurrently:
  - `cuda:0`: Streams 1–4, 13–15 (Seed 0 streams, 70 event evaluations)
  - `cuda:1`: Streams 5–8, 16–18 (Seed 1 streams, 70 event evaluations)
  - `cuda:2`: Streams 9–12, 19–21 (Seed 2 streams, 70 event evaluations)
- **Estimated Runtime:** ~45 minutes total wall-clock time.
- **Estimated Disk Usage:** ~35 GB for compressed prediction shards (`.npz`).

### 4.3 Recovery Instructions
If an interruption occurs during final test execution:
1. Do **not** delete existing output folders. Completed streams will be automatically verified and preserved.
2. Incomplete streams can be identified by missing `stream_summary.json`. Remove only the specific incomplete stream directory before relaunching.
3. Relaunching the command will resume seamlessly from the first uncompleted stream.
