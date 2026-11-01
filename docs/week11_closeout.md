# ChangeGuard Week 11 Closeout & Evidence Report

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Reviews & Audits:**  
- [`docs/reviews/week11a_cpu_validation.md`](reviews/week11a_cpu_validation.md) (Independent CPU Validation v1)  
- [`docs/reviews/week11a_cpu_validation_v2.md`](reviews/week11a_cpu_validation_v2.md) (Independent CPU Validation v2)  
- [`docs/reviews/week11a_cpu_validation_v3.md`](reviews/week11a_cpu_validation_v3.md) (Independent CPU Validation v3: PASS)  
- [`docs/reviews/week11c_smoke_infrastructure_validation.md`](reviews/week11c_smoke_infrastructure_validation.md) (Independent Infrastructure Validation v1)  
- [`docs/reviews/week11c_smoke_infrastructure_validation_v2.md`](reviews/week11c_smoke_infrastructure_validation_v2.md) (Independent Infrastructure Validation v2)  
- [`docs/week11c_infrastructure_corrections_v1.md`](week11c_infrastructure_corrections_v1.md) (Corrections Note v1)  
- [`docs/week11c_infrastructure_corrections_v2.md`](week11c_infrastructure_corrections_v2.md) (Corrections Note v2)  
- [`docs/week11c_implementer_self_audit_v1.md`](week11c_implementer_self_audit_v1.md) (Implementer-Owned Self-Audit v1: PASS)  
**Governing Protocol & Configs:**  
- Protocol: [`docs/protocols/week11_evidence_consistency_protocol_v1.md`](protocols/week11_evidence_consistency_protocol_v1.md)  
- Approved Config: [`configs/changeguard/week11_smoke_event520_approved_v1.json`](../configs/changeguard/week11_smoke_event520_approved_v1.json) (`SHA-256: 98c2395159853402116a92fe29d2be74d5971b76dd7ea0f76abf4902c359ab72`)  
- Draft Config (Preserved Unauthorized): [`configs/changeguard/week11_smoke_event520_draft.json`](../configs/changeguard/week11_smoke_event520_draft.json) (`SHA-256: 4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55`)  
**Status:** **CLOSED & ACCEPTED (Week 11 Bounded Smoke)**  

---

## 1. Executive Summary & Verification Disposition

Week 11 transitioned the ChangeGuard test-time adaptation research from CPU synthetic algorithm validation (Weeks 11A/11B) to bounded real-data smoke infrastructure execution on Event 520 (Week 11C).

The bounded real-data smoke check was authorized, preflight-audited, and executed on GPU 0 (`cuda:0`). It processed all 50 target samples across 4 batches (`[16, 16, 16, 2]`), applied 4 finite optimizer updates with zero skips and zero failures, persisted ID-bound predictions prior to ground-truth access, and completed offline double-masked scoring in **25.92 seconds**.

All 12 smoke output artifacts in [`outputs/week11_real_smoke/approved_smoke_event520_v1/`](../outputs/week11_real_smoke/approved_smoke_event520_v1/) and the execution log in [`logs/week11_real_smoke/approved_smoke_event520_v1/smoke_execution.log`](../logs/week11_real_smoke/approved_smoke_event520_v1/smoke_execution.log) have been audited and cryptographically pinned in [`outputs/week11_closeout/artifact_sha256.txt`](../outputs/week11_closeout/artifact_sha256.txt).

---

## 2. Invariant & Artifact Verification Audit

An automated audit of the completed smoke directory verified all required invariants:

1. **Completion Status:**
   - [`completion_status.json`](../outputs/week11_real_smoke/approved_smoke_event520_v1/completion_status.json) records `"status": "COMPLETED"`, `completed_at_utc: "2026-10-18T15:58:07.153352+00:00"`, `elapsed_seconds: 26.021`.
2. **Sample Population & Uniqueness:**
   - [`frozen_predictions.npz`](../outputs/week11_real_smoke/approved_smoke_event520_v1/frozen_predictions.npz) contains exactly 50 distinct keys corresponding to the deterministic alphanumeric ascending head-50 sample IDs of Event 520.
3. **Batch Execution & Progress Tracking:**
   - [`adaptation_progress.json`](../outputs/week11_real_smoke/approved_smoke_event520_v1/adaptation_progress.json) records `batch_sizes_completed: [16, 16, 16, 2]`, `samples_completed: 50`, `updates_applied: 4`, `updates_skipped: 0`, `updates_failed: 0`, and `status: "ADAPTATION_COMPLETED_PENDING_OFFLINE_SCORING"`.
4. **Finite Diagnostics & Diagnostic Timing:**
   - [`smoke_step_records.json`](../outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_step_records.json) records finite scalar diagnostics for every batch:
     - Mean loss: `0.02334` (finite across all 4 steps: `0.0384`, `0.0169`, `0.0215`, `0.0166`).
     - Pre-update normalized entropy: finite (`[0.2241, 0.2227, 0.2335, 0.2464]`).
     - Pre-update max confidence: finite (`[0.9168, 0.9171, 0.9137, 0.9090]`).
     - Pre-update source disagreement rate: finite (`[0.0031, 0.0039, 0.0041, 0.0049]`).
     - Post-update null score: finite (`[0.0000, 0.0000, 0.0000, 0.0000]`).
5. **Confusion-Count Conservation Invariant:**
   - In [`smoke_summary.json`](../outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_summary.json):
     $$\text{TP} (155,993) + \text{FP} (56,585) + \text{FN} (58,904) + \text{TN} (1,633,152) = 1,904,634 \equiv \text{valid\_pixels}$$
     Exact conservation holds; no invalid or unannotated pixels are counted as true negatives.
6. **Code, Config, Checkpoint, and Manifest Identities:**
   - Git Commit: `9d45e4ee507e3ff376987f1242667a81f7db35f1`
   - Config SHA-256: `98c2395159853402116a92fe29d2be74d5971b76dd7ea0f76abf4902c359ab72`
   - Checkpoint Expected/Actual SHA-256: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
   - KuroSiwo Manifest SHA-256: `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`

---

## 3. Scientific Event 520 Smoke Performance

Evaluated strictly offline on frozen predictions using canonical double-masking (`valid = (MK0_MNA == 1) & (MK0_MLU != 3)`):

| Metric | Event 520 Smoke Value |
|---|---|
| **Evaluated Samples** | 50 (ascending head-50) |
| **Evaluated Valid Pixels** | 1,904,634 |
| **Ground Truth Flood Pixels** | 214,897 |
| **Predicted Flood Pixels** | 212,578 |
| **True Positives (TP)** | 155,993 |
| **False Positives (FP)** | 56,585 |
| **False Negatives (FN)** | 58,904 |
| **True Negatives (TN)** | 1,633,152 |
| **Flood Precision** | **0.7338** |
| **Flood Recall** | **0.7259** |
| **Flood F1** | **0.7298** |
| **Flood IoU** | **0.5746** |

---

## 4. Honest Disclosure of Technical Incidents, Runtime & Boundaries

1. **Runtime Environment & Execution Interpreter:**
   - The successful final smoke execution was conducted using Python environment `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`, PyTorch `2.13.0+cu130`, NumPy `2.2.6`).
   - Physical execution took place on `gpu22`, assigned to idle physical GPU 0 (`cuda:0`, NVIDIA GeForce RTX 3090, 24 GiB VRAM). Process PID was `1226015`. GPU memory returned to idle (16 MiB) immediately upon process termination.
2. **Initial Execution Attempt & Output Directory Cleanup:**
   - An initial execution attempt was launched under the `sar_cd` environment (`/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python`, Python 3.8, NumPy 1.x).
   - This attempt failed immediately during `INITIALIZATION` inside `torch.load` with `ModuleNotFoundError: No module named 'numpy._core'`. As recorded in the failure handler log, `final_checkpoint.pt` was serialized with NumPy 2.x, which requires NumPy $\ge 2.0$ for deserialization.
   - The failure handling engineered in Phase 1 functioned as designed: it caught the exception, wrote `execution_failure.json` detailing the failing phase (`INITIALIZATION`), error message, and traceback, and withheld `completion_status.json`.
   - Because the runner strictly enforces `output_dir_clean: true` (forbidding `--allow_overwrite`), the partial output directory `outputs/week11_real_smoke/approved_smoke_event520_v1/` and log were deleted to allow a clean retry under `vfm_env`. The details of this failure are recovered exclusively from retained terminal records; no unrecorded state was generated.
3. **Verification Nature:**
   - The final infrastructure verification before launch was **implementer-owned** ([`docs/week11c_implementer_self_audit_v1.md`](week11c_implementer_self_audit_v1.md)), following independent reviews `v1` and `v2`.
4. **Scope & Algorithmic Boundaries:**
   - **No EMA Teacher:** In accordance with the approved protocol, no Exponential Moving Average (EMA) teacher model was constructed or updated; consistency was evaluated solely between weak and strong views of the single candidate branch.
   - **Null Probes:** The implemented runner evaluates primary null-pair probes `(pre_event_2, pre_event_2)` and `(post_event, post_event)` state-neutrally post-update. The broader multi-temporal erasure expansion family was explicitly deferred.
   - **No Multi-Event Matrix:** This smoke execution was strictly bounded to seed 0, Event 520, head-50 samples. No multi-event or multi-seed matrix was run in Week 11.

---

## 5. Artifact Hashes (outputs/week11_closeout/artifact_sha256.txt)

```text
807c04c9ae4c4d50ccb25c80e75685ccb8d367417ace34a55a72c9025af9fadb  outputs/week11_real_smoke/approved_smoke_event520_v1/adaptation_progress.json
03775f15d3c1962a2c7ef67737be9acd81a72f2e5b00ff52d02082b70f9286e0  outputs/week11_real_smoke/approved_smoke_event520_v1/adaptation_step_records.csv
2b0961eb3f8b89b75e4e00473d1cf56121a18d4fec57688cdc75b0a735357e8b  outputs/week11_real_smoke/approved_smoke_event520_v1/adaptation_step_records.json
600259aef40b148818fe533f30a288061af8816fda95115846d23ed9c9cb5d35  outputs/week11_real_smoke/approved_smoke_event520_v1/completion_status.json
9fdf8bdab352d3485edc37b996f4b0876340913c9971aa298df8d5406ab3ae4b  outputs/week11_real_smoke/approved_smoke_event520_v1/frozen_predictions.npz
3e86571d50be91e86e89f700440bab371243c47f4a6e5259aadb91b0e25f8d97  outputs/week11_real_smoke/approved_smoke_event520_v1/initial_manifest.json
6e187983de2523e0dd59941b8ca217bdb063ea85858c01b6bab2c0cbc85ac042  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_event_metrics.csv
5946eea1a2ecd0cf54938587a294f4cf4021c0ed6a150261060a7fae1b1aa3da  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_event_metrics.json
ca67896ce452b15edfb0d1d57729b1e78116d53307fb94b3b814fc4e87cd6b27  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_manifest.json
d6d88c1f49b94b10d86b0b851d55b819c03302cf8c26a0f8fb788a66a6e2b4f6  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_step_records.csv
888f806705f197d12e40927473939711020c17becd5e74c777f95d62c097b123  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_step_records.json
2f37dbf61420d8b5d383bafa7265e481d97a20fcdd5b62a536f4cf0a3c067a6c  outputs/week11_real_smoke/approved_smoke_event520_v1/smoke_summary.json
dfa2467c08b1e392c0510a297d969a202d96dd3459148d71feece9b68c2fc5e1  logs/week11_real_smoke/approved_smoke_event520_v1/smoke_execution.log
```

---

## 6. Disposition & Readiness for Week 12

Week 11 is formally closed. The core evidence consistency mechanism, fail-closed preflight, durable pre-GT persistence, and offline double-masked evaluation are validated and operational. ChangeGuard is ready for the bounded Week 12 exploratory ablation study across the six development events.
