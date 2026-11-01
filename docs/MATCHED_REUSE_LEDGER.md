# ChangeGuard V3: Matched Reuse Ledger

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md)  
**Active Work Package:** `E1` (Extended Recorder, Matched Baselines & Normalization Checks)  
**Status:** Implementer-Verified Reuse Ledger (Updated Post-E1.4 Baseline Audit)  

---

## 1. Principles of Scientific Reuse

Under ChangeGuard Plan V3 (Section 3.4 & Section 5.3), a historical or existing experimental cell may be reused in comparative claims **only** when all execution and data parameters strictly match the active scientific protocol:
1. **Source Checkpoint Identity:** Exact SHA-256 parameter digest matching the frozen all-27 canonical checkpoint.
2. **Sample IDs & Sequence Order:** Identical sample membership, order within each event (`order_a` / `order_b`), and identical event sequence.
3. **Batch Sizing & Tail Handling:** Physical batch size 16 with natural tail handling (no artificial padding or truncation).
4. **Input Preprocessing Pipeline:** Strict ordering: linear amplitude $\to$ clamp $[0.0, 0.15]$ $\to$ non-finite replacement $\to$ frozen $z$-score standardization ($\mu_{\text{VV}}=0.0953, \sigma_{\text{VV}}=0.0427, \mu_{\text{VH}}=0.0264, \sigma_{\text{VH}}=0.0215$).
5. **Sensory Masking & Valid Pixels:** Double-masking ($\text{valid} = (\text{MK0\_MNA} == 1) \land (\text{label} \ne 3)$).
6. **Prediction Timing Convention:** `PRE_UPDATE` primary prediction timing for online streaming decisions.
7. **Reset Semantics:** Explicit `reset_per_event` or `continual` state isolation across events.
8. **Algorithm & Objective Settings:** Identical loss formulation, learning rates, and optimizer state management.

### Reuse Eligibility Classifications:
- **`REUSE_VERIFIED_PRIMARY`:** Meets all criteria. Full spatial predictions and metrics verified on disk. Eligible for primary comparative tables.
- **`REUSE_VERIFIED_DIAGNOSTIC`:** Verified under its specific diagnostic context (e.g. normalization contrast), but lacks certain extended cache fields (e.g. probability maps or post-update predictions). Eligible for component/mechanism claims, but not as a stand-in for a full policy.
- **`REUSE_UNVERIFIED_HISTORICAL`:** Completed historical experiment that lacked spatial prediction caches, pooled confusion cell counts, or used an alternative prediction timing (e.g. POST_UPDATE). Requires matching verification or replay under E1.
- **`INELIGIBLE_FOR_REUSE`:** Incompatible protocol, differing architecture, or unverified input provenance.

---

## 2. Reused Condition Ledger

### 2.1 Canonical Source Checkpoints

| Item ID | Checkpoint Identity / SHA-256 | Data Split / Events | Status | Approved Reuse Scope | Invariance / Matching Evidence |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **CKPT-OFFICIAL** | `kuro_snunet_official.pt`<br>`f6064f77...` | Upstream KuroSiwo | `HISTORICAL_REF` | Historical motivation only. | Input channel provenance incomplete. Not eligible for primary controlled source claims. |
| **CKPT-A27-S0** | `source_snunet_vvvh_all27_seed0`<br>`247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | All 27 Source Events<br>(21,462 patches, 16 ep) | `REUSE_VERIFIED_PRIMARY` | Primary Canonical Checkpoint Seed 0. Baseline for all dev6 TTA runs. | Loaded on CPU, strict `load_state_dict`, 100% finite tensors. Backed up in external storage. |
| **CKPT-A27-S1** | `source_snunet_vvvh_all27_seed1`<br>`72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` | All 27 Source Events<br>(21,462 patches, 16 ep) | `REUSE_VERIFIED_PRIMARY` | Canonical Checkpoint Seed 1 for confirmation studies (E1/E8). | Verified identical training protocol, 100% finite tensors. Backed up in external storage. |
| **CKPT-A27-S2** | `source_snunet_vvvh_all27_seed2`<br>`0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` | All 27 Source Events<br>(21,462 patches, 16 ep) | `REUSE_VERIFIED_PRIMARY` | Canonical Checkpoint Seed 2 for confirmation studies (E1/E8). | Verified identical training protocol, 100% finite tensors. Backed up in external storage. |
| **CKPT-A22-SLOPE** | `source_snunet_vvvh_slope_seed0`<br>Inner-split checkpoint | 22 Inner-Train Events | `REUSE_VERIFIED_DIAGNOSTIC` | Retained strictly as an inner-split source ablation. | 3-seed inner A/B comparison documented in `docs/week3_seed012_source_comparison.md`. |

---

### 2.2 Historical Baseline Runs (Weeks 7–9)

| Item ID | Method | Checkpoint | Factors (Seed/Order/Reset) | Timing | Artifacts on Disk | Reuse Status | Reuse Limitations & Missing Fields |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **RUN-W7-SRC** | Source-Only | All-27 S0/S1/S2 | S0,1,2 / Order A / Reset per event | Frozen BN | `stream_event_metrics.csv` | `REUSE_UNVERIFIED_HISTORICAL` | Ran on dev6 events. Event metrics exist, but raw spatial prediction arrays were not cached. Seed 0 re-evaluated in Week 12. |
| **RUN-W8-ADABN** | AdaBN | All-27 S0/S1/S2 | S0,1,2 / Orders A, B / Reset & Continual | Candidate BN | `stream_event_metrics.csv`, `stream_matrix_summary.json` | `REUSE_UNVERIFIED_HISTORICAL` | Full matrix ran (12 streams), but no spatial prediction cache saved. Missing raw pooled confusion counts. Requires protocol matching verification. |
| **RUN-W8-TENT** | Tent | All-27 S0/S1/S2 | S0,1,2 / Orders A, B / Reset & Continual | `POST_UPDATE` | `stream_event_metrics.csv`, `stream_matrix_summary.json` | `INELIGIBLE_FOR_PRIMARY` | Uses POST_UPDATE predictions. Cannot be directly compared against PRE_UPDATE streams without explicit disclosure. |
| **RUN-W9-EATA** | EATA ($d=0.05$) | All-27 S0/S1/S2 | S0,1,2 / Orders A, B / Reset & Continual | `POST_UPDATE` | `batch_diagnostics.jsonl`, `stream_event_metrics.csv` | `REUSE_UNVERIFIED_HISTORICAL` | Uses V2 Fisher artifacts. High-density batch diagnostics saved, but raw spatial predictions absent. Prediction timing was POST_UPDATE. |
| **RUN-W9-SAR** | SAR | All-27 S0/S1/S2 | S0,1,2 / Orders A, B / Reset & Continual | `POST_UPDATE` | `batch_diagnostics.jsonl`, `stream_event_metrics.csv` | `REUSE_UNVERIFIED_HISTORICAL` | Reliable sample filtering and entropy thresholding. Same timing/cache limitations as EATA. |

---

### 2.3 Completed Exploratory Adaptation & Diagnostic Runs (Weeks 11–14)

All runs below executed on **Canonical Seed 0 Checkpoint**, **`order_a`**, **`reset_per_event`**, across the **6 Development Events** ($N=2,826$ samples, $135,859,080$ valid pixels) using **`vfm_env`** and strict **`PRE_UPDATE`** prediction timing:

| Item ID | Condition Name | Algorithm & Hyperparameters | Prediction Timing | Cached Artifacts | Reuse Status | Eligible Claims & Limitations |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **RUN-W11-SMK** | `approved_smoke_event520_v1` | Head 50 samples, consistency, bs=16 | `PRE_UPDATE` | Step records, event metrics, predictions | `REUSE_VERIFIED_PRIMARY` | Bounded infrastructure smoke proof. Not a full-event evaluation. |
| **RUN-W12-SRC** | `source_only` | Frozen training BN, 0 updates | N/A (Frozen) | `frozen_predictions.npz`, `event_metrics.json` | `REUSE_VERIFIED_PRIMARY` | Primary source baseline for Seed 0. Full spatial predictions cached. Evaluated offline in Week 14. |
| **RUN-W12-BN** | `matched_bn_no_update_v1` | Current-batch BN stats, 0 updates, lr=0 | Current Batch | `frozen_predictions.npz`, `event_metrics.json` | `REUSE_VERIFIED_DIAGNOSTIC` | Authoritative normalization control for Seed 0. Demonstrates batch-stat degradation without gradient updates. |
| **RUN-W12-TENT** | `matched_tent_pre_update_v1` | Tent entropy minimization, lr=1e-4, Adam | `PRE_UPDATE` | `frozen_predictions.npz`, `event_metrics.json` | `REUSE_VERIFIED_PRIMARY` | Authoritative matched PRE_UPDATE Tent baseline for Seed 0. Full spatial predictions cached. |
| **RUN-W12-C1** | `C1_consistency_no_conf` | Consistency loss (weak/strong flip), lr=1e-4, no conf filter | `PRE_UPDATE` | `frozen_predictions.npz`, `event_metrics.json` | `REUSE_VERIFIED_PRIMARY` | Evaluated consistency adaptation without confidence mask. Reusable in E2 confidence grid. |
| **RUN-W12-C2** | `C2_consistency_conf90` | Consistency loss, lr=1e-4, conf $\ge 0.90$ | `PRE_UPDATE` | `frozen_predictions.npz`, `event_metrics.json` | `REUSE_VERIFIED_PRIMARY` | Predefined primary reference for consistency. Reusable in E2 confidence grid. |
| **RUN-W13-C3** | `C3_anchored_consistency_v1` | C2 consistency + source KL anchor ($\lambda=0.50$) | `PRE_UPDATE` | `frozen_predictions.npz`, `event_metrics.json`, `step_records.json` | `REUSE_VERIFIED_PRIMARY` | Single probability-anchor cell for Seed 0. Reusable in E3 anchor comparison. Does not cover $\lambda \in \{0.10, 1.00\}$. |
| **RUN-W14-DIAG** | `week14_offline_diagnosis` | Offline $3 \times 3$ confusion decomposition across above 6 runs | Offline GT Read | `class_confusion_3x3.csv`, `error_breakdown.csv`, `comparative_shifts.csv` | `REUSE_VERIFIED_PRIMARY` | Authoritative error-channel decomposition. Direct input for E4 class-aware intervention design. |

---

### 2.4 Matched Baselines Executed under Work Package E1.4 (October 2026)

Under Work Package E1.4, the full 51-stream matched baseline matrix was evaluated and audited under the unified ChangeGuard Plan V3 PRE_UPDATE measurement contract:
- **Coverage:** 51 streams total (Source-Only: 3, AdaBN: 12, Tent: 12, EATA: 12, SAR: 12 across Seeds 0, 1, 2 $\times$ Orders A, B $\times$ Reset/Continual).
- **Accounting:** 2 `REUSED_VERIFIED` (`source_only_s0`, `tent_s0_oa_re`), 49 `EXECUTED_NEW`.
- **Validation:** 100% confusion conservation ($TP+FP+FN+TN == \text{valid\_pixels}$), 3x3 consistency, and update counts verified via deterministic CPU verifier (`tools/verify_e1_4_integrity.py`).
- **Artifacts:** Master coverage tables saved in `outputs/e1_matched_baselines/v1/master_coverage_table.json` and `.md`; prediction caches (`frozen_predictions.npz`) and step records saved for all 49 executed streams.
- **Classification:** All 51 streams are now classified as `REUSE_VERIFIED_PRIMARY` for comparative adaptation claims.

---

## 3. Detailed Parameter & Protocol Matching Matrix

| Condition | Checkpoint SHA-256 Match | Sample IDs & Order Match | Batch Sizing (16, tail) | Preprocessing Match | Double Masking Match | Prediction Timing Match | Reset Mode Match | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_only` (W12) | Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | Frozen (`PRE`) | `reset_per_event` | `REUSE_VERIFIED_PRIMARY` |
| `matched_bn_no_update_v1` | Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | Candidate (`PRE`) | `reset_per_event` | `REUSE_VERIFIED_DIAGNOSTIC` |
| `matched_tent_pre_update_v1` | Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | `PRE_UPDATE` | `reset_per_event` | `REUSE_VERIFIED_PRIMARY` |
| `C1_consistency_no_conf` | Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | `PRE_UPDATE` | `reset_per_event` | `REUSE_VERIFIED_PRIMARY` |
| `C2_consistency_conf90` | Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | `PRE_UPDATE` | `reset_per_event` | `REUSE_VERIFIED_PRIMARY` |
| `C3_anchored_consistency_v1`| Yes (`247d3584...`) | Yes (`order_a`, 2826 IDs) | Yes ([16...tail]) | Yes | Yes | `PRE_UPDATE` | `reset_per_event` | `REUSE_VERIFIED_PRIMARY` |
| `week8_adabn` (W8) | Yes (`247d3584...`) | Unverified IDs/order | Yes (16) | Yes | Unverified | Candidate (`PRE`) | Both evaluated | `REUSE_UNVERIFIED_HISTORICAL` |
| `week8_tent` (W8) | Yes (`247d3584...`) | Unverified IDs/order | Yes (16) | Yes | Unverified | `POST_UPDATE` (Differs) | Both evaluated | `INELIGIBLE_FOR_PRIMARY` |
| `week9_eata` (W9) | Yes (`247d3584...`) | Unverified IDs/order | Yes (16) | Yes | Unverified | `POST_UPDATE` (Differs) | Both evaluated | `REUSE_UNVERIFIED_HISTORICAL` |
| `week9_sar` (W9) | Yes (`247d3584...`) | Unverified IDs/order | Yes (16) | Yes | Unverified | `POST_UPDATE` (Differs) | Both evaluated | `REUSE_UNVERIFIED_HISTORICAL` |

---

## 4. Reuse Execution Rules for Work Package E1

1. **Retain Verified Seed 0 Runs:** The six completed Week 12–13 runs (`source_only`, `matched_bn_no_update_v1`, `matched_tent_pre_update_v1`, `C1`, `C2`, `C3`) are frozen and cryptographically pinned. They will **not** be re-executed for seed 0 screening.
2. **Replay Historical Baselines under PRE_UPDATE:** Because historical Week 8/9 AdaBN, Tent, EATA, and SAR used POST_UPDATE or lacked spatial prediction caching, they cannot be mixed into primary PRE_UPDATE comparison tables. When baseline claims are made in E1/E8, they will be executed under the unified PRE_UPDATE protocol with full cache export.
3. **Multi-Seed Coverage:** Checkpoint Seeds 1 and 2 must be evaluated under the unified protocol for all methods entering the final comparison matrix.

