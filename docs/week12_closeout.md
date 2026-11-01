# ChangeGuard Week 12 Closeout & Evidence Report

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Protocols & Configurations:**  
- Ablation Config: [`configs/changeguard/week12_ablation_dev6_v1.json`](../configs/changeguard/week12_ablation_dev6_v1.json)  
- BN Diagnostic Config: [`configs/changeguard/week12_bn_diagnostic_dev6_v1.json`](../configs/changeguard/week12_bn_diagnostic_dev6_v1.json)  
**Governing Evidence Reports:**  
- Exploratory Results (v2): [`docs/week12_exploratory_results_v2.md`](week12_exploratory_results_v2.md)  
- BN Diagnostic Comparison (v2): [`docs/week12_bn_diagnostic_comparison_v2.md`](week12_bn_diagnostic_comparison_v2.md)  
- Artifact Inventory: [`outputs/week12_closeout/artifact_sha256.txt`](../outputs/week12_closeout/artifact_sha256.txt)  
**Status:** **CLOSED & ACCEPTED (Week 12 Exploratory Ablation & BN Diagnostic)**  

---

## 1. Executive Summary & Verification Disposition

Week 12 executed a controlled 4-condition exploratory ablation across all 6 KuroSiwo development events (`[279, 437, 520, 559, 1111003, 1111008]`), followed by a targeted diagnostic condition (`matched_bn_no_update_v1`) to isolate the effect of mini-batch normalization statistics from test-time gradient updates.

### Conditions Evaluated
1. `source_only`: Frozen SNUNet_ECAM VV/VH seed 0 checkpoint in `eval()` mode using training population statistics; 0 updates.
2. `matched_tent_pre_update_v1`: Entropy minimization over sensory-valid pixels in batch-stat BN mode; 179 updates.
3. `C1_consistency_no_conf`: Weak-strong paired horizontal flip consistency over all valid pixels; 179 updates.
4. `C2_consistency_conf90`: Weak-strong consistency filtered to confident pixels ($\tau \ge 0.90$); 179 updates.
5. `matched_bn_no_update_v1`: Evaluated under batch-stat BN mode (`track_running_stats = False`) with strictly 0 backward passes and 0 updates.

All 130 artifacts across [`outputs/week12_ablation/approved_dev6_s0_v1/`](../outputs/week12_ablation/approved_dev6_s0_v1/) and [`outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1/`](../outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1/) are inventoried and cryptographically hashed in [`outputs/week12_closeout/artifact_sha256.txt`](../outputs/week12_closeout/artifact_sha256.txt).

---

## 2. Invariant & Artifact Audit

1. **Coverage & Sample Identity:**
   - Evaluated across all 6 development events (`279, 437, 520, 559, 1111003, 1111008`).
   - Exactly 2,826 samples per condition, partitioned into identical batches (`[149, 11, 6, 3, 2, 8]`, total 179 batches of size 16 with natural tails).
   - Sample IDs and presentation order (`order_a`, ascending sorted path order) were verified identical across all conditions.
2. **Double-Masking Offline Evaluation:**
   - Double-masking strictly applied: $\text{valid} = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$.
   - Evaluated valid pixels: exactly **135,859,080** pixels across all conditions.
3. **Confusion Conservation:**
   - For all 5 conditions and every event: $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels}$.
4. **Code, Config, Checkpoint, and Manifest Identities:**
   - Checkpoint: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` (`SHA-256: 247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`)
   - Manifest: `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` (`SHA-256: dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`)
   - Runtime: Python environment `vfm_env` (`cuda:0`).

---

## 3. Summary of Scientific Findings

| Condition | Updates | Macro $F_1$ | Pooled $F_1$ | Worst Event $F_1$ (1111003) | Total FP | Total TP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `source_only` | 0 | **0.594880** | **0.519331** | 0.256835 | **1,970,172** | 1,322,310 |
| `matched_bn_no_update_v1` | 0 | 0.573298 | 0.441744 | **0.265045** | 2,909,127 | 1,334,937 |
| `matched_tent_pre_update_v1` | 179 | 0.574088 | 0.446873 | **0.265045** | 2,832,369 | 1,332,811 |
| `C1_consistency_no_conf` | 179 | 0.573270 | 0.441606 | **0.265045** | 2,911,192 | **1,334,989** |
| `C2_consistency_conf90` | 179 | 0.573318 | 0.441912 | **0.265045** | 2,906,575 | 1,334,864 |

1. **Harmful Adaptation on Long Streams:** Continuous test-time adaptation degraded performance relative to the frozen source on 4 of 6 events, suffering massive false positive surges on Event 279 (FP increased by ~920k pixels).
2. **Role of Normalization Context:** Evaluating under batch-statistic BatchNorm with 0 updates applied reproduces a similar aggregate deterioration ($F_1 = 0.5733$ vs $0.5949$).
3. **Prediction Closeness vs Parameter Drift:** Predictions under C1 and C2 differed from the no-update BN control on fewer than 6,200 pixels (<0.005% of all pixels). While predictions and metrics remained very close, this establishes output similarity under these specific settings and does not represent direct measurement of parameter distance.

---

## 4. Scope Limitations & Deferred Work

- **Exploratory Parameters Only:** Evaluated on source checkpoint seed 0 only, single sequence (`order_a`), and per-event reset (`reset_per_event = true`).
- **Deferred Components:** Exponential Moving Average (EMA) teacher models, Sharpness-Aware Minimization (SAM), and EATA Fisher information weighting remain deferred.
- **Quarantine:** All 10 sealed test events remain strictly quarantined.

