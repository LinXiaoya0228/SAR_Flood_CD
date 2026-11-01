# ChangeGuard Week 13 Closeout & Evidence Report

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Protocol & Configuration:**  
- Protocol: [`docs/protocols/week13_source_anchor_protocol_v1.md`](protocols/week13_source_anchor_protocol_v1.md)  
- Config: [`configs/changeguard/week13_source_anchor_dev6_v1.json`](../configs/changeguard/week13_source_anchor_dev6_v1.json)  
**Governing Evidence Reports:**  
- Source Anchor Results (v2): [`docs/week13_source_anchor_results_v2.md`](week13_source_anchor_results_v2.md)  
- Artifact Inventory: [`outputs/week13_closeout/artifact_sha256.txt`](../outputs/week13_closeout/artifact_sha256.txt)  
**Status:** **CLOSED & ACCEPTED (Week 13 Source-Anchored Exploratory Adaptation)**  

---

## 1. Executive Summary & Verification Disposition

Week 13 implemented and executed Condition **C3** (`C3_anchored_consistency_v1`), combining C2 weak-strong consistency ($\tau \ge 0.90$) with an explicit frozen-source probability anchor:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{consistency}} + 0.50 \cdot \frac{1}{|\Omega_{\text{valid}}|} \sum_{i \in \Omega_{\text{valid}}} D_{\text{KL}}\left( p_{\text{source\_detached}, i} \,\|\, p_{\text{candidate\_aligned\_strong}, i} \right)$$

Execution was carried out in detached tmux on GPU 0 (`cuda:0`) using Python environment `vfm_env`. The run processed all 2,826 development samples across 179 batches, applying 179 finite updates with 0 skips and 0 failures in **324.44 seconds**.

All 27 output artifacts in [`outputs/week13_source_anchor/approved_dev6_s0_v1/`](../outputs/week13_source_anchor/approved_dev6_s0_v1/) are inventoried and cryptographically hashed in [`outputs/week13_closeout/artifact_sha256.txt`](../outputs/week13_closeout/artifact_sha256.txt).

---

## 2. Invariant & Artifact Audit

1. **Completion Status:**
   - [`completion_status.json`](../outputs/week13_source_anchor/approved_dev6_s0_v1/completion_status.json) records `"status": "COMPLETED"`, `elapsed_seconds: 324.44`.
2. **Coverage & Sample Identity:**
   - Evaluated across all 6 development events (`279, 437, 520, 559, 1111003, 1111008`).
   - Exactly 2,826 samples, partitioned into identical batches (`[149, 11, 6, 3, 2, 8]`).
   - Sample IDs and presentation order (`order_a`) match Week 12 exactly.
3. **Double-Masking Offline Evaluation:**
   - Double-masking strictly applied: $\text{valid} = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$.
   - Evaluated valid pixels: exactly **135,859,080** pixels.
4. **Confusion Conservation:**
   - Total confusion matrix sum:
     $$\text{TP} (1,334,536) + \text{FP} (2,894,691) + \text{FN} (465,340) + \text{TN} (131,164,513) = 135,859,080 \equiv \text{valid\_pixels}$$
   - Exact conservation verified across all 6 events individually.
5. **Update Counts & Finite Diagnostics:**
   - Updates applied: 179 / 179 (0 skipped, 0 failed).
   - All recorded loss and diagnostic scalars are finite.
6. **Code, Config, Checkpoint, and Manifest Identities:**
   - Checkpoint: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` (`SHA-256: 247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`)
   - Manifest: `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` (`SHA-256: dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`)
   - Config: `configs/changeguard/week13_source_anchor_dev6_v1.json`

---

## 3. Scientific Performance Summary

| Condition | Updates | Macro $F_1$ | Pooled $F_1$ | Worst Event $F_1$ (1111003) | Total FP | Total TP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `source_only` | 0 | **0.594880** | **0.519331** | 0.256835 | **1,970,172** | 1,322,310 |
| `matched_bn_no_update_v1` | 0 | 0.573298 | 0.441744 | **0.265045** | 2,909,127 | **1,334,937** |
| `matched_tent_pre_update_v1` | 179 | 0.574088 | 0.446873 | **0.265045** | 2,832,369 | 1,332,811 |
| `C2_consistency_conf90` | 179 | 0.573318 | 0.441912 | **0.265045** | 2,906,575 | 1,334,864 |
| `C3_anchored_consistency_v1` | 179 | 0.573441 | 0.442698 | **0.265045** | 2,894,691 | 1,334,536 |

### Core Scientific Takeaways & Interpretation Boundaries

1. **Harm Reduction on Long Streams:**
   Condition C3 reduced false positive pixels on Event 279 by 11,882 pixels relative to C2 (and 14,438 relative to BN No-Update), improving Event 279 $F_1$ from 0.300410 to 0.301130 and Pooled $F_1$ from 0.441912 to 0.442698.
2. **Preservation of Beneficial Adaptation:**
   On Events 559 and 1111003, where batch normalization improved over source running stats, C3 preserved the full gain without degradation.
3. **Precision on Loss Share:**
   The anchor's approximately 70.4% share refers strictly to the scalar loss function value ($0.50 \times 0.021953 / 0.015599$), **not** to the vector gradient contribution.
4. **Distinction Between Agreement and Accuracy:**
   Reduced disagreement with the frozen source model is an alignment metric; it is not equivalent to ground-truth corrected predictions.
5. **Source Calibration Status:**
   Source probability calibration has not been formally established.
6. **Promotion Threshold:**
   Current exploratory evidence does **not** justify promoting C3 as an improved main method over the frozen source baseline (Macro $F_1 = 0.5734$ vs $0.5949$). Hyperparameter tuning and experiment reruns remain unauthorized.

---

## 4. Scope Limitations

- Evaluated strictly on seed 0, `order_a`, `reset_per_event = true`, and fixed exploratory $\lambda_{\text{anc}} = 0.50$.
- Evaluated strictly within the 6 development events; 10 sealed test events remain untouched.

