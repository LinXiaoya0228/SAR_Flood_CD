# Week 13 Frozen-Source Probability-Anchor Results Report

**Protocol Version:** `week13_source_anchor_dev6_v1`  
**Configuration File:** [`configs/changeguard/week13_source_anchor_dev6_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week13_source_anchor_dev6_v1.json)  
**Execution Timestamp:** 2026-10-19T10:47:54Z – 2026-10-19T10:53:18Z (Elapsed: 324.44s)  
**Evidence Level:** `CONTROLLED_DEVELOPMENT_ABLATION`  
**Execution Target:** `cuda:0` (NVIDIA GeForce RTX 3090, 24GB VRAM)  
**Artifact Directory:** [`outputs/week13_source_anchor/approved_dev6_s0_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week13_source_anchor/approved_dev6_s0_v1/)  
**Analysis Outputs:** [`outputs/week13_source_anchor/analysis_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week13_source_anchor/analysis_v1/)  

---

## 1. Executive Summary & Core Results

Week 13 evaluated Condition **C3** (`C3_anchored_consistency_v1`), which pairs Condition C2 (weak-strong consistency with confidence filtering $\tau \ge 0.90$) with an explicit frozen-source probability anchor:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{consistency}} + 0.50 \cdot \frac{1}{|\Omega_{\text{valid}}|} \sum_{i \in \Omega_{\text{valid}}} D_{\text{KL}}\left( p_{\text{source\_detached}, i} \,\|\, p_{\text{candidate\_aligned\_strong}, i} \right)$$

### Key Findings

1. **Measurable Harm Reduction on the Longest Stream (Event 279):**
   - On Event 279 (2,381 samples, 149 batches, 116.2M valid pixels), unconstrained adaptation had previously suffered severe false-positive inflation (FP surged from 1,779,878 in `source_only` to 2,698,558 in `matched_bn_no_update_v1` and 2,696,002 in C2).
   - Under C3 ($\lambda_{\text{anc}} = 0.50$), false positives decreased to **2,684,120**, representing a **reduction of 11,882 false positive pixels** relative to C2 (and **14,438 fewer FP** relative to BN No-Update).
   - Per-event $F_1$ on Event 279 improved from **0.300410** (C2) to **0.301130** (C3) ($\Delta F_1 = +0.000720$).
   - Overall Pooled $F_1$ (heavily weighted by Event 279) improved from **0.441912** (C2) to **0.442698** (C3) ($\Delta \text{Pooled } F_1 = +0.000786$).

2. **Complete Preservation of Previously Beneficial Adaptation:**
   - On Event 559, where batch normalization improved performance over the source model ($F_1 = 0.692433$ vs $0.672577$), C3 maintained the exact benefit ($F_1 = 0.692433$), demonstrating zero loss of beneficial adaptation.
   - On Event 1111003, where batch normalization also improved over the source model ($F_1 = 0.265045$ vs $0.256835$), C3 matched the exact score ($F_1 = 0.265045$).
   - Across Events 437, 520, and 1111008, C3 maintained performance identical to C2 within $< 0.00001 F_1$.

3. **Macro-Level Impact & Structural Ceiling:**
   - Event-Macro $F_1$ reached **0.573441** (vs **0.573318** for C2, **0.573298** for BN No-Update, and **0.594880** for `source_only`).
   - Macro IoU reached **0.433048** (vs **0.432962** for C2).
   - While the anchor reduced false positives and improved metrics on Event 279 without degrading short streams, a 0.50 anchor weight did not reverse the larger performance deficit caused by computing batch statistics rather than utilizing calibrated training population statistics.

---

## 2. Quantitative Summary Across All Conditions

Metrics are computed offline from full-precision confusion cell counts over double-masked valid pixels ($\text{valid\_mask} == 1 \land \text{label} \ne 3$), strictly satisfying $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels} = 135,859,080$.

| Condition ID | Description | Updates (App/Skp/Fail) | Macro $F_1$ | Pooled $F_1$ | Worst Event $F_1$ (Event 1111003) | Macro IoU | Pooled IoU | Macro Prec | Macro Rec |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `source_only` | Frozen source, eval running stats | 0 / 0 / 0 | **0.594880** | **0.519331** | 0.256835 | **0.452959** | **0.350741** | **0.591729** | **0.675319** |
| `matched_bn_no_update_v1` | Batch-stat BN, zero updates | 0 / 0 / 0 | 0.573298 | 0.441744 | **0.265045** | 0.432950 | 0.283486 | 0.562260 | 0.672148 |
| `matched_tent_pre_update_v1` | Tent entropy minimization | 179 / 0 / 0 | 0.574088 | 0.446873 | **0.265045** | 0.433500 | 0.287725 | 0.562929 | 0.671734 |
| `C2_consistency_conf90` | Consistency with $\tau \ge 0.90$ | 179 / 0 / 0 | 0.573318 | 0.441912 | **0.265045** | 0.432962 | 0.283624 | 0.562274 | 0.672129 |
| `C3_anchored_consistency_v1` | Consistency + Source Anchor ($\lambda = 0.50$) | 179 / 0 / 0 | 0.573441 | 0.442698 | **0.265045** | 0.433048 | 0.284272 | 0.562380 | 0.672068 |

### Aggregate Controlled Differences for C3

- **C3 vs C2 (`C2_consistency_conf90`):**
  - $\Delta \text{Macro } F_1 = +0.000123$
  - $\Delta \text{Pooled } F_1 = +0.000786$
  - $\Delta \text{Macro IoU} = +0.000086$
  - $\Delta \text{Pooled IoU} = +0.000648$
  - Net False Positive Change: **$-11,884$ pixels**
  - Net True Positive Change: **$-328$ pixels**
- **C3 vs Matched BN No-Update (`matched_bn_no_update_v1`):**
  - $\Delta \text{Macro } F_1 = +0.000143$
  - $\Delta \text{Pooled } F_1 = +0.000954$
  - Net False Positive Change: **$-14,436$ pixels**
- **C3 vs Frozen Source (`source_only`):**
  - $\Delta \text{Macro } F_1 = -0.021438$
  - $\Delta \text{Pooled } F_1 = -0.076633$

---

## 3. Per-Event Granular Results

### Event 279 (2,381 samples, 149 batches, 116,199,247 valid pixels)
- `source_only`: $F_1 = 0.376893$ | TP = 615,011 | FP = 1,779,878 | FN = 253,686
- `matched_bn_no_update_v1`: $F_1 = 0.300256$ | TP = 630,147 | FP = 2,698,558 | FN = 238,550
- `matched_tent_pre_update_v1`: $F_1 = 0.304978$ | TP = 628,033 | FP = 2,621,823 | FN = 240,664
- `C2_consistency_conf90`: $F_1 = 0.300410$ | TP = 630,075 | FP = 2,696,002 | FN = 238,622
- `C3_anchored_consistency_v1`: $F_1 = \mathbf{0.301130}$ | TP = 629,747 | FP = $\mathbf{2,684,120}$ | FN = 238,950
*Observation:* On this long stream, the source anchor actively restrained false positive growth, cutting 11,882 false positives relative to C2 and increasing $F_1$ by $+0.000720$.

### Event 437 (169 samples, 11 batches, 7,859,430 valid pixels)
- `source_only`: $F_1 = 0.699365$
- `matched_bn_no_update_v1`: $F_1 = 0.634441$
- `matched_tent_pre_update_v1`: $F_1 = 0.634460$
- `C2_consistency_conf90`: $F_1 = 0.634424$
- `C3_anchored_consistency_v1`: $F_1 = 0.634432$ ($\Delta \text{ vs C2} = +0.000009$)

### Event 520 (92 samples, 6 batches, 3,455,248 valid pixels)
- `source_only`: $F_1 = 0.713750$
- `matched_bn_no_update_v1`: $F_1 = 0.699581$
- `matched_tent_pre_update_v1`: $F_1 = 0.699590$
- `C2_consistency_conf90`: $F_1 = 0.699574$
- `C3_anchored_consistency_v1`: $F_1 = 0.699574$ ($\Delta \text{ vs C2} = 0.000000$)

### Event 559 (42 samples, 3 batches, 1,843,520 valid pixels)
- `source_only`: $F_1 = 0.672577$
- `matched_bn_no_update_v1`: $F_1 = 0.692433$
- `matched_tent_pre_update_v1`: $F_1 = 0.692433$
- `C2_consistency_conf90`: $F_1 = 0.692425$
- `C3_anchored_consistency_v1`: $F_1 = 0.692433$ ($\Delta \text{ vs C2} = +0.000008$, exact match to BN No-Update)

### Event 1111003 (19 samples, 2 batches, 815,490 valid pixels)
- `source_only`: $F_1 = 0.256835$ (Worst event for source)
- `matched_bn_no_update_v1`: $F_1 = 0.265045$
- `matched_tent_pre_update_v1`: $F_1 = 0.265045$
- `C2_consistency_conf90`: $F_1 = 0.265045$
- `C3_anchored_consistency_v1`: $F_1 = 0.265045$ ($\Delta \text{ vs C2} = 0.000000$)

### Event 1111008 (123 samples, 8 batches, 5,686,145 valid pixels)
- `source_only`: $F_1 = 0.849858$
- `matched_bn_no_update_v1`: $F_1 = 0.848030$
- `matched_tent_pre_update_v1`: $F_1 = 0.848023$
- `C2_consistency_conf90`: $F_1 = 0.848033$
- `C3_anchored_consistency_v1`: $F_1 = 0.848033$ ($\Delta \text{ vs C2} = -0.000000$)

---

## 4. Loss & Runtime Profiles

Losses and timing are extracted directly from saved `step_records.json` across all 179 batches.

| Event ID | Batches | Mean Total Loss | Mean Consistency Loss | Mean Anchor Loss | Mean Step Time | Total Step Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 279 | 149 | 0.016693 | 0.004862 | 0.023662 | 910.1 ms | 135.61 s |
| 437 | 11 | 0.006185 | 0.001818 | 0.008734 | 846.6 ms | 9.31 s |
| 520 | 6 | 0.015306 | 0.004496 | 0.021620 | 859.0 ms | 5.15 s |
| 559 | 3 | 0.006354 | 0.002922 | 0.006864 | 790.8 ms | 2.37 s |
| 1111003 | 2 | 0.014787 | 0.004683 | 0.020209 | 527.2 ms | 1.05 s |
| 1111008 | 8 | 0.012065 | 0.004740 | 0.014651 | 872.5 ms | 6.98 s |
| **Overall** | **179** | **0.015599** | **0.004623** | **0.021953** | **896.5 ms** | **160.48 s** |

- **Loss Composition:** Over all 179 batches, the anchor term contributed $\lambda_{\text{anc}} \cdot \bar{\mathcal{L}}_{\text{anc}} = 0.50 \times 0.021953 = 0.010976$, which accounted for $\approx 70.4\%$ of the total gradient penalty, providing sustained stabilizing pressure toward the frozen source model.
- **Runtime:** Adapter step time totaled 160.48s (average 896.5 ms per batch of 16), with total end-to-end execution across all 6 events requiring 324.44s (~5.4 minutes).

---

## 5. Pixel Divergence Analysis

| Event ID | Samples | Total Grid Pixels | Diff (C3 vs Source) | Frac (C3 vs Source) | Diff (C3 vs BN) | Frac (C3 vs BN) | Diff (C3 vs C2) | Frac (C3 vs C2) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 279 | 2,381 | 119,469,056 | 2,188,151 | 1.832% | 32,389 | 0.0271% | 26,552 | 0.0222% |
| 437 | 169 | 8,479,744 | 42,967 | 0.507% | 11 | 0.0001% | 15 | 0.0002% |
| 520 | 92 | 4,616,192 | 50,213 | 1.088% | 27 | 0.0006% | 24 | 0.0005% |
| 559 | 42 | 2,107,392 | 6,104 | 0.290% | 0 | 0.0000% | 1 | 0.0000% |
| 1111003 | 19 | 953,344 | 12,015 | 1.260% | 0 | 0.0000% | 0 | 0.0000% |
| 1111008 | 123 | 6,171,648 | 123,102 | 1.995% | 29 | 0.0005% | 29 | 0.0005% |
| **Total** | **2,826** | **141,797,376** | **2,422,552** | **1.708%** | **32,456** | **0.0229%** | **26,621** | **0.0188%** |

- Across all events, C3 shifted **26,621 pixels** relative to C2 (99.7% of them concentrated in Event 279).
- Total discrepancy against the frozen source model decreased from 2,442,504 pixels (for BN No-Update) down to 2,422,552 pixels (for C3), reflecting an absolute recovery of nearly 20,000 pixels towards the frozen source prediction map.

---

## 6. Methodological Conclusions & Scope Limitations

1. **Direction of Effect:**
   The frozen-source probability anchor successfully operates in the intended scientific direction: it pulls test-time predictions closer to the frozen source model, reducing false positive pixel inflation on long streams without degrading short streams or destroying beneficial adaptation.
2. **Exploratory Scope Limitations:**
   - Evaluated on seed 0 only;
   - Tested under a single deterministic sequence (`order_a`) with `reset_per_event = true`;
   - Tested with a single exploratory anchor weight ($\lambda_{\text{anc}} = 0.50$);
   - Evaluated strictly within the 6 KuroSiwo development events; sealed test events remain untouched.
3. **Implications for TTA Architecture:**
   Because the performance drop relative to source is primarily rooted in evaluating under mini-batch statistics rather than unanchored gradient drift, regularizing parameter updates alone provides incremental protection (reducing thousands of false positives) but cannot fully eliminate the batch normalization domain shift.

