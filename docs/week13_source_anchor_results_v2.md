# Week 13 Frozen-Source Probability-Anchor Results Report (v2)

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

### Key Findings & Methodological Boundaries

1. **Measurable Harm Reduction on the Longest Stream (Event 279):**
   - On Event 279 (2,381 samples, 149 batches, 116.2M valid pixels), unconstrained adaptation had previously suffered severe false-positive inflation (FP surged from 1,779,878 in `source_only` to 2,698,558 in `matched_bn_no_update_v1` and 2,696,002 in C2).
   - Under C3 ($\lambda_{\text{anc}} = 0.50$), false positives decreased to **2,684,120**, representing a **reduction of 11,882 false positive pixels** relative to C2 (and **14,438 fewer FP** relative to BN No-Update).
   - Per-event $F_1$ on Event 279 improved from **0.300410** (C2) to **0.301130** (C3) ($\Delta F_1 = +0.000720$).
   - Overall Pooled $F_1$ (heavily weighted by Event 279) improved from **0.441912** (C2) to **0.442698** (C3) ($\Delta \text{Pooled } F_1 = +0.000786$).

2. **Preservation of Beneficial Adaptation on Short Streams:**
   - On Event 559, where batch normalization improved performance over the source model ($F_1 = 0.692433$ vs $0.672577$), C3 maintained the exact benefit ($F_1 = 0.692433$).
   - On Event 1111003, where batch normalization also improved over the source model ($F_1 = 0.265045$ vs $0.256835$), C3 matched the exact score ($F_1 = 0.265045$).
   - Across Events 437, 520, and 1111008, C3 maintained performance identical to C2 within $< 0.00001 F_1$.

3. **Macro-Level Impact & Scientific Qualification:**
   - Event-Macro $F_1$ reached **0.573441** (vs **0.573318** for C2, **0.573298** for BN No-Update, and **0.594880** for `source_only`).
   - Macro IoU reached **0.433048** (vs **0.432962** for C2).
   - **Method Status:** Current empirical evidence does **not** justify promoting C3 as an improved main method over the frozen source baseline. While the anchor provided small harm reduction on Event 279, aggregate Macro $F_1$ remains substantially below the source baseline. Tuning anchor weights or rerunning experiments is neither authorized nor justified at this stage.

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

---

## 3. Per-Event Granular Results

### Event 279 (2,381 samples, 149 batches, 116,199,247 valid pixels)
- `source_only`: $F_1 = 0.376893$ | TP = 615,011 | FP = 1,779,878 | FN = 253,686
- `matched_bn_no_update_v1`: $F_1 = 0.300256$ | TP = 630,147 | FP = 2,698,558 | FN = 238,550
- `matched_tent_pre_update_v1`: $F_1 = 0.304978$ | TP = 628,033 | FP = 2,621,823 | FN = 240,664
- `C2_consistency_conf90`: $F_1 = 0.300410$ | TP = 630,075 | FP = 2,696,002 | FN = 238,622
- `C3_anchored_consistency_v1`: $F_1 = \mathbf{0.301130}$ | TP = 629,747 | FP = $\mathbf{2,684,120}$ | FN = 238,950

### Events 437, 520, 559, 1111003, 1111008
- **Event 437:** Source $0.699365 \to$ BN $0.634441 \to$ C2 $0.634424 \to$ C3 $0.634432$
- **Event 520:** Source $0.713750 \to$ BN $0.699581 \to$ C2 $0.699574 \to$ C3 $0.699574$
- **Event 559:** Source $0.672577 \to$ BN $0.692433 \to$ C2 $0.692425 \to$ C3 $0.692433$
- **Event 1111003:** Source $0.256835 \to$ BN $0.265045 \to$ C2 $0.265045 \to$ C3 $0.265045$
- **Event 1111008:** Source $0.849858 \to$ BN $0.848030 \to$ C2 $0.848033 \to$ C3 $0.848033$

---

## 4. Loss Breakdown, Runtime & Technical Disclosures

| Event ID | Batches | Mean Total Loss | Mean Consistency Loss | Mean Anchor Loss | Mean Step Time | Total Step Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 279 | 149 | 0.016693 | 0.004862 | 0.023662 | 910.1 ms | 135.61 s |
| 437 | 11 | 0.006185 | 0.001818 | 0.008734 | 846.6 ms | 9.31 s |
| 520 | 6 | 0.015306 | 0.004496 | 0.021620 | 859.0 ms | 5.15 s |
| 559 | 3 | 0.006354 | 0.002922 | 0.006864 | 790.8 ms | 2.37 s |
| 1111003 | 2 | 0.014787 | 0.004683 | 0.020209 | 527.2 ms | 1.05 s |
| 1111008 | 8 | 0.012065 | 0.004740 | 0.014651 | 872.5 ms | 6.98 s |
| **Overall** | **179** | **0.015599** | **0.004623** | **0.021953** | **896.5 ms** | **160.48 s** |

### Critical Technical Disclosures
1. **Scalar Loss Value vs Gradient Contribution:**
   Across all 179 batches, the anchor term contributed $0.50 \times 0.021953 = 0.010976$ out of the total scalar loss of $0.015599$ ($\approx 70.4\%$). **This percentage strictly describes the scalar value of the loss function, not the relative magnitude or direction of the vector gradient contribution**, which varies non-linearly across candidate layers and parameters.
2. **Source Agreement vs Corrected Predictions:**
   C3 predictions differed from C2 on 26,621 pixels, reducing discrepancy with the frozen source by ~20,000 pixels. However, **reduced disagreement with the frozen source model is an alignment measure; it is not equivalent to ground-truth corrected predictions**.
3. **Source Calibration:**
   Source probability calibration has **not** been established on these SAR target events.
4. **Experimental Scope Restrictions:**
   All conclusions apply strictly to the specific combination of: seed 0, `order_a`, `reset_per_event = true`, and $\lambda_{\text{anc}} = 0.50$.

