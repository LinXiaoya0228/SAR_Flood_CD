# Week 12 Matched BN Diagnostic Comparison Report (v2)

**Protocol Reference:** `week12_bn_diagnostic_dev6_v1`  
**Execution Timestamp:** 2026-10-19T10:13:38Z – 2026-10-19T10:17:11Z (Elapsed: 205.32s)  
**Evidence Level:** `IMPLEMENTER_SELF_VERIFIED_DIAGNOSTIC`  
**Target Device:** `cuda:0` (NVIDIA RTX A6000)  
**Artifact Directory:** [`outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1/)  
**Analysis Outputs:** [`outputs/week12_ablation/analysis_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/analysis_v1/)  

---

## 1. Executive Summary & Corrected Interpretation

The **Matched BN No-Update Diagnostic** (`matched_bn_no_update_v1`) was designed to evaluate the frozen source model under **current-batch statistics** ($\mu_B, \sigma_B^2$ with `track_running_stats = False`) with strictly **zero backward passes and zero optimizer updates**.

### Key Observations & Interpretation Boundaries

1. **Batch-Stat Normalization Reproduces Similar Aggregate Deterioration:**
   - Evaluating under batch-statistic BatchNorm with 0 updates applied yields an aggregate Macro $F_1$ of **0.573298** and Pooled $F_1$ of **0.441744**, compared to the frozen source benchmark of **0.594880** Macro $F_1$ and **0.519331** Pooled $F_1$.
   - The unanchored updating conditions achieve very similar aggregate scores:
     - Tent: Macro $F_1 = 0.574088$, Pooled $F_1 = 0.446873$
     - C1 (no conf): Macro $F_1 = 0.573270$, Pooled $F_1 = 0.441606$
     - C2 ($\tau \ge 0.90$): Macro $F_1 = 0.573318$, Pooled $F_1 = 0.441912$
   - Thus, simply switching the normalization context to mini-batch statistics without modifying any network weights reproduces an aggregate performance decline of similar magnitude to that seen during continuous test-time adaptation.

2. **Prediction Differences vs Parameter Drift:**
   - Across all 135,859,080 valid pixels (2,826 samples over 6 events), `matched_bn_no_update_v1` differs from `source_only` on **2,442,504 pixels** (1.80% of all evaluated pixels).
   - In contrast, discrete predictions under C1 and C2 differed from `matched_bn_no_update_v1` on only **4,622 pixels** and **6,146 pixels** respectively (less than 0.005% of evaluated pixels).
   - **Boundary on Causal Claims:** While discrete prediction outputs and aggregate metrics remain extremely close between `matched_bn_no_update_v1` and C1/C2, this observation measures output similarity under these specific test settings; it does not constitute direct measurement of parameter distance or weight trajectory drift.
   - These results represent controlled experimental contrasts rather than an exact additive causal decomposition, as feature normalization and weight updates interact dynamically throughout adaptation.

---

## 2. Quantitative Summary Across All 5 Conditions

All metrics are computed from full-precision confusion cell counts over double-masked sensory-valid pixels ($\text{valid\_mask} == 1 \land \text{label} \ne 3$), strictly satisfying $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels} = 135,859,080$.

| Condition ID | Description | Updates (App/Skp/Fail) | Macro $F_1$ | Pooled $F_1$ | Worst Event $F_1$ (Event 1111003) | Macro IoU | Pooled IoU | Macro Prec | Macro Rec |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `source_only` | Frozen source, eval running stats | 0 / 0 / 0 | **0.594880** | **0.519331** | 0.256835 | **0.452959** | **0.350741** | **0.591729** | 0.675319 |
| `matched_bn_no_update_v1` | Batch-stat BN, zero updates | 0 / 0 / 0 | 0.573298 | 0.441744 | **0.265045** | 0.432950 | 0.283486 | 0.562260 | 0.672148 |
| `matched_tent_pre_update_v1` | Tent entropy minimization | 179 / 0 / 0 | 0.574088 | 0.446873 | **0.265045** | 0.433500 | 0.287725 | 0.562929 | 0.671734 |
| `C1_consistency_no_conf` | Consistency without confidence filter | 179 / 0 / 0 | 0.573270 | 0.441606 | **0.265045** | 0.432927 | 0.283373 | 0.562234 | **0.672154** |
| `C2_consistency_conf90` | Consistency with $\tau \ge 0.90$ | 179 / 0 / 0 | 0.573318 | 0.441912 | **0.265045** | 0.432962 | 0.283624 | 0.562274 | 0.672129 |

### Aggregate Metric Differences

#### Versus `source_only`
- `matched_bn_no_update_v1` vs `source_only`: $\Delta \text{Macro } F_1 = -0.021582$, $\Delta \text{Pooled } F_1 = -0.077587$
- `matched_tent_pre_update_v1` vs `source_only`: $\Delta \text{Macro } F_1 = -0.020791$, $\Delta \text{Pooled } F_1 = -0.072458$
- `C1_consistency_no_conf` vs `source_only`: $\Delta \text{Macro } F_1 = -0.021610$, $\Delta \text{Pooled } F_1 = -0.077725$
- `C2_consistency_conf90` vs `source_only`: $\Delta \text{Macro } F_1 = -0.021561$, $\Delta \text{Pooled } F_1 = -0.077419$

#### Updating Conditions Versus `matched_bn_no_update_v1` (Controlled Baseline)
- `matched_tent_pre_update_v1` vs BN No-Update: $\Delta \text{Macro } F_1 = +0.000791$, $\Delta \text{Pooled } F_1 = +0.005129$
- `C1_consistency_no_conf` vs BN No-Update: $\Delta \text{Macro } F_1 = -0.000028$, $\Delta \text{Pooled } F_1 = -0.000137$
- `C2_consistency_conf90` vs BN No-Update: $\Delta \text{Macro } F_1 = +0.000021$, $\Delta \text{Pooled } F_1 = +0.000168$

---

## 3. Per-Event Granular Results

### Event 279 (2,381 samples, 149 batches, 116,199,247 valid pixels)
- `source_only`: $F_1 = 0.376893$ (FP: 1,779,878 | TP: 615,011)
- `matched_bn_no_update_v1`: $F_1 = 0.300256$ (FP: 2,698,558 | TP: 630,147) $\to \Delta \text{ vs source} = -0.076637$
- `matched_tent_pre_update_v1`: $F_1 = 0.304978$ (FP: 2,621,823 | TP: 628,033) $\to \Delta \text{ vs BN} = +0.004722$
- `C1_consistency_no_conf`: $F_1 = 0.300132$ (FP: 2,700,608 | TP: 630,203) $\to \Delta \text{ vs BN} = -0.000124$
- `C2_consistency_conf90`: $F_1 = 0.300410$ (FP: 2,696,002 | TP: 630,075) $\to \Delta \text{ vs BN} = +0.000154$

### Event 437 (169 samples, 11 batches, 7,859,430 valid pixels)
- `source_only`: $F_1 = 0.699365$
- `matched_bn_no_update_v1`: $F_1 = 0.634441$ ($\Delta \text{ vs source} = -0.064924$)
- `matched_tent_pre_update_v1`: $F_1 = 0.634460$ ($\Delta \text{ vs BN} = +0.000019$)
- `C1_consistency_no_conf`: $F_1 = 0.634416$ ($\Delta \text{ vs BN} = -0.000025$)
- `C2_consistency_conf90`: $F_1 = 0.634424$ ($\Delta \text{ vs BN} = -0.000018$)

### Event 520 (92 samples, 6 batches, 3,455,248 valid pixels)
- `source_only`: $F_1 = 0.713750$
- `matched_bn_no_update_v1`: $F_1 = 0.699581$ ($\Delta \text{ vs source} = -0.014169$)
- `matched_tent_pre_update_v1`: $F_1 = 0.699590$ ($\Delta \text{ vs BN} = +0.000008$)
- `C1_consistency_no_conf`: $F_1 = 0.699566$ ($\Delta \text{ vs BN} = -0.000015$)
- `C2_consistency_conf90`: $F_1 = 0.699574$ ($\Delta \text{ vs BN} = -0.000007$)

### Event 559 (42 samples, 3 batches, 1,843,520 valid pixels)
- `source_only`: $F_1 = 0.672577$
- `matched_bn_no_update_v1`: $F_1 = 0.692433$ ($\Delta \text{ vs source} = +0.019856$)
- `matched_tent_pre_update_v1`: $F_1 = 0.692433$ ($\Delta \text{ vs BN} = 0.000000$)
- `C1_consistency_no_conf`: $F_1 = 0.692433$ ($\Delta \text{ vs BN} = 0.000000$)
- `C2_consistency_conf90`: $F_1 = 0.692425$ ($\Delta \text{ vs BN} = -0.000008$)

### Event 1111003 (19 samples, 2 batches, 815,490 valid pixels)
- `source_only`: $F_1 = 0.256835$ (Worst event for source)
- `matched_bn_no_update_v1`: $F_1 = 0.265045$ ($\Delta \text{ vs source} = +0.008210$)
- `matched_tent_pre_update_v1`: $F_1 = 0.265045$ ($\Delta \text{ vs BN} = 0.000000$)
- `C1_consistency_no_conf`: $F_1 = 0.265045$ ($\Delta \text{ vs BN} = 0.000000$)
- `C2_consistency_conf90`: $F_1 = 0.265045$ ($\Delta \text{ vs BN} = 0.000000$)

### Event 1111008 (123 samples, 8 batches, 5,686,145 valid pixels)
- `source_only`: $F_1 = 0.849858$
- `matched_bn_no_update_v1`: $F_1 = 0.848030$ ($\Delta \text{ vs source} = -0.001828$)
- `matched_tent_pre_update_v1`: $F_1 = 0.848023$ ($\Delta \text{ vs BN} = -0.000006$)
- `C1_consistency_no_conf`: $F_1 = 0.848027$ ($\Delta \text{ vs BN} = -0.000003$)
- `C2_consistency_conf90`: $F_1 = 0.848033$ ($\Delta \text{ vs BN} = +0.000004$)

---

## 4. Pixel Divergence Analysis

| Event ID | Samples | Total Grid Pixels | Diff (BN vs Source) | Frac (BN vs Source) | Diff (Tent vs BN) | Diff (C1 vs BN) | Diff (C2 vs BN) | Diff (C2 vs C1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 279 | 2,381 | 119,469,056 | 2,207,049 | 1.847% | 175,818 | 4,563 | 6,100 | 10,097 |
| 437 | 169 | 8,479,744 | 42,972 | 0.507% | 42 | 12 | 12 | 8 |
| 520 | 92 | 4,616,192 | 50,222 | 1.088% | 27 | 20 | 11 | 19 |
| 559 | 42 | 2,107,392 | 6,104 | 0.290% | 0 | 0 | 1 | 1 |
| 1111003 | 19 | 953,344 | 12,015 | 1.260% | 0 | 0 | 0 | 0 |
| 1111008 | 123 | 6,171,648 | 123,099 | 1.995% | 97 | 27 | 22 | 31 |
| **Total** | **2,826** | **141,797,376** | **2,442,504** | **1.723%** | **175,984** | **4,622** | **6,146** | **10,156** |

---

## 5. Summary Findings

1. **Batch Statistics Normalization Shift:** Calculating mean and variance on small evaluation mini-batches (size 16) produces a substantial shift in predictions relative to source running statistics, altering over 2.4 million pixels and generating a comparable aggregate performance drop to continuous adaptation.
2. **Output Closeness of Updating Conditions:** Under the tested settings (learning rate $1.25 \times 10^{-4}$, 1 step per batch of 16), C1 and C2 predictions remain within 0.005% of the un-updated batch-stat predictions across the 6 events.
3. **Causal Interpretation:** These observations confirm that batch normalization statistics alone reproduce a similar aggregate drop; they do not establish that parameter updates are inert or that parameter drift is zero.

