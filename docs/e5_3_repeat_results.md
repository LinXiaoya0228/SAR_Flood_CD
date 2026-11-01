# ChangeGuard Work Package E5.3: Cross-Cell Repeat Results and Analysis

> [!IMPORTANT]
> **Work Package E5.3 Status: EXECUTED & AUDITED**
> - **Execution Commit**: `4f150ccaacb94df1e1fa483a07a5ccfa9b488a2b` (`feat(e5): freeze orthogonal repeat cells and verify prelaunch readiness`)
> - **Governing Plans & Specifications**:
>   - [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md)
>   - [`docs/e5_3_repeat_prelaunch.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_3_repeat_prelaunch.md)
>   - [`docs/e5_2c_analysis_contract.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_2c_analysis_contract.md)
>   - [`docs/e5_2_final_closeout.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_2_final_closeout.md)
> - **Evaluated Cells ($N=3$)**:
>   1. **Canonical Cell (`s0_oa`)**: Seed 0, Stream Order A (`[279, 437, 520, 559, 1111003, 1111008]`), ascending sample IDs.
>   2. **Order Repeat Cell R1 (`s0_ob`)**: Seed 0, Stream Order B (`[1111008, 1111003, 559, 520, 437, 279]`), descending sample IDs.
>   3. **Seed Repeat Cell R2 (`s1_oa`)**: Seed 1, Stream Order A (`[279, 437, 520, 559, 1111003, 1111008]`), ascending sample IDs.
> - **Evidence & Shard Completeness**: 179 authoritative `.npz` batch shards (2,826 samples) per cell; 537 shards total; zero failure records; zero sealed-test access.
> - **Terminal Route Verdict**: `E5 COMPLETE — E6 CONTINUES WITH REDUCED SCOPE`

---

## 1. Executive Summary & Lifecycle Audits

Work Package E5.3 was established to test the stability and generalization of temporal adaptation risks and pre-update risk indicators under two strictly orthogonal variations:
1. **Order Variation (Cell R1: `s0_ob`)**: Same source model checkpoint (Seed 0), but reverse temporal event order and descending within-event sample order.
2. **Seed Variation (Cell R2: `s1_oa`)**: Same canonical event sequence (Order A), but a distinct source checkpoint initialization (Seed 1, SHA-256 `72c4be7887373f7360249b6b7cf95d023f03b22cf3ff3411b93f1bc5570fa561`).

Both formal evidence collections executed on dedicated physical GPUs with bounded memory (peak GPU memory $\le 2.92$ GB), zero memory leaks, and zero data loss. Both offline analyses executed deterministically with the frozen analyzer [`tools/analyze_e5_temporal_risk.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/analyze_e5_temporal_risk.py).

### 1.1 Run Telemetry Summary

| Metric / Dimension | Canonical Cell (`s0_oa`) | Order Repeat R1 (`s0_ob`) | Seed Repeat R2 (`s1_oa`) |
| :--- | :---: | :---: | :---: |
| **Run ID** | `e5_temporal_risk_s0_oa_v1` | `e5_temporal_risk_s0_ob_v1` | `e5_temporal_risk_s1_oa_v1` |
| **Physical Hardware** | Physical GPU 0 | Physical GPU 0 | Physical GPU 1 |
| **Source Checkpoint Seed** | Seed 0 | Seed 0 | Seed 1 |
| **Stream Event Order** | Order A | Order B (Reversed) | Order A |
| **Within-Event Sorting** | Ascending `sample_id` | Descending `sample_id` | Ascending `sample_id` |
| **Samples Processed** | 2,826 | 2,826 | 2,826 |
| **Batches / Shards** | 179 | 179 | 179 |
| **Collection Wall Time** | ~17.5 min | 1,067.3s (17.79 min) | 972.0s (16.20 min) |
| **Completion Status** | `COMPLETED_RUN` | `COMPLETED_RUN` | `COMPLETED_RUN` |
| **Failure Records** | 0 | 0 | 0 |
| **Sealed-Test Incursions** | 0 | 0 | 0 |
| **Analysis Status** | `COMPLETED_ANALYSIS` | `COMPLETED_ANALYSIS` | `COMPLETED_ANALYSIS` |

---

## 2. Cross-Cell Empirical Findings

### 2.1 Full Cross-Cell Event Outcome Comparison

The primary evaluation assesses whether test-time adaptation (Candidate context: confidence threshold 0.90, augmentations, lr $10^{-4}$, reset per event) improves or harms segmentation performance relative to the static frozen Source model and the Batch Normalization (BN) baseline.

| Event ID | Geography | Cell | Valid Pixels | Source F1 | BN Ctrl F1 | Cand F1 | $\Delta \text{F1}_{\text{Cand-Src}}$ | $\Delta \text{F1}_{\text{Cand-BN}}$ | Outcome Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | Spain (Ebro) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 116,199,247<br>116,199,247<br>116,199,247 | 0.3769<br>0.3769<br>0.3339 | 0.3758<br>0.3767<br>0.3337 | 0.3224<br>0.3274<br>0.4926 | **-0.0545**<br>**-0.0495**<br>**+0.1588** | -0.0533<br>-0.0493<br>+0.1589 | **Harmful**<br>**Harmful**<br>**Beneficial** |
| **437** | UK (Severn) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 7,859,430<br>7,859,430<br>7,859,430 | 0.6994<br>0.6994<br>0.7126 | 0.6925<br>0.6928<br>0.7091 | 0.6352<br>0.6393<br>0.6590 | **-0.0642**<br>**-0.0601**<br>**-0.0537** | -0.0573<br>-0.0535<br>-0.0501 | **Harmful**<br>**Harmful**<br>**Harmful** |
| **520** | USA (Florence) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 3,455,248<br>3,455,248<br>3,455,248 | 0.7138<br>0.7137<br>0.7010 | 0.7106<br>0.7139<br>0.7052 | 0.6997<br>0.6995<br>0.7213 | **-0.0140**<br>**-0.0143**<br>**+0.0203** | -0.0108<br>-0.0143<br>+0.0162 | **Harmful**<br>**Harmful**<br>**Beneficial** |
| **559** | USA (Midwest) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 1,843,520<br>1,843,520<br>1,843,520 | 0.6726<br>0.6726<br>0.6640 | 0.6841<br>0.6806<br>0.6471 | 0.6923<br>0.6890<br>0.6377 | **+0.0197**<br>**+0.0164**<br>**-0.0263** | +0.0082<br>+0.0084<br>-0.0094 | **Beneficial**<br>**Beneficial**<br>**Harmful** |
| **1111003** | Bangladesh (Ganges) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 815,490<br>815,490<br>815,490 | 0.2568<br>0.2568<br>0.2790 | 0.2665<br>0.2790<br>0.2809 | 0.2650<br>0.2723<br>0.2803 | **+0.0082**<br>**+0.0155**<br>**+0.0013** | -0.0014<br>-0.0067<br>-0.0006 | **Neutral**<br>**Beneficial**<br>**Neutral** |
| **1111008** | Myanmar (Sagaing) | `s0_oa`<br>`s0_ob`<br>`s1_oa` | 5,686,145<br>5,686,145<br>5,686,145 | 0.8499<br>0.8499<br>0.8398 | 0.8496<br>0.8488<br>0.8378 | 0.8479<br>0.8465<br>0.8299 | **-0.0019**<br>**-0.0034**<br>**-0.0100*** | -0.0016<br>-0.0023<br>-0.0079 | **Neutral**<br>**Neutral**<br>**Neutral** |

*\*Note on Event 1111008 classification in s1_oa\*: In `s1_oa`, candidate F1 is 0.82985853 and source F1 is 0.83983545, giving an unrounded candidate delta of $\Delta \text{F1} = 0.82985853 - 0.83983545 = -0.0099769197$. Under the analysis contract, an event is classified Harmful strictly if $\Delta \text{F1} \le -0.01000000$. Because $-0.0099769... > -0.0100000$, this outcome falls strictly inside the $(-0.010, +0.010)$ Neutral band. When rounded to four decimal places for compact display, it appears as approximately `-0.0100`, but its exact unrounded value is strictly classified as Neutral under the analysis contract.*

### 2.2 Macro Summary Across Cells

| Metric | Canonical Cell (`s0_oa`) | Order Repeat R1 (`s0_ob`) | Seed Repeat R2 (`s1_oa`) |
| :--- | :---: | :---: | :---: |
| **Source Macro F1** | 0.5949 | 0.5949 | 0.5884 |
| **BN Control Macro F1** | 0.5965 | 0.5986 | 0.5856 |
| **Candidate Macro F1** | 0.5771 | 0.5790 | 0.6035 |
| **Macro F1 $\Delta_{\text{Cand-Src}}$** | **-0.0178** | **-0.0159** | **+0.0151** |
| **Harmful Events Count** | 3 / 6 (50.0%) | 3 / 6 (50.0%) | 2 / 6 (33.3%) |
| **Beneficial Events Count** | 1 / 6 (16.7%) | 2 / 6 (33.3%) | 2 / 6 (33.3%) |
| **Neutral Events Count** | 2 / 6 (33.3%) | 1 / 6 (16.7%) | 2 / 6 (33.3%) |
| **Batch Harm Prevalence** | 40.78% (73 / 179) | 43.58% (78 / 179) | 11.17% (20 / 179) |
| **Baseline Pooled Pixel F1** | 0.4578 | 0.4616 | 0.5831 |

---

## 3. Direct Answers to Core Scientific Questions

### Q1: Does harmful adaptation persist under reverse temporal ordering (Order B)?
**YES.** Under reverse temporal ordering (Order B, Cell R1), harmful adaptation persists with near-identical magnitude and profile. Events 279 ($\Delta \text{F1} = -0.0495$), 437 ($\Delta \text{F1} = -0.0601$), and 520 ($\Delta \text{F1} = -0.0143$) remain strictly harmful. Macro candidate F1 is 0.5790 (vs 0.5771 in Order A), and batch-level harm prevalence remains virtually unchanged at 43.58% (78/179 batches) compared to 40.78% (73/179 batches) in Order A.

### Q2: Does harmful adaptation persist under a different checkpoint initialization (Seed 1)?
**YES, but with a critical structural divergence.** Harmful adaptation does not disappear on Seed 1, but its event distribution changes dramatically. While Event 437 experiences severe degradation ($\Delta \text{F1} = -0.0537$) and Event 559 degrades ($\Delta \text{F1} = -0.0263$), Event 279 undergoes a **complete polarity inversion** from severe collapse on Seed 0 ($\Delta \text{F1} = -0.0545$) to large observed benefit on Seed 1 ($\Delta \text{F1} = +0.1588$). Consequently, overall batch harm prevalence drops to 11.17% (20/179 batches) and Candidate Macro F1 improves to 0.6035.

### Q3: Are the harmful events (279, 437, 520) consistently harmful across orders and seeds?
**NO.** Only **Event 437** (UK / Severn) is consistently harmful across all three cells ($\Delta \text{F1} = -0.0642$ in `s0_oa`, $-0.0601$ in `s0_ob`, $-0.0537$ in `s1_oa`).
- **Event 279**: Consistently harmful across orders on Seed 0, but flips to strongly beneficial on Seed 1 ($+0.1588$).
- **Event 520**: Consistently harmful across orders on Seed 0 ($-0.0140$ and $-0.0143$), but flips to beneficial on Seed 1 ($+0.0203$).
- **Event 559**: Benefited under Seed 0 ($+0.0197$ and $+0.0164$), but became harmful under Seed 1 ($-0.0263$).
This proves that catastrophic adaptation failure is not an inherent static property of an event geography alone, but arises from the interaction between the specific initialization landscape and the incoming domain shift.

### Q4: Which risk signals maintain discrimination across both repeat cells?
**NONE.** No risk indicator achieves positive discrimination across both repeat cells. All pre-update risk indicators that exhibited high AUROC on Seed 0 fail or invert on Seed 1.

### Q5: Does `source_to_candidate_kl` maintain positive correlation with harm across both cells?
**NO. It suffers a complete sign inversion.**
- On `s0_oa`: Spearman $\rho = -0.8857$, Event $\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$.
- On `s0_ob`: Spearman $\rho = -0.7714$, Event $\text{AUROC} = 0.8889$, $\text{AUPRC} = 0.9167$.
- On `s1_oa`: Spearman $\rho = \mathbf{+0.8857}$, Event $\text{AUROC} = \mathbf{0.0000}$, $\text{AUPRC} = \mathbf{0.2667}$.
**Mechanism**: `source_to_candidate_kl` measures the magnitude of predictive displacement caused by adaptation. On Seed 0, large adaptation steps destabilized representations, causing catastrophic collapse on Event 279. On Seed 1, large adaptation steps successfully corrected the severe domain mismatch on Event 279. Because large divergence flags large adaptation regardless of whether that adaptation is beneficial or harmful, high KL divergence is positively correlated with F1 gain on Seed 1, producing an inverted AUROC of 0.0000.

### Q6: Does `s_flood_expansion` maintain discrimination across both cells?
**NO.** While `s_flood_expansion` maintains perfect separation under Order B on Seed 0 ($\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$), its discrimination collapses on Seed 1 ($\text{AUROC} = 0.2500$, $\text{AUPRC} = 0.3333$, Spearman $\rho = -0.0286$). Under Seed 1, flood expansion occurs during valid recovery adaptation, rendering expansion rate an unreliable discriminator of harmful collapse.

### Q7: Does `s_null` show utility in either repeat cell?
**INCONCLUSIVE — weak and unstable.** Across all three cells, `s_null` demonstrates weak, noisy, and unstable discrimination:
- `s0_oa`: $\rho = -0.6000$, Event $\text{AUROC} = 0.6667$, $\text{AUPRC} = 0.6389$.
- `s0_ob`: $\rho = -0.7714$, Event $\text{AUROC} = 0.7778$, $\text{AUPRC} = 0.8056$.
- `s1_oa`: $\rho = -0.2319$, Event $\text{AUROC} = 0.4375$, $\text{AUPRC} = 0.4500$.
`s_null` is classified as **INCONCLUSIVE — weak and unstable** rather than definitively negative, as it produces fluctuating, low-magnitude signal across cells and suffers from undefined values on batches lacking valid water-masked regions, offering no reliable cross-cell predictive utility.

### Q8: How does selective rejection budget ($b=0.20$) perform across all three cells?
- **Under Seed 0 (`s0_oa` and `s0_ob`)**: Selective rejection works effectively. High-risk batch fallback to Source prevents 32.9%–34.6% of harmful batches with 0%–5.6% lost benefits, boosting net realized pixel F1 from ~0.46 to ~0.50.
- **Under Seed 1 (`s1_oa`)**: Selective rejection based on KL divergence or flood expansion **causes severe harm**. At $b=0.20$, rejecting the highest KL batches rejects 30 beneficial batches and 0 harmful batches (29.1% lost benefits), causing net realized pixel F1 to crash from **0.5831 down to 0.5362** (a net destruction of $-0.0469$ F1).

### Q9: What is the formal route decision for E6?
**`E5 COMPLETE — E6 CONTINUES WITH REDUCED SCOPE`**.
A high-capacity, multi-signal calibration controller or aggressive gating mechanism is empirically contraindicated because pre-update adaptation signals are seed-contingent and invert between initializations. Tuning complex gating on one seed risks catastrophic suppression of beneficial adaptation on another. Therefore, E6 should continue with a reduced-scope conservative policy whose exact mechanisms and bounds must be predeclared in the E6 protocol.

---

## 4. Formal Signal Classification

Based on empirical evidence across all three orthogonal evaluation cells ($N = 3 \times 179 = 537$ batch shards), all candidate risk indicators are classified under the frozen three-tier taxonomy:

| Signal Key | Display Name | Family | `s0_oa` AUROC | `s0_ob` AUROC | `s1_oa` AUROC | Cross-Cell Classification | Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `source_to_candidate_kl` | Prediction KL Divergence | Generic | 1.0000 | 0.8889 | 0.0000 | **INCONCLUSIVE** | Highly predictive under Seed 0, but completely inverts on Seed 1 ($\text{AUROC} = 0.0000$). Measures adaptation magnitude rather than direction. |
| `hard_disagreement_rate` | Hard Label Disagreement | Generic | 0.7778 | 0.6667 | 0.0000 | **INCONCLUSIVE** | Divergence proxy; collapses and inverts on Seed 1 due to valid label reassignment. |
| `abs_delta_flood_pixels` | Absolute Flood Pixel Shift | Change | 0.7778 | 0.7778 | 0.2500 | **INCONCLUSIVE** | Sensitive to large shifts; fails on Seed 1 when large shifts correspond to genuine adaptation gain. |
| `s_flood_expansion` | Flood Area Expansion Rate | Change | 1.0000 | 1.0000 | 0.2500 | **INCONCLUSIVE** | Robust across orders on Seed 0, but fails to generalize to Seed 1 ($\text{AUROC} = 0.2500$). |
| `s_null` | Null-Flood Sensitivity Probe | Change | 0.6667 | 0.7778 | 0.4375 | **INCONCLUSIVE — weak and unstable** | Weak and unstable correlation across cells; synthetic probe introduces artifactual variance and data sparsity without reliable cross-cell discrimination. |
| `mean_entropy` | Predictive Entropy | Generic | 0.4444 | 0.2222 | 0.0000 | **NEGATIVE** | Completely uninformative ($\text{AUROC} \le 0.44$); reflects image texture rather than adaptation instability. |
| `mean_max_confidence` | Mean Max Confidence | Generic | 0.4444 | 0.2222 | 0.0000 | **NEGATIVE** | Uninformative; model remains overconfident even during catastrophic representational collapse. |
| `aug_inconsistency` | Augmentation Inconsistency | Generic | 0.4444 | 0.2222 | 0.0000 | **NEGATIVE** | No discrimination for streaming adaptation risk across any cell. |
| `bn_weight_displacement` | BN Weight Displacement | Generic | 0.8889 | 0.8889 | 0.5000 | **NEGATIVE** | Driven entirely by cumulative step count; uninformative at step 0 for individual batches. |
| `s_flood_disappearance` | Flood Disappearance Rate | Change | 0.4444 | 0.4444 | 0.1250 | **NEGATIVE** | Sub-random performance across all cells. |

> [!NOTE]
> **Summary of Classification**:
> - **Positive**: **Zero (0)** signals qualify as globally robust positive discriminators across both seed and order.
> - **Inconclusive**: **Five (5)** signals (`source_to_candidate_kl`, `s_flood_expansion`, `hard_disagreement_rate`, `abs_delta_flood_pixels`, `s_null`) fail to demonstrate stable cross-cell discrimination. Specifically, `s_null` is classified as **INCONCLUSIVE — weak and unstable** due to fluctuating metrics and coverage sparsity.
> - **Negative**: **Five (5)** signals (`mean_entropy`, `mean_max_confidence`, `aug_inconsistency`, `bn_weight_displacement`, `s_flood_disappearance`) fail consistently across all cells and are rejected from further consideration.

---

## 5. Rejection Budget Cross-Cell Performance

### 5.1 Realized Performance at Budget $b = 0.20$ (Rejecting Top 36 Batches)

| Policy / Indicator | `s0_oa` Net F1 (Base: 0.4578) | `s0_ob` Net F1 (Base: 0.4616) | `s1_oa` Net F1 (Base: 0.5831) | Cross-Cell Generalization |
| :--- | :---: | :---: | :---: | :--- |
| **`source_to_candidate_kl`** | **0.4959** (+0.0381) | **0.4953** (+0.0337) | **0.5362** (**-0.0469**) | **Catastrophic Failure on Seed 1** |
| **`s_flood_expansion`** | **0.5000** (+0.0422) | **0.5012** (+0.0396) | **0.5674** (**-0.0157**) | **Degrades Performance on Seed 1** |
| **`abs_delta_flood_pixels`** | **0.5017** (+0.0439) | **0.5032** (+0.0416) | **0.5338** (**-0.0493**) | **Catastrophic Failure on Seed 1** |
| **`hard_disagreement_rate`** | **0.4953** (+0.0375) | **0.4947** (+0.0331) | **0.5408** (**-0.0423**) | **Catastrophic Failure on Seed 1** |
| **`s_null`** | **0.4775** (+0.0197) | **0.4828** (+0.0212) | **0.6090** (+0.0259)* | Inconsistent coverage (rejected only 11 batches in s1_oa) |
| **`mean_entropy`** | **0.4889** (+0.0311) | **0.4864** (+0.0248) | **0.5679** (**-0.0152**) | Fails on Seed 1 |
| **`aug_inconsistency`** | **0.4875** (+0.0297) | **0.4890** (+0.0274) | **0.5654** (**-0.0177**) | Fails on Seed 1 |

*Note: In `s1_oa`, `s_null` had only 11 batches with non-null values exceeding threshold; thus it rejected only 11 batches.*

---

## 6. Synthesis and Epistemic Route Decision

The orthogonal repeat evaluation resolves a central open question for test-time adaptation governance: **Can test-time adaptation risks be reliably detected and gated by uncalibrated pre-update displacement signals across arbitrary model checkpoints?**

The empirical answer is definitively **no**:
1. **Displacement measures adaptation magnitude, not adaptation correctness.** When a model adapts successfully to a large domain shift (as Seed 1 does on Event 279), predictive displacement and divergence signals fire intensely.
2. **Aggressive gating destroys high-performing models.** Applying a rejection gate calibrated on Seed 0 failures to Seed 1 strips away the large observed $+0.1588$ F1 gain on Event 279, severely depressing overall accuracy.
3. **Implications for Block E6**:
   - The original speculative goal of training a complex multivariate calibration gate (e.g., logistic regression or random forest over pre-update signals) must be abandoned as ungrounded and prone to negative transfer.
   - E6 should continue with a reduced-scope conservative policy whose exact mechanisms and bounds must be predeclared in the E6 protocol.

---

## 7. Terminal Closeout Statement

All work package commitments for E5.3 have been fulfilled in accordance with the governing plan. All 537 formal evidence shards remain intact on disk.

```text
E5 COMPLETE — E6 CONTINUES WITH REDUCED SCOPE
```

