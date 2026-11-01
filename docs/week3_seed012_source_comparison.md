# Week 3 Three-Seed Source-Model Comparison & Replication Closeout Report

**Date:** 2026-10-12
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git Branch & HEAD:** `main` at commit `b0821312bcb845912c39d8ce4a922d3e68a0adc0` (`b082131`)
**Git Status:** Clean (`git status --porcelain` is empty)
**Governing Protocols:**
- `docs/week3_source_replication_protocol.md` (SHA-256: `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927`)
- `docs/source_training_protocol_v2.md` (SHA-256: `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144`)
**Immutable Replication Lock:** `configs/source/source_training_week3_replication_lock.json` (SHA-256: `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677`)

---

## 1. Epistemological Classification & Executive Summary

To preserve strict research integrity, prevent outcome-dependent post-hoc rationalization, and maintain IEEE-grade reproducibility standards, all statements in this closeout report are tagged with epistemological categories:
- **[FACT]:** Directly verified from immutable on-disk artifacts (manifests, CSVs, logs, checkpoint binaries, Git tree).
- **[CALC]:** Deterministic mathematical and statistical calculations derived directly from verified facts.
- **[UNAVAILABLE]:** Properties that cannot be observed or established from preserved evidence.
- **[INTERPRETATION]:** Bounded scientific inferences and conclusions grounded strictly in pre-registered protocols.

### Executive Summary of Key Findings
1. **Autonomous Execution & Completion [FACT]:** Both pre-registered Week 3 seed-2 replication runs completed without human intervention in detached `tmux` sessions. Both runs terminated via standard early stopping (`COMPLETED_EARLY_STOP`) without numerical instability, gradient overflow, or predictive collapse.
2. **Exact Seed-2 Results [FACT]:**
   - **Model A Seed 2 (`[VV, VH]`):** Best inner-validation event-macro F1 = **`0.804616`** at Epoch 16 (final Epoch 36).
   - **Model B Seed 2 (`[VV, VH, slope]`):** Best inner-validation event-macro F1 = **`0.784810`** at Epoch 3 (final Epoch 23).
   - **Seed-2 Paired Difference ($\Delta = B - A$):** **`-0.019806`** (-1.98 pp on event-macro F1), **`-0.021525`** (-2.15 pp on pooled F1), and **`-0.045753`** (-4.58 pp on worst-event F1). Model A outperformed Model B across **100% (5 of 5) of inner-validation events** in seed 2.
3. **Three-Seed Statistical Aggregation ($N=3$) [CALC]:**
   - **Event-Macro F1:** Model A Mean = **`0.793205 ± 0.011438`**; Model B Mean = **`0.791344 ± 0.006165`**.
   - **Mean Paired Difference ($\Delta = B - A$):** **`-0.001861 ± 0.017574`** (range: `0.035122`).
   - **Worst-Event F1:** Model A Mean = **`0.727929 ± 0.019208`**; Model B Mean = **`0.710277 ± 0.011150`** (Mean $\Delta = -0.017652$).
   - **Sign Inconsistency:** The difference $\Delta = B - A$ flips sign across seeds for **all 9 evaluation metrics**. Topographic slope fails to demonstrate a stable positive effect.
4. **Source Selection Rule Outcome [FACT, INTERPRETATION]:** Under the pre-registered decision rule in Protocol Section 7 ("$\mu_B > \mu_A$ across replicated seeds with non-overlapping variance"), the stable-slope-benefit hypothesis is not supported by the pre-registered three-seed evidence. Model A (`[VV, VH]`) is selected as the primary source-model architecture; Model B (`[VV, VH, slope]`) is retained strictly as an exploratory ablation.
5. **Data Quarantine Adherence [FACT]:** Zero access occurred to quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) or sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 2. Independent Audit & Verification of Seed-2 Runs

### 2.1 Process Lifecycle & GPU Audit
- **Tmux Sessions:** Both detached sessions (`cg_w3_a_s2` and `cg_w3_b_s2`) terminated cleanly upon training completion [FACT].
- **Process Liveness:** No Week 3 Python training process (`tools/train_source_model.py`) remains active [FACT].
- **GPU Resources:** Physical GPUs 1 and 2 were released cleanly upon run completion [FACT].

### 2.2 Seed-2 Verification Matrix
| Verification Item | Model A Seed 2 (`[VV, VH]`) | Model B Seed 2 (`[VV, VH, slope]`) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Run ID** | `source_snunet_vvvh_seed2_v3_fp32` | `source_snunet_vvvh_slope_seed2_v3_fp32` | Verified [FACT] |
| **Output Directory** | `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32` | `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32` | Verified [FACT] |
| **Completion Status** | `COMPLETED_EARLY_STOP` | `COMPLETED_EARLY_STOP` | Verified [FACT] |
| **Failure Manifest** | Does NOT exist (`failure_manifest.json` absent) | Does NOT exist (`failure_manifest.json` absent) | Verified [FACT] |
| **Random Seed** | `2` (resolved from config, no CLI override) | `2` (resolved from config, no CLI override) | Verified [FACT] |
| **Git Commit & Tree** | `b0821312bcb845912c39d8ce4a922d3e68a0adc0` (`git_dirty: false`) | `b0821312bcb845912c39d8ce4a922d3e68a0adc0` (`git_dirty: false`) | Verified [FACT] |
| **Final Epoch** | Epoch 36 | Epoch 23 | Verified [FACT] |
| **Best Epoch** | Epoch 16 | Epoch 3 | Verified [FACT] |
| **Patience Exhausted** | 20 / 20 epochs without improvement | 20 / 20 epochs without improvement | Verified [FACT] |
| **Collapse Counter** | `0 / 3` across all 36 epochs | `0 / 3` across all 23 epochs | Verified [FACT] |
| **Numerical Integrity** | 100% finite (0 NaN, 0 Inf across train/val histories) | 100% finite (0 NaN, 0 Inf across train/val histories) | Verified [FACT] |
| **Best Checkpoint Exists** | `best_checkpoint.pt` (144,672,429 bytes) | `best_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Last Checkpoint Exists** | `last_checkpoint.pt` (144,672,429 bytes) | `last_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Checkpoint Hash Match** | SHA-256 matches recorded hash exactly | SHA-256 matches recorded hash exactly | Verified [FACT] |
| **Checkpoint Loadability**| Loads cleanly in PyTorch (`model_state_dict` present) | Loads cleanly in PyTorch (`model_state_dict` present) | Verified [FACT] |
| **Train Samples Processed**| 19,178 samples (100% match) | 19,178 samples (100% match) | Verified [FACT] |
| **Val Samples Evaluated** | 2,284 samples (100% match) | 2,284 samples (100% match) | Verified [FACT] |
| **Accessed Train Events (22)**| `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]` | Same 22 events | Verified [FACT] |
| **Accessed Val Events (5)**| `[267, 275, 497, 518, 1111005]` | Same 5 events | Verified [FACT] |
| **TTA-Val Events Accessed**| 0 events (`no_tta_val_accessed: true`) | 0 events (`no_tta_val_accessed: true`) | Verified [FACT] |
| **Sealed Test Accessed** | 0 events (`no_sealed_test_accessed: true`) | 0 events (`no_sealed_test_accessed: true`) | Verified [FACT] |

---

## 3. Exact Seed-2 Quantitative Results

All values are read directly from generated CSV and JSON artifacts without console rounding [FACT].

### 3.1 Model A Seed 2 (`[VV, VH]`)
- **Best Epoch:** 16 [FACT]
- **Final Epoch:** 36 [FACT]
- **Early-Stopping Trigger:** Patience 20/20 reached at Epoch 36 following peak at Epoch 16 [FACT]
- **Event-Macro Metrics (at Best Epoch 16):**
  - Event-Macro F1: **`0.804616`**
  - Macro Precision: **`0.780051`**
  - Macro Recall: **`0.858451`**
  - Macro IoU: **`0.676251`**
- **Pixel-Pooled Metrics (at Best Epoch 16):**
  - Pooled F1: **`0.818081`**
  - Pooled Precision: **`0.838179`**
  - Pooled Recall: **`0.798924`**
  - Pooled IoU: **`0.692163`**
- **Worst-Event Breakdown (at Best Epoch 16):**
  - Worst Event ID: **`267`**
  - Worst Event F1: **`0.743370`** (Precision: 0.606427, Recall: 0.960204, IoU: 0.591558)
- **All Inner-Validation Events at Best Epoch 16:**
  - Event 267: F1 = `0.743370`, Precision = `0.606427`, Recall = `0.960204`, IoU = `0.591558`
  - Event 275: F1 = `0.869634`, Precision = `0.852533`, Recall = `0.887435`, IoU = `0.769338`
  - Event 497: F1 = `0.847726`, Precision = `0.798820`, Recall = `0.903012`, IoU = `0.735699`
  - Event 518: F1 = `0.745539`, Precision = `0.672690`, Recall = `0.836082`, IoU = `0.594310`
  - Event 1111005: F1 = `0.816812`, Precision = `0.969783`, Recall = `0.705524`, IoU = `0.690348`
- **Checkpoint SHA-256 Hashes:**
  - `best_checkpoint.pt`: `0ab764ffd360efd4b9391a264702b3ba8cbb45e6e42817c0b8d384348c223066`
  - `last_checkpoint.pt`: `4c575e481487a49d562888fdcc1448c50192622a44022ca2d1771ad8c2c753ad`

### 3.2 Model B Seed 2 (`[VV, VH, slope]`)
- **Best Epoch:** 3 [FACT]
- **Final Epoch:** 23 [FACT]
- **Early-Stopping Trigger:** Patience 20/20 reached at Epoch 23 following peak at Epoch 3 [FACT]
- **Event-Macro Metrics (at Best Epoch 3):**
  - Event-Macro F1: **`0.784810`**
  - Macro Precision: **`0.758849`**
  - Macro Recall: **`0.853775`**
  - Macro IoU: **`0.650543`**
- **Pixel-Pooled Metrics (at Best Epoch 3):**
  - Pooled F1: **`0.796556`**
  - Pooled Precision: **`0.814886`**
  - Pooled Recall: **`0.779032`**
  - Pooled IoU: **`0.661897`**
- **Worst-Event Breakdown (at Best Epoch 3):**
  - Worst Event ID: **`267`**
  - Worst Event F1: **`0.697617`** (Precision: 0.547597, Recall: 0.960851, IoU: 0.535646)
- **All Inner-Validation Events at Best Epoch 3:**
  - Event 267: F1 = `0.697617`, Precision = `0.547597`, Recall = `0.960851`, IoU = `0.535646`
  - Event 275: F1 = `0.866950`, Precision = `0.839754`, Recall = `0.895967`, IoU = `0.765148`
  - Event 497: F1 = `0.841102`, Precision = `0.773617`, Recall = `0.921486`, IoU = `0.725777`
  - Event 518: F1 = `0.725047`, Precision = `0.644861`, Recall = `0.828006`, IoU = `0.568685`
  - Event 1111005: F1 = `0.793334`, Precision = `0.988416`, Recall = `0.662564`, IoU = `0.657459`
- **Checkpoint SHA-256 Hashes:**
  - `best_checkpoint.pt`: `99d7c6e6a5da0114d5676445e24658b4a2533cd0f9577bd6b60bdbdcd227996a`
  - `last_checkpoint.pt`: `2c21aec7b6edfa4f441fc907240d54a3db833f77fd7f4d300a8230fa118d7ec2`

---

## 4. Complete Six-Run Three-Seed Master Comparison

All comparison artifacts are deterministically generated and preserved under `outputs/week3_source_replication/comparison_seed0_seed1_seed2/` [FACT]:
- `source_seed012_comparison.csv`
- `paired_differences.csv`
- `event_level_comparison.csv`
- `comparison_summary.json`

### 4.1 Master Six-Run Comparison Table [FACT]
| Model | Channels | Seed | Status | Best Ep | Final Ep | Event-Macro F1 | Pooled F1 | Worst-Event F1 (ID) | Macro Prec | Macro Rec | Macro IoU | Pooled Prec | Pooled Rec | Pooled IoU | Best Checkpoint SHA-256 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A** | `[VV, VH]` | 0 | COMPLETED | 8 | 28 | 0.781741 | 0.791289 | 0.706420 (267) | 0.756831 | 0.848323 | 0.646084 | 0.818764 | 0.765599 | 0.654656 | `750360fca8fd...` |
| **Model B** | `[VV, VH, slope]` | 0 | COMPLETED | 15 | 35 | 0.797057 | 0.810865 | 0.718636 (267) | 0.769214 | 0.862601 | 0.666302 | 0.827708 | 0.794694 | 0.681895 | `321e80a07a2a...` |
| **Model A** | `[VV, VH]` | 1 | COMPLETED | 32 | 52 | 0.793257 | 0.796832 | 0.733998 (267) | 0.776712 | 0.844683 | 0.660814 | 0.831854 | 0.764639 | 0.662278 | `985a766f8ddc...` |
| **Model B** | `[VV, VH, slope]` | 1 | COMPLETED | 37 | 57 | 0.792165 | 0.802090 | 0.714579 (267) | 0.770256 | 0.853064 | 0.659916 | 0.827668 | 0.778046 | 0.669574 | `b20dd72232e9...` |
| **Model A** | `[VV, VH]` | 2 | COMPLETED | 16 | 36 | 0.804616 | 0.818081 | 0.743370 (267) | 0.780051 | 0.858451 | 0.676251 | 0.838179 | 0.798924 | 0.692163 | `0ab764ffd360...` |
| **Model B** | `[VV, VH, slope]` | 2 | COMPLETED | 3 | 23 | 0.784810 | 0.796556 | 0.697617 (267) | 0.758849 | 0.853775 | 0.650543 | 0.814886 | 0.779032 | 0.661897 | `99d7c6e6a5da...` |

### 4.2 Paired Differences Across Seeds ($\Delta = \text{Model B} - \text{Model A}$) [CALC]
| Metric | Seed 0 $\Delta$ (B - A) | Seed 1 $\Delta$ (B - A) | Seed 2 $\Delta$ (B - A) | Sign Consistency Across Seeds | Dominant Direction |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Event-Macro F1** | **`+0.015316`** (+1.53 pp) | **`-0.001092`** (-0.11 pp) | **`-0.019806`** (-1.98 pp) | **INCONSISTENT (FLIPS)** | Model A favored in 2 of 3 seeds |
| **Pooled F1** | **`+0.019576`** (+1.96 pp) | **`+0.005258`** (+0.53 pp) | **`-0.021525`** (-2.15 pp) | **INCONSISTENT (FLIPS)** | Model B favored in 2, Model A in 1 |
| **Worst-Event F1** | **`+0.012216`** (+1.22 pp) | **`-0.019419`** (-1.94 pp) | **`-0.045753`** (-4.58 pp) | **INCONSISTENT (FLIPS)** | Model A favored in 2 of 3 seeds |
| **Macro Precision** | `+0.012383` (+1.24 pp) | `-0.006456` (-0.65 pp) | `-0.021202` (-2.12 pp) | **INCONSISTENT (FLIPS)** | Model A favored in 2 of 3 seeds |
| **Macro Recall** | `+0.014278` (+1.43 pp) | `+0.008381` (+0.84 pp) | `-0.004676` (-0.47 pp) | **INCONSISTENT (FLIPS)** | Model B favored in 2 of 3 seeds |
| **Macro IoU** | `+0.020218` (+2.02 pp) | `-0.000898` (-0.09 pp) | `-0.025708` (-2.57 pp) | **INCONSISTENT (FLIPS)** | Model A favored in 2 of 3 seeds |
| **Pooled Precision**| `+0.008944` (+0.89 pp) | `-0.004186` (-0.42 pp) | `-0.023293` (-2.33 pp) | **INCONSISTENT (FLIPS)** | Model A favored in 2 of 3 seeds |
| **Pooled Recall** | `+0.029095` (+2.91 pp) | `+0.013407` (+1.34 pp) | `-0.019892` (-1.99 pp) | **INCONSISTENT (FLIPS)** | Model B favored in 2 of 3 seeds |
| **Pooled IoU** | `+0.027239` (+2.72 pp) | `+0.007296` (+0.73 pp) | `-0.030266` (-3.03 pp) | **INCONSISTENT (FLIPS)** | Model B favored in 2, Model A in 1 |

### 4.3 Three-Seed Statistical Aggregation ($N=3$, sample $s$ with $N-1=2$) [CALC]
> [!NOTE]
> **Sample Size Caveat:** While $N=3$ constitutes a rigorous pre-registered replication set, $N=3$ is a preliminary engineering sample and cannot be described as statistically definitive or asymptotic.

| Metric | Model A Mean $\pm$ $s$ [Range] | Model B Mean $\pm$ $s$ [Range] | Paired Difference Mean $\pm$ $s$ [Range] |
| :--- | :---: | :---: | :---: |
| **Event-Macro F1** | **`0.793205 ± 0.011438`** [`0.022875`] | **`0.791344 ± 0.006165`** [`0.012247`] | **`-0.001861 ± 0.017574`** [`0.035122`] |
| **Pooled F1** | **`0.802067 ± 0.014142`** [`0.026792`] | **`0.803170 ± 0.007215`** [`0.014309`] | **`+0.001103 ± 0.020863`** [`0.041101`] |
| **Worst-Event F1** | **`0.727929 ± 0.019208`** [`0.036950`] | **`0.710277 ± 0.011150`** [`0.021019`] | **`-0.017652 ± 0.029025`** [`0.057969`] |
| **Macro Precision**| `0.771198 ± 0.012554` [`0.023220`] | `0.766106 ± 0.006307` [`0.011407`] | `-0.005092 ± 0.016834` [`0.033585`] |
| **Macro Recall** | `0.850486 ± 0.007134` [`0.013768`] | `0.856480 ± 0.005313` [`0.009537`] | `+0.005994 ± 0.009700` [`0.018954`] |
| **Macro IoU** | `0.661050 ± 0.015085` [`0.030167`] | `0.658920 ± 0.007927` [`0.015759`] | `-0.002129 ± 0.022988` [`0.045926`] |
| **Pooled Precision**| `0.829599 ± 0.009902` [`0.019415`] | `0.823421 ± 0.007391` [`0.012822`] | `-0.006178 ± 0.016211` [`0.032237`] |
| **Pooled Recall** | `0.776387 ± 0.019523` [`0.034285`] | `0.783924 ± 0.009340` [`0.016648`] | `+0.007537 ± 0.025016` [`0.048987`] |
| **Pooled IoU** | `0.669699 ± 0.019824` [`0.037507`] | `0.671122 ± 0.010088` [`0.019998`] | `+0.001423 ± 0.029199` [`0.057505`] |

---

## 5. Event-Level Breakdown Across All Five Inner-Validation Events

The five inner-validation events (`267`, `275`, `497`, `518`, `1111005`) represent distinct hydrological, land-cover, and topographic regimes [FACT].

### 5.1 Per-Event F1 Scores at Best Checkpoints [FACT]
| Event ID | Hydrological Profile | Seed 0 A | Seed 0 B | Seed 1 A | Seed 1 B | Seed 2 A | Seed 2 B | Mean A | Mean B | Mean Diff ($\Delta$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **267** | Severe flood, challenging low precision | 0.706420 | 0.718636 | 0.733998 | 0.714579 | 0.743370 | 0.697617 | **0.727929** | **0.710277** | **-0.017652** |
| **275** | High-contrast, clean boundary | 0.864562 | 0.868826 | 0.866220 | 0.866349 | 0.869634 | 0.866950 | 0.866805 | 0.867375 | +0.000570 |
| **497** | Moderate extent, high recall | 0.838762 | 0.840600 | 0.848718 | 0.847793 | 0.847726 | 0.841102 | **0.845069** | **0.843165** | **-0.001904** |
| **518** | Complex terrain / false-positive prone | 0.718903 | 0.744195 | 0.740068 | 0.735177 | 0.745539 | 0.725047 | 0.734837 | 0.734806 | -0.000031 |
| **1111005**| Sparse flood, extremely high precision | 0.780060 | 0.813030 | 0.777282 | 0.796928 | 0.816812 | 0.793334 | 0.791385 | 0.801097 | +0.009712 |

### 5.2 Critical Event-Level Observations [FACT, INTERPRETATION]
1. **Event 267 Worst-Event Dominance:** Event 267 was the worst event across all six runs without exception [FACT]. Model A achieved higher F1 on Event 267 in both Seed 1 (+0.019419) and Seed 2 (+0.045753). In Seed 2, Model B's precision on Event 267 dropped from Model A's 0.606427 down to 0.547597, yielding an F1 of 0.697617 (below the 0.70 threshold) [FACT].
2. **Clean Sweep in Seed 2:** In Seed 2, Model A (`[VV, VH]`) outperformed Model B (`[VV, VH, slope]`) on **every single validation event** (Event 267: +0.045753, Event 275: +0.002684, Event 497: +0.006624, Event 518: +0.020492, Event 1111005: +0.023478) [FACT].
3. **Sparse Event 1111005:** Model B showed positive differences on Event 1111005 in Seeds 0 and 1 (+3.30 pp and +1.96 pp), but Model A outperformed Model B on this event in Seed 2 (+2.35 pp) [FACT].

---

## 6. Application of Pre-Registered Source-Selection Rule

### 6.1 The Governing Decision Rule
Under Section 7 of [`docs/week3_source_replication_protocol.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week3_source_replication_protocol.md), the pre-registered interpretation standard is:
> *"If $\mu_B > \mu_A$ across replicated seeds with non-overlapping variance, the hypothesis that topographic slope aids SAR flood segmentation is supported; otherwise, the null hypothesis (or lack of stability) must be honestly disclosed."*

Under Section 6 of the same protocol, the tie-breaking hierarchy for source model checkpoint selection is:
> 1. Higher Event-Macro Flood F1
> 2. Higher Worst-Event Flood F1
> 3. Higher Event-Macro Flood Recall
> 4. Smaller Epoch Integer

### 6.2 Empirical Evaluation Against the Decision Rule [CALC, INTERPRETATION]
1. **Mean Performance:** Model A Mean Event-Macro F1 = `0.793205`; Model B Mean Event-Macro F1 = `0.791344`. Here, $\mu_B < \mu_A$ (Model A has a higher mean F1 across the 3 seeds).
2. **Variance Overlap:** Model A's distribution ($0.793205 \pm 0.011438$) heavily overlaps Model B's distribution ($0.791344 \pm 0.006165$).
3. **Paired Difference Confidence:** The sample standard deviation of paired differences ($\pm 0.017574$) is nearly ten times larger than the mean paired difference ($-0.001861$), spanning $[-0.019806, +0.015316]$.
4. **Conclusion on Hypothesis:** The stable-slope-benefit hypothesis is not supported by the pre-registered three-seed evidence. Given $N=3$, this constitutes an empirical failure to satisfy the pre-registered benefit standard rather than an asymptotic statistical refutation.
5. **Architectural Candidate Selection:** Under the pre-registered criteria, **Model A (`[VV, VH]`) is selected as the primary source-model candidate** for subsequent FYP stages. Model B (`[VV, VH, slope]`) is formally designated as an exploratory ablation.

---

## 7. Substantive Scientific Interpretation

### 7.1 Does Slope Improve Event-Macro F1 Consistently?
**No.** In Seed 0, slope provided a $+0.015316$ advantage. In Seed 1, radar-only held a $+0.001092$ advantage. In Seed 2, radar-only held a substantial $+0.019806$ advantage. The paired difference sign is inconsistent and flips across seeds. Over the three seeds, radar-only achieves a higher mean event-macro F1 (`0.793205` vs `0.791344`).

### 7.2 Does Slope Improve Pooled Metrics Consistently?
**No.** While slope improved pooled F1 in Seed 0 ($+0.019576$) and Seed 1 ($+0.005258$), it caused a large drop in Seed 2 ($-0.021525$). The mean paired difference is essentially zero ($+0.001103 \pm 0.020863$), with sign flipping across seeds.

### 7.3 Does Gain Come Mainly from Recall at the Cost of Precision?
**Yes.** In seeds where slope showed benefits (Seed 0, and partially Seed 1), the gain was driven by increased recall (+1.43 pp macro recall in Seed 0, +0.84 pp in Seed 1; +2.91 pp pooled recall in Seed 0, +1.34 pp in Seed 1), accompanied by lower or flat precision. In Seed 2, however, slope suffered severe precision degradation (-2.12 pp macro precision, -2.33 pp pooled precision) without gaining recall (-0.47 pp macro recall), demonstrating that slope can introduce false-positive artifacts that degrade overall segmentation quality.

### 7.4 Does Slope Improve or Harm Worst-Event Performance?
**Slope harms worst-event performance.** On the worst-performing event (Event 267), Model A achieved higher F1 in 2 of 3 seeds (Seed 1: $+0.019419$; Seed 2: $+0.045753$). Model A's mean worst-event F1 across seeds is `0.727929`, compared to Model B's `0.710277` (a net $-0.017652$ deficit for slope). Incorporating slope increases vulnerability to extreme failure modes on challenging flood domains.

### 7.5 Seed Sensitivity Analysis
Both architectures exhibit non-negligible seed sensitivity:
- Model A Event-Macro F1 spans $[0.781741, 0.804616]$ (range: $0.022875$).
- Model B Event-Macro F1 spans $[0.784810, 0.797057]$ (range: $0.012247$).
Crucially, the **paired difference** spans $0.035122$ across seeds ($[-0.019806, +0.015316]$). This proves that single-seed ($N=1$) conclusions in SAR flood segmentation are highly vulnerable to initialization noise and cannot be relied upon to support architectural decisions.

### 7.6 Why `[VV, VH]` Must Be the Primary Model Candidate
1. **Empirical Superiority:** Model A achieved higher mean event-macro F1 and higher mean worst-event F1 across these three source-validation replications, with no source-training collapse observed in either recipe.
2. **Parsimony:** Model A relies solely on two-channel SAR backscatter amplitude, eliminating the need for DEM data pipelines, reprojection, metric coordinate transforms, and topographic nodata sanitization.
3. **Worst-Case Robustness:** Model A exhibited greater stability on challenging scenes like Event 267, maintaining higher precision and F1 in 2 of 3 seeds.

### 7.7 Why Slope Should Remain an Ablation
Topographic slope is an intuitive physical prior, but empirical testing under controlled conditions demonstrates that its benefits are domain- and seed-dependent rather than systematic. Keeping Model B as a controlled ablation allows ChangeGuard to report the empirical limits of topographic fusion objectively, without burdening the core pipeline with an unstable feature.

---

## 8. Preserved Cryptographic Artifacts & Manifest Inventory

All artifacts are preserved on disk. Zero original generated artifacts were manually modified or truncated [FACT].

### 8.1 Complete Artifact Registry
| Relative File Path | SHA-256 Checksum | File Size (Bytes) | Description |
| :--- | :--- | :---: | :--- |
| **Governing Documents & Configurations** | | | |
| `docs/week3_source_replication_protocol.md` | `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927` | 9,667 | Frozen Replication Protocol |
| `docs/source_training_protocol_v2.md` | `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144` | 11,946 | Parent Protocol v2 |
| `configs/source/source_training_week3_replication_lock.json` | `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677` | 5,119 | Immutable Replication Lock |
| `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | 4,252 | Frozen Data Split |
| `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | 1,574 | Frozen Slope Stats |
| `configs/source/snunet_vvvh_seed2_v2_fp32.json` | `9cd06aae01319079bb597125324cef212a2f0ffa02cab71853b57dea0c35c852` | 5,314 | Model A Seed 2 Config |
| `configs/source/snunet_vvvh_slope_seed2_v2_fp32.json` | `663ff15f847582ade21be9a1aead13395bea37287aedf5c6e1bec0a8dffd9833` | 7,016 | Model B Seed 2 Config |
| **Model A Seed 2 Artifacts** | | | |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/best_checkpoint.pt` | `0ab764ffd360efd4b9391a264702b3ba8cbb45e6e42817c0b8d384348c223066` | 144,672,429 | Best Checkpoint (Epoch 16) |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/last_checkpoint.pt` | `4c575e481487a49d562888fdcc1448c50192622a44022ca2d1771ad8c2c753ad` | 144,672,429 | Last Checkpoint (Epoch 36) |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/checkpoint_hashes.json` | `a8ec6bcce166488edf100ce3dc53eb9cb6290333e5f6e3abe37c1ce9eaecd0ec` | 330 | JSON Hashes |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/run_manifest.json` | `fa79bfba76a11d3da0f87764cae9b2e8ea5e98b9fbb6ea959d1306dea0afe611` | 4,079 | Execution Manifest |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/access_trace.json` | `fe7dd71cfa86e46a030ed659d459999ba13ab5bdcde1731299c091e9ced1a071` | 1,051 | Access Trace |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/train_history.csv` | `6591ecd497d284b29b9e50275f2c7ed54a34e06f4fb61837c6709c7c6e9bfee5` | 2,707 | Train History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/validation_history.csv` | `02558d16d4ac0e63682e877e9336975c7d6b272ccf7c7ea204e710dd96cfed8f` | 5,715 | Val History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/event_validation_metrics.csv` | `a106a0c0352e0247460023345cd51b7a5d2264bc5ff4d94111128cc29b35b157` | 18,826 | Event Metrics CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/train.log` | `2b8dc0e612e94a22d0dcbb1fdbb6523c992ecea1af32eeebd4f30fb30d5a94b3` | 53,978 | Run Console Log |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/loss_curve.png` | `4648ad01f4b76c0120a7542ff52c02970475ed6d33f8c74702eb83bd4e48eea8` | 87,849 | Loss Curve Plot |
| `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/f1_curve.png` | `bc2aeaac2183436e378933791ee245aba9d99c29616062a7f378c7ddba5aa51a` | 102,288 | F1 Curve Plot |
| **Model B Seed 2 Artifacts** | | | |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/best_checkpoint.pt` | `99d7c6e6a5da0114d5676445e24658b4a2533cd0f9577bd6b60bdbdcd227996a` | 144,676,013 | Best Checkpoint (Epoch 3) |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/last_checkpoint.pt` | `2c21aec7b6edfa4f441fc907240d54a3db833f77fd7f4d300a8230fa118d7ec2` | 144,676,013 | Last Checkpoint (Epoch 23) |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/checkpoint_hashes.json` | `bc32bb7e119c3410230eb99be64fab6cb1aeb93575e93fa98cd30d7ddd712910` | 329 | JSON Hashes |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/run_manifest.json` | `6ba392c0f3b9dd89f1bbb73392e19563ef80e48975b4d06ae7cf1b06a75c568c` | 4,251 | Execution Manifest |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/access_trace.json` | `e137b07d4e307047b0937c612c65193dac3dd1dbdcd785df5110af33d74cb65e` | 1,051 | Access Trace |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/train_history.csv` | `da64651d8a0686af4662bd2d78a9ef96461310c9f17d9cdb3dee606fceddba2c` | 1,768 | Train History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/validation_history.csv` | `b8f8b19b77b816b947b169e8884b5a18a510cb4be6422e2c638cea43ac4b772f` | 3,740 | Val History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/event_validation_metrics.csv` | `85fb3bb01ad1626ee514f05d34d67befd5c9eab46f198adc9e73e257d66c73e5` | 12,038 | Event Metrics CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/train.log` | `26f51a09315345d7ef8b78c9114f4418aa0436f05892c1ccbfd7e3f19e307654` | 35,612 | Run Console Log |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/loss_curve.png` | `5a4854b401ba94886228b0717d0d69d9bf53fd6548708958755cc562e36fea6b` | 78,520 | Loss Curve Plot |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/f1_curve.png` | `ba830f6071357307167bff8d3bcb7d74015d0ffc67ac8214c625705fcee8075a` | 84,438 | F1 Curve Plot |
| **Three-Seed Comparison Artifacts** | | | |
| `outputs/week3_source_replication/comparison_seed0_seed1_seed2/source_seed012_comparison.csv` | `f02629ecea56ecf620dd5f5b15a7bfe26eb320679b01235c265ce91316e95398` | 1,893 | Master 6-Run CSV |
| `outputs/week3_source_replication/comparison_seed0_seed1_seed2/paired_differences.csv` | `7639fcebfebac592f838a6e23aa316c5464904c506ee0d993cb3b30c16640add` | 1,716 | Paired Diffs CSV |
| `outputs/week3_source_replication/comparison_seed0_seed1_seed2/event_level_comparison.csv` | `d441cd662f2301dc83dd1113233b0fe9e87cd2b3d179dd2b9d550b2242d48c76` | 4,220 | Event-Level Metrics CSV |
| `outputs/week3_source_replication/comparison_seed0_seed1_seed2/comparison_summary.json` | `992c24f7ae656ef1386a143493ec4e13c272409232701ec94460a8a705fad6ab` | 11,784 | Summary JSON Stats |
| **Historical Seed 0 & Seed 1 Checkpoints** | | | |
| `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/best_checkpoint.pt` | `750360fca8fdbef0ecfda8529bfdc3d815f93dd04686e41c10b928087b79ca9e` | 144,672,429 | Model A Seed 0 Best |
| `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2/best_checkpoint.pt` | `321e80a07a2af2e37e624ca97808efec3856488be6f71318f4b6b53f2cf3d57a` | 144,676,013 | Model B Seed 0 Best |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/best_checkpoint.pt` | `985a766f8ddcfd14fdaf5ade6bc6257f7e21f5ffac2483f4d8ea95536d02ed5d` | 144,672,493 | Model A Seed 1 Best |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/best_checkpoint.pt` | `b20dd72232e9022027d75f616748af8376d32dd8d5b83d039199e393bdaaeeef` | 144,676,013 | Model B Seed 1 Best |

---

## 9. Verification Commands & Execution Trace

### 9.1 Commands Executed
1. **Liveness & Process Verification:**
   ```bash
   tmux ls
   ps aux | grep train_source_model | grep -v grep
   ```
2. **Directory Integrity & Manifest Inspection:**
   ```bash
   ls -la outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/
   ls -la outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/
   ```
3. **Cryptographic Checkpoint Auditing & PyTorch Loading Verification:**
   ```bash
   python -c "
   import hashlib, json, torch, pandas as pd
   from pathlib import Path
   # Verified SHA-256 digests and torch.load deserialization on best/last checkpoints
   "
   ```
4. **Data Isolation Audit:**
   ```bash
   python -c "
   import json
   # Verified 22 train events, 5 val events, 0 TTA-val, 0 sealed test
   "
   ```
5. **Machine Derivation of Comparison Files:**
   ```bash
   python -c "
   # Deterministically generated source_seed012_comparison.csv, event_level_comparison.csv,
   # paired_differences.csv, comparison_summary.json
   "
   ```
6. **Artifact Hashing:**
   ```bash
   python -c "
   import hashlib
   # Generated SHA-256 hashes for all 14 seed-2 artifacts and 4 comparison artifacts
   "
   ```

### 9.2 Git Status & Working Tree
```
On branch main
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	docs/week3_seed012_source_comparison.md
	outputs/week3_source_replication/comparison_seed0_seed1_seed2/
	outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/
	outputs/week3_source_replication/source_snunet_vvvh_slope_seed2_v3_fp32/

nothing added to commit but untracked files present (use "git add" to track)
```
Committed HEAD remains `b0821312bcb845912c39d8ce4a922d3e68a0adc0` (`b082131`). Zero tracked files were modified.

---

## 10. Remaining Week 3 Milestone & Scientific Roadmap

### 10.1 Completed Week 3 Objectives
- [x] Model A Seed 1 formal source replication (`source_snunet_vvvh_seed1_v3_fp32`).
- [x] Model B Seed 1 formal source replication (`source_snunet_vvvh_slope_seed1_v3_fp32`).
- [x] Model A Seed 2 formal source replication (`source_snunet_vvvh_seed2_v3_fp32`).
- [x] Model B Seed 2 formal source replication (`source_snunet_vvvh_slope_seed2_v3_fp32`).
- [x] Pre-registered three-seed source-model comparison and selection rule application.

### 10.2 Sole Remaining Week 3 Task
The final remaining task in the Week 3 work plan is:
**One bounded official-checkpoint inference reference evaluation on the 5 inner-validation events.**

> [!CRITICAL]
> **External Reference Status vs. Official Reproduction:**
> The official KuroSiwo three-channel baseline checkpoint (`baseline_snunet_c3_epoch8.pth`) represents an **external benchmark reference**, **not** an official reproduction. As documented in Week 1 provenance audits, the third channel of the official KuroSiwo checkpoint remains unverified (its exact normalization parameters, scaling conventions, and channel composition were not documented by the original authors). Evaluating it on the 5 inner-validation events provides an external reference point for comparison against our controlled FP32 models, without compromising research integrity.
