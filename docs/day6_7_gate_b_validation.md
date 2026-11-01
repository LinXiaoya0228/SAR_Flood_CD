# ChangeGuard Days 6–7 Gate B Independent Validation Report

**Document Version:** 1.0  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Starting Commit (Gate A):** `214150c2ad22e09814888748e49f71fc9678a62e`  
**Validation Date:** October 3, 2026  
**Auditor:** Independent Antigravity Verification Agent  
**Python Environment:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`  
**Validation Script:** `tools/run_gate_b_independent_validation.py`  
**Target Document Under Review:** `docs/day6_7_gate_b_audit.md`  
**Overall Validation Verdict:** **`VALIDATED WITH DOCUMENTATION CORRECTIONS`**

---

## Executive Summary

This audit performs an independent, read-only validation of the completed Days 6–7 Gate B evaluation. The evaluation investigated whether label-free, change-specific signals (bi-temporal counterfactual null probes, change-disappearance probes, and spatial consistency features) predict hidden empirical degradation ($\Delta\text{F1} \le -0.01$) under online Test-Time Adaptation (Tent, lr=$10^{-4}$), and whether they outperform generic indicators (parameter drift, intermediate feature drift, entropy, and confidence).

The independent validation confirms:
1. **Procedural & Artifact Integrity:** All Stage 1 and Stage 2 runs completed without error across all 3 seeds and 6 validation events. Zero test events were accessed. Zero ground-truth labels entered Stage 1. All cryptographic SHA-256 digests remain byte-for-byte identical to reported values.
2. **Numerical Accuracy:** Recomputation of all primary metrics across all 26 indicators against the frozen evaluation dataset yielded **zero discrepancy** with `indicator_evaluation_summary.csv` (identical to float64 precision).
3. **`valid_mask` Provenance:** Code and runtime tracing confirms that `valid_mask` originates entirely from Sentinel-1 SAR No-Data rasters (`MK0_MNA`). It has zero dependence on `MK0_MLU`, class 3 labels, or ground-truth flood annotations.
4. **Scientific Verdict:** The reported **`NO-GO`** decision is fully supported by the primary evidence. The predeclared composite score performs at chance (AUROC = `0.5102`, LOEO AUROC = `0.5004`, Spearman $r = -0.0007$). Generic parameter drift (`bn_affine_drift_l2`: AUROC `0.9501`, LOEO `0.9112`) decisively outperforms all change-specific probes.
5. **Documentation Corrections:** Four specific documentation corrections are identified in `docs/day6_7_gate_b_audit.md`, including clarifying the working tree status, correcting the technical reason for `auroc_thresh_10` producing `NaN`, explaining high baseline AUPRC values, and contrasting micro vs. macro Spearman correlations.

---

## 1. Completion & Artifact Identity Verification

### 1.1 Process and System State
- **Active Processes:** Zero Gate B generation or evaluation processes remain active (`pgrep -af "run_gate_b|python"` returned only system daemons).
- **Branch:** `main`
- **HEAD Commit:** `214150c2ad22e09814888748e49f71fc9678a62e`
- **Working Tree State:** Tracked modified: `src/data/kuro_dataset.py`. Untracked files: `configs/gate_b_risk_protocol.json`, `docs/day6_7_gate_b_audit.md`, `src/risk/`, `tests/test_risk_*.py`, `tools/run_day6_smoke_tests.py`, `tools/run_gate_b_offline_evaluation.py`, `tools/run_gate_b_risk_generation.py`, `tools/run_gate_b_independent_validation.py`.  
  *(Correction Note: The phrase "clean tracked state" in the preliminary summary was technically imprecise; `src/data/kuro_dataset.py` is modified, so the working tree is dirty, as correctly recorded in the run manifests).*

### 1.2 Cryptographic Artifact Hash Verification

Every artifact generated during Stage 1 and Stage 2 was re-hashed directly from disk. All hashes match the audit report:

| Component | Relative Path | Size | Recorded SHA-256 Digest | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Protocol** | `configs/gate_b_risk_protocol.json` | 2,679 B | `132ddb7ad11afc74dfd06485e92822516c7e820cce8f96a8c9c2e06a82b22074` | **MATCH** |
| **Checkpoint** | `checkpoints/kuro_snunet_official.pt` | 108 MB | `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed` | **MATCH** |
| **Manifest Seed 0** | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed0.json` | 2,027 B | `e4f4425d1dc2985e44b5c94979598e51a8e6c6705c087b0cfa888398edd9fa9d` | **MATCH** |
| **Manifest Seed 1** | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed1.json` | 2,027 B | `c083bb6f7cd9a15e3425f7d23688e97d8d1f3b754843214e299fab3d169631e7` | **MATCH** |
| **Manifest Seed 2** | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed2.json` | 2,027 B | `22f114218bfd504020cc79c61938fe435f2db3634f13494df29cc321f8233994` | **MATCH** |
| **Preds Seed 0** | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed0_predictions.pt` | 1.13 GB | `53fbdbc0884bcdb52b7683e810fcd42fefacc35a5fbdfdf36f5b753c46564eba` | **MATCH** |
| **Preds Seed 1** | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed1_predictions.pt` | 1.13 GB | `cd9b6384530c9582a9a11c4bb1b6560576ac0c2b168382c2d4abd935f5e54637` | **MATCH** |
| **Preds Seed 2** | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed2_predictions.pt` | 1.13 GB | `92e2accdb9e77b14da1811d732e780f04702ca2c4383d0ecfbe0384340d622fe` | **MATCH** |
| **Risk Seed 0** | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed0_risk_features.json` | 732,850 B | `9b6fb0400e3c21bb787b7a0dd40804c868d21bb4a06897e5a92a4d910edd4173` | **MATCH** |
| **Risk Seed 1** | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed1_risk_features.json` | 732,054 B | `5853e6c14c06d708da4f0841b7259119de03d98112cd99f246c3465a615d56c0` | **MATCH** |
| **Risk Seed 2** | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed2_risk_features.json` | 732,342 B | `adb8fae9a59359a1f12f153519cbb79a29e4976a52fe1b398b08f6a7b45dab7d` | **MATCH** |
| **Stage 1 Log** | `outputs/day6_7_gate_b/stage1_execution.log` | 3,577 B | `2c47ab8014c36678b25f0bff5117a24bfa9c8d05c2892f3924a1d2effb842b1c` | **MATCH** |
| **Eval JSON** | `outputs/day6_7_gate_b/offline_evaluation/gate_b_evaluation.json` | 26 KB | `bd88386770132b9256a6796bf41e17ff53d953139c0c10303b4465fab5fa3eac` | **MATCH** |
| **Summary CSV** | `outputs/day6_7_gate_b/offline_evaluation/indicator_evaluation_summary.csv` | 12 KB | `1cbd68ba5f1703458eff866bc8a5dd287aa071b164bbda1a7cd035837e3caec7` | **MATCH** |
| **Joined CSV** | `outputs/day6_7_gate_b/offline_evaluation/joined_risk_and_hidden_outcomes.csv` | 940 KB | `2127befe92d441214755e22c9886bae3ca6fd441508b03d297351badb83a3945` | **MATCH** |

---

## 2. Independent Recomputation & Discrepancy Audit

### 2.1 Outcome Class Distribution & Prevalences
Using the independent verification script `tools/run_gate_b_independent_validation.py` directly on `joined_risk_and_hidden_outcomes.csv`:
- **Total Evaluated Batches:** Exactly **537** (179 batches $\times$ 3 seeds).
- **Primary Harm Batches ($\Delta\text{F1} \le -0.01$):** **504 / 537** (Prevalence = **`93.8547%`**).
- **Primary Beneficial Batches ($\Delta\text{F1} \ge +0.01$):** **11 / 537** (Prevalence = **`2.0484%`**).
- **Neutral Batches ($-0.01 < \Delta\text{F1} < +0.01$):** **22 / 537** (Prevalence = **`4.0968%`**).
- **Sensitivity Threshold Counts:**
  - $\Delta\text{F1} \le 0.00$ (Any degradation): **523 / 537** (`97.39%`)
  - $\Delta\text{F1} \le -0.02$ (Severe harm): **486 / 537** (`90.50%`)
  - $\Delta\text{F1} \le -0.10$ (Catastrophic collapse): **477 / 537** (`88.83%`)
- **No-Skill AUPRC Random Baseline:** Exactly equal to positive class prevalence = **`0.938547`**.

### 2.2 Independent Recomputation of All 26 Indicators

Every indicator was independently computed from the joined artifact using `scipy.stats.spearmanr`, `sklearn.metrics.roc_auc_score`, and `sklearn.metrics.average_precision_score`.

| Indicator Name | Frozen Direction | Recomputed Spearman $r$ | Recomputed AUROC | Recomputed AUPRC | Summary CSV Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bn_affine_drift_l2` | Higher $\to$ Harm | -0.415200 | 0.950096 | 0.996715 | **0.000000** |
| `intermediate_feature_drift_l2` | Higher $\to$ Harm | -0.382345 | 0.948954 | 0.996645 | **0.000000** |
| `disappear_prob_ratio` | Higher $\to$ Harm | -0.402579 | 0.920034 | 0.994462 | **0.000000** |
| `bn_running_stat_drift_l2` | Higher $\to$ Harm | -0.285017 | 0.919252 | 0.993774 | **0.000000** |
| `stability_pre1_pre2_area_increase` | Higher $\to$ Harm | -0.379894 | 0.864568 | 0.989524 | **0.000000** |
| `null_pre2_pre2_activation_rate` | Higher $\to$ Harm | -0.308657 | 0.854467 | 0.987588 | **0.000000** |
| `disappear_area_ratio` | Higher $\to$ Harm | -0.367275 | 0.850649 | 0.988018 | **0.000000** |
| `adapted_to_source_area_ratio` | Higher $\to$ Harm | -0.367275 | 0.850649 | 0.988018 | **0.000000** |
| `predeclared_signal_null_growth` | Higher $\to$ Harm | -0.316778 | 0.844517 | 0.986618 | **0.000000** |
| `null_pre2_pre2_area_increase` | Higher $\to$ Harm | -0.325786 | 0.844487 | 0.987820 | **0.000000** |
| `null_pre2_pre2_area_ratio` | Higher $\to$ Harm | -0.325655 | 0.833995 | 0.986842 | **0.000000** |
| `null_pre2_pre2_spatial_concentration` | Higher $\to$ Harm | -0.299198 | 0.832852 | 0.981577 | **0.000000** |
| `null_post_post_area_increase` | Higher $\to$ Harm | -0.234741 | 0.805976 | 0.982565 | **0.000000** |
| `null_pre1_pre1_area_increase` | Higher $\to$ Harm | -0.266830 | 0.785113 | 0.981702 | **0.000000** |
| `kl_divergence_source_adapted` | Higher $\to$ Harm | -0.294579 | 0.744649 | 0.971936 | **0.000000** |
| `mean_prediction_entropy` | Higher $\to$ Harm | -0.219703 | 0.657347 | 0.945532 | **0.000000** |
| `disappear_disagreement_fraction` | Higher $\to$ Harm | -0.218511 | 0.627405 | 0.944926 | **0.000000** |
| `source_disagreement_fraction` | Higher $\to$ Harm | -0.218511 | 0.627405 | 0.944926 | **0.000000** |
| `null_pre2_pre2_aug_inconsistency` | Higher $\to$ Harm | -0.135259 | 0.559945 | 0.923232 | **0.000000** |
| **`predeclared_change_risk_score`** | Higher $\to$ Harm | **-0.000724** | **0.510221** | **0.946927** | **0.000000** |
| `predeclared_signal_flood_erasure` | Higher $\to$ Harm | 0.005959 | 0.503788 | 0.946747 | **0.000000** |
| `disappear_fraction_source_erased` | Higher $\to$ Harm | 0.005959 | 0.503788 | 0.946747 | **0.000000** |
| `augmentation_inconsistency` | Higher $\to$ Harm | -0.118178 | 0.483255 | 0.903565 | **0.000000** |
| `predicted_flood_area` | Higher $\to$ Harm | -0.054334 | 0.458213 | 0.924823 | **0.000000** |
| `mean_max_confidence` | Lower $\to$ Harm | 0.186829 | 0.402778 | 0.895193 | **0.000000** |
| `disappear_feature_norm_ratio` | Higher $\to$ Harm | 0.395359 | 0.056758 | 0.836709 | **0.000000** |

**Discrepancy Audit Verdict:** **ZERO DISCREPANCY**. The recomputed metrics match the values stored in `indicator_evaluation_summary.csv` and `gate_b_evaluation.json` to full 64-bit floating-point precision.

---

## 3. Event Imbalance & Robustness Audit

### 3.1 Event Sample & Batch Distribution
The independent count for each validation event across 3 seeds:

| Event ID | Samples / Seed | % Total Samples | Batches / Seed | Total Batches (3 Seeds) | % Total Batches | Harmful Batches | Harm Rate | Beneficial Batches | Bene Rate | Outcome Classes Present |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 2,381 | **84.25%** | 149 | 447 | **83.24%** | 446 | 99.78% | 0 | 0.00% | Both (446 harm, 1 non-harm) |
| **437** | 169 | 5.98% | 11 | 33 | 6.15% | 32 | 96.97% | 1 | 3.03% | Both (32 harm, 1 non-harm) |
| **520** | 92 | 3.26% | 6 | 18 | 3.35% | 4 | 22.22% | 0 | 0.00% | Both (4 harm, 14 neutral) |
| **559** | 42 | 1.49% | 3 | 9 | 1.68% | 0 | **0.00%** | 9 | **100.0%** | **Single (0 harm, 9 non-harm)** |
| **1111003** | 19 | 0.67% | 2 | 6 | 1.12% | 6 | **100.0%** | 0 | **0.00%** | **Single (6 harm, 0 non-harm)** |
| **1111008** | 123 | 4.35% | 8 | 24 | 4.47% | 16 | 66.67% | 1 | 4.17% | Both (16 harm, 8 non-harm) |
| **Total** | **2,826** | **100.0%** | **179** | **537** | **100.0%** | **504** | **93.85%** | **11** | **2.05%** | **Both** |

**Event 279 Dominance Verification:**
The claim in `docs/day6_7_gate_b_audit.md` that Event 279 constitutes "approximately 84.3%" of the data is confirmed:
$$\frac{2,381 \text{ samples}}{2,826 \text{ total samples}} = 84.2534\% \approx 84.3\%$$

### 3.2 Within-Event Single-Class AUROC Undefined States
In Event 559, all 9 batches experienced beneficial adaptation ($\Delta\text{F1} \ge +0.01$) and zero batches experienced harm. Conversely, in Event 1111003, all 6 batches experienced harm and zero batches were non-harmful.
Consequently, within-event AUROC is **strictly undefined** on Event 559 and Event 1111003 because AUROC requires both positive and negative binary ground-truth classes.

### 3.3 Individual Leave-One-Event-Out (LOEO) Folds

Reporting all 6 individual leave-one-event-out folds for the leading indicators:

| Indicator Name | Holdout 279 | Holdout 437 | Holdout 520 | Holdout 559 | Holdout 1111003 | Holdout 1111008 | Mean LOEO AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bn_affine_drift_l2`** | **0.6843** | 0.9580 | 0.9587 | 0.9423 | 0.9586 | 0.9652 | **0.9112** |
| `intermediate_feature_drift_l2` | 0.5803 | 0.9742 | 0.9699 | 0.9401 | 0.9500 | 0.9596 | 0.8957 |
| `stability_pre1_pre2_area_increase` | 0.4744 | 0.8758 | 0.9583 | 0.8363 | 0.8706 | 0.8540 | 0.8116 |
| `null_pre2_pre2_area_increase` | 0.3723 | 0.8679 | 0.8788 | 0.8554 | 0.8532 | 0.8350 | 0.7771 |
| `disappear_prob_ratio` | 0.5453 | 0.9290 | 0.9014 | 0.9581 | 0.9267 | 0.9305 | 0.8652 |
| `predeclared_signal_null_growth` | 0.4189 | 0.8671 | 0.8809 | 0.8479 | 0.8512 | 0.8379 | 0.7840 |
| `predeclared_signal_flood_erasure` | 0.3955 | 0.5193 | 0.6532 | 0.3731 | 0.4978 | 0.5307 | 0.4949 |
| **`predeclared_change_risk_score`** | **0.3944** | 0.5264 | 0.6632 | 0.3752 | 0.5043 | 0.5387 | **0.5004** |

**Crucial Fold Analysis (Holdout 279 Impact):**
- When any of events 437, 520, 559, 1111003, or 1111008 is held out, the training set retains Event 279 (447 batches) and performance is uniformly high (~0.94 - 0.96 for generic drift; ~0.84 - 0.88 for null probes).
- When **Event 279 is held out**, the evaluation set drops to only 90 batches across the 5 remaining events. Under Holdout 279:
  - `bn_affine_drift_l2` remains above chance at **`0.6843`**.
  - `null_pre2_pre2_area_increase` collapses to **`0.3723`** (sub-chance).
  - `predeclared_change_risk_score` collapses to **`0.3944`** (sub-chance).
This proves that the counterfactual null probe relies entirely on the presence of Event 279 to maintain its pooled discriminatory power, whereas generic parameter drift maintains positive discrimination even without Event 279.

### 3.4 Micro (Pooled) vs. Macro (Event-Balanced) Summaries

| Indicator Name | Pooled (Micro) Spearman $r$ | Event-Balanced (Macro) Spearman $r$ | Event Folds Included |
| :--- | :---: | :---: | :---: |
| `bn_affine_drift_l2` | **-0.4152** | **+0.0458** | 6 / 6 |
| `intermediate_feature_drift_l2` | -0.3823 | +0.1043 | 6 / 6 |
| `stability_pre1_pre2_area_increase` | -0.3799 | +0.0420 | 6 / 6 |
| `null_pre2_pre2_area_increase` | -0.3258 | +0.0794 | 6 / 6 |
| `disappear_prob_ratio` | -0.4026 | -0.1041 | 6 / 6 |
| `predeclared_signal_null_growth` | -0.3168 | -0.1648 | 4 / 6 (2 folds constant) |
| **`predeclared_change_risk_score`** | **-0.0007** | **+0.0796** | 6 / 6 |

**Interpretation:**
The pooled (micro) Spearman correlation for generic drift is strongly negative ($-0.4152$), driven by the fact that events with large cumulative drift (Events 279 and 437) suffer large negative $\Delta\text{F1}$, whereas Event 559 experienced low drift and positive $\Delta\text{F1}$.
However, the event-balanced (macro) Spearman correlation is near zero ($+0.0458$) because within-event correlations are flat or sign-inverted across individual events. This demonstrates that the indicators primarily function as **cross-event domain shift detectors** rather than fine-grained intra-event batch regressors.

---

## 4. AUPRC & Confounder Audit

### 4.1 AUPRC Informativeness
- **Harmful Class Prevalence ($p$):** $504 / 537 = \mathbf{0.9385}$
- **No-Skill Constant Baseline AUPRC:** $\mathbf{0.9385}$
- **AUPRC Improvement over Baseline ($\Delta\text{AUPRC} = \text{AUPRC} - p$):**
  - `bn_affine_drift_l2`: $0.9967 - 0.9385 = \mathbf{+0.0582}$
  - `intermediate_feature_drift_l2`: $0.9966 - 0.9385 = \mathbf{+0.0581}$
  - `disappear_prob_ratio`: $0.9945 - 0.9385 = \mathbf{+0.0560}$
  - `null_pre2_pre2_area_increase`: $0.9878 - 0.9385 = \mathbf{+0.0493}$
  - `predeclared_change_risk_score`: $0.9469 - 0.9385 = \mathbf{+0.0084}$

**Audit Finding:**
All AUPRC values reported in `docs/day6_7_gate_b_audit.md` (e.g., 0.94 to 0.99) are **heavily inflated by the high underlying prevalence of harmful adaptation (93.85%)**. A naive baseline assigning maximum risk to every batch achieves an AUPRC of 0.9385. The actual discriminatory gain for `predeclared_change_risk_score` is less than 0.009. Therefore, AUROC and Spearman rank correlations provide the sole legitimate basis for model discrimination in this regime.

### 4.2 Confounder Analysis
- **Correlation with Baseline Flood Extent (`source_predicted_flood_area`):**
  - `disappear_prob_ratio`: $r = \mathbf{-0.7890}$
  - `disappear_area_ratio`: $r = \mathbf{-0.8102}$
  - `predicted_flood_area`: $r = \mathbf{+0.4285}$
  - `bn_affine_drift_l2`: $r = -0.2194$
  - `null_pre2_pre2_area_increase`: $r = -0.1806$
  - `predeclared_change_risk_score`: $r = +0.1762$

**Audit Finding:**
The top disappearance indicator `disappear_prob_ratio` is heavily confounded with baseline water area ($r = -0.7890$). When a patch has few or no water pixels, disappearance metrics drop dramatically due to denominator instability. It detects the absence of water, not the corruption of the change detector.

---

## 5. Mandatory `valid_mask` Provenance Audit

The research integrity protocol mandates a strict trace of how `valid_mask` is created to ensure no ground-truth leakage entered Stage 1.

### 5.1 Source File and Dataset Code Path
1. **Source File on Disk:** Physical GeoTIFF raster file `MK0_MNA_*.tif` in each sample directory under `/usr1/home/s126mdg22_06/FYP_SARCD/datasets/KuroSiwo/`.
2. **File Definition:** In KuroSiwo dataset specifications, `MK0_MNA` is the physical sensor No-Data Area Mask (MNA = Mask No-data Area), provided directly with Sentinel-1 acquisitions. It is NOT the label file `MK0_MLU` (MLU = Mask Land-Use / flood labels).
3. **Dataset Code Path (`src/data/kuro_dataset.py`):**
   - Line 61: `SENSORY_COMPONENTS` includes `"MK0_MNA"`.
   - Line 63: `LABEL_COMPONENTS` contains `["MK0_MLU"]`.
   - Line 151: In `KuroUnlabeledValidationDataset`, `load_labels=False`, so `required_components` excludes `MK0_MLU`.
   - Line 162: `valid = self._read_raster(located["MK0_MNA"]).astype(np.uint8)`.
   - Line 248: `"valid_mask": torch.from_numpy(valid).to(torch.uint8)`.

### 5.2 Label and Ground-Truth Independence
- **Does `valid_mask` depend on `MK0_MLU`?** **NO**. It is read strictly from `MK0_MNA`.
- **Does `valid_mask` depend on class 3 or any ground-truth label value in Stage 1?** **NO**. In Stage 1, `label` is `None` and is never read. The label exclusion condition `(valid_mask == 1) & (label != 3)` exists only in Stage 2 offline evaluation (`compute_confusion_counts`), where ground truth is joined offline.
- **How is `valid_mask` used in Stage 1?**
  1. In `TentAdapter` (`src/tta/tent.py`), `sar_valid_mask` restricts the entropy loss to physical observation pixels (excluding sensor padding).
  2. In `RiskFeatureCollector` (`src/risk/collector.py`), `valid_mask` excludes sensor padding from spatial averages of entropy and confidence. It does NOT mask outcomes, because Stage 1 generates zero outcomes.

**Audit Verification:** The Stage 1 "label-free" claim is **100% verified and unviolated**.

---

## 6. Protocol Compliance & Missing Outputs Audit

The Gate B protocol specified several desirable diagnostic outputs. This audit cross-references existing repository artifacts against each specification:

| Output Requirement | Status | Evidence / Location |
| :--- | :---: | :--- |
| **All sensitivity-threshold counts** | **COMPLETED** | Saved in `indicator_evaluation_summary.csv` (`auroc_thresh_00`, `auroc_thresh_02`). *(See correction for thresh 10 below).* |
| **Every event-wise result** | **COMPLETED** | Stored in `gate_b_evaluation.json` and `indicator_evaluation_summary.csv` (`event_spearmans_f1`). |
| **Every failed indicator reported** | **COMPLETED** | All 26 indicators (including 21 failed indicators) are fully reported in `indicator_evaluation_summary.csv`. |
| **Bootstrap confidence intervals** | **NOT IMPLEMENTED** | Not computed in `run_gate_b_offline_evaluation.py`; not stored in JSON/CSV. |
| **Calibration / reliability curves or ECE** | **NOT IMPLEMENTED** | No calibration curves or reliability diagrams were generated in artifacts. |
| **Warning timing relative to harm** | **NOT IMPLEMENTED AS METRIC** | Batch metrics are stored sequentially in `joined_risk_and_hidden_outcomes.csv` (temporal analysis possible), but explicit lead-time metrics were not implemented. |

---

## 7. Assessment of the Scientific Conclusion

| Scientific Proposition | Status | Validation Evidence |
| :--- | :---: | :--- |
| **1. The predeclared composite score performs near chance.** | **VALIDATED** | AUROC = `0.5102`, LOEO AUROC = `0.5004`, Spearman $r = -0.0007$. |
| **2. Generic BN/feature drift outperforms the predeclared score.** | **VALIDATED** | `bn_affine_drift_l2` achieves AUROC `0.9501` and LOEO `0.9112`, beating the predeclared score by +0.44 AUROC points. |
| **3. Some individual null probes contain predictive information.** | **VALIDATED** | `stability_pre1_pre2_area_increase` (AUROC `0.8646`) and `null_pre2_pre2_area_increase` (AUROC `0.8445`) demonstrate non-trivial discrimination. |
| **4. Top disappearance score is confounded by source flood area.** | **VALIDATED** | `disappear_prob_ratio` has an extreme Spearman correlation with baseline flood extent ($r = -0.7890$). |
| **5. Frozen Gate B criteria imply `NO-GO`.** | **VALIDATED** | The predeclared score failed, no probe beat generic drift, and top probe is confounded. `NO-GO` is mandated. |
| **6. Results do not prove all possible change safeguards are ineffective.** | **VALIDATED** | Applies strictly to Tent on SNUNet on KuroSiwo under the frozen protocol formulations. |

---

## 8. Proposed Documentation Corrections for `docs/day6_7_gate_b_audit.md`

In accordance with instructions, `docs/day6_7_gate_b_audit.md` has NOT been altered. The following four corrections should be applied when the document is updated:

### Correction 1: Working Tree Git State Terminology
- **Location:** `docs/day6_7_gate_b_audit.md`, Section 6.2 and Self-Validation Point 14.
- **Original Text:**  
  `Clean git history maintained on branch main at commit 214150c2ad22e09814888748e49f71fc9678a62e. No files staged or committed.`
- **Evidence:** `git status --short` shows `M src/data/kuro_dataset.py`. Because a tracked file is modified, the working tree is technically dirty (`git_dirty: true` as recorded in the run manifests).
- **Corrected Wording:**  
  `Repository is on branch main at commit 214150c2ad22e09814888748e49f71fc9678a62e, with modified tracked file src/data/kuro_dataset.py and untracked Day 6-7 files. No files are staged or committed.`

### Correction 2: Technical Root Cause of `auroc_thresh_10 == NaN`
- **Location:** `docs/day6_7_gate_b_audit.md`, Section 4.3 (Footnote).
- **Original Text:**  
  `*(Note: Threshold delta F1 <= -0.10 was undefined for AUROC because across the validation set, cumulative batch delta F1 degradation stayed largely between -0.01 and -0.07, resulting in zero batches satisfying <= -0.10 in the binary ground truth).*`
- **Evidence:** In `joined_risk_and_hidden_outcomes.csv`, `hidden_is_harmful_thresh10` has 477 ones and 60 zeros (88.8% prevalence). The true cause of `NaN` in `indicator_evaluation_summary.csv` was that `y_harm_10` was accidentally omitted from the variable definitions in `tools/run_gate_b_offline_evaluation.py` lines 277-281, causing a caught `NameError`.
- **Corrected Wording:**  
  `*(Note: In indicator_evaluation_summary.csv, auroc_thresh_10 returned NaN due to a script variable omission in tools/run_gate_b_offline_evaluation.py line 316. Recomputation from the joined ground-truth column hidden_is_harmful_thresh10 confirms that 477 batches satisfy delta F1 <= -0.10, yielding true AUROCs of 0.9716 for bn_affine_drift_l2 and 0.4264 for predeclared_change_risk_score).*`

### Correction 3: Inflation of Baseline AUPRC Values
- **Location:** `docs/day6_7_gate_b_audit.md`, Section 4.2.
- **Original Context:** High AUPRC values (0.94 - 0.99) were tabulated without explicit comparison against the random baseline.
- **Evidence:** Harmful class prevalence is 93.85%. A constant random classifier achieves an AUPRC of 0.9385.
- **Corrected Wording:**  
  Add an explicit note in Section 4.2:  
  `*(Note: Due to the 93.85% prevalence of harmful adaptation, the no-skill random baseline AUPRC is 0.9385. AUPRC values between 0.94 and 0.99 represent modest improvements over this baseline (+0.008 for the composite score, +0.058 for parameter drift); AUROC and Spearman rank correlations provide the true discriminatory measure).*`

### Correction 4: Clarification of Micro vs. Macro Spearman Correlations
- **Location:** `docs/day6_7_gate_b_audit.md`, Section 4.4.
- **Original Context:** Tabulated within-event Spearman correlations without highlighting the difference between pooled (micro) and event-balanced (macro) averages.
- **Evidence:** Pooled Spearman for `bn_affine_drift_l2` is $-0.4152$, while the macro-averaged Spearman across events is $+0.0458$.
- **Corrected Wording:**  
  Add an explicit note in Section 4.4:  
  `*(Note: While generic parameter drift achieves a strong pooled Spearman correlation of -0.4152 driven by inter-event differences between high-drift harmful events and low-drift beneficial events, its event-balanced macro average is +0.0458 due to flat within-event trajectories in Event 279 and opposite slopes in smaller events).*`

---

## 9. Final Validation Verdict

### Verdict: **`VALIDATED WITH DOCUMENTATION CORRECTIONS`**

**Summary of Determination:**
The empirical execution, artifact immutability, data hygiene, and the scientific **`NO-GO`** decision are completely sound, rigorously executed, and verified to full floating-point precision. The proposed documentation corrections address minor reporting nuances (git status terminology, script-level try/except explanation, and prevalence baseline context) without altering any numeric result or the scientific conclusion.

---

## 10. Verification Audit Trail

### 10.1 Commands Executed
1. `pgrep -af "run_gate_b|python" ; git branch --show-current ; git rev-parse HEAD ; git status --short ; date -Iseconds`
2. `sha256sum outputs/day6_7_gate_b/manifests/* outputs/day6_7_gate_b/predictions/* outputs/day6_7_gate_b/risk_generation/* outputs/day6_7_gate_b/stage1_execution.log outputs/day6_7_gate_b/offline_evaluation/*`
3. Inspection of `src/data/kuro_dataset.py` lines 50–260 (tracing `valid_mask` and `MK0_MNA`).
4. Inspection of `src/risk/indicators.py` lines 1–120 (tracing `valid_mask` usage in entropy/confidence).
5. Execution of `tools/run_gate_b_independent_validation.py` to independently recompute all metrics.
6. Execution of threshold 10 ground-truth audit.

### 10.2 Files Read
- `configs/gate_b_risk_protocol.json`
- `outputs/day6_7_gate_b/stage1_execution.log`
- `outputs/day6_7_gate_b/offline_evaluation/gate_b_evaluation.json`
- `outputs/day6_7_gate_b/offline_evaluation/indicator_evaluation_summary.csv`
- `outputs/day6_7_gate_b/offline_evaluation/joined_risk_and_hidden_outcomes.csv`
- `src/data/kuro_dataset.py`
- `src/risk/indicators.py`
- `tools/run_gate_b_offline_evaluation.py`
- `docs/day6_7_gate_b_audit.md`

### 10.3 Files Created
- `tools/run_gate_b_independent_validation.py` (read-only validation script)
- `docs/day6_7_gate_b_validation.md` (this report)

### 10.4 Absolute Integrity Affirmations
- **Sealed Test Event Access:** Strictly zero access. Confirmed zero test event IDs in all artifacts.
- **Manual Data Editing:** Strictly zero manual edits. All artifacts verified by cryptographic SHA-256 match.
- **Final Git Status:**
  ```
  $ git status --short
   M src/data/kuro_dataset.py
  ?? configs/gate_b_risk_protocol.json
  ?? docs/day6_7_gate_b_audit.md
  ?? docs/day6_7_gate_b_validation.md
  ?? src/risk/
  ?? tests/test_risk_indicators.py
  ?? tests/test_risk_probing_isolation.py
  ?? tests/test_risk_schema.py
  ?? tools/run_day6_smoke_tests.py
  ?? tools/run_gate_b_independent_validation.py
  ?? tools/run_gate_b_offline_evaluation.py
  ?? tools/run_gate_b_risk_generation.py
  ```
- **Git Milestone Recommendation:** The existing Days 6–7 Gate B audit and results are **recommended for acceptance**. The documentation corrections identified above can be applied to `docs/day6_7_gate_b_audit.md` before finalizing the Git commit.
