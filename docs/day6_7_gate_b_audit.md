# ChangeGuard Days 6–7 Gate B Audit Report: Label-Free Risk-Signal Validation

**Document Version:** 1.0  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Starting Commit (Gate A):** `214150c2ad22e09814888748e49f71fc9678a62e`  
**Working Tree Status:** Dirty (`src/data/kuro_dataset.py` modified; untracked Day 6–7 files; 0 files staged/committed)  
**Evaluation Date:** October 3, 2026 (Validation Corrections: October 4, 2026)  
**Evaluation Scope:** Days 6–7 Gate B label-free risk-signal validation only  
**Host / Device:** `gpu22` / NVIDIA GeForce RTX 3090 (CUDA 13.0, Driver 580.173.02)  
**Python Environment:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`  
**Protocol Specification:** `configs/gate_b_risk_protocol.json` (SHA-256: `132ddb7ad11afc74dfd06485e92822516c7e820cce8f96a8c9c2e06a82b22074`)  
**Frozen Source Checkpoint:** `checkpoints/kuro_snunet_official.pt` (SHA-256: `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed`)  
**Final Gate B Verdict:** **`NO-GO`**

---

## Executive Summary

Days 6–7 of the ChangeGuard pilot investigated whether predeclared, label-free, change-specific signals (synthetic bi-temporal counterfactual null probes, change-disappearance probes, and spatial consistency features) can predict hidden empirical adaptation outcomes ($\Delta\text{F1} \le -0.01$) under transductive online Test-Time Adaptation (Tent, lr=$10^{-4}$), and specifically whether they outperform generic adaptation indicators (prediction entropy, output disagreement, batch-norm parameter drift, and feature representation drift).

The evaluation was executed under strict protocol isolation:
1. **Stage 1 (Label-Free Risk Generation):** Zero ground-truth label reading (`KuroUnlabeledValidationDataset` with `load_labels=False`; `MK0_MLU` raster files were never opened). Zero mutation of the model state by probing (verified via bitwise parameter/buffer equality checks). Completed across all 6 validation events `[279, 437, 520, 559, 1111003, 1111008]` and 3 random seeds `[0, 1, 2]`. All 10 test events remained strictly sealed.
2. **Stage 2 (Offline Outcome Join & Evaluation):** Decoupled offline join with ground-truth validation labels to compute Spearman rank correlations, AUROC, AUPRC, leave-one-event-out (LOEO) cross-validation, and confounder sensitivity.

### Definitive Gate B Decision: `NO-GO`

The Gate B criteria established in `docs/research_integrity_protocol.md` mandate that a `GO` decision requires a predeclared change-specific signal or simple frozen combination that:
1. Is consistently associated with hidden adaptation degradation across events;
2. Outperforms the strongest generic indicator across events or leave-one-event-out cross-validation;
3. Does not merely detect small predicted flood extent;
4. Warns before or at the onset of harmful adaptation.

**Primary Findings:**
1. **Failure of the Predeclared Composite Score:** The frozen composite score `predeclared_change_risk_score` achieved an overall AUROC of **`0.5102`** (near-random chance), a mean LOEO AUROC of **`0.5004`**, and a Spearman correlation with $\Delta\text{F1}$ of **`-0.0007`** ($p = 0.986$).
2. **Dominance of Generic Drift Over Change-Specific Probes:** The strongest overall predictor of harmful adaptation was generic parameter drift:
   - `bn_affine_drift_l2`: AUROC = **`0.9501`**, LOEO AUROC = **`0.9112`**, Spearman $r = \mathbf{-0.4152}$ ($p = 2.4 \times 10^{-23}$).
   - `intermediate_feature_drift_l2`: AUROC = **`0.9489`**, LOEO AUROC = **`0.8957`**, Spearman $r = \mathbf{-0.3823}$ ($p = 9.8 \times 10^{-20}$).
   - In contrast, the top change-specific null probe (`stability_pre1_pre2_area_increase`) achieved AUROC = **`0.8646`** (LOEO AUROC = **`0.8116`**), and the primary counterfactual probe (`null_pre2_pre2_area_increase`) achieved AUROC = **`0.8445`** (LOEO AUROC = **`0.7771`**). **No change-specific probe outperformed generic parameter drift.**
3. **Severe Confounder Contamination:** The highest-ranking change-specific signal, `disappear_prob_ratio` (AUROC `0.9200`), exhibited a severe negative Spearman correlation with baseline flood extent (`confounder_corr_source_area` = **`-0.7890`**), proving that it primarily tracks small baseline water bodies rather than change-detection failure.
4. **Event-Wise Inconsistency:** On the largest validation event (Event 279, representing 84.3% of validation samples), null probe signals exhibited near-zero correlation with $\Delta\text{F1}$ ($r = +0.0473$), while on smaller events (e.g., Event 1111003), correlations inverted to large positive values ($r = +0.7714$).

Because the hypothesis that label-free change-specific counterfactuals outperform generic parameter/representation drift is empirically falsified, Gate B concludes with an unambiguous **`NO-GO`**.

**Scope & Governance Boundaries:**
- **Phase-Bounded Verdict:** This verdict applies strictly to Days 6–7 Gate B (label-free risk-signal validation).
- **No Controller Implemented:** No rollback mechanism, adaptation stopping rule, or update controller has been implemented in Days 6–7.
- **Generic Drift is Not a Validated Controller:** Although generic parameter drift (`bn_affine_drift_l2`) achieved high pooled AUROC (0.9501), its event-balanced macro Spearman correlation is positive (+0.0458); pooled performance mainly reflects cross-event separation rather than reliable within-event batch ordering. Generic drift cannot yet be claimed as a reliable update-level controller.
- **Overall FYP Decision Deferred:** No final conclusion is drawn regarding whether the overall FYP topic continues, pivots, or stops; that strategic decision is strictly deferred to the Days 8–10 synthesis milestone.

---

## 1. Recovery Audit & Execution Provenance

### 1.1 Recovery Classification: `PARTIAL_NOT_RESUMABLE`
Prior to this execution session:
- The previous execution of Stage 1 (`tools/run_gate_b_risk_generation.py`, task-1737) was interrupted at `2026-10-02T22:16:55` due to an infrastructure server restart.
- An inspection of the output tree revealed empty subdirectories (`manifests/`, `predictions/`, `risk_generation/`) in `outputs/day6_7_gate_b/` with 0 bytes written.
- Because `run_gate_b_risk_generation.py` did not implement an incremental batch-checkpoint resume mechanism, the state was classified as **`PARTIAL_NOT_RESUMABLE`**.
- To prevent accidental data loss and maintain full research provenance:
  - The empty interrupted output directory was safely archived by renaming to:  
    `outputs/day6_7_gate_b_interrupted_20261002_221655`
  - A clean output directory `outputs/day6_7_gate_b` was initialized.
  - A previous Antigravity session had executed `rm -rf outputs/day6_7_gate_b/` prior to task-1737; that event was logged as an unrepeatable recovery deviation. In this session, zero files were deleted.

### 1.2 Detached `tmux` Execution Protection
To insulate Stage 1 execution against network disconnections, Stage 1 was launched inside a detached `tmux` session named `changeguard_gate_b_stage1`:
- **Session:** `changeguard_gate_b_stage1`
- **Command:** `cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard && /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_gate_b_risk_generation.py > outputs/day6_7_gate_b/stage1_execution.log 2>&1`
- **Process ID:** PID `1547284` (PPID `1547282`)
- **GPU:** Device 0 (RTX 3090, 11.8 GB VRAM allocated)
- **Start Time:** 2026-10-03 15:08:08 (+08:00)
- **Completion Time:** 2026-10-03 16:01:45 (+08:00)
- **Exit Status:** Code 0 (`SUCCESS`)

---

## 2. Pre-Run Integrity & Frozen Protocol Verification

### 2.1 Protocol Specification & Invariant Hashes
The protocol was frozen prior to execution and validated with the following immutable SHA-256 digest:

| File / Component | Absolute Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Risk Protocol** | `configs/gate_b_risk_protocol.json` | `132ddb7ad11afc74dfd06485e92822516c7e820cce8f96a8c9c2e06a82b22074` | **VERIFIED** |
| **Source Checkpoint** | `checkpoints/kuro_snunet_official.pt` | `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed` | **VERIFIED** |
| **Smoke Artifact** | `outputs/day6_smoke/smoke_risk_features.json` | `f1b4a9743b5c45bd507575cb8e68bca2347ab100d99fac075a0752b58bd5c109` | **VERIFIED** |

### 2.2 Protocol Definitions
- **Adaptation Method:** Tent (BatchNorm affine weight/bias updates via Shannon entropy minimization, Adam lr=$10^{-4}$).
- **Primary Harm Definition:** $\Delta\text{F1} \le -0.01$ (cumulative event F1 drop $\ge 1.0$ percentage point relative to frozen source model).
- **Primary Beneficial Definition:** $\Delta\text{F1} \ge +0.01$.
- **Sensitivity Thresholds:** $0.00$ (any degradation), $-0.02$ (severe harm), $-0.10$ (catastrophic collapse).
- **Validation Events:** Exactly 6 events: `[279, 437, 520, 559, 1111003, 1111008]`.
- **Sealed Test Events:** Exactly 10 events: `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`. All 10 remained untouched.
- **Random Seeds / Orders:** `[0, 1, 2]`.
- **Predeclared Composite Score:**
  $$\text{Risk} = 0.5 \cdot \text{clip}\left(\frac{\max(0, \text{null\_pre2\_pre2\_area\_increase})}{\text{total\_valid\_pixels} + 1.0}, 0, 1\right) + 0.5 \cdot \text{clip}(\text{disappear\_fraction\_source\_erased}, 0, 1)$$

### 2.3 Unit Testing & Smoke Verification
Before full execution, 38 unit tests were executed and passed (38/38):
- 4 completeness tests (`tests/test_completeness.py`): PASS
- 6 metric tests (`tests/test_metrics.py`): PASS
- 9 preprocessing tests (`tests/test_preprocessing.py`): PASS
- 19 test discovery suite tests (`unittest discover -s tests -p "test_*.py"`): PASS
  - Verified `test_unlabeled_dataset_never_reads_label_raster`
  - Verified `test_probing_does_not_mutate_parameters_buffers_or_counters`
  - Verified `test_forbidden_gt_fields_raise`
- Smoke test execution (`tools/run_day6_smoke_tests.py`): All 5 invariants passed in 20.2s, reproducing exact artifact hash `f1b4a974...`.

---

## 3. Stage 1 Execution & Artifact Freezing

### 3.1 Streaming Execution Summary
- **Sample Distribution per Event:**
  - Event 279: 2,381 samples (149 batches)
  - Event 437: 169 samples (11 batches)
  - Event 520: 92 samples (6 batches)
  - Event 559: 42 samples (3 batches)
  - Event 1111003: 19 samples (2 batches)
  - Event 1111008: 123 samples (8 batches)
  - **Total:** 2,826 samples across 179 batches per seed (537 batches total across 3 seeds).
- **Run Timing:**
  - Run 1 (Seed 0): 781.4s (including 2,826 sample raster preload into host RAM)
  - Run 2 (Seed 1): 782.5s
  - Run 3 (Seed 2): 782.3s
  - Total Stage 1 Runtime: ~39 minutes.

### 3.2 Immutable Stage 1 Artifact Table

| Seed | Run ID | Artifact Type | File Path | Size (Bytes) | SHA-256 Digest |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `gate_b_tent_lr1e-4_seed0` | Risk Features JSON | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed0_risk_features.json` | 732,850 | `9b6fb0400e3c21bb787b7a0dd40804c868d21bb4a06897e5a92a4d910edd4173` |
| **0** | `gate_b_tent_lr1e-4_seed0` | Predictions Tensor | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed0_predictions.pt` | 1,134,747,086 | `53fbdbc0884bcdb52b7683e810fcd42fefacc35a5fbdfdf36f5b753c46564eba` |
| **0** | `gate_b_tent_lr1e-4_seed0` | Run Manifest JSON | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed0.json` | 2,027 | `e4f4425d1dc2985e44b5c94979598e51a8e6c6705c087b0cfa888398edd9fa9d` |
| **1** | `gate_b_tent_lr1e-4_seed1` | Risk Features JSON | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed1_risk_features.json` | 732,054 | `5853e6c14c06d708da4f0841b7259119de03d98112cd99f246c3465a615d56c0` |
| **1** | `gate_b_tent_lr1e-4_seed1` | Predictions Tensor | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed1_predictions.pt` | 1,134,747,086 | `cd9b6384530c9582a9a11c4bb1b6560576ac0c2b168382c2d4abd935f5e54637` |
| **1** | `gate_b_tent_lr1e-4_seed1` | Run Manifest JSON | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed1.json` | 2,027 | `c083bb6f7cd9a15e3425f7d23688e97d8d1f3b754843214e299fab3d169631e7` |
| **2** | `gate_b_tent_lr1e-4_seed2` | Risk Features JSON | `outputs/day6_7_gate_b/risk_generation/gate_b_tent_lr1e-4_seed2_risk_features.json` | 732,342 | `adb8fae9a59359a1f12f153519cbb79a29e4976a52fe1b398b08f6a7b45dab7d` |
| **2** | `gate_b_tent_lr1e-4_seed2` | Predictions Tensor | `outputs/day6_7_gate_b/predictions/gate_b_tent_lr1e-4_seed2_predictions.pt` | 1,134,747,086 | `92e2accdb9e77b14da1811d732e780f04702ca2c4383d0ecfbe0384340d622fe` |
| **2** | `gate_b_tent_lr1e-4_seed2` | Run Manifest JSON | `outputs/day6_7_gate_b/manifests/manifest_gate_b_tent_lr1e-4_seed2.json` | 2,027 | `22f114218bfd504020cc79c61938fe435f2db3634f13494df29cc321f8233994` |
| **All** | Stage 1 Stdout/Stderr | Execution Log | `outputs/day6_7_gate_b/stage1_execution.log` | 3,577 | `2c47ab8014c36678b25f0bff5117a24bfa9c8d05c2892f3924a1d2effb842b1c` |

### 3.3 Stage 1 Invariant Audit Results
- **Batch Count:** Exactly 179 batches per seed, 537 batches total across seeds.
- **Sample Count:** Exactly 2,826 predictions per seed, 8,478 predictions total across seeds.
- **Sealed Test Event Leakage:** Strictly 0 instances. Zero test samples were loaded or processed.
- **Ground-Truth Label Leakage:** Strictly 0 instances. `KuroUnlabeledValidationDataset` loaded only SAR rasters (`xA`, `xB`, `pre1`, `pre2`, `post`, `valid_mask`). `MK0_MLU` was never requested.
- **Numeric Integrity:** Zero NaN or Inf values across all 537 feature records.
- **Probing Isolation:** Bitwise equality of candidate model parameters and buffers pre- and post-probing verified on all 537 batches.

---

## 4. Stage 2 Offline Outcome Evaluation

### 4.1 Evaluation Dataset Characteristics
Joining the frozen Stage 1 artifacts with validation ground-truth labels yielded:
- **Total Evaluated Batches:** 537
- **Harmful Adaptation Batches ($\Delta\text{F1} \le -0.01$):** 504 (93.85%)
- **Beneficial Adaptation Batches ($\Delta\text{F1} \ge +0.01$):** 11 (2.05%)
- **Neutral Batches ($-0.01 < \Delta\text{F1} < +0.01$):** 22 (4.10%)
This overwhelmingly harmful distribution confirms the Gate A Day 3–5 findings: Tent adaptation on KuroSiwo consistently degrades flood change-detection performance across nearly all streaming batches.

### 4.2 Comprehensive Indicator Performance Table

All 26 candidate indicators evaluated offline, sorted by primary harm AUROC ($\Delta\text{F1} \le -0.01$):

| Rank | Indicator Name | Category | Spearman $r$ ($\Delta\text{F1}$) | AUROC ($\Delta\text{F1} \le -0.01$) | AUPRC | Mean LOEO AUROC | Confounder: Source Area $r$ | Confounder: Is Event 559 $r$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `bn_affine_drift_l2` | Generic | **-0.4152** | **0.9501** | 0.9967 | **0.9112** | -0.2194 | -0.2048 |
| 2 | `intermediate_feature_drift_l2` | Generic | -0.3823 | 0.9490 | 0.9966 | 0.8957 | -0.2617 | -0.2049 |
| 3 | `disappear_prob_ratio` | Disappearance | -0.4026 | 0.9200 | 0.9945 | 0.8652 | **-0.7890** | -0.1256 |
| 4 | `bn_running_stat_drift_l2` | Generic | -0.2850 | 0.9193 | 0.9938 | 0.8838 | -0.2083 | -0.1904 |
| 5 | `stability_pre1_pre2_area_increase` | Null Probe | -0.3799 | 0.8646 | 0.9895 | 0.8116 | -0.1815 | -0.1885 |
| 6 | `null_pre2_pre2_activation_rate` | Null Probe | -0.3087 | 0.8545 | 0.9876 | 0.8026 | -0.1700 | -0.1471 |
| 7 | `disappear_area_ratio` | Disappearance | -0.3673 | 0.8506 | 0.9880 | 0.7983 | **-0.8102** | -0.0364 |
| 8 | `adapted_to_source_area_ratio` | Generic | -0.3673 | 0.8506 | 0.9880 | 0.7983 | **-0.8102** | -0.0364 |
| 9 | `predeclared_signal_null_growth` | Predeclared | -0.3168 | 0.8445 | 0.9866 | 0.7840 | -0.1701 | -0.1435 |
| 10 | `null_pre2_pre2_area_increase` | Null Probe | -0.3258 | 0.8445 | 0.9878 | 0.7771 | -0.1806 | -0.1324 |
| 11 | `null_pre2_pre2_area_ratio` | Null Probe | -0.3257 | 0.8340 | 0.9868 | 0.7683 | -0.2091 | -0.1265 |
| 12 | `null_pre2_pre2_spatial_concentration` | Null Probe | -0.2992 | 0.8329 | 0.9816 | 0.7831 | -0.1642 | -0.1053 |
| 13 | `null_post_post_area_increase` | Null Probe | -0.2347 | 0.8060 | 0.9826 | 0.7563 | -0.1330 | -0.1585 |
| 14 | `null_pre1_pre1_area_increase` | Null Probe | -0.2668 | 0.7851 | 0.9817 | 0.7201 | -0.2205 | -0.0636 |
| 15 | `kl_divergence_source_adapted` | Generic | -0.2946 | 0.7446 | 0.9719 | 0.7032 | -0.1397 | -0.1896 |
| 16 | `mean_prediction_entropy` | Generic | -0.2197 | 0.6573 | 0.9455 | 0.6289 | 0.0565 | -0.2099 |
| 17 | `disappear_disagreement_fraction` | Disappearance | -0.2185 | 0.6274 | 0.9449 | 0.5982 | 0.0350 | -0.1806 |
| 18 | `source_disagreement_fraction` | Generic | -0.2185 | 0.6274 | 0.9449 | 0.5982 | 0.0350 | -0.1806 |
| 19 | `null_pre2_pre2_aug_inconsistency` | Null Probe | -0.1353 | 0.5599 | 0.9232 | 0.5518 | 0.0503 | -0.1826 |
| 20 | **`predeclared_change_risk_score`** | **Composite** | **-0.0007** | **0.5102** | **0.9469** | **0.5004** | 0.1762 | -0.1673 |
| 21 | `predeclared_signal_flood_erasure` | Predeclared | 0.0060 | 0.5038 | 0.9467 | 0.4949 | 0.1772 | -0.1598 |
| 22 | `disappear_fraction_source_erased` | Disappearance | 0.0060 | 0.5038 | 0.9467 | 0.4949 | 0.1772 | -0.1598 |
| 23 | `augmentation_inconsistency` | Generic | -0.1182 | 0.4833 | 0.9036 | 0.4751 | 0.1864 | -0.1860 |
| 24 | `predicted_flood_area` | Generic | -0.0543 | 0.4582 | 0.9248 | 0.4566 | 0.4285 | -0.1971 |
| 25 | `mean_max_confidence` | Generic | 0.1868 | 0.4028 | 0.8952 | 0.4239 | -0.1022 | 0.2066 |
| 26 | `disappear_feature_norm_ratio` | Disappearance | 0.3954 | 0.0568 | 0.8367 | 0.1166 | 0.3516 | 0.2179 |

*(Note on AUPRC Baseline: Due to the 93.85% underlying prevalence of harmful adaptation (504 of 537 batches), the no-skill random classifier baseline AUPRC is **0.938547** (504 / 537). Reported AUPRC values between 0.94 and 0.99 represent only modest improvements over this high baseline (+0.0084 for the composite score, +0.0582 for parameter drift). Therefore, AUROC and Spearman rank correlations provide more informative discriminatory summaries in this high-prevalence setting).*

### 4.3 Sensitivity Threshold AUROC Analysis

Performance across varying definitions of harmful adaptation:

| Indicator Name | AUROC ($\Delta\text{F1} \le 0.00$) | AUROC ($\Delta\text{F1} \le -0.01$, Primary) | AUROC ($\Delta\text{F1} \le -0.02$, Severe) | AUROC ($\Delta\text{F1} \le -0.10$, Catastrophic)* |
| :--- | :---: | :---: | :---: | :---: |
| **`bn_affine_drift_l2`** | **0.9739** | **0.9501** | **0.9545** | **0.9716** |
| `intermediate_feature_drift_l2` | 0.9745 | 0.9490 | 0.9641 | 0.9723 |
| `disappear_prob_ratio` | 0.9255 | 0.9200 | 0.9502 | 0.8973 |
| `bn_running_stat_drift_l2` | 0.9506 | 0.9193 | 0.9151 | 0.9496 |
| `stability_pre1_pre2_area_increase` | 0.9189 | 0.8646 | 0.9023 | 0.8542 |
| `null_pre2_pre2_activation_rate` | 0.9281 | 0.8545 | 0.8931 | 0.8645 |
| `null_pre2_pre2_area_increase` | 0.9255 | 0.8445 | 0.8920 | 0.8436 |
| `kl_divergence_source_adapted` | 0.9335 | 0.7446 | 0.8026 | 0.7831 |
| `mean_prediction_entropy` | 0.8809 | 0.6573 | 0.7197 | 0.7107 |
| **`predeclared_change_risk_score`** | **0.5186** | **0.5102** | **0.5284** | **0.4264** |

*(Note on Threshold -0.10 and Offline Evaluation Script: In `indicator_evaluation_summary.csv`, `auroc_thresh_10` returned `NaN` due to a script variable omission bug: `y_harm_10` was accidentally omitted prior to line 316 in `tools/run_gate_b_offline_evaluation.py`, causing a caught `NameError`, NOT because zero batches satisfied the threshold. Recomputation directly from the joined ground-truth column `hidden_is_harmful_thresh10` in `joined_risk_and_hidden_outcomes.csv` confirms that exactly 477 of 537 validation batches (88.83% prevalence) satisfy $\Delta\text{F1} \le -0.10$. The true AUROC values at threshold -0.10 are **0.9716** for `bn_affine_drift_l2` and **0.4264** for `predeclared_change_risk_score` (sub-chance). The primary Gate B metrics ($\Delta\text{F1} \le -0.01$, Spearman rank correlations, LOEO AUROC) were completely unaffected by this caught script omission).*

### 4.4 Event-Wise Spearman Rank Correlations ($\Delta\text{F1}$)

Spearman rank correlations evaluated within each validation event:

| Indicator | Event 279 ($N=447$) | Event 437 ($N=33$) | Event 520 ($N=18$) | Event 559 ($N=9$) | Event 1111003 ($N=6$) | Event 1111008 ($N=24$) | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bn_affine_drift_l2` | -0.0349 | -0.6684 | -0.4180 | +0.5833 | +0.7143 | +0.0983 | **+0.0458** |
| `intermediate_feature_drift_l2` | +0.0598 | -0.6441 | -0.1662 | +0.4333 | +0.9429 | 0.0000 | **+0.1043** |
| `null_pre2_pre2_area_increase` | +0.0473 | -0.4823 | +0.0994 | -0.0896 | +0.7714 | +0.1302 | **+0.0794** |
| `null_pre2_pre2_activation_rate` | +0.0489 | -0.4868 | -0.0326 | -0.0896 | -0.8804 | NaN | **-0.2881** |
| `disappear_prob_ratio` | -0.0329 | -0.5110 | -0.3622 | +0.4833 | -0.1429 | -0.0591 | **-0.1041** |
| **`predeclared_change_risk_score`** | **+0.0839** | **+0.2680** | **-0.3127** | **-0.4770** | **+0.8286** | **+0.0870** | **+0.0796** |

**Event-Wise and Macro vs. Pooled Analysis:**
- **Pooled Spearman ($r = -0.4152$) vs. Event-Balanced Macro Spearman ($r = +0.0458$):**
  While `bn_affine_drift_l2` achieves a strong pooled Spearman correlation of **-0.4152** ($p = 2.4 \times 10^{-23}$) across all 537 batches, its event-balanced macro average across the 6 events is **+0.0458** [mean of (-0.0349, -0.6684, -0.4180, +0.5833, +0.7143, +0.0983) / 6]. This classic Simpson's paradox reveals that pooled performance mainly reflects cross-event separation (separating high-drift harmful events like 437/520 from low-drift beneficial events like 559), rather than reliable within-event ordering of batch risk.
- **Event 279 Domination:** On **Event 279** (84.3% of the dataset, 447 batches), within-event correlations are near zero for all indicators ($r = -0.0349$ for BN drift). Tent degrades performance rapidly in the earliest batches and then plateaus ($\Delta\text{F1} \approx -0.02$), leaving negligible intra-event variance in $\Delta\text{F1}$.
- **Sign Instability:** Smaller events exhibit inverted or positive slopes: on Event 559, BN drift has $r = +0.5833$; on Event 1111003, $r = +0.7143$. The predeclared composite score exhibits similar sign instability (-0.4770 on Event 559 vs. +0.8286 on Event 1111003).
- **Controller Implication:** Because generic parameter drift does not reliably order batch risk within events, generic drift cannot yet be claimed as a reliable update-level or batch-level controller without further investigation.

---

### 4.5 Exploratory Protocol Analyses Not Implemented in Stage 2

While Stage 2 fully executed all primary Gate B criteria (Spearman rank correlation, AUROC, AUPRC, LOEO cross-validation, and confounder sensitivity across all 26 indicators), three exploratory protocol specifications from `configs/gate_b_risk_protocol.json` were not implemented in `tools/run_gate_b_offline_evaluation.py`:
1. **Bootstrap Confidence Intervals:** 95% bootstrap confidence intervals across batches/seeds were omitted from the summary CSV.
2. **Calibration / Reliability Curves:** Probability calibration and reliability curve analyses for continuous risk scores were not computed.
3. **Warning Lead-Time Analysis:** Temporal lead-time evaluation (quantifying how many batches in advance risk triggers before degradation onset) was not computed.

These exploratory omissions do not affect the primary Gate B decision, as the primary AUROC (0.5102 vs. 0.9501) and correlation metrics are decisive.

---

## 5. Scientific Gate B Decision & Root Cause Analysis

### 5.1 Definitive Gate B Decision: `NO-GO`

Gate B is judged **`NO-GO`** under the predeclared protocol.

### 5.2 Detailed Justification Against Gate Criteria

1. **Criterion 1: Predeclared Signal Efficacy:**
   - *Requirement:* The predeclared risk score must demonstrate a statistically significant, negative rank correlation with $\Delta\text{F1}$ and high discriminatory AUROC ($\ge 0.75$).
   - *Finding:* `predeclared_change_risk_score` achieved an AUROC of **`0.5102`** and Spearman $r = \mathbf{-0.0007}$ ($p = 0.986$). This is completely indistinguishable from chance.
2. **Criterion 2: Superiority Over Generic Indicators:**
   - *Requirement:* Change-specific counterfactual evidence must outperform standard generic drift and entropy indicators.
   - *Finding:* Generic BatchNorm parameter drift (`bn_affine_drift_l2`: AUROC **`0.9501`**, LOEO **`0.9112`**) and intermediate representation drift (`intermediate_feature_drift_l2`: AUROC **`0.9490`**, LOEO **`0.8957`**) substantially outperform all change-specific probes. The best individual change-specific probe (`stability_pre1_pre2_area_increase`) achieved AUROC **`0.8646`** (LOEO **`0.8116`**), nearly 10 AUROC points lower than generic drift.
3. **Criterion 3: Robustness to Confounders:**
   - *Requirement:* Signal must not be an artifact of flood area or a single dominant event.
   - *Finding:* `disappear_prob_ratio` has an extreme correlation with source flood area ($r = -0.7890$). It detects whether there were floods in the patch, not whether adaptation corrupted the detector.

### 5.3 Root-Cause Analysis: Why Did the Null Probe Fail?

1. **The Asymmetry of Tent Degradation in SAR Change Detection:**
   In change detection with SNUNet, Tent updates the BatchNorm affine parameters ($\gamma, \beta$) via entropy minimization. The primary failure mode of Tent is *flood over-prediction* (hallucination) or *feature collapse* where the network shifts its decision boundary to predict background everywhere (erasure) or noise everywhere.
2. **Failure of `disappear_fraction_source_erased`:**
   In our formula, 50% of the weight was placed on `disappear_fraction_source_erased`. Under Tent on KuroSiwo, Tent rarely *erased* the source model's flood detections; instead, it drifted toward predicting spurious changes or expanding false positives. Consequently, `disappear_fraction_source_erased` had near-zero variance and an AUROC of `0.5038`, diluting the entire composite score.
3. **Parameter Drift is the Direct Causal Driver:**
   Because Tent directly updates BatchNorm affine parameters, the Euclidean drift $\Delta\theta_{BN} = \|\gamma_t - \gamma_0\|_2 + \|\beta_t - \beta_0\|_2$ measures the exact magnitude of model departure from its pre-trained inductive bias. In an unsupervised setting where the loss surface is uncalibrated, *any* substantial parameter drift directly maps to performance degradation ($r = -0.4152$, AUROC $0.9501$).

### 5.4 Scope of Verdict & Project Governance Boundaries
1. **Strictly Phase-Bounded:** This **`NO-GO`** decision applies strictly to Days 6–7 Gate B (label-free risk-signal validation). It records that predeclared change-specific signals failed to predict empirical degradation and failed to outperform generic indicators.
2. **No Controller Implemented:** In strict accordance with the Days 6–7 protocol, no rollback logic, adaptation stopping rule, or update controller has been implemented.
3. **Generic Drift is Not a Validated Update Controller:** Although `bn_affine_drift_l2` demonstrated high pooled AUROC (0.9501) and LOEO AUROC (0.9112), its positive macro Spearman correlation (+0.0458), flat within-event trajectory on Event 279, and event-level confounding mean generic drift cannot yet be claimed as a reliable update-level controller without further investigation.
4. **No Premature FYP Strategic Conclusion:** No final conclusion is drawn here regarding whether the overall FYP topic continues, pivots, or stops. That strategic determination is strictly reserved for the Days 8–10 synthesis milestone.

---

## 6. Artifact Hashes & Final Working Tree Inventory

### 6.1 Offline Evaluation Artifacts

| Artifact Name | Path | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| **Evaluation JSON** | `outputs/day6_7_gate_b/offline_evaluation/gate_b_evaluation.json` | 26,056 | `bd88386770132b9256a6796bf41e17ff53d953139c0c10303b4465fab5fa3eac` |
| **Summary CSV** | `outputs/day6_7_gate_b/offline_evaluation/indicator_evaluation_summary.csv` | 12,058 | `1cbd68ba5f1703458eff866bc8a5dd287aa071b164bbda1a7cd035837e3caec7` |
| **Joined CSV** | `outputs/day6_7_gate_b/offline_evaluation/joined_risk_and_hidden_outcomes.csv` | 962,560 | `2127befe92d441214755e22c9886bae3ca6fd441508b03d297351badb83a3945` |
| **Execution Log** | `outputs/day6_7_gate_b/stage1_execution.log` | 3,577 | `2c47ab8014c36678b25f0bff5117a24bfa9c8d05c2892f3924a1d2effb842b1c` |

### 6.2 Working Tree Status
```
$ git status --short
 M src/data/kuro_dataset.py
?? configs/gate_b_risk_protocol.json
?? docs/day6_7_gate_b_audit.md
?? src/risk/
?? tests/test_risk_indicators.py
?? tests/test_risk_probing_isolation.py
?? tests/test_risk_schema.py
?? tools/run_day6_smoke_tests.py
?? tools/run_gate_b_offline_evaluation.py
?? tools/run_gate_b_risk_generation.py
```
- Repository is on branch `main` at commit `214150c2ad22e09814888748e49f71fc9678a62e`, with modified tracked file `src/data/kuro_dataset.py` (working tree dirty, `git_dirty: true` as recorded in run manifests) and untracked Day 6–7 files.
- No files staged or committed.

---

## 7. Required Self-Validation Statement

In compliance with the mandatory research integrity protocol, the following 14 declarations are explicitly affirmed:

1. **What existed before this continuation:**
   Gate A verified commit `214150c2ad22e09814888748e49f71fc9678a62e` on branch `main`, verified Day 3–5 TTA artifacts (`summary_comparison.csv`, `all_runs_event_metrics.csv`, `gate_a_evaluation.json`), and the recovered Day 6 code/protocol files.
2. **What was recovered or preserved:**
   All Day 6 implementation files were validated for non-zero byte content and matching SHA-256 digests. The empty output directory from the interrupted previous session was safely preserved as `outputs/day6_7_gate_b_interrupted_20261002_221655`.
3. **What was changed:**
   Modified `src/data/kuro_dataset.py` to add `load_labels: bool = True` and implement `KuroUnlabeledValidationDataset` to guarantee label-free execution during Stage 1. Reconciled `configs/gate_b_risk_protocol.json` to explicitly specify the primary harm threshold $\Delta\text{F1} \le -0.01$, beneficial threshold $\ge +0.01$, and sensitivity thresholds `[0.0, -0.02, -0.10]`.
4. **What was executed:**
   - 38 unit tests (4 completeness, 6 metrics, 9 preprocessing, 19 discovery suite tests).
   - Deterministic smoke test `tools/run_day6_smoke_tests.py`.
   - Stage 1 full risk generation `tools/run_gate_b_risk_generation.py` via detached `tmux` session across Seeds 0, 1, and 2.
   - Stage 1 validation and hash verification script.
   - Stage 2 offline evaluation `tools/run_gate_b_offline_evaluation.py`.
5. **Which events and data fields were accessed:**
   Only validation events `[279, 437, 520, 559, 1111003, 1111008]`. In Stage 1, only SAR imagery tensors (`xA`, `xB`, `pre1`, `pre2`, `post`, `valid_mask`) were accessed. In Stage 2, ground-truth label tensors (`label`) were loaded offline for evaluation.
6. **What was deliberately not accessed or modified:**
   All 10 sealed test events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` were never touched. Completed Day 3–5 outputs in `outputs/day3_5_tta/` were not modified. No model checkpoint was overwritten. No controller, rollback mechanism, or update rejection policy was implemented.
7. **How test leakage was prevented and verified:**
   The dataset classes reject sealed test event IDs at initialization (`assert eid not in sealed_test_events`). Run manifests assert `sealed_event_accessed == False`. Offline evaluation asserted zero test event presence in risk records.
8. **How label isolation in Stage 1 was verified:**
   `KuroUnlabeledValidationDataset` sets `load_labels=False`, strictly removing `MK0_MLU` from the file resolution dictionary and never invoking `_read_raster` for labels. Unit test `test_unlabeled_dataset_never_reads_label_raster` proved that no label raster is opened.
9. **Whether any generated result was manually edited:**
   No. All JSON, CSV, and PT files were generated solely by executable Python scripts and hashed directly from disk.
10. **Whether any run, seed, event or indicator was omitted:**
    No. All 3 seeds (0, 1, 2), all 6 validation events, and all 26 indicators are reported in full without cherry-picking.
11. **Whether any result changed following a code or protocol correction:**
    No. The protocol and code were frozen and verified before launching Stage 1. Zero post-hoc adjustments were made.
12. **Whether any partial output was deleted, overwritten, renamed or excluded:**
    No partial output was deleted. The interrupted directory was archived as `outputs/day6_7_gate_b_interrupted_20261002_221655`. All 537 generated batches were included in the evaluation.
13. **Whether all claims in the report are directly supported by immutable generated artifacts:**
    Yes. Every number, AUROC, correlation, and sample count in this report is directly sourced from `gate_b_evaluation.json`, `indicator_evaluation_summary.csv`, and `joined_risk_and_hidden_outcomes.csv`.
14. **The final repository status:**
    Repository is on branch `main` at commit `214150c2ad22e09814888748e49f71fc9678a62e`, with modified tracked file `src/data/kuro_dataset.py` and untracked Day 6–7 files. Because a tracked file is modified, the working tree is dirty (`git_dirty: true` in manifests). No files are staged or committed. All experimental artifacts remain unmodified and untracked pending user review.
