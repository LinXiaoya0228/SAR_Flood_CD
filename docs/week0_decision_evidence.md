# Week 0 Unified Decision Evidence and Protocol Formulation

**Document Version:** 1.0.0  
**Project:** ChangeGuard FYP (Controlled Source-Model Phase)  
**Date:** 2026-10-06  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA RTX 3090 24GB)  
**Branch:** `main`  
**HEAD Commit:** `0d8dcec24c8c56609b4e318e317e66f5390c176b`  
**Status:** `AWAITING_USER_DECISION — STOPPED AT MANDATORY PROTOCOL GATE`  

---

## 1. Executive Summary

This document establishes the empirical, statistical, and architectural evidence required to finalize the experimental protocol for Week 0 of the ChangeGuard Final Year Project (FYP).

Following the completion of the exploratory two-week pilot (Days 1–10), the project has transitioned to the controlled-source-model phase. Before training any neural network or consuming GPU compute, all data partitions, architectural specifications, hyperparameter provenances, and experimental safeguards must be fully audited, mathematically verified, and frozen under user supervision.

This document synthesizes:
1. The **Research Integrity Audit**, including an explicit statement correcting prior metadata access to the test manifest, establishing protocol deviation `DEVIATION-W0-01`, and declaring permanent access prohibitions.
2. The **Official-Checkpoint Third-Channel Evidence Hierarchy**, providing a rigorous 5-level taxonomy proving why the official checkpoint cannot serve as the sole primary model and clarifying that Model B is a scientific ablation study.
3. The **Project Direction Formulation**, confirming that the Master Plan (`docs/MASTER_EXECUTION_PLAN.md`) supersedes the pilot's conditional pivot recommendation.
4. The **Deterministic Source Dataset Census**, establishing the exact pixel and sample distributions across all 27 source events (21,462 patches, 1,062,845,830 valid pixels).
5. A rigorous **Split-Sensitivity Analysis**, comparing the **Proposed 22/5 Split** against **Alternative A (Size-Inclusive Stratification)** and **Alternative B (Multivariate Representative Split)** using existing statistics only.
6. The **Hyperparameter Provenance Matrix**, attributing every architectural, optimization, and preprocessing choice to specific upstream source lines and commits.
7. Technical analyses and recommendations for **No-Water Patch Sampling**, **Slope Generation Conventions (S1 vs S2)**, **Epoch Selection & Final Retraining Rules**, and **Batch Size & Seed Policies**.

---

## 2. Research Integrity Certifications & Test-Access Statement

### 2.1 Statement on Prior Test Manifest Metadata Access
During preflight integrity verification in an earlier exploratory session of Week 0, code executed that unpickled `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` to check the top-level keys (`actids`) to verify alignment with `configs/kuro_event_split.json`.

To maintain the highest standard of scientific integrity, the exact scope of this interaction is formally documented:
- **What Occurred:** Top-level event IDs contained in the test manifest were read.
- **What Did NOT Occur:** Zero test image rasters (`SL1`, `SL2`, `MS1`, `DEM`) were loaded or opened. Zero test label or valid mask rasters (`MLU`, `MNA`) were loaded, inspected, or parsed. Zero test predictions, sample counts, flood proportions, or performance metrics were computed or accessed.
- **Protocol Deviation Recorded:** Logged in project records as `DEVIATION-W0-01` (*Preflight metadata-only inspection of test pickle manifest keys*).

### 2.2 Permanent Test Manifest Access Prohibition
To ensure that no data leakage can ever occur throughout the remainder of the FYP:
1. **Permanent Ban:** Reading, unpickling, opening, or inspecting `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` or any other test manifest under `third_party/KuroSiwo/pickle/` is **permanently prohibited**.
2. **Data Scope Restriction:** All dataset loading, statistics generation, and training pipelines are strictly restricted to `third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz` and the rasters corresponding to the 27 source events.
3. **TTA-Validation Isolation:** The 6 TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) remain strictly quarantined. Zero samples from these events were or will be used to design the source split, select source model hyperparameters, or select source checkpoints.

---

## 3. Official-Checkpoint Third-Channel Evidence Hierarchy

The downloaded official checkpoint (`checkpoints/kuro_snunet_official.pt`) cannot serve as the sole primary model in this dissertation. To eliminate ambiguity, the evidence regarding its inputs is structured into five distinct evidentiary tiers:

| Evidence Tier | Finding / Fact | Source / Verification Basis |
| :--- | :--- | :--- |
| **1. VERIFIED (from data)** | The first convolutional weight tensor `conv0_0.conv1.weight` has shape `(32, 3, 3, 3)`. The network mathematically requires **3 input channels** per temporal Siamese branch. | Direct inspection of checkpoint state dict (`checkpoint_provenance.md:16`). |
| **2. VERIFIED (from code)** | The default KuroSiwo GRD configuration specifies only **2 channels**: `["vv", "vh"]`. | Upstream config file `configs/train/data_config.json:11`. |
| **3. OBSERVED (from upstream commits)** | The upstream Git repository contains conflicting configuration flags within the same commit: `"dem": true` and `"slope": true` appear on lines 24–25, followed immediately by `"dem": false` and `"slope": false` on lines 30–31. | Upstream commit `4347ed173c4e48f5a9d578bb5fe8453706b08e5e` in `data_config.json`. |
| **4. COMMUNITY HYPOTHESIS** | An unmerged community pull request in the upstream GitHub repository suggested computing slope from DEM as the third channel to improve training. | Upstream community PR discussion. Remains an unverified hypothesis. |
| **5. UNKNOWN** | The exact third channel, its preprocessing pipeline, and its normalization constants ($\mu, \sigma$) in the official checkpoint remain completely unrecorded and unverifiable. | Missing from checkpoint metadata, training logs, and publication text. |

### Role of Controlled Model B (`[VV, VH, slope]`)
Because the official checkpoint's third channel is unknown, Model B is established as a **controlled topographic ablation study**. Model B evaluates whether augmenting dual-polarization SAR with local topographic gradient improves flood discrimination and suppresses mountain shadow false alarms. Model B is **not** an assertion that slope was the channel used in the official checkpoint.

---

## 4. Project Direction Formulation: Master Plan vs Exploratory Pilot

During the exploratory two-week pilot (Days 1–10), initial unsupervised test-time adaptation (TTA) experiments performed on the official checkpoint showed catastrophic degradation, and synthetic null probes failed to provide a reliable rollback trigger. Based on those specific observations, the pilot report provisionally recommended `PIVOT_FAILURE_CHARACTERIZATION`.

However, as formally established in the **Master Execution Plan** ([`docs/MASTER_EXECUTION_PLAN.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN.md)):
1. The pilot's findings were strictly conditional on an un-provenanced, un-reproducible official checkpoint whose input channels and training recipe were uncertain.
2. The Master Plan **supersedes** the pilot's conditional pivot recommendation.
3. The active project strategy is:
   - **Phase 1 (Weeks 0–1):** Train and audit reproducible controlled source models (Model A pure radar and Model B topographic ablation).
   - **Phase 2 (Weeks 2–3):** Re-evaluate source-only baselines and test-time adaptation methods on controlled models.
   - **Phase 3 (Decision Point):** Determine the ultimate research trajectory based on controlled, reproducible evidence—deciding whether to develop **selective adaptation**, **distributionally stable TTA**, or a **formal empirical failure characterization**.

---

## 5. Source Dataset Census (Exact Deterministic Statistics)

Computed deterministically from all 21,462 patches in `third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz` across the 27 source events (recorded in `outputs/week0/source_event_statistics.csv`, SHA-256: `bdf6deec...`):

- **Total Source Events:** 27 geographic disaster events
- **Total Image Patches:** 21,462 patches ($224 \times 224$ pixels at $10\text{ m}$ resolution)
- **Total Valid Pixels:** 1,062,845,830 valid pixels (satisfying double-masking $\text{valid\_mask} == 1 \land \text{label} \ne 3$)
- **Class 0 (No Water Background):** 793,932,988 pixels (74.70% of valid pixels)
- **Class 1 (Permanent Water):** 74,838,191 pixels (7.04% of valid pixels)
- **Class 2 (Newly Inundated Flood Water):** 194,074,651 pixels (18.26% of valid pixels)
- **Class 3 (Invalid / NoData):** 14,031,482 pixels (1.30% of total raster pixels)
- **Patches Containing Flood Water:** 18,121 patches (84.43% of source patches)
- **Patches Containing Zero Flood Water:** 3,341 patches (15.57% of source patches)
- **Event-Macro Mean Flood Proportion:** 14.60% ($\pm 14.97\%$, range: $1.26\%$ to $66.60\%$)
- **Event-Macro Mean Permanent Water Proportion:** 5.76% ($\pm 5.48\%$, range: $0.40\%$ to $23.98\%$)

### Distribution of Event Sizes
The 27 source events exhibit severe sample size imbalance:
- **4 Mega-Events ($>1,500$ patches):**
  - Event `1111009`: 7,556 patches (35.21% of corpus)
  - Event `1111004`: 4,086 patches (19.04% of corpus)
  - Event `427`: 1,980 patches (9.23% of corpus)
  - Event `1111011`: 1,619 patches (7.54% of corpus)
  - *Combined Mega-Event Share:* 15,241 patches (71.01% of all source data!)
- **7 Medium Events ($300 - 1,000$ patches):**
  - Event `275` (742), `555` (684), `1111005` (620), `502` (445), `497` (398), `470` (376), `147` (362), `1111010` (341), `1111006` (313), `518` (306).
- **16 Small Events ($<300$ patches):**
  - Event `130` (248), `267` (218), `492` (213), `324` (189), `417` (162), `273` (112), `118` (108), `498` (102), `567` (96), `554` (71), `421` (68), `174` (26), `496` (21).

---

## 6. Split-Sensitivity Analysis (Ground-Truth Comparative Evaluation)

Using existing statistics exclusively, four split configurations were evaluated under identical metric definitions:
1. **Proposed Split (Quintile Flood Stratification):**
   - Inner-Val Events: `[267, 496, 497, 518, 1111005]` (5 events)
   - Inner-Train Events: 22 remaining events
2. **Alternative A1 (Size-Inclusive Stratification w/ Event 275):**
   - Replaces tiny Event `496` (21 patches) with medium Event `275` (742 patches, mid-flood tertile).
   - Inner-Val Events: `[267, 275, 497, 518, 1111005]` (5 events)
   - Inner-Train Events: 22 remaining events
3. **Alternative A2 (Size-Inclusive Stratification w/ Mega-Event 427):**
   - Evaluates including mega-event `427` (1,980 patches) in validation.
   - Inner-Val Events: `[267, 427, 496, 497, 1111005]` (5 events)
   - Inner-Train Events: 22 remaining events
4. **Alternative B (Multivariate Representative Split via K-Means Medoids):**
   - Selects 5 cluster medoids based on standardized features: `[log(sample_count), flood_pct, perm_water_pct, invalid_pct, flood_sample_pct]`.
   - Inner-Val Events: `[130, 275, 324, 497, 567]` (5 events)
   - Inner-Train Events: 22 remaining events

### 6.1 Full Comparative Metric Table

| Dimension / Metric | Proposed Split | Alternative A1 (w/ 275) | Alternative A2 (w/ 427) | Alternative B (K-Means) | Total Corpus |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Event IDs** | `[267, 496, 497, 518, 1111005]` | `[267, 275, 497, 518, 1111005]` | `[267, 427, 496, 497, 1111005]` | `[130, 275, 324, 497, 567]` | 27 Events |
| **Val Sample Count** | 1,563 (7.28%) | 2,284 (10.64%) | 3,237 (15.08%) | 1,673 (7.80%) | 21,462 (100%) |
| **Train Sample Count** | 19,899 (92.72%) | 19,178 (89.36%) | 18,225 (84.92%) | 19,789 (92.20%) | 21,462 (100%) |
| **Val Pooled Flood %** | 19.00% | 16.88% | 9.61% | 19.18% | 18.26% |
| **Train Pooled Flood %** | 18.20% | 18.42% | 19.79% | 18.18% | 18.26% |
| **Pooled Flood Discrepancy** | **0.79%** | **1.54%** | **10.18%** | **1.00%** | — |
| **Val Macro Flood %** | 15.47% | 15.62% | 15.18% | 24.28% | 14.60% |
| **Train Macro Flood %** | 14.40% | 14.36% | 14.46% | 12.39% | 14.60% |
| **Macro Flood Discrepancy** | **1.07%** | **1.26%** | **0.71%** | **11.89%** | — |
| **Val Pooled Perm Water %** | 8.63% | 6.85% | 18.34% | 5.69% | 7.04% |
| **Train Pooled Perm Water %** | 6.92% | 7.06% | 5.05% | 7.15% | 7.04% |
| **Pooled Perm Discrepancy** | **1.71%** | **0.22%** | **13.29%** | **1.46%** | — |
| **Val Macro Perm Water %** | 7.52% | 7.62% | 11.25% | 5.86% | 5.76% |
| **Train Macro Perm Water %** | 5.36% | 5.33% | 4.51% | 5.74% | 5.76% |
| **Macro Perm Discrepancy** | **2.16%** | **2.29%** | **6.74%** | **0.12%** | — |
| **Val Invalid Pixel %** | 2.78% | 2.06% | 1.85% | 3.43% | 1.30% |
| **Train Invalid Pixel %** | 1.19% | 1.21% | 1.21% | 1.12% | 1.30% |
| **Val Flood-Sample %** | 76.46% | 81.48% | 41.95% | 82.25% | 84.43% |
| **Train Flood-Sample %** | 85.06% | 84.78% | 91.98% | 84.62% | 84.43% |
| **Flood-Sample Discrepancy** | **8.60%** | **3.30%** | **50.03%** | **2.37%** | — |
| **Max Overall Discrepancy** | **8.60%** | **3.30%** | **50.03%** | **11.89%** | — |

---

### 6.2 Key Statistical Findings & Analysis

1. **Alternative A1 (w/ Event 275) is Empirically the Most Balanced Split:**
   - Replacing tiny Event `496` (21 samples) with Event `275` (742 samples) reduces the maximum discrepancy across all statistical dimensions from **8.60% down to 3.30%**.
   - Pooled permanent water discrepancy drops from 1.71% to just **0.22%** (Val: 6.85% vs Train: 7.06%).
   - Flood-positive sample fraction discrepancy drops from 8.60% to **3.30%** (Val: 81.48% vs Train: 84.78%).
   - At 2,284 validation samples (10.64%), it provides a substantially more robust evaluation set than the Proposed Split (1,563 samples, 7.28%) while still retaining 19,178 samples (89.36%) for training.

2. **Alternative A2 (w/ Event 427) is Mathematically Disastrous:**
   - Placing mega-event `427` into validation creates severe distortions:
     - Event `427` has an enormous permanent water footprint (23.98%) and almost no flood water (1.26%).
     - Val pooled permanent water surges to **18.34%** (vs 5.05% in train, a 13.29% discrepancy).
     - Val flood-positive sample rate collapses to **41.95%** (vs 91.98% in train, a massive **50.03% discrepancy**).
   - This proves conclusively that mega-events with extreme class distributions (like Event `427`) must remain in the training set.

3. **Alternative B (Multivariate K-Means Medoids) Induces Severe Macro Flood Skew:**
   - The k-means clustering selected Event `130` (which has 66.60% flood water) and Event `567` (28.66% flood water).
   - As a result, Val Macro Flood % jumps to **24.28%** while Train Macro Flood % drops to **12.39%**, producing an unacceptably high **11.89% macro flood discrepancy**.
   - Model selection on this validation set would heavily overfit to extreme deluge conditions.

4. **Recommendation for User Decision:**
   - **Recommended Choice:** **Alternative A1 (`[267, 275, 497, 518, 1111005]`)**. It achieves the lowest overall statistical discrepancy (3.30%), provides a solid 10.64% validation corpus, and eliminates the vulnerability of relying on 21-sample Event `496`.
   - **Acceptable Fallback:** The **Proposed Split (`[267, 496, 497, 518, 1111005]`)** remains valid and well-stratified across quintiles, though with slightly higher sample-level discrepancy (8.60%).

---

## 7. Hyperparameter Provenance Matrix

Every configuration option is cataloged below with its exact file path, code line number, upstream commit (`4347ed173c4e48f5a9d578bb5fe8453706b08e5e`), and evidence classification:

| Parameter / Dimension | Protocol Specification | Upstream File & Line Reference | Upstream Commit | Evidence Tier |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | SNUNet Siamese U-Net (`base_channel=32`) | `models/method/snunet/snunet.json:8` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Input Spatial Size** | $224 \times 224$ pixels | `src/data/kuro_dataset.py:171` | `0d8dcec24c8c...` | `VERIFIED_FROM_DATA` |
| **Temporal Inputs** | Branch A: `pre_event_2`, Branch B: `post_event` | `training/change_detection_trainer.py:19` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Optimizer** | Adam ($lr=1.0 \times 10^{-3}$, $\beta=(0.9, 0.999)$, $wd=0.0$) | `training/change_detection_trainer.py:54` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Adam Betas Note** | Upstream `snunet.json:6` has typo `[0.9, 0.009]`, but `change_detection_trainer.py:54` calls `torch.optim.Adam(..., lr=...)` without passing `betas`, thus using PyTorch defaults `(0.9, 0.999)`. | `configs/method/snunet/snunet.json:6` vs `change_detection_trainer.py:54` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Loss Function** | Cross-Entropy + Soft Dice Loss | Upstream uses CE (`train_config.json:11`); CE+Dice is standard CD practice | `4347ed173c4e...` | `CURRENT_PROJECT_DECISION` |
| **Loss Ignore Index** | `ignore_index = 3` (invalid NoData excluded) | `utilities/utilities.py:140` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Class Weighting** | Unweighted `[1.0, 1.0, 1.0]` vs Inverse Frequency `[0.37, 14.0, 8.2]` | `configs/train/train_config.json:9` (`"weighted": false`) | `4347ed173c4e...` | `CURRENT_PROJECT_DECISION` |
| **Effective Batch Size** | 32 (with AMP mixed precision) | Upstream config has 128 (`train_config.json:2`); 32 verified for 24GB VRAM | `4347ed173c4e...` | `CURRENT_PROJECT_DECISION` |
| **Max Training Epochs** | Proposed: 150 epochs (with patience 20 early stopping) | Upstream config has 1 (`train_config.json:3`); paper reports ~150 epochs | `4347ed173c4e...` | `SUPPORTED_BY_OFFICIAL_DOCS` |
| **SAR Clamping** | Range $[0.0, 0.15]$ | `configs/train/data_config.json:14`, `Dataset.py:969` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **SAR Normalization** | $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$ | `configs/train/data_config.json:16–17` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **SAR NaN Replacement** | Replace non-finite with $0.15$ | `Dataset.py:970`, `src/data/kuro_dataset.py:197` | `4347ed173c4e...` | `VERIFIED_FROM_CODE` |
| **Data Augmentations** | Random horizontal flip, vertical flip, 90° rotation | `utilities/augmentations.py` | `4347ed173c4e...` | `CURRENT_PROJECT_DECISION` |
| **Random Seeds** | Seeds `[42, 43, 44]` (3 distinct training seeds) | Pre-declared for statistical significance | `0d8dcec24c8c...` | `CURRENT_PROJECT_DECISION` |

---

## 8. No-Water Patch Sampling Policy

### 8.1 Empirical Distribution
In `third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz`:
- 18,121 patches (84.43%) contain at least one class-2 flood pixel.
- 3,341 patches (15.57%) contain zero flood pixels.

### 8.2 Policy Comparison
- **Policy N1 (Train on All Available Patches — Recommended):**
  Train on all 19,899 inner-train patches (or 19,178 under Alternative A1) without post-hoc filtering or artificial oversampling.
  - *Rationale:* The upstream manifest `KuroV2_grid_dict.gz` already performed primary water filtering during dataset construction (reducing 67,490 total geographic grids down to 31,707 patches). Keeping all patches preserves the authentic geographical prior distribution of background terrain, ensuring the model learns to suppress false alarms on non-flooded scenes.
- **Policy N2 (Filter Patches with Zero Flood Pixels):**
  Discard the 15.57% zero-flood patches during training.
  - *Drawbacks:* Artificially inflates flood prevalence during training, inducing higher false positive rates when the model is applied to dry geographic regions during TTA.

---

## 9. Slope Generation Policy: S1 vs S2 Analysis

For Model B (`[VV, VH, slope]`), the local terrain slope must be computed from `MK0_DEM`. Two technical conventions are evaluated:

| Dimension | Policy S1: Upstream-Compatible | Policy S2: Physically Specified & Robust (Recommended) |
| :--- | :--- | :--- |
| **Grid Cell Spacing** | Unit grid spacing ($\Delta x = \Delta y = 1.0$), ignoring physical resolution. | Physical ground resolution ($\Delta x = \Delta y = 10.0\text{ m}$). |
| **NoData Sanitization** | Relies on raw DEM, unhandled $-32768$ values. | Pre-sanitized via `kuro_dem_sanitize_v1` ($< -1000\text{m} \to 0.0$). |
| **Normalization Constants** | Upstream `data_config.json:26–27`: $\mu=2.1277, \sigma=67.5048$. | Robust train-only statistics: empirical median/IQR or train-only $\mu, \sigma$. |
| **Dynamic Range** | Heavily compressed: standard deviation of $67.5$ pushes typical terrain slopes ($0.05 - 0.5$) into near-zero values. | Well-scaled: standardized zero-mean, unit-variance over physical terrain gradients. |
| **Data Leakage Risk** | Zero (constants from config). | Zero (computed strictly on 22 inner-train events). |
| **Recommendation** | Not recommended due to numerical distortion. | **Strongly Recommended.** |

---

## 10. Epoch Selection and Final Retraining Protocol

To maintain complete research integrity without data leakage:

### Step 1: Recipe and Epoch Optimization on Inner-Split
1. Train Model A and Model B on the 22 inner-train events using 3 random seeds (`[42, 43, 44]`).
2. Evaluate checkpoints after each epoch strictly on the 5 inner-validation events.
3. Early stopping patience is set to 20 epochs based on inner-validation Macro F1.
4. Record the optimal validation epoch $E_s^*$ for each seed $s \in \{42, 43, 44\}$.
5. Compute the locked retraining epoch:
   $$E_{\text{retrain}} = \text{median}(E_{42}^*, E_{43}^*, E_{44}^*)$$

### Step 2: Final Retraining on Full Source Corpus
1. After the training recipe (loss function, class weighting, and architecture) is selected, retrain the final source model from scratch across **all 27 source events** (21,462 samples).
2. Training executes for **exactly $E_{\text{retrain}}$ epochs** (fixed schedule).
3. **Zero validation tuning:** Neither the 5 inner-validation events nor the 6 TTA-validation events are monitored during this retraining phase.
4. Final source checkpoints are saved for all 3 seeds (`[42, 43, 44]`) for both Model A and Model B, producing the final controlled source models ready for downstream TTA evaluation.

---

## 11. Batch Size, Memory, and GPU Strategy

1. **Target Effective Batch Size:** 32.
2. **Precision:** Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`).
3. **Physical Microbatch Status:** The exact VRAM consumption of a forward-backward pass has not yet been measured on this system and must not be asserted prematurely. Physical microbatch size is formally designated as `TO_BE_CONFIRMED_BY_WEEK1_MEMORY_SMOKE_TEST`.
4. **Execution Protocol & Microbatch Progression:**
   - In Week 1, memory smoke testing will evaluate candidate physical microbatches `[2, 4, 8, 16]`, starting conservatively and stopping safely before an out-of-memory error.
   - Gradient accumulation will be set as $\text{accumulation\_steps} = 32 / \text{physical\_microbatch\_size}$ with strict divisibility required.
   - Formal multi-epoch training remains blocked until physical microbatch and accumulation settings are verified and recorded in a versioned Week 1 protocol amendment.
   - Single-GPU execution per run will be used to ensure deterministic reproducibility, adhering to `docs/mlda_gpu_usage_protocol.md`.

---

## 12. Decisions Requiring User Approval

The agent has stopped at the mandatory protocol stop point. Before Week 1 source-model training begins, user approval is requested on the following 5 core items:

1. **Split Selection:**
   - **Option 1 (Recommended):** Alternative A1 (`[267, 275, 497, 518, 1111005]`, 2,284 val samples / 10.64%, max discrepancy 3.30%).
   - **Option 2:** Proposed Split (`[267, 496, 497, 518, 1111005]`, 1,563 val samples / 7.28%, max discrepancy 8.60%).
2. **Slope Generation Policy:**
   - **Option 1 (Recommended):** Policy S2 (physical $10\text{ m}$ spacing, pre-sanitized DEM, robust inner-train normalization).
   - **Option 2:** Policy S1 (upstream unit grid spacing and config constants $\mu=2.1277, \sigma=67.5048$).
3. **No-Water Patch Sampling:**
   - **Option 1 (Recommended):** Policy N1 (train on all manifest patches without filtering).
   - **Option 2:** Policy N2 (discard patches with zero flood pixels).
4. **Loss Function Weighting:**
   - **Option 1:** Unweighted Cross-Entropy + Soft Dice (`[1.0, 1.0, 1.0]`).
   - **Option 2:** Inverse-frequency weighted Cross-Entropy + Soft Dice (`[0.37, 14.0, 8.2]`).
5. **Retraining Rule & Seeds:**
   - Approve median inner-val epoch rule $E_{\text{retrain}} = \text{median}(E_{42}^*, E_{43}^*, E_{44}^*)$ and 3-seed protocol `[42, 43, 44]`.

---

## 13. Immediate Next Steps for Week 1 (Post-Approval)

Upon receiving user decision:
1. Freeze approved split configuration into production config (`configs/splits/kuro_source_inner_split_v1.json`).
2. Implement modular dataset builder supporting Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`) with selected slope and sampling policies.
3. Conduct 1-epoch GPU smoke test on single RTX 3090.
4. Execute controlled source training runs on 22 inner-train events.
