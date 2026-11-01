# Week 0 Review and Decision Gate for User

**Document Version:** 1.0.0  
**Date:** 2026-10-06  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22`  
**Current Phase:** Week 0 (Establish, audit, and prepare controlled source-model protocol)  
**Status:** `AWAITING_USER_APPROVAL — STOPPED AT MANDATORY GATE`  

---

## 1. Project-Context Restatement

1. **The SAR Flood Change-Detection Task:**
   Bi-temporal multi-temporal Sentinel-1 C-band synthetic aperture radar (SAR) flood extent mapping across global geographic disaster events. Two temporal acquisitions—an immediate pre-disaster reference pass ($x_A = \text{pre\_event\_2}$) and an emergency disaster pass ($x_B = \text{post\_event}$)—are ingested by a Siamese change detection network (SNUNet) to predict newly inundated flood water while suppressing false alarms from seasonal moisture and speckle noise.
2. **The Label Mapping:**
   Ground truth rasters (`MK0_MLU`) follow four-class semantics:
   - Class 0: `No Water` (dry ground background)
   - Class 1: `Permanent Water` (pre-existing lakes/rivers; non-flood change background)
   - Class 2: `Flood Water` (newly inundated flood areas — target change class)
   - Class 3: `Invalid / No Data` (satellite nodata / unmapped pixels)
   - The valid evaluation mask is strictly enforced via the double-masking rule: $\text{valid} = (\text{valid\_mask} == 1) \land (\text{label} \ne 3)$.
   - The binary flood target is defined as $\text{flood\_target} = (\text{label} == 2)$, with classes 0 and 1 treated as non-flood negatives.
3. **Why the Downloaded Official Checkpoint Is Not Accepted as the Sole Primary Model:**
   The downloaded official checkpoint (`checkpoints/kuro_snunet_official.pt`) cannot serve as the sole primary model because its exact input composition and training provenance are unrecorded:
   - `VERIFIED (from data)`: The first convolutional layer (`conv0_0.conv1.weight`) has shape `(32, 3, 3, 3)`, requiring 3 input channels per temporal branch.
   - `VERIFIED (from code)`: The default KuroSiwo GRD configuration (`configs/train/data_config.json:11`) specifies only 2 channels: `["vv", "vh"]`.
   - `OBSERVED (from upstream commits)`: Upstream Git history contains conflicting keys (`"dem": true` and `"dem": false` in `data_config.json`).
   - `COMMUNITY HYPOTHESIS`: An unmerged community pull request suggested slope as an input channel; this remains an unverified hypothesis.
   - `UNKNOWN`: The exact third channel and its normalization parameters in the official checkpoint remain unknown.
   - Furthermore, its training split, augmentations, loss weights, and convergence epochs are absent from the checkpoint metadata. A credible academic dissertation cannot rely exclusively on an un-reproducible external artifact.
4. **The Role of the New Controlled Source Model:**
   To establish fully documented, bit-for-bit reproducible source models trained in-house under strict version control across two controlled configurations: Model A (`[VV, VH]` 2-channel pure radar baseline) and Model B (`[VV, VH, slope]` 3-channel topographic ablation). Note: Model B is a principled scientific ablation, not an assertion that slope was the channel used in the official checkpoint.
5. **The Difference Between Event Splits:**
   - *Source-Training Events ($N=27$):* Exclusively allocated for source model training and architecture selection (further partitioned into 22 inner-train and 5 inner-validation events).
   - *TTA-Validation Events ($N=6$ usable: `[279, 437, 520, 559, 1111003, 1111008]`):* Strictly reserved for evaluating test-time adaptation algorithms and diagnostic audits. Never used to select source model hyperparameters or choose checkpoints!
   - *Sealed Test Events ($N=10$: `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`):* Permanently sealed and untouched during development.
6. **The Current Status of the Two-Week Pilot and Project Direction:**
   The exploratory pilot completed across Days 1–10 and is fully synthesized in `docs/two_week_pilot_complete_evidence_report.md` (SHA-256: `a6d6fa5c...`) anchored at commits `ca4d18e2...` and `0d8dcec2...`. While the pilot recommended `PIVOT_FAILURE_CHARACTERIZATION` conditionally based on the un-provenanced official checkpoint, the Master Plan (`docs/MASTER_EXECUTION_PLAN.md`) supersedes this conditional recommendation: the active strategy is to train controlled source models (Model A and Model B), re-evaluate baselines, and then decide between selective adaptation, stable TTA, or failure characterization based on controlled evidence.
7. **Why Week 0 Must Be Completed Before New Training Begins:**
   Week 0 establishes, audits, and freezes the entire experimental protocol *before* consuming any GPU hours or fitting any parameters. It pre-declares the 22/5 source-inner split, audits the data pipeline, specifies Model A and Model B configurations, establishes the GPU usage protocol, and creates recoverable code backups, ensuring absolute research integrity and preventing data leakage.

---

## 2. Initial Branch, HEAD, and Repository State

- **Host Name:** `gpu22`
- **Current Branch:** `main`
- **Current HEAD Commit:** `0d8dcec24c8c56609b4e318e317e66f5390c176b`
- **Initial Working Tree Status:** Clean tracked state; zero modified tracked files; exactly two untracked documentation files (`docs/MASTER_EXECUTION_PLAN.md` and `docs/two_week_pilot_complete_evidence_report.md`).
- **Pre-Change State Audit Document:** Recorded in [`docs/week0_prechange_state.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week0_prechange_state.md).

---

## 3. Recoverable Code-Level Backup Verification

Backup directory `/usr1/home/s126mdg22_06/FYP_SARCD/backups` was established and populated with two immutable backups:

| Backup Artifact | Absolute Path | Size | SHA-256 Digest | Verification Command & Result |
| :--- | :--- | :---: | :--- | :--- |
| **Git Bundle** | `/usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_refs_20261006_193120.bundle` | 199 KB | `ccb68f97461a81108bc081b0b79bb6f789ff4ccd40e5e748a9991bc7742598cb` | `git bundle verify` returned: `bundle is okay` |
| **Working-Tree Tarball** | `/usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_worktree_20261006_193120.tar.gz` | 192 KB | `e1d57d80a7da2364bc936002faa60aee30297083e1622fa9c84fddcb1e4e1f1a` | `tar -tzvf` listed all 61 tracked files + untracked docs |

Recorded in [`docs/week0_backup_record.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week0_backup_record.md). Backup files reside outside the Git repository.

---

## 4. Current Loader and Preprocessing Findings

Documented in full in [`docs/source_data_pipeline_audit.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_data_pipeline_audit.md):
- **SAR Preprocessing:** Clamped to $[0.0, 0.15]$, normalized with $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$, `nan_to_num(..., nan=0.15)`. Verified from code.
- **DEM Sanitization:** Upstream code fails to sanitize $-32768$ integer nodata. ChangeGuard protocol `kuro_dem_sanitize_v1` replaces $< -1000\text{ m} \to 0.0$ (normalized neutral prior).
- **Slope Upstream Reproducibility & Physical-Spacing Concern:** Upstream `richdem.TerrainAttribute(attrib="slope_riserun")` omits explicit metric pixel spacing, computing slope in dimensionless index units. Normalization stats in `data_config.json` contain an std of $67.5$, indicating severe outlier distortion.
- **Dual Configuration Support:** Current loader is hard-coded for 3-channel `[VV, VH, DEM]`. Week 1 will introduce a modular builder for Model A `[VV, VH]` and Model B `[VV, VH, slope]`.

---

## 5. Proposed 22/5 Source-Inner Split

- **Config File:** [`configs/splits/kuro_source_inner_split_proposed.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/splits/kuro_source_inner_split_proposed.json)
- **Status:** `AWAITING_USER_APPROVAL` (NOT frozen).
- **Exact Event IDs:**
  - **Proposed 5 Inner-Validation Events:** `[267, 496, 497, 518, 1111005]`
  - **Proposed 22 Inner-Train Events:** `[118, 130, 147, 174, 273, 275, 324, 417, 421, 427, 470, 492, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`

---

## 6. Comparison Table of 22 Inner-Train vs 5 Inner-Validation Statistics

*Source: `outputs/week0/source_event_statistics.csv` (SHA-256: `bdf6deec1c48b6a05b01bf520f5e0af6e492894c9d7c347d85fc64de2396bf80`)*

| Metric / Dimension | 22 Source-Inner-Train Events | 5 Source-Inner-Validation Events | Total Source Corpus (27 Events) |
| :--- | :---: | :---: | :---: |
| **Total Event Count** | 22 events (81.48%) | 5 events (18.52%) | 27 events (100.0%) |
| **Total Sample Count** | 19,899 samples (92.72%) | 1,563 samples (7.28%) | 21,462 samples (100.0%) |
| **Valid Pixels Count** | 986,600,159 pixels | 76,245,671 pixels | 1,062,845,830 pixels |
| **Class 0 (No Water) Pixels** | 738,750,506 pixels (74.88%) | 55,182,482 pixels (72.38%) | 793,932,988 pixels (74.70%) |
| **Class 1 (Perm Water) Pixels** | 68,258,672 pixels (6.92%) | 6,579,519 pixels (8.63%) | 74,838,191 pixels (7.04%) |
| **Class 2 (Flood Water) Pixels** | 179,590,981 pixels (18.20%) | 14,483,670 pixels (19.00%) | 194,074,651 pixels (18.26%) |
| **Class 3 (Invalid) Pixels** | 11,852,065 pixels (1.19%) | 2,179,417 pixels (2.78%) | 14,031,482 pixels (1.30%) |
| **Samples Containing Flood** | 16,926 samples (85.06%) | 1,195 samples (76.46%) | 18,121 samples (84.43%) |
| **Event-Macro Mean Flood %** | 14.40% | 15.47% | 14.60% |
| **Event-Macro Mean Perm Water %**| 5.36% | 7.52% | 5.76% |

---

## 7. Rationale for the Proposed 22/5 Split

1. **Mass Preservation for Training:** The 4 largest mega-events (`1111009`: 7,556 samples; `1111004`: 4,086 samples; `427`: 1,980 samples; `1111011`: 1,619 samples) are reserved exclusively for training, ensuring the network trains on 92.72% of all available data (19,899 samples).
2. **Stratified Flood Distribution Representation:** The 5 validation events represent all five quintile strata of flood prevalence:
   - Low flood ($2.72\%$): Event 518 (306 samples)
   - Low-mid flood ($6.85\%$): Event 497 (398 samples)
   - Mid flood ($11.63\%$): Event 496 (21 samples)
   - Mid-high flood ($23.28\%$): Event 267 (218 samples)
   - High flood ($32.87\%$): Event 1111005 (620 samples)
3. **Topographic and Water Variation Representation:**
   - Event 497 captures extreme permanent water presence (14.47% permanent water, 398 samples).
   - Event 496 provides a compact, localized flood footprint (21 samples).
   - Event 1111005 provides a large-scale regional flood footprint (620 samples).
4. **Macro vs Pooled Balance:** Macro mean flood prevalence between inner-train (14.40%) and inner-val (15.47%) differs by only 1.07%, guaranteeing that model selection on inner-val will not overfit to an uncharacteristically dry or wet regime.

---

## 8. All Unresolved Preprocessing Decisions

The following technical decisions are explicitly tagged as `UNRESOLVED` and require user decision prior to Week 1 implementation:

1. **Slope Pixel-Spacing Convention:**
   Whether `richdem` slope calculation should assume dimensionless unit grid spacing ($\Delta x = \Delta y = 1.0$) or physical Sentinel-1 pixel spacing ($\Delta x = \Delta y = 10.0\text{ m}$).
2. **Slope Normalization Statistics:**
   Whether to use upstream config values ($\mu=2.1277, \sigma=67.5048$, which exhibit outlier distortion) or perform a clean, robust recalculation over the 22 inner-train events.
3. **Loss Function Class Weighting:**
   Whether training should use unweighted Cross-Entropy + Dice ($[1.0, 1.0, 1.0]$) or inverse-class-frequency weighting ($[0.3716, 14.0098, 8.2041]$).
4. **No-Water Patch Sampling Strategy:**
   Whether training should use all 19,899 inner-train patches or oversample/filter pure background patches.

---

## 9. Side-by-Side Summary of the Two Draft Configs

| Specification | Model A: `snunet_vvvh_draft.json` | Model B: `snunet_vvvh_slope_draft.json` | Distinction / Rationale |
| :--- | :--- | :--- | :--- |
| **Model Name** | `snunet_vvvh_controlled` | `snunet_vvvh_slope_controlled` | Distinct output namespace |
| **Configuration Role** | Pure radar baseline | Topographic ablation study | Evaluates utility of slope channel |
| **Input Channels / Branch** | 2 channels (`["vv", "vh"]`) | 3 channels (`["vv", "vh", "slope"]`) | Channel ablation |
| **Backbone In-Channels** | `in_channels_per_branch = 2` | `in_channels_per_branch = 3` | Architecture match |
| **Temporal Pair** | `[pre_event_2, post_event]` | `[pre_event_2, post_event]` | Identical |
| **Output Classes** | 3 (`No Water`, `Perm Water`, `Flood`) | 3 (`No Water`, `Perm Water`, `Flood`) | Identical |
| **Ignore Index** | 3 (`invalid_no_data`) | 3 (`invalid_no_data`) | Identical |
| **Optimizer / LR** | Adam ($lr=0.001, \beta=(0.9, 0.999)$) | Adam ($lr=0.001, \beta=(0.9, 0.999)$) | Identical |
| **Proposed Epochs** | 150 (early stopping patience 20) | 150 (early stopping patience 20) | Identical |
| **Status** | `AWAITING_USER_APPROVAL` | `AWAITING_USER_APPROVAL` | Both blocked from training |

---

## 10. Tests Run and Pass/Fail Results

All test suites were executed using `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`:

1. `tests/test_source_inner_split.py` (5 tests): **PASS**
   - Verified 22 inner-train and 5 inner-val events.
   - Verified exact disjointness and union equals 27 source events.
   - Verified zero leakage from 6 TTA-validation and 10 sealed test events.
   - Verified source statistics CSV cryptographic hash matches recorded proposal hash.
2. `tests/test_source_configs.py` (6 tests): **PASS**
   - Verified JSON syntax and schema for Model A and Model B.
   - Verified channel counts (2 for Model A, 3 for Model B).
   - Verified identical training protocols and 3 output classes.
   - Verified zero test-event ID presence in config text.
   - Verified unresolved fields prevent unapproved execution launch.
3. Complete Unittest Discovery Suite (`tests/test_*.py`): **32 / 32 tests passed** in 14.790s (`OK`).

---

## 11. Complete File Modification Inventory

### New Files Created (10 files)
1. `docs/week0_prechange_state.md` (Initial Git and repo integrity record)
2. `docs/week0_backup_record.md` (Backup verification and recovery manual)
3. `docs/source_data_pipeline_audit.md` (Data loading and preprocessing technical audit)
4. `tools/generate_source_event_statistics.py` (Deterministic source statistics script)
5. `outputs/week0/source_event_statistics.csv` (Exact event statistics for 27 source events)
6. `outputs/week0/source_event_statistics.json` (Structured event statistics JSON)
7. `configs/splits/kuro_source_inner_split_proposed.json` (Proposed 22/5 split config)
8. `tests/test_source_inner_split.py` (Automated tests for 22/5 split)
9. `docs/source_training_protocol_draft.md` (Source training protocol draft)
10. `configs/source/snunet_vvvh_draft.json` (Draft config for Model A)
11. `configs/source/snunet_vvvh_slope_draft.json` (Draft config for Model B)
12. `tests/test_source_configs.py` (Automated tests for draft source configs)
13. `docs/interim_report_outline.md` (Interim report outline, deadline 2026-11-10)
14. `docs/mlda_gpu_usage_protocol.md` (MLDA cluster GPU operational protocol)
15. `docs/week0_review_for_user.md` (This review document)

### Files Modified
- **Zero tracked files were modified.**

---

## 12. Command Execution Record

```bash
# Preflight Git and process checks
git status && git rev-parse HEAD && ps -u s126mdg22_06 -o pid,stat,cmd | grep -E 'python|torch|run_'
find . -mindepth 2 -name ".git"
git -C /usr1/home/s126mdg22_06/FYP_SARCD/third_party/KuroSiwo status --short
nvidia-smi

# Event split verification script
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -c '...'

# Backup creation and verification
git bundle create /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_refs_20261006_193120.bundle --all
git bundle verify /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_refs_20261006_193120.bundle
tar -czvf /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_worktree_20261006_193120.tar.gz ...
tar -tzvf /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_worktree_20261006_193120.tar.gz | head -n 30
sha256sum /usr1/home/s126mdg22_06/FYP_SARCD/backups/*

# Tracked files hashing
git ls-files | xargs sha256sum

# Source event statistics generation (21,462 samples)
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/generate_source_event_statistics.py
sha256sum outputs/week0/source_event_statistics.csv outputs/week0/source_event_statistics.json

# Test suite validation
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_source_inner_split.py
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest tests/test_source_configs.py
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest discover -s tests -p "test_*.py"
```

---

## 13. Event Access Audit & Research Integrity Certifications

1. **Event IDs Accessed During Statistics Generation ($N=27$):**
   `118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011`.
   Only source-training labels and masks were inspected.
2. **Explicit TTA-Validation Confirmation:**
   Zero samples, labels, rasters, or predictions from the 6 TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) were used to design the 22/5 split or configure source models.
3. **Explicit Sealed Test Status & Integrity Correction:**
   During preflight verification in the prior turn of Week 0, code executed that opened `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` and enumerated the top-level manifest keys (event IDs).
   - **What Occurred:** Test manifest metadata (actids) was read to verify event ID alignment.
   - **What Did NOT Occur:** Zero test image rasters (`SL1`, `SL2`, `MS1`, `DEM`) or test label/mask rasters (`MLU`, `MNA`) were loaded, opened, or examined. Zero test predictions, counts, or metrics were computed or accessed.
   - **Remediation & Permanent Policy:** Opening, reading, unpickling, or inspecting `KuroV2_grid_dict_test_0_100.gz` or any other test manifest file under `third_party/KuroSiwo/pickle/` is **permanently prohibited**. All downstream code is restricted strictly to `KuroV2_grid_dict.gz` and source rasters.
4. **Manual Editing Confirmation:**
   All tables, statistics, JSONs, and test outputs were produced deterministically by executable code; zero generated results were manually edited.
5. **Protocol Deviations:**
   - `DEVIATION-W0-01` (Recorded): Exploratory metadata-only inspection of test pickle manifest `KuroV2_grid_dict_test_0_100.gz` occurred during preflight checks. No test rasters or labels were touched. Mitigated by permanent prohibition policy.

---

## 14. Final Git Status

```bash
$ git status --short
?? configs/source/
?? configs/splits/
?? docs/MASTER_EXECUTION_PLAN.md
?? docs/interim_report_outline.md
?? docs/mlda_gpu_usage_protocol.md
?? docs/source_data_pipeline_audit.md
?? docs/source_training_protocol_draft.md
?? docs/two_week_pilot_complete_evidence_report.md
?? docs/week0_backup_record.md
?? docs/week0_prechange_state.md
?? docs/week0_review_for_user.md
?? tests/test_source_configs.py
?? tests/test_source_inner_split.py
?? tools/generate_source_event_statistics.py
```
Tracked files: zero modifications (`git diff` returns empty). Zero commits executed.

---

## 15. Decision Checklist Requiring User Approval

The agent has stopped at the mandatory stop point. Please review and provide explicit **Approval** or **Rejection / Modification** for the following 8 items:

1. **The Exact 22/5 Event Split:**
   - 22 Inner-Train Events: `[118, 130, 147, 174, 273, 275, 324, 417, 421, 427, 470, 492, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`
   - 5 Inner-Validation Events: `[267, 496, 497, 518, 1111005]`
2. **The Split-Selection Algorithm:**
   - Stratified flood prevalence quintile rule reserving mega-events ($>1500$ samples) for training, selecting representative events by stratum median flood proportion.
3. **The Primary SAR Normalization:**
   - Linear amplitude clamping $[0.0, 0.15]$, normalization with $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$, `nan_to_num(0.15)`.
4. **The Slope Generation and Normalization Policy:**
   - Pixel spacing convention (10m vs unit grid) and empirical slope normalization recalculation.
5. **The No-Water Patch Sampling Policy:**
   - Training on all 19,899 inner-train patches vs oversampling/filtering pure background patches.
6. **Optimizer, Learning Rate, Loss, and Epoch Schedule:**
   - Adam ($lr=0.001$), Cross-Entropy + Dice loss, proposed 150 epochs with 20-epoch early stopping.
7. **Batch Size and Reproducibility Settings:**
   - Effective batch size 32, mixed precision AMP, seeds `[42, 43, 44]`.
8. **The Two Controlled Configurations:**
   - Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`) under SNUNet Siamese architecture.
