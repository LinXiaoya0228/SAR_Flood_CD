# Checkpoint Provenance and Semantic Audit

**Document version:** 1.0  
**Project:** ChangeGuard (FYP_SARCD)  
**Date:** 23 September 2026  
**Audited Checkpoint:** `ChangeGuard/checkpoints/kuro_snunet_official.pt`  
**Official Repository:** `third_party/KuroSiwo`  

---

## 1. Verified Checkpoint Facts

1. **File Type & Top-Level Keys**:
   * The file is a PyTorch checkpoint dictionary containing exactly:
     * `epoch`: `13`
     * `model_state_dict`: 134 parameter/buffer tensors
     * `optimizer_state_dict`: Adam optimizer state (learning rate: `0.001`, betas: `(0.9, 0.999)`, weight_decay: `0`)
     * `lr_scheduler_state_dict`: LambdaLR state (`base_lrs: [0.001]`, `last_epoch: 0`)
     * `loss`: `0.14699462056159973`
2. **Architecture Matching**:
   * The state dict loads strictly (`strict=True`) into `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
   * Total parameters: 12,035,107 across 30 `BatchNorm2d` layers and nested residual dense blocks.
   * Input layers: `conv0_0.conv1.weight` has shape `(32, 3, 3, 3)`.
   * Final projection: `conv_final.weight` has shape `(3, 128, 1, 1)` and `conv_final.bias` has shape `(3,)`.
3. **Execution Forward Pass**:
   * When provided two temporal branches `(xA, xB)` each of shape `(B, 3, H, W)`, the model outputs `(B, 3, H, W)` logits.

---

## 2. Verified Current-Code Semantics

1. **Dual Branch Input (`xA`, `xB`)**:
   * In `third_party/KuroSiwo/training/change_detection_trainer.py`, the training forward pass is:
     ```python
     output = model(*inputs)
     ```
     where `len(inputs) == 2` is asserted (`assert len(configs['inputs']) == 2`).
2. **Channel Assembly**:
   * In `third_party/KuroSiwo/utilities/utilities.py`:
     ```python
     if (config['task'] == 'cd') or (config['method'] == 'convlstm'):
         config['num_channels'] = len(config['channels'])
         if config['dem']:
             config['num_channels'] += 1
     ```
   * When `config['channels'] == ['vv', 'vh']` (length 2) and `config['dem'] == True`, `num_channels` is `3`.
3. **Branch Concatenation**:
   * In `change_detection_trainer.py` (lines 120–133):
     ```python
     if configs['dem']:
         inputs.append(torch.cat((pre_event, dem), dim=1))
     ```
     Each branch thus receives `[VV, VH, DEM]`.
4. **Filename vs Task**:
   * In `change_detection_trainer.py` (line 318), the best change-detection model is hardcoded to save as `best_segmentation.pt`:
     ```python
     torch.save(..., Path(configs['checkpoint_path']) / 'best_segmentation.pt')
     ```
     Thus, the filename `best_segmentation.pt` in Kuro Siwo does NOT imply a single-image semantic segmentation model; it is the standard checkpoint name used for change-detection models as well.

---

## 3. Checkpoint-Era Configuration Evidence

* **Release Date in Repository**:
  * In Git commit `043a36abc03b4f626f73f3a5e915abc0fc6379da` (Date: `Thu Jan 4 15:38:05 2024 +0200`, message: *"add checkpoints"*), the Dropbox download link for SNUNet (`best_segmentation.pt`) was added to `README.md`.
* **Configuration State at Commit `043a36a`**:
  * `configs/train/data_config.json` contained:
    ```json
    {
      "track": "RandomEvents",
      "train_pickle": "pickle/grid_dict.pkl",
      "test_pickle": "pickle/grid_dict.pkl",
      "inputs": ["pre_event_1", "pre_event_2", "post_event"],
      "channels": ["vv", "vh"],
      "clamp_input": 0.15,
      "scale_input": "normalize",
      "data_mean": [0.0953, 0.0264],
      "data_std": [0.0427, 0.0215],
      "dem_mean": 67.0293,
      "dem_std": 1765.0062,
      "dem": false,
      "slope": false
    }
    ```
* **Contradictory Duplicate Keys in Current Code**:
  * In `configs/train/data_config.json` (as of commit `1663f3a`, March 2025), duplicate keys exist:
    * Line 24: `"dem": true`
    * Line 30: `"dem": false`
  * Standard JSON parsers overwrite line 24 with line 30, meaning `"dem"` evaluates to `false` by default unless overridden via the CLI flag `--dem=True`.

---

## 4. Third-Channel Evidence

* **Candidate A: DEM (`[VV, VH, DEM]`)** — *Strong Documentary Evidence (`inferred`)*:
  1. The Kuro Siwo paper (Bountos et al., NeurIPS 2024) specifically evaluates change-detection models with and without auxiliary DEM.
  2. The code explicitly implements DEM concatenation in `change_detection_trainer.py` when `--dem=True`, giving 3 channels per branch (`[VV, VH, DEM]`).
  3. `README.md` provides the exact usage example: `python main.py ... --dem=True`.
* **Candidate B: Polarimetric Ratio (`[VV, VH, VH/VV]`)** — *Weak Evidence*:
  1. `dataset/Dataset.py` supports `"vh/vv"` in `self.configs["channels"]`.
  2. However, in `data_config.json`, `channels` is explicitly `["vv", "vh"]` throughout git history.
  3. No script or config in the repository sets `channels` to `["vv", "vh", "vh/vv"]` for SNUNet.
* **Provisional Scientific Decision**:
  * The third channel is provisionally inferred as **DEM**, but strictly tagged as `inferred`, NOT `verified`. First-layer weights must not be treated as modality proof.

---

## 5. Normalization Evidence

* **SAR Normalization**:
  * Clamping: `torch.clamp(image, min=0.0, max=0.15)` followed by `torch.nan_to_num(image, 0.15)`.
  * Normalization: `Normalize(mean=[0.0953, 0.0264], std=[0.0427, 0.0215])`.
  * Status: **VERIFIED** across all repository commits.
* **DEM Normalization**:
  * Checkpoint-era (Jan 2024): `dem_mean = 67.0293`, `dem_std = 1765.0062`.
  * KuroSiwo v2 (Mar 2025): `dem_mean = 93.4313`, `dem_std = 1410.8382`.
  * Status: **INFERRED** (historical v1 stats `67.0293 / 1765.0062` correspond to the checkpoint creation date, but exact train-time normalization is not recorded in the checkpoint).

---

## 6. Temporal-Pairing Evidence

* The manifest exposes two pre-event images: `pre_event_1` (`SL1_I*`) and `pre_event_2` (`SL2_I*`), plus `post_event` (`MS1_I*`).
* In `change_detection_trainer.py`, the loop requires exactly two inputs.
* When `configs['inputs']` is configured for change detection, it takes `[pre_event_n, post_event]`.
* Standard Sentinel-1 acquisition ordering typically designates `pre_event_2` as the acquisition closest in time prior to the flood event, while `pre_event_1` is an earlier reference.
* Status: **INFERRED**. The canonical evaluation pair is provisionally designated as `(pre2, post)` with `(pre1, post)` documented as the alternative reference.

---

## 7. Class and Ignore-Label Semantics

* Output Channels (`conv_final`): 3 channels.
  * Channel 0: No Water
  * Channel 1: Permanent Water
  * Channel 2: Flood Water
* Label 3 in Ground Truth (`MK0_MLU`): Invalid / No Data.
* In `utilities/utilities.py`:
  * `nn.CrossEntropyLoss(..., ignore_index=3)`
  * `Accuracy(num_classes=4, ignore_index=3)`
  * `JaccardIndex(num_classes=4, ignore_index=3)`
* Double-Masking Requirement:
  * Audit revealed 91,899 pixels where `valid_mask == 1` while `label == 3`.
  * The valid pixel filter must strictly be: `valid = (valid_mask == 1) & (label != 3)`.
* Status: **VERIFIED**.

---

## 8. Weighted-Loss Evidence

* In `utilities/utilities.py`:
  * If `"weighted": true`, class weights are `[0.3716, 14.0098, 8.2041]`.
  * If `"weighted": false`, class weights are `[1.0, 1.0, 1.0]`.
* In `configs/train/train_config.json`:
  * `"weighted": false`
* The checkpoint file records `loss: 0.14699`. Whether the authors trained with weighted loss is not preserved in the state dict.
* Status: **UNKNOWN**.

---

## 9. Remaining Unknowns

1. Whether the checkpoint used DEM v1 stats (`67.0293 / 1765.0062`) or an unnormalized DEM during training.
2. Whether `pre1` or `pre2` was the canonical training branch for Siamese input `xA`.
3. Whether random flipping or data augmentations were active during checkpoint training.
4. Exact training split event list at the time of Jan 2024 training (the v1 pickle `grid_dict.pkl` vs v2 `KuroV2_grid_dict.gz`).

---

## 10. Current Provisional Assumption

* **Input A**: `[pre_event_2_VV, pre_event_2_VH, DEM]`
* **Input B**: `[post_event_VV, post_event_VH, DEM]`
* **SAR Normalization**: Clamp to `[0.0, 0.15]`, normalize with `mean=[0.0953, 0.0264]`, `std=[0.0427, 0.0215]`.
* **DEM Normalization**: Checkpoint-era stats `mean=67.0293`, `std=1765.0062`.
* **Output Decoding**: `argmax(dim=1)` over logits; binary flood is `pred == 2`.
* **Evaluation Mask**: `valid = (valid_mask == 1) & (label != 3)`.
* **Status**: Clearly labeled as **PROVISIONAL / INFERRED**.

---

## 11. Why the Provisional Assumption Is Not an Exact Official Reproduction

* The official Kuro Siwo training script was not version-locked with run-specific config snapshots.
* Contradictory keys exist in the configuration files (`"dem": true` and `"dem": false`).
* Checkpoint metadata does not store input argument flags or preprocessing pipelines.
* Claiming "exact reproduction" would violate scientific truthfulness; the setup represents our best evidence-grounded inference.

---

## 12. Conditions Requiring Training a New Documented Source Model

A new source model must be trained under fully documented, controlled conditions if:
1. The frozen official checkpoint exhibits catastrophic prediction failure (e.g. 0% Recall or collapsing to all-background across validation events).
2. The model exhibits severe sensitivity where slight changes in DEM scaling drastically alter predictions.
3. Reviewers or benchmarks demand 100% reproducible training provenance from raw Sentinel-1 SLC/GRD to final checkpoint.

---

## 13. Official-Repository Commit Hash

* Current repository HEAD of `third_party/KuroSiwo`:
  `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`

---

## 14. Canonical Baseline Distinction: Raw Baseline vs Corrected Baseline

To ensure scientific reproducibility and transparency, we explicitly establish and distinguish two baseline regimes:

1. **Official-Compatible Raw Baseline (`outputs/day2/`, `outputs/day2_corrected/raw_full_val_event_metrics.csv`)**:
   - Reproduces the available official-style loader behavior, including its unhandled `-32768` integer NoData values that normalize into extreme negative inputs ($\sim -18.6$).
   - Stored in `outputs/day2/val_event_metrics.csv`, `outputs/day2/val_summary.json`, and `outputs/day2_corrected/raw_full_val_event_metrics.csv`.
   - Results: Macro F1 = 0.5237, Macro IoU = 0.4110, Pooled F1 = 0.6859.
   - Preserved as a historical reference to document the effect of the upstream data loader anomaly.

2. **Corrected Canonical Baseline (`outputs/day2_corrected/corrected_full_val_event_metrics.csv`)**:
   - Implements explicit DEM NoData sanitization (`kuro_dem_sanitize_v1`).
   - Replaces non-finite values (NaN, +Inf, -Inf) and raw DEM values $< -1000.0\text{ m}$ (including $-32768$) with the training mean elevation ($\mu_{DEM} = 67.0293\text{ m}$), mapping to normalized value `0.0` (neutral standardized prior).
   - Preserves all valid physical elevations without clipping (including below-sea-level depressions like -430 m and high elevations up to 1800 m+).
   - SAR channels, labels, and evaluation masks remain identical.
   - Stored in `outputs/day2_corrected/corrected_full_val_event_metrics.csv` and `outputs/day2_corrected/summary.json`.
   - Results: Macro F1 = 0.5646 (+0.0409), Macro IoU = 0.4385 (+0.0275), Pooled F1 = 0.7020 (+0.0161).
   - **All subsequent Test-Time Adaptation (TTA) methods must evaluate against this corrected canonical baseline.**

---

## 15. Evaluation Completeness and Sample Traceability Guarantee

To eliminate sample omission, duplication, or accidental subsetting:
1. **Automated Completeness Tracker**: Every evaluation run now executes through `CompletenessTracker` (`src/evaluation/completeness.py`), verifying every sample ID against the filtered event manifest.
2. **Audit Verification (`outputs/day2_corrected/evaluation_completeness.json`)**:
   - Total expected samples: 2,826
   - Total processed samples: 2,826
   - Event 279: 2,381 expected / 2,381 processed (0 missing, 0 duplicates)
   - Event 437: 169 expected / 169 processed (0 missing, 0 duplicates)
   - Event 520: 92 expected / 92 processed (0 missing, 0 duplicates)
   - Event 559: 42 expected / 42 processed (0 missing, 0 duplicates)
   - Event 1111003: 19 expected / 19 processed (0 missing, 0 duplicates)
   - Event 1111008: 123 expected / 123 processed (0 missing, 0 duplicates)
   - Status: `PASS` across both raw and corrected baseline runs.
3. **Discrepancy Provenance Note**: In early draft notes, pixel counts for Event 279 (`valid_pixels: 4,906,255`) were transcribed from a 100-sample diagnostic slice generated in `tools/audit_day2_deep.py:202` for computing fast preprocessing statistics (`outputs/day2_audit/preprocessing_statistics.csv`). The actual baseline evaluation (`tools/eval_val_baseline.py` and `outputs/day2/val_event_metrics.csv`) had executed all 2,381 samples (`valid_pixels: 116,199,247`). Both the raw baseline and corrected baseline have now been fully and fair-mindedly verified on the full 2,826 validation samples.



