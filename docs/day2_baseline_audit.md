# ChangeGuard: Day 2 Baseline Audit and Root Cause Report

**Document version:** 2.0 (Calibrated & Supplemental Audit)  
**Project:** ChangeGuard (Safe Test-Time Adaptation for SAR Flood Change Detection)  
**Date:** 24 September 2026  
**Audited Checkpoint:** `ChangeGuard/checkpoints/kuro_snunet_official.pt`  
**Evaluation Scope:** Event-Disjoint Kuro Siwo Validation Split (6 present events: 279, 437, 520, 559, 1111003, 1111008)  
**Test Set Status:** Strictly sealed (10 test events untouched)  

---

## 1. Active Environment and Verification

The audit and evaluation pipeline were executed within the verified project environment:

| Property | Value / Status |
| :--- | :--- |
| **Current Working Directory** | `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard` |
| **Git Repository Root** | `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard` |
| **Git Branch & Commit** | `main` @ `c895b66` (`chore: establish audited ChangeGuard frozen baseline`) |
| **Working Tree Status** | Clean (supplemental audit tools and documentation staged for review) |
| **Python Executable** | `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` (Python 3.10.20) |
| **PyTorch & CUDA** | PyTorch 2.13.0+cu130, CUDA available: `True`, Device Count: 4 |
| **Primary GPU** | `NVIDIA GeForce RTX 3090` (GPU 0) |
| **Metric Unit Tests** | `tests/test_metrics.py` executed: **All 6/6 unit tests passed** |

### Unit Test Verification Summary
All unit tests in `ChangeGuard/tests/test_metrics.py` passed cleanly:
1. `test_perfect_flood_prediction`: Precision=1.0, Recall=1.0, F1=1.0, IoU=1.0.
2. `test_all_background_prediction`: Precision=0.0, Recall=0.0, F1=0.0, IoU=0.0.
3. `test_all_flood_prediction`: Precision matches prevalence, Recall=1.0.
4. `test_no_ground_truth_flood`: When no flood is present and none predicted, metrics correctly resolve to 1.0 (true negative agreement).
5. `test_class3_marked_valid_exclusion`: Pixels with `valid_mask == 1` and `label == 3` are strictly excluded from confusion counts.
6. `test_permanent_water_exclusion_from_flood_target`: Permanent water (class 1) is strictly a non-flood negative class.

---

## 2. Reproduced Frozen Baseline Results

The frozen source baseline was evaluated using the official checkpoint `kuro_snunet_official.pt` across all 6 available validation events (2,826 samples total).

### 2.1 Verified Full Raw Baseline Performance Table (All 2,826 Samples)

Every single sample in the validation split was evaluated under the `official-compatible raw baseline` (canonical `pre_event_2`, official DEM normalization without NoData sanitation, double mask applied, frozen weights). Verified via `CompletenessTracker` (0 duplicate IDs, 0 missing IDs, 2,826/2,826 samples).

| Event ID | Samples | Valid Pixels | GT Flood Pixels | Prevalence | Pred Flood Pixels | TP | FP | FN | TN | Precision | Recall | Flood F1 | Flood IoU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 2,381 | 116,199,247 | 868,697 | 0.75% | 1,033,388 | 572,221 | 461,167 | 296,476 | 114,869,383 | 0.5537 | 0.6587 | **0.6017** | 0.4303 |
| **437** | 169 | 7,859,430 | 64,002 | 0.81% | 60,077 | 46,413 | 13,664 | 17,589 | 7,781,764 | 0.7726 | 0.7252 | **0.7481** | 0.5976 |
| **520** | 92 | 3,455,248 | 329,243 | 9.53% | 282,996 | 234,477 | 48,519 | 94,766 | 3,077,486 | 0.8286 | 0.7122 | **0.7660** | 0.6207 |
| **559** | 42 | 1,843,520 | 50,672 | 2.75% | 2,191 | 2,158 | 33 | 48,514 | 1,792,815 | **0.9849** | **0.0426** | **0.0816** | **0.0426** |
| **1111003** | 19 | 815,490 | 6,961 | 0.85% | 4,091 | 610 | 3,481 | 6,351 | 805,048 | **0.1491** | **0.0876** | **0.1104** | **0.0584** |
| **1111008** | 123 | 5,686,145 | 480,301 | 8.45% | 472,183 | 397,520 | 74,663 | 82,781 | 5,131,181 | 0.8419 | 0.8276 | **0.8347** | 0.7163 |

### 2.2 Aggregate Summary (Full Raw Baseline)
- **Macro-averaged Precision**: **0.6885**
- **Macro-averaged Recall**: **0.5090**
- **Macro-averaged F1**: **0.5237**
- **Macro-averaged IoU**: **0.4110**
- **Median F1**: **0.6749**
- **Pooled (Micro-averaged) F1**: **0.6859** (Precision: 0.6757, Recall: 0.6964, IoU: 0.5219; Total Valid: 135,859,080, TP: 1,253,399, FP: 601,527, FN: 546,477)
- **Validation Event 514**: Present in official Kuro Siwo `val_acts` config, but completely absent from the downloaded Kuro Siwo event directories. Recorded and excluded from evaluation.
- **Sealed Test Events**: 10 events (`205`, `277`, `321`, `411`, `445`, `561`, `562`, `1111002`, `1111007`, `1111013`) remain strictly sealed and untouched.

### 2.3 Provenance Note: Resolution of Event 279 Historical Transcription Discrepancy
In early drafts of this audit document, the confusion pixel counts for Event 279 were reported as `valid_pixels = 4,906,255`, `gt_flood = 61,893`, `pred_flood = 44,383`, `tp = 34,746`, `fp = 9,637`, `fn = 27,147`, `tn = 4,834,725`, alongside the full event sample count `2,381` and the full F1 `0.6017`.
- **Exact Root Cause**: In `tools/audit_day2_deep.py:202`:
  ```python
  eval_indices = event_indices if len(event_indices) <= 200 else event_indices[:100]
  ```
  The deep audit tool subsampled Event 279 to its first 100 samples when extracting summary statistics into `outputs/day2_audit/preprocessing_statistics.csv` to save time. Those 100-sample pixel counts were accidentally transcribed into the audit table instead of the full run counts.
- **Verification**: The actual raw evaluation script `tools/eval_val_baseline.py` had evaluated all 2,381 samples without subsampling, correctly producing `outputs/day2/val_event_metrics.csv` (`valid_pixels: 116,199,247`, `gt_flood: 868,697`, `pred_flood: 1,033,388`, `tp: 572,221`, `fp: 461,167`, `fn: 296,476`, `tn: 114,869,383`). Note that $572,221 / (572,221 + 0.5 \times (461,167 + 296,476)) = 0.601678$, yielding the exact reported F1 of `0.6017` and IoU of `0.4303`.
- **Completeness Enforcement**: Moving forward, the `CompletenessTracker` rigorously validates all evaluation runs and asserts that all 2,826 samples are processed with zero duplicates and zero missing samples.

---

## 3. Implementation and Data Path Audit

### 3.1 Sample Tracing and Pure NumPy Hand-Checks
Raw rasters were loaded directly from disk using `rasterio` and passed through the model. Hand-computed NumPy confusion counts (`TP`, `FP`, `FN`, `TN`) matched the evaluation pipeline counts **identically with 0 discrepancy** across traced samples in events 559, 1111003, and 1111008.

### 3.2 Double-Masking Logic
* `MK0_MNA` is the binary valid mask (1: valid, 0: invalid/margin).
* `MK0_MLU` has class 3 as `Invalid/Ignore`.
* **Audit Finding**: In Kuro Siwo, **91,899 pixels** have `valid_mask == 1` while simultaneously having `label == 3`. Both our dataset loader and metric computation implement the strict double-mask rule: `valid = (valid_mask == 1) & (label != 3)`.

### 3.3 Clarification of the "Zero-DEM" Intervention
In the initial diagnostic script `tools/audit_day2_deep.py`, the "zero-DEM" ablation set the normalized DEM tensor slice to zero:
```python
xA[:, 2, :, :] = 0.0
xB[:, 2, :, :] = 0.0
```
Because DEM normalization is defined as:
$$\text{DEM}_{norm} = \frac{\text{DEM}_{raw} - \mu_{DEM}}{\sigma_{DEM}}$$
setting the normalized tensor slice to `0.0` corresponds to setting elevation to the **training-mean elevation** ($\mu_{DEM} \approx 67.03\text{ m}$ in v1, or $93.43\text{ m}$ in v2), **not literal 0 meters elevation above sea level**. This diagnostic is properly described as a **"training-mean / neutral normalized DEM"** intervention.

---

## 4. Full Training-Event Elevation Distribution Audit

To verify the elevation characteristics of the training data, an exhaustive census was conducted across **all 48,768 DEM rasters** in all 27 training events and 6 validation events. Known NoData sentinels (`NaN` and values $< -1000\text{ m}$) were excluded when computing valid elevation statistics.

The full event-wise table is preserved in:
[`outputs/day2_audit/train_event_dem_statistics.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/day2_audit/train_event_dem_statistics.csv)

### Selected Event-Wise Elevation Statistics

| Split | Event ID | Rasters Examined | Valid Min (m) | 1st %ile (m) | Median (m) | Mean (m) | 99th %ile (m) | Valid Max (m) | % NaN | % -32768 | % < -1000 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **train** | 130 | 373 | -2.6 | 2.0 | 6.8 | 24.5 | 145.4 | 206.0 | 4.98% | 0.00% | 0.00% |
| **train** | 470 | 821 | 95.1 | 106.3 | 134.7 | 135.4 | 173.2 | 193.0 | 0.05% | 0.00% | 0.00% |
| **train** | 555 | 2,874 | 0.1 | 154.2 | 425.9 | 462.9 | 1,101.1 | **1,469.5** | 0.00% | 0.00% | 0.00% |
| **train** | 118 | 342 | 207.0 | 216.0 | 533.5 | 504.7 | 1,074.6 | **1,268.7** | 3.44% | 0.00% | 0.00% |
| **train** | 174 | 120 | 218.1 | 222.9 | 278.2 | 348.6 | 926.9 | **1,050.7** | 1.70% | 0.00% | 0.00% |
| **train** | 324 | 548 | -989.3 | 2.4 | 80.7 | 105.2 | 412.3 | 594.0 | 4.52% | 0.03% | 0.05% |
| **train** | 427 | 5,283 | -997.7 | 6.5 | 117.6 | 124.2 | 299.8 | 352.9 | 0.01% | 0.00% | 0.00% |
| **train** | 1111011 | 3,308 | -999.9 | 1.9 | 99.0 | 211.2 | 1,288.1 | **1,831.9** | 0.01% | 0.18% | 0.25% |
| **train** | 1111009 | 18,660 | -42.0 | 19.5 | 56.3 | 93.1 | 706.4 | **1,678.2** | 0.00% | 0.00% | 0.00% |
| **train** | 1111005 | 1,369 | 702.0 | 752.0 | **858.8** | **895.1** | 1,330.6 | **1,576.8** | 0.03% | 0.00% | 0.00% |
| **val** | 520 | 92 | 3.3 | 10.3 | 22.5 | 22.7 | 39.4 | 58.3 | 1.00% | 0.00% | 0.00% |
| **val** | 559 | 42 | 882.2 | 889.4 | **922.8** | **930.7** | 1,038.5 | **1,093.1** | 3.51% | 0.00% | 0.00% |
| **val** | 1111003 | 19 | -999.5 | -0.6 | 13.1 | 22.4 | 81.2 | 114.5 | 0.00% | **5.88%** | **7.35%** |
| **val** | 1111008 | 123 | -997.1 | 0.7 | 10.7 | 12.6 | 56.6 | 76.0 | 0.52% | **1.06%** | **1.32%** |

### Answers to Distribution Questions
1. **Is the statement that training events are between 0 m and 65 m correct?**
   **No. That statement was incorrect.** While many training events are coastal or low-elevation floodplains (median $< 50\text{ m}$), the training split contains significant high-elevation coverage.
2. **Is Event 559 outside the entire training elevation range?**
   **No.** Event 559 (range: $882.2\text{ m}$ to $1,093.1\text{ m}$, mean: $930.7\text{ m}$) falls within the elevation envelope of several training events:
   - Event 1111011 reaches up to **1,831.9 m** (99th %ile: 1,288.1 m).
   - Event 1111009 reaches up to **1,678.2 m**.
   - Event 1111005 is predominantly a high-altitude plateau with a median elevation of **858.8 m**, mean of **895.1 m**, and maximum of **1,576.8 m**.
   - Event 555 reaches up to **1,469.5 m** (99th %ile: 1,101.1 m).
   - Event 118 reaches up to **1,268.7 m** (99th %ile: 1,074.6 m).
3. **Is Event 559 outside most training events but overlapping some of them?**
   **Yes.** Most training events have median elevations below 150 m. However, Event 559 overlaps substantially with high-elevation training events, especially Event 1111005 ($702\text{ m} - 1,576\text{ m}$) and the upper bounds of Events 555, 118, and 1111011.

---

## 5. Elevation Counterfactual Sweep for Event 559

To test whether the model's prediction on Event 559 is causally sensitive to the DEM feature input, we conducted a controlled causal probe across the 25 Event 559 samples containing ground-truth flood. The SAR pre/post pairs were held strictly fixed while only the DEM channel was set to constant elevations $E \in \{0, 50, 100, 250, 500, 750, 1000\}\text{ m}$, normalized using $\text{DEM}_{norm} = (E - 67.0293) / 1765.0062$.

The exact results are preserved in:
[`outputs/day2_audit/event559_elevation_sweep.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/day2_audit/event559_elevation_sweep.csv)

### Sweep Results (Fixed SAR Observations, Varying DEM Elevation)

| Constant Elevation | Normalized Input | GT Flood Pixels | Mean Flood Prob on GT | Predicted Flood Pixels | TP | FP | FN | Precision | Recall | Flood F1 | Flood IoU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 m** | -0.0380 | 50,672 | **0.5286** | 29,650 | 27,090 | 2,560 | 23,582 | 0.9137 | **0.5346** | **0.6745** | 0.5089 |
| **50 m** | -0.0096 | 50,672 | **0.5063** | 28,058 | 25,934 | 2,124 | 24,738 | 0.9243 | **0.5118** | **0.6588** | 0.4912 |
| **100 m** | +0.0187 | 50,672 | **0.4812** | 26,374 | 24,651 | 1,723 | 26,021 | 0.9347 | **0.4865** | **0.6399** | 0.4705 |
| **250 m** | +0.1037 | 50,672 | **0.4017** | 20,962 | 20,113 | 849 | 30,559 | 0.9595 | **0.3969** | **0.5615** | 0.3904 |
| **500 m** | +0.2453 | 50,672 | **0.2563** | 12,077 | 11,837 | 240 | 38,835 | 0.9801 | **0.2336** | **0.3773** | 0.2325 |
| **750 m** | +0.3870 | 50,672 | **0.1446** | 4,401 | 4,331 | 70 | 46,341 | 0.9841 | **0.0855** | **0.1573** | 0.0854 |
| **1000 m** | +0.5286 | 50,672 | **0.0865** | 1,526 | 1,500 | 26 | 49,172 | 0.9830 | **0.0296** | **0.0575** | 0.0296 |

### Findings from the Counterfactual Probe
* **Strict Monotonicity**: As elevation increases from $0\text{ m}$ to $1,000\text{ m}$, the model's output demonstrates a strictly monotonic decline in:
  - Mean flood probability inside ground-truth flood pixels: **$0.5286 \rightarrow 0.0865$**
  - Predicted flood pixels: **$29,650 \rightarrow 1,526$**
  - Flood Recall: **$53.46\% \rightarrow 2.96\%$**
  - Flood F1: **$0.6745 \rightarrow 0.0575$**
* **Calibrated Interpretation**: This causal probe confirms that holding SAR inputs constant while providing higher DEM values monotonically suppresses flood prediction. Because Event 559 lies at ~930 m elevation, this suppression explains the low baseline recall (0.0426). This is strong empirical evidence of elevation-conditioned prediction suppression on this scene. It is a diagnostic causal probe, not a proposed baseline.

---

## 6. DEM NoData Semantics and Policy Evaluation (Event 1111003)

### 6.1 Official Loader and Raster Metadata Investigation
1. **Raster-Declared Metadata**: The GeoTIFF files declare `nodata: nan` (`float32`).
2. **Origin of `-32768.0`**: The underlying DEM sources (Copernicus DEM / SRTM) originally used 16-bit signed integer format with `-32768` as NoData. When cast to `float32`, the values remained `-32768.0`.
3. **Official Code Behavior**:
   - In `Dataset.py:733-739`, the code checked `dem.isnull()`, which only detects IEEE NaNs, leaving `-32768.0` unhandled.
   - In `Dataset.py:1139-1149`, the authors noted: *"Nodata values are not NaN but a rather high float number... convert nodata to NaN and interpolate"*. However, because `dem.rio.nodata` is `nan`, the comparison `dem == nodata` evaluates to false for IEEE floats.
4. **Valid Mask Disconnect**: In Event 1111003, `valid_mask` (`MK0_MNA`) was generated from SAR orbital margins, not DEM coverage. Consequently, **22,163 pixels** with `DEM < -1000` fall inside `valid_mask == 1`.

### 6.2 Validation Evaluation of NoData Policies
We evaluated three diagnostic policies across all 6 validation events:
- **Policy 1 (Raw)**: Existing behavior (sentinels $<-1000$ untouched, normalized to $\sim -18.6$).
- **Policy 2 (Replace with Mean)**: Sentinel values ($< -1000\text{ m}$ or `NaN`) replaced with training mean elevation ($\mu_{DEM}$), resulting in a neutral normalized DEM input of `0.0`.
- **Policy 3 (Exclude from Evaluation)**: Excluding DEM-invalid pixels from the evaluation mask.

The full results are preserved in:
[`outputs/day2_audit/dem_nodata_policy_evaluation.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/day2_audit/dem_nodata_policy_evaluation.csv)

### Policy Comparison Table

| Policy | Event ID | Affected Pixels in Eval | % of Eval Area | Precision | Recall | Flood F1 | Flood IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy 1: Raw** | 279 | 50,253 | 0.04% | 0.5537 | 0.6587 | 0.6017 | 0.4303 |
| | 437 | 0 | 0.00% | 0.7726 | 0.7252 | 0.7481 | 0.5976 |
| | 520 | 0 | 0.00% | 0.8286 | 0.7122 | 0.7660 | 0.6207 |
| | 559 | 0 | 0.00% | 0.9849 | 0.0426 | 0.0816 | 0.0425 |
| | **1111003** | **22,163** | **2.72%** | **0.1490** | **0.0876** | **0.1103** | **0.0584** |
| | 1111008 | 5,223 | 0.09% | 0.8419 | 0.8276 | 0.8347 | 0.7163 |
| **Policy 2: Replace with Mean** | 279 | 50,253 | 0.04% | 0.6039 | 0.6587 | **0.6301** | 0.4600 |
| | 437 | 0 | 0.00% | 0.7726 | 0.7252 | 0.7481 | 0.5976 |
| | 520 | 0 | 0.00% | 0.8286 | 0.7122 | 0.7660 | 0.6207 |
| | 559 | 0 | 0.00% | 0.9849 | 0.0426 | 0.0816 | 0.0425 |
| | **1111003** | **22,163** | **2.72%** | **0.2761** | **0.4070** | **0.3290** | **0.1969** |
| | 1111008 | 5,223 | 0.09% | 0.8361 | 0.8290 | 0.8326 | 0.7131 |
| **Policy 3: Exclude Invalid** | 279 | 50,253 | 0.04% | 0.5537 | 0.6587 | 0.6017 | 0.4303 |
| | 437 | 0 | 0.00% | 0.7726 | 0.7252 | 0.7481 | 0.5976 |
| | 520 | 0 | 0.00% | 0.8286 | 0.7122 | 0.7660 | 0.6207 |
| | 559 | 0 | 0.00% | 0.9849 | 0.0426 | 0.0816 | 0.0425 |
| | **1111003** | **22,163** | **2.72%** | **0.1490** | **0.0876** | **0.1103** | **0.0584** |
| | 1111008 | 5,223 | 0.09% | 0.8419 | 0.8276 | 0.8347 | 0.7163 |

### Recommended NoData Policy and Justification
* **Recommendation: Policy 2 (Replace sentinels $< -1000\text{ m}$ and `NaN` with training DEM mean, normalized to `0.0`)**.
* **Justification**:
  1. Sentinel values like $-32,768\text{ m}$ are non-physical fill flags that distort convolution activations when normalized into extreme negative space ($-18.6$).
  2. Imputing with the dataset mean elevation provides an uninformative / neutral prior ($0.0$ in standardized space), preventing extreme gradient or activation spikes while letting SAR channels drive prediction.
  3. Policy 3 (simply excluding pixels at evaluation) leaves the model's convolutional receptive fields corrupted by the raw negative spikes, failing to address the artifact.
  4. The SAR data and ground-truth flood labels in these areas are scientifically valid; dropping them from evaluation is neither dataset-semantically nor methodologically necessary.

---

## 7. Corrected Failure Classifications

Using the originally specified category definitions:
- **Category A**: Genuine frozen-model cross-event failure
- **Category B**: Likely preprocessing or configuration mismatch
- **Category C**: Dataset or label anomaly
- **Category D**: Mixed or unresolved

### Event 559: Provisional Category A (Genuine Frozen-Model Cross-Event Failure)
* **Justification**:
  1. Implementation, data loader, SAR preprocessing, and label mapping are verified.
  2. The elevation counterfactual sweep confirms that the frozen model exhibits strong elevation-conditioned prediction suppression on this scene.
  3. The training census confirms that high elevations ($> 1,000\text{ m}$) do exist in training (e.g. Events 555, 118, 1111005). The failure represents a genuine cross-event domain generalization issue of the frozen model on this event.
  4. It is classified as *provisional Category A* pending confirmation that the official training checkpoint was indeed trained with identical DEM normalization parameters.

### Event 1111003: Category D (Mixed / Unresolved)
* **Justification**:
  1. **Data/preprocessing artifact**: The DEM GeoTIFF contains unmasked $-32,768\text{ m}$ NoData values affecting 2.72% of the evaluation area. Replacing these with the neutral training mean increases F1 from $0.1103$ to $0.3290$.
  2. **Model/radiometric failure**: Even when NoData is neutralized, F1 remains low ($0.3290$, Precision $0.2761$) due to low flood prevalence ($0.85\%$) and severe confusion between dry bare earth and water (60,170 pixels of dry land misclassified as permanent water).
  3. Because data quality artifacts and genuine model limitations coexist, it is properly classified as Category D.

---

## 8. Precise GO / NO-GO Decisions

To ensure scientific rigor, we distinguish the project decision into four clear gates:

1. **Preprocessing Policy: GO**
   - Adopt Policy 2 (sanitizing DEM values $< -1000\text{ m}$ and `NaN` to the training mean, i.e., neutral normalized value `0.0`) across all pipelines and freeze preprocessing.
2. **Generic TTA Baselines: GO**
   - Proceed with running standard unsupervised test-time adaptation baselines (Tent updating batch normalization affine parameters, pseudo-labeling) on validation events to measure whether they improve or harm performance.
3. **Claiming Harmful Adaptation: NO-GO**
   - Harmful adaptation remains an **untested empirical hypothesis**. We make no claims that Tent fails catastrophically or erases flood changes until Tent has actually been implemented and evaluated on validation events.
4. **Implementing ChangeGuard Update Controller: NO-GO**
   - The proposed update controller and change-preservation probes must **not** be implemented or claimed until at least one generic TTA method produces measurable, documented event-wise harm on the validation set.

---

## 9. Staged Files and Git Status

### Committed Audit Files
- `ChangeGuard/docs/day2_baseline_audit.md` (Committed at `99e6ce5`)
- `ChangeGuard/tools/audit_day2_deep.py` (Committed at `99e6ce5`)
- `ChangeGuard/tools/run_day2_supplemental_audit.py` (Committed at `99e6ce5`)

---

## 10. Canonical Baseline Hierarchy: Raw vs Corrected

We maintain an explicit separation between:

1. **Official-Compatible Raw Baseline (`outputs/day2/`, `outputs/day2_corrected/raw_full_val_event_metrics.csv`)**:
   - Matches available official loader behavior; preserves raw unhandled `-32768` values (normalized to $\approx -18.6$).
   - Macro F1: `0.5237`, Macro IoU: `0.4110`, Pooled F1: `0.6859`.
   - Verified over all 2,826 samples via `CompletenessTracker` (`outputs/day2_corrected/evaluation_completeness.json`).

2. **Corrected Canonical Baseline (`outputs/day2_corrected/corrected_full_val_event_metrics.csv`)**:
   - Implements `kuro_dem_sanitize_v1`: raw DEM values $< -1000.0\text{ m}$ and non-finite values are sanitized to normalized `0.0` (training mean elevation).
   - Valid physical elevations (both depressions down to -430 m and elevations up to 1800 m+) are preserved without clipping.
   - SAR observations, ground-truth labels, and validity masks remain untouched.
   - Macro F1: `0.5646` (+0.0409), Macro IoU: `0.4385` (+0.0275), Pooled F1: `0.7020` (+0.0161).
   - Verified over all 2,826 samples via `CompletenessTracker`.
   - **This corrected canonical baseline is the frozen benchmark against which all subsequent test-time adaptation (TTA) algorithms must be compared.**

### Full Event-Wise Corrected Baseline Table (All 2,826 Samples)

| Event ID | Samples | Valid Pixels | GT Flood Pixels | Pred Flood Pixels | TP | FP | FN | TN | Precision | Recall | Flood F1 | Flood IoU | Invalid DEM Pixels |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 2,381 | 116,199,247 | 868,697 | 947,538 | 572,221 | 375,317 | 296,476 | 114,955,233 | 0.6039 | 0.6587 | **0.6301** | 0.4600 | 50,253 |
| **437** | 169 | 7,859,430 | 64,002 | 60,077 | 46,413 | 13,664 | 17,589 | 7,781,764 | 0.7726 | 0.7252 | **0.7481** | 0.5976 | 0 |
| **520** | 92 | 3,455,248 | 329,243 | 282,996 | 234,477 | 48,519 | 94,766 | 3,077,486 | 0.8286 | 0.7122 | **0.7660** | 0.6207 | 0 |
| **559** | 42 | 1,843,520 | 50,672 | 2,191 | 2,158 | 33 | 48,514 | 1,792,815 | **0.9849** | **0.0426** | **0.0816** | **0.0426** | 0 |
| **1111003** | 19 | 815,490 | 6,961 | 10,257 | 2,833 | 7,424 | 4,128 | 801,105 | **0.2762** | **0.4070** | **0.3291** | **0.1969** | 22,163 |
| **1111008** | 123 | 5,686,145 | 480,301 | 476,253 | 398,189 | 78,064 | 82,112 | 5,127,780 | 0.8361 | 0.8290 | **0.8325** | 0.7131 | 5,223 |

### Fair Event-Wise Comparison: Raw vs Corrected

| Event ID | Samples | Invalid DEM | Raw Prec | Corr Prec | $\Delta$ Prec | Raw Rec | Corr Rec | $\Delta$ Rec | Raw F1 | Corr F1 | $\Delta$ F1 | Raw IoU | Corr IoU | $\Delta$ IoU | Raw Pred Area | Corr Pred Area | $\Delta$ Pred Area |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 2,381 | 50,253 | 0.5537 | 0.6039 | +0.0502 | 0.6587 | 0.6587 | +0.0000 | 0.6017 | 0.6301 | **+0.0284** | 0.4303 | 0.4600 | +0.0297 | 1,033,388 | 947,538 | -85,850 (-8.31%) |
| **437** | 169 | 0 | 0.7726 | 0.7726 | 0.0000 | 0.7252 | 0.7252 | 0.0000 | 0.7481 | 0.7481 | **0.0000** | 0.5976 | 0.5976 | 0.0000 | 60,077 | 60,077 | 0 (0.00%) |
| **520** | 92 | 0 | 0.8286 | 0.8286 | 0.0000 | 0.7122 | 0.7122 | 0.0000 | 0.7660 | 0.7660 | **0.0000** | 0.6207 | 0.6207 | 0.0000 | 282,996 | 282,996 | 0 (0.00%) |
| **559** | 42 | 0 | 0.9849 | 0.9849 | 0.0000 | 0.0426 | 0.0426 | 0.0000 | 0.0816 | 0.0816 | **0.0000** | 0.0426 | 0.0426 | 0.0000 | 2,191 | 2,191 | 0 (0.00%) |
| **1111003** | 19 | 22,163 | 0.1491 | 0.2762 | +0.1271 | 0.0876 | 0.4070 | +0.3194 | 0.1104 | 0.3291 | **+0.2187** | 0.0584 | 0.1969 | +0.1385 | 4,091 | 10,257 | +6,166 (+150.72%) |
| **1111008** | 123 | 5,223 | 0.8419 | 0.8361 | -0.0058 | 0.8276 | 0.8290 | +0.0014 | 0.8347 | 0.8325 | **-0.0022** | 0.7163 | 0.7131 | -0.0032 | 472,183 | 476,253 | +4,070 (+0.86%) |
| **Macro** | 2,826 | 77,639 | 0.6885 | 0.7170 | +0.0285 | 0.5090 | 0.5624 | +0.0534 | 0.5237 | 0.5646 | **+0.0409** | 0.4110 | 0.4385 | +0.0275 | 1,854,926 | 1,779,252 | -75,674 (-4.08%) |
| **Pooled**| 2,826 | 77,639 | 0.6757 | 0.7061 | +0.0304 | 0.6964 | 0.6980 | +0.0016 | 0.6859 | 0.7020 | **+0.0161** | 0.5219 | 0.5408 | +0.0189 | 1,854,926 | 1,779,252 | -75,674 (-4.08%) |

### Key Scientific Deductions from Fair Comparison
1. **Event 559 Confirmed as Frozen-Source Model Failure**: Contains 0 invalid DEM pixels. All metrics (Precision: 0.9849, Recall: 0.0426, F1: 0.0816, IoU: 0.0426) remain strictly identical ($\Delta = 0.0000$). Event 559 is a confirmed frozen-source model failure under the evaluated pipeline. Its metrics are unaffected by DEM NoData sanitation, and controlled input intervention demonstrates strong elevation-conditioned flood-prediction suppression. The precise physical and environmental cause remains unresolved.
2. **Event 1111003 Remains Category D (Mixed Quality and Difficult Domain)**: Replacing `-32768` sentinels with neutral mean elevation restores recall from 8.76% to 40.70% and boosts F1 from 0.1104 to 0.3291. However, precision remains low (27.62%) and F1 remains poor due to severe bare arid soil backscatter ambiguity and low flood prevalence (0.85%).
3. **Events 437 and 520 are Unaffected**: Zero invalid DEM pixels; delta is 0.0000 across all metrics.
4. **Event 279 False Positives Reduced**: 50,253 invalid DEM pixels existed in Event 279. Sanitizing them eliminated 85,850 spurious flood predictions, reducing false positives from 461,167 to 375,317, which raised precision from 0.5537 to 0.6039 and boosted F1 from 0.6017 to 0.6301.


