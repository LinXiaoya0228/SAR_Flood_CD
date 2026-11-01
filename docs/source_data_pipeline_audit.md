# Audit of the Current Source-Data Pipeline

**Document Version:** 1.0.0  
**Project:** ChangeGuard FYP (Phase: Week 0 Controlled Source Model Preparation)  
**Date:** 2026-10-06  
**Audited Repositories:**  
- `ChangeGuard` (HEAD `0d8dcec24c8c56609b4e318e317e66f5390c176b`)
- `third_party/KuroSiwo` (commit `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`)

---

## 1. Executive Summary

This audit establishes the ground-truth technical state of the KuroSiwo data loading, preprocessing, normalization, and channel assembly pipeline across both the internal `ChangeGuard` codebase and the upstream `third_party/KuroSiwo` reference implementation. Every design dimension is classified according to our strict evidence taxonomy:
- `VERIFIED_FROM_CODE`: Directly documented and verified from explicit lines of executable code.
- `VERIFIED_FROM_DATA`: Confirmed by programmatic inspection of on-disk rasters or checkpoints.
- `SUPPORTED_BY_OFFICIAL_DOCUMENTATION`: Documented in peer-reviewed papers or official READMEs.
- `CURRENT_PROJECT_DECISION`: Explicitly chosen protocol convention established within ChangeGuard.
- `UNRESOLVED`: Conflicting, missing, or under-specified details requiring formal project decisions.

---

## 2. Dimension-by-Dimension Technical Audit

### 2.1 Temporal Pairing Resolution (`pre_event_1`, `pre_event_2`, `post_event`)
- **Upstream KuroSiwo (`third_party/KuroSiwo/dataset/Dataset.py:770–777`):**
  Locates files by prefix: `SL1_` (secondary 1 / pre 1), `SL2_` (secondary 2 / pre 2), and `MS1_` (master 1 / post). In `change_detection_trainer.py:19`, asserts `len(configs['inputs']) == 2`.
- **ChangeGuard (`src/data/kuro_dataset.py:57–62, 227–234`):**
  Loads `SL1_IVV`, `SL1_IVH`, `SL2_IVV`, `SL2_IVH`, `MS1_IVV`, `MS1_IVH`. The canonical baseline pair uses `canonical_preevent = "pre_event_2"` (`xA = pre2_3ch`, `xB = post_3ch`).
- **Classification:** `VERIFIED_FROM_CODE`

### 2.2 SAR Polarizations (VV and VH) Loading
- **Upstream KuroSiwo (`Dataset.py:716–729`):**
  Loads TIFF files using `rioxarray.open_rasterio()`.
- **ChangeGuard (`src/data/kuro_dataset.py:140–160`):**
  Reads band 1 via `rasterio.open().read(1)` as `float32`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.3 Radiometric Representation
- **Physical Representation:**
  Sentinel-1 GRD linear amplitude backscatter (square root of intensity). Raw pixel values span $[0.0, \sim 0.8]$, with over 95% of pixels falling within $[0.0, 0.15]$. Values are NOT in decibels ($dB$) and NOT calibrated backscatter power ($\sigma^0$).
- **Classification:** `SUPPORTED_BY_OFFICIAL_DOCUMENTATION` (Bountos et al., NeurIPS 2024) / `VERIFIED_FROM_DATA`

### 2.4 Clamping
- **Upstream KuroSiwo (`configs/train/data_config.json:14`, `Dataset.py:969`):**
  `"clamp_input": 0.15`. Code applies `torch.clamp(image, min=0.0, max=0.15)`.
- **ChangeGuard (`src/data/kuro_dataset.py:196`):**
  `np.clip(sar, 0.0, self.sar_clamp_max)` with `sar_clamp_max = 0.15`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.5 Normalization
- **Upstream KuroSiwo (`configs/train/data_config.json:16–17`, `Dataset.py:972`):**
  `"data_mean": [0.0953, 0.0264]`, `"data_std": [0.0427, 0.0215]`. Normalized via `torchvision.transforms.Normalize()`.
- **ChangeGuard (`src/data/kuro_dataset.py:198`):**
  `(sar - self.sar_mean) / self.sar_std`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.6 NaN and NoData Handling
- **Upstream KuroSiwo (`Dataset.py:970`):**
  `torch.nan_to_num(image, 0.15)`.
- **ChangeGuard (`src/data/kuro_dataset.py:197`):**
  `np.nan_to_num(sar, nan=0.15, posinf=0.15, neginf=0.0)`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.7 Label Mapping
- **Semantic Classes:**
  - Class 0: `No Water` (dry background)
  - Class 1: `Permanent Waters` (pre-existing water bodies)
  - Class 2: `Floods` (newly inundated flood water — primary target)
  - Class 3: `Invalid pixels` (missing/corrupt/unmapped data)
- **Citations:** `third_party/KuroSiwo/training/change_detection_trainer.py:15`, `ChangeGuard/configs/kuro_event_split.json:2–7`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.8 Valid-Mask Handling
- **Mask Raster:** `MK0_MNA` (binary valid mask: 1 = valid coverage, 0 = no satellite coverage).
- **Double-Masking Rule:**
  Audit revealed 91,899 pixels in the validation set where `valid_mask == 1` while `label == 3`. ChangeGuard enforces strict double masking:
  `valid = (valid_mask == 1) & (label != 3)`.
- **Citations:** `docs/checkpoint_provenance.md:143–146`, `src/evaluation/metrics.py:48`.
- **Classification:** `VERIFIED_FROM_CODE` / `CURRENT_PROJECT_DECISION`

### 2.9 Class-3 (Invalid) Handling in Training
- **Loss Computation:**
  In `third_party/KuroSiwo/utilities/utilities.py:140`, cross-entropy is initialized with `nn.CrossEntropyLoss(ignore_index=3)`. Class 3 generates zero gradient.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.10 DEM Loading and Sanitization
- **Upstream KuroSiwo (`Dataset.py:731–740`):**
  Loads `MK0_DEM` with `rio.open_rasterio()`, interpolates interior NaNs with `rio.interpolate_na()`. Crucially, does NOT sanitize negative integer NoData values (e.g. $-32768$), which normalize to $\sim -18.6$ and degrade training stability.
- **ChangeGuard Protocol (`src/data/kuro_dataset.py:14–45`, `kuro_dem_sanitize_v1`):**
  Replaces non-finite values and values $< -1000.0\text{ m}$ with normalized value $0.0$ (neutral prior), preserving genuine physical depressions (e.g. Dead Sea $-430\text{ m}$) and high elevations.
- **Classification:** `VERIFIED_FROM_CODE` / `CURRENT_PROJECT_DECISION`

### 2.11 Slope Loading and Computation
- **Upstream KuroSiwo Implementation (`Dataset.py:748–761`):**
  When `configs["slope"]` is enabled:
  ```python
  rd_dem = rd.rdarray(dem.squeeze(), no_data=nodata)
  slope = rd.TerrainAttribute(rd_dem, attrib="slope_riserun")
  slope = np.asarray(slope.data)
  slope = einops.rearrange(slope, "h w -> 1 h w")
  normalization = transforms.Normalize(mean=configs["slope_mean"], std=configs["slope_std"])
  dem = normalization(torch.from_numpy(slope))
  ```
  Slope replaces DEM in the 3rd channel position.
- **Critical Unresolved Issues in Upstream Slope Code:**
  1. *Grid Spacing Assumption:* `richdem.TerrainAttribute(..., attrib="slope_riserun")` is called without specifying ground pixel spacing. In Sentinel-1 GRD patches, pixel resolution is $10\text{ m}$. Without a spacing parameter, `richdem` computes slope assuming $\Delta x = \Delta y = 1.0$ (index units), yielding dimensionless values scaled $10\times$ higher than physical rise-over-run.
  2. *Contradictory Normalization Statistics:* `configs/train/data_config.json:26–29` defines `"slope_mean": 2.1277, "slope_std": 67.5048`, but also defines `"slc_slope_mean": 0.3977, "slc_slope_std": 0.4946`. An std of $67.5$ indicates severe outliers in the upstream statistics computation.
- **Classification:** `UNRESOLVED` (Must be formalized in Week 1 before training Model B).

### 2.12 Spatial Alignment and Shape Checks
- **Spatial Dimension:** Exactly $224 \times 224$ pixels across all channels.
- **Assertion:** `src/data/kuro_dataset.py:168–181` programmatically verifies that all rasters (`SL1`, `SL2`, `MS1`, `DEM`, `MNA`, `MLU`) have shape `(224, 224)`.
- **Classification:** `VERIFIED_FROM_CODE`

### 2.13 Data Augmentation
- **Upstream KuroSiwo (`Dataset.py:44–50`):**
  Albumentations transforms (horizontal/vertical flip, random 90-degree rotations). Disabled by default in `data_config.json:13` (`"data_augmentations": false`).
- **Classification:** `VERIFIED_FROM_CODE`

### 2.14 Filtering of No-Water Patches
- **Training Manifest Filtering:**
  In `third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz`, patches with 0% total water are largely pre-filtered (31,707 samples total).
  - *Integrity Note regarding Test Manifest:* The total sample count of the unconstrained test pickle (67,490) was observed in an earlier session during preflight manifest inspection (`DEVIATION-W0-01`). Zero test image or label rasters were loaded. Further access to `KuroV2_grid_dict_test_0_100.gz` or any test manifest is **permanently prohibited**.
- **Classification:** `VERIFIED_FROM_DATA` / `VERIFIED_FROM_CODE`

### 2.15 Training Patch Size
- $224 \times 224$ pixels at $10\text{ m}$ pixel resolution ($2.24\text{ km} \times 2.24\text{ km}$ spatial footprint).
- **Classification:** `VERIFIED_FROM_DATA`

### 2.16 SNUNet Expected Channel Count & Third-Channel Provenance
- **Architecture Weight Tensor:**
  Official checkpoint `checkpoints/kuro_snunet_official.pt` has first convolutional layer `conv0_0.conv1.weight` of shape `(32, 3, 3, 3)`. The checkpoint requires exactly 3 input channels per temporal branch.
- **Evidence Hierarchy for Official Checkpoint Inputs:**
  - `VERIFIED_FROM_DATA`: First convolution tensor shape requires 3 input channels per temporal branch.
  - `VERIFIED_FROM_CODE`: Default KuroSiwo GRD config (`data_config.json:11`) specifies 2 channels: `["vv", "vh"]`.
  - `OBSERVED`: Upstream Git history contains conflicting keys (`"dem": true` and `"dem": false` in `data_config.json`).
  - `COMMUNITY_HYPOTHESIS`: An unmerged community pull request suggested slope as the third channel; this remains an unverified hypothesis.
  - `UNKNOWN`: The exact third channel and its normalization parameters in the official checkpoint remain unknown.
- **Classification:** `VERIFIED_FROM_DATA` / `UNKNOWN`

### 2.17 Temporal Branch Channel Configuration
- **Model A (Controlled 2-Channel Baseline):**
  Per-time input channels: `[VV, VH]` (2 channels per branch, 4 channels total). Pure radar baseline.
- **Model B (Controlled 3-Channel Ablation):**
  Per-time input channels: `[VV, VH, slope]` (3 channels per branch, 6 channels total). Topographic ablation study (not an assertion of the official checkpoint).
- **Classification:** `CURRENT_PROJECT_DECISION`

### 2.18 Dual Configuration Support in Current Code
- Current `src/data/kuro_dataset.py` is hard-coded to produce 3 channels (`[VV, VH, DEM]`) for evaluating the official checkpoint.
- To support both Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`), the loader must be refactored into a configurable dataset builder that accepts an explicit channel specification list (`channels=["vv", "vh"]` vs `channels=["vv", "vh", "slope"]`).
- **Classification:** `CURRENT_PROJECT_DECISION` / `UNRESOLVED` (requires modular dataset implementation in Week 1).

---

## 3. Summary Table of Audit Classifications

| Dimension | Classification | Primary File Reference | Notes |
| :--- | :--- | :--- | :--- |
| **Temporal Pairing** | `VERIFIED_FROM_CODE` | `kuro_dataset.py:227` | Canonical pre-event is `pre_event_2` |
| **SAR Polarization Loading** | `VERIFIED_FROM_CODE` | `kuro_dataset.py:155` | Band 1 read as `float32` via rasterio |
| **Radiometric Units** | `SUPPORTED_BY_OFFICIAL_DOCUMENTATION` | NeurIPS 2024 paper | Linear amplitude GRD backscatter |
| **SAR Clamping** | `VERIFIED_FROM_CODE` | `Dataset.py:969` | Clamped to `[0.0, 0.15]` |
| **SAR Normalization** | `VERIFIED_FROM_CODE` | `data_config.json:16` | $\mu = [0.0953, 0.0264], \sigma = [0.0427, 0.0215]$ |
| **SAR NaN/NoData** | `VERIFIED_FROM_CODE` | `kuro_dataset.py:197` | `nan_to_num(..., nan=0.15)` |
| **Label Mapping** | `VERIFIED_FROM_CODE` | `trainer.py:15` | 0=NoWater, 1=PermWater, 2=Flood, 3=Invalid |
| **Valid Masking** | `CURRENT_PROJECT_DECISION` | `metrics.py:48` | Double-masking: `(vld == 1) & (lbl != 3)` |
| **Class-3 Ignore** | `VERIFIED_FROM_CODE` | `utilities.py:140` | `ignore_index = 3` in CrossEntropyLoss |
| **DEM Sanitization** | `CURRENT_PROJECT_DECISION` | `kuro_dataset.py:211` | `kuro_dem_sanitize_v1` replaces $< -1000\text{m} \to 0.0$ |
| **Slope Computation** | `UNRESOLVED` | `Dataset.py:748` | `richdem` unit cell size and stats require normalization |
| **Patch Dimensions** | `VERIFIED_FROM_DATA` | Sentinel-1 rasters | Uniform $224 \times 224$ pixels |
| **Dual-Config Support** | `CURRENT_PROJECT_DECISION` | ChangeGuard | Model A `[VV,VH]` vs Model B `[VV,VH,slope]` |
