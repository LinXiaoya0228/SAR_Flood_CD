# GEOID-Flood Compatibility and Event Independence Audit (v1)

**Date**: November 1, 2026  
**Status**: Authoritative Scientific Audit  
**Target Benchmark**: GEOID-Flood (`links-ads/geoid-flood`, Release `v1.1`)  
**Evaluated Models**: Frozen ChangeGuard All-27-Source SNUNet-ECAM Models (R0 vs. R2)

---

## 1. Executive Summary

This document establishes the formal compatibility, radiometric alignment, and strict event independence of the **GEOID-Flood** benchmark (`links-ads/geoid-flood` v1.1) for zero-shot external evaluation of the frozen ChangeGuard source-trained models.

The primary objective of this external evaluation is to answer:
> **Does the robust source-training recipe (R2: event-balanced sampling + class-weighted loss) demonstrate superior cross-event generalization compared to standard source training (R0) when evaluated on an independent, post-2022 global SAR flood change detection benchmark without any fine-tuning, adaptation, or threshold recalibration?**

All evaluation is strictly post-hoc and inference-only (`model.eval()`, frozen BatchNorm, `torch.no_grad()`). Zero weights, hyperparameters, decision thresholds, or normalization parameters are modified or tuned on external data.

---

## 2. Radiometric and Physical Product Compatibility

### 2.1 Sensor and Acquisition Mode
- **Sensor**: Sentinel-1 C-band Synthetic Aperture Radar (SAR), constellation satellites Sentinel-1A and Sentinel-1C.
- **Acquisition Mode**: Interferometric Wide Swath (IW), Ground Range Detected High Resolution (GRDH).
- **Polarizations**: Dual-polarization VV (vertical transmit, vertical receive) and VH (vertical transmit, horizontal receive).
- **Pixel Spacing**: 10 meters per pixel.

Both the Kuro Siwo training corpus and GEOID-Flood utilize identical Sentinel-1 IW GRDH dual-polarization observations.

### 2.2 Product Selection: `s1grd` vs. `s1rtc`
GEOID-Flood provides two distinct SAR products:
1. `s1rtc`: Radiometrically Terrain Corrected $\gamma^0$ (Gamma Nought, backscatter per unit area perpendicular to line-of-sight).
2. `s1grd`: Range-Doppler Terrain Corrected linear $\sigma^0$ (Sigma Nought, radar backscatter coefficient per unit ground area).

**ChangeGuard Training Lineage Audit**:
Auditing `third_party/KuroSiwo/configs/grd_preprocessing.xml` confirms the ESA SNAP preprocessing workflow applied to the Kuro Siwo training data:
- `Apply-Orbit-File`
- `ThermalNoiseRemoval`
- `Calibration` (output: Sigma0 band, linear power)
- `Terrain-Correction` (Range-Doppler with SRTM/Copernicus 30m DEM, output: linear Sigma0 power)

Therefore, **`s1grd` is the mathematically and physically exact product matching ChangeGuard's source training domain**. The decision to use `s1grd` was formally locked *prior* to any inference or model scoring.

### 2.3 Input Normalization Pipeline
ChangeGuard ingests 2-channel SAR tensors `[VV, VH]` preprocessed identically to the canonical Kuro Siwo pipeline:
1. **Clipping**: Values clamped to $[0.0, 0.15]$.
2. **NaN / Inf Handling**: `torch.nan_to_num(x, nan=0.0, posinf=0.15, neginf=0.0)`.
3. **Standardization**:
   $$\mu = [0.0953, 0.0264], \quad \sigma = [0.0427, 0.0215]$$
   $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

Empirical verification of 1,301 held-out GEOID-Flood `s1grd` rasters confirms:
- Mean VV backscatter: $\approx 0.10 - 0.13$
- Mean VH backscatter: $\approx 0.028 - 0.034$

The numerical scale of GEOID-Flood `s1grd` matches the training distribution of Kuro Siwo.

---

## 3. Label Ontology and Evaluation Invariants

### 3.1 Class Ontology Mapping
| Class ID | GEOID-Flood Semantics | ChangeGuard Model Head | ChangeGuard Evaluation Role |
| :---: | :--- | :--- | :--- |
| **0** | Background (Land / Non-water) | Class 0: Background | Negative (Non-flood) |
| **1** | Permanent Water | Class 1: Permanent Water | Negative (Non-flood) |
| **2** | Flooded Water | Class 2: Flood Water | **Positive (Flood Target)** |
| **255** | Invalid / Outside AOI | Ignored (Class 3 in training) | **Excluded from Evaluation** |

### 3.2 Masking Invariant
In GEOID-Flood, each tile is accompanied by a binary `validity` raster:
- `validity == 1`: Valid observation within Copernicus Emergency Management Service (CEMS) delineation area.
- `validity == 0`: Invalid pixel, outside AOI, or corrupted observation.

The evaluation mask is strictly defined as:
$$\text{evaluable\_mask} = (\text{validity} == 1) \land (\text{label} \neq 255)$$

Pixels failing this predicate are excluded entirely from confusion accumulators (TP, FP, FN, TN).

---

## 4. Spatial Tiling Architecture

- **Native GEOID Tile Geometry**: $1024 \times 1024$ pixels at 10 m resolution ($10.24 \times 10.24 \text{ km}$).
- **Model Receptive Geometry**: SNUNet-ECAM is fully convolutional and accepts inputs divisible by $2^4 = 16$.
- **Tiling Scheme**: Each $1024 \times 1024$ raster decomposes into a deterministic $4 \times 4$ grid of sixteen non-overlapping $256 \times 256$ chips:
  - Rows: `[0:256, 256:512, 512:768, 768:1024]`
  - Cols: `[0:256, 256:512, 512:768, 768:1024]`
- **Full Coverage Guarantee**: Exactly $16 \times (256 \times 256) = 1,048,576$ pixels per tile. Every pixel in the original $1024 \times 1024$ tile is evaluated once and only once with zero spatial resampling, zero distortion, and zero boundary padding.

---

## 5. Event Independence Verification

### 5.1 Temporal Independence
- **Kuro Siwo Range**: 2014 through February 2022 (highest Copernicus activation: `EMSR567`).
- **GEOID-Flood Held-out Range**: January 2026 through April 2026 (activations `EMSR857` through `EMSR871`).
- **Temporal Separation**: Greater than 3.9 years. Zero temporal overlap exists.

### 5.2 Geographic and Activation Independence
The held-out partition contains 1,301 tiles across 9 independent Copernicus Emergency Management Service (CEMS) activations:

| Activation ID | Countries | Event Date | Tiles | Valid Pixels | Flood Pixels | Flood % | Perm Water % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EMSR857** | Mozambique, South Africa | 2026-01-10 | 208 | 127,216,970 | 24,159,571 | 18.99% | 3.91% |
| **EMSR858** | Italy | 2026-01-20 | 10 | 2,111,777 | 31,226 | 1.48% | 9.92% |
| **EMSR861** | Portugal, Spain | 2026-01-28 | 580 | 435,341,294 | 7,539,308 | 1.73% | 2.94% |
| **EMSR864** | Portugal | 2026-02-03 | 157 | 68,612,676 | 5,788,020 | 8.44% | 5.62% |
| **EMSR865** | Colombia | 2026-01-26 | 51 | 38,218,599 | 2,412,868 | 6.31% | 2.72% |
| **EMSR866** | France | 2026-02-12 | 191 | 151,113,728 | 3,563,280 | 2.36% | 1.71% |
| **EMSR868** | Greece | 2026-02-18 | 31 | 16,344,370 | 882,960 | 5.40% | 2.08% |
| **EMSR870** | Ecuador | 2026-03-02 | 31 | 14,463,700 | 495,456 | 3.43% | 0.62% |
| **EMSR871** | Italy | 2026-03-31 | 42 | 19,667,414 | 175,512 | 0.89% | 1.28% |
| **Total** | **9 Activations** | **2026 Q1** | **1,301** | **873,090,528** | **45,048,201** | **5.16%** | **2.97%** |

All 9 activations are 100% spatially and physically disjoint from all 43 Kuro Siwo activations.

### 5.3 Sensitivity Analysis Protocol for Iberian Events
Activations `EMSR861` (Jan 28, 2026, Portugal/Spain) and `EMSR864` (Feb 3, 2026, Portugal) occurred within 6 days of each other in the Iberian Peninsula. To ensure robustness against potential meteorological auto-correlation:
- **Primary Reporting**: Canonical unweighted macro average over all 9 distinct CEMS activations.
- **Sensitivity Check**: Grouped evaluation combining `EMSR861` and `EMSR864` into a single Iberian winter flood event (yielding 8 macro events).

---

## 6. Frozen Model Checkpoint Manifest

All evaluated checkpoints are bit-identical to the authoritative RST9/RST10 ledger:

| Recipe | Seed | Checkpoint Path | SHA-256 Hash |
| :--- | :---: | :--- | :--- |
| **R0** | 0 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| **R0** | 1 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` |
| **R0** | 2 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` |
| **R2** | 0 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1/final_checkpoint.pt` | `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7` |
| **R2** | 1 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s1_v1/final_checkpoint.pt` | `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2` |
| **R2** | 2 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s2_v1/final_checkpoint.pt` | `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893` |

---

## 7. Audit Sign-off

- [x] Product selection audited and locked (`s1grd` linear $\sigma^0$).
- [x] Normalization verified against frozen training distribution.
- [x] Double-masking logic validated across 1,301 held-out tiles.
- [x] Tiling architecture verified without spatial distortion ($16 \times 256 \times 256$).
- [x] Temporal, spatial, and activation independence confirmed.
- [x] Checkpoint hashes cryptographically verified.
