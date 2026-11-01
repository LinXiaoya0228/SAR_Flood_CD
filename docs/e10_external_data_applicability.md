# ChangeGuard Work Package E10: External-Data Compatibility and Applicability Audit

**Document Version:** `e10_external_data_applicability_v1`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E10 (Section 15)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `COMPLETED — STRICT INTEGRITY COMPLIANT`  

---

## 1. Executive Summary & Objective

In accordance with Block E10 Section 15 of the Master Execution Plan V3, this audit evaluates candidate external benchmarks and datasets for their potential use in stress-testing test-time adaptation (TTA) algorithms on multi-temporal Synthetic Aperture Radar (SAR) flood change detection.

Specifically, the audit examines whether any existing external dataset satisfies the necessary operational criteria to serve as an out-of-domain evaluation benchmark for ChangeGuard models without violating scientific integrity or introducing confounding variables.

---

## 2. Technical Compatibility Criteria

To serve as a scientifically valid out-of-distribution benchmark for ChangeGuard's source-trained models (SNUNet-ECAM and SiamUnet_diff), an external dataset must satisfy all five of the following strict criteria:

1. **Sensor & Modality Match:** Must consist of dual-polarimetric (VV and VH) Synthetic Aperture Radar (C-band, preferably Sentinel-1 Interferometric Wide Swath / Ground Range Detected). Optical imagery (RGB/NIR) cannot be ingested by 2-channel SAR backbones.
2. **Temporal Structure:** Must provide true bitemporal paired observations: an aligned pre-event SAR acquisition ($T_1$) and post-event SAR acquisition ($T_2$). Single-temporal flood mapping datasets cannot test change detection architectures.
3. **Taxonomy & Class Definitions:** Must provide or be strictly convertible to the 3-class flood change taxonomy:
   - Class 0: No Water (background land / non-water)
   - Class 1: Permanent Water (pre-existing water bodies)
   - Class 2: Flooded / Changed Water (new inundation)
   - Class 3: Ignored / Invalid (no-data / nodata mask)
4. **Spatial Calibration & Scale:** Must be resampled or natively provided at 10 m pixel spacing with standardized linear amplitude calibration ($[0.0, 0.15]$ dynamic range).
5. **No Data Leakage / Quarantine Integrity:** Must not overlap with or leak information from the 10 sealed Kuro Siwo test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 3. Evaluation of Candidate Datasets

### 3.1 Sen1Floods11 (Bonafilia et al., CVPRW 2020)
- **Modality:** Sentinel-1 C-band SAR (VV, VH) and Sentinel-2 optical imagery.
- **Coverage:** 11 global flood events, 4,831 chips ($512 \times 512$ at 10 m resolution).
- **Temporal Pairing:** **FAILED.** Sen1Floods11 is structured as a *single-temporal* flood detection dataset. Chips contain only post-flood Sentinel-1 images. There are no co-registered, aligned pre-event SAR chips provided.
- **Label Taxonomy:** **FAILED.** Binary labels only ($0 = \text{Non-water}, 1 = \text{Water}$). Sen1Floods11 does not distinguish between permanent surface water (Class 1) and flood-induced change (Class 2).
- **Verdict:** **INCOMPATIBLE.** Ingesting Sen1Floods11 would require either synthesizing an artificial pre-event image (introducing uncalibrated artifacts) or collapsing the 3-class change detection task into single-image segmentation, which would fundamentally violate the bitemporal change-detection formulation.

### 3.2 FloodNet (Rahnemoonfar et al., IEEE Access 2021)
- **Modality:** High-resolution optical Unmanned Aerial Vehicle (UAV / drone) RGB imagery.
- **Spatial Resolution:** Sub-meter ($1.5 - 3.0$ cm/pixel).
- **Sensor:** DJI Phantom 4 Pro camera (visible light).
- **Temporal Pairing:** Single-epoch post-disaster RGB photography. Zero SAR data.
- **Verdict:** **INCOMPATIBLE.** Optical drone photography lacks radar backscatter physics, polarization channels, and orbital geometry.

### 3.3 SpaceNet 8: Flood Detection Challenge (Hansch et al., 2022)
- **Modality:** Very High Resolution (VHR) Maxar optical satellite imagery ($0.5$ m RGB / Pan-sharpened).
- **Coverage:** Hurricane Ida (Louisiana) and Germany floods.
- **Task Focus:** Building damage segmentation and flooded road network graph extraction.
- **Sensor:** Optical spaceborne sensors. Zero SAR imagery.
- **Verdict:** **INCOMPATIBLE.** SpaceNet 8 is strictly an optical infrastructure-mapping dataset.

### 3.4 OpenIFS / Global Flood Monitoring (GFM) / CEMS Raw Activations
- **Modality:** Heterogeneous spaceborne radar (Sentinel-1, COSMO-SkyMed, TerraSAR-X, RADARSAT-2).
- **Temporal Pairing:** Variable; pre-event data must be manually queried, downloaded, and co-registered from ESA / Copernicus SciHub.
- **Labels:** Vector polygons of emergency delineation without standardized per-pixel 3-class ground truth or standardized quality-assurance masking.
- **Processing Requirements:** Requires custom end-to-end preprocessing pipelines (orbit correction, thermal noise removal, calibration, terrain correction, resampling), introducing arbitrary spatial interpolation differences and projection mismatches.
- **Verdict:** **INCOMPATIBLE FOR CONTROLLED TTA.** Introducing unstandardized ad-hoc raw activations would introduce massive confounders from differing preprocessing pipelines rather than isolating the effect of domain shift on adaptation.

### 3.5 Kuro Siwo Sealed Test Split (Events 205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013)
- **Modality & Compatibility:** Natively identical (Sentinel-1 VV/VH, 10 m, 3-class labels).
- **Quarantine Status:** **STRICTLY PROHIBITED.**
- **Verdict:** In accordance with the ChangeGuard Research Integrity Protocol, access to the 10 sealed test events is strictly reserved for the final Block E11 confirmation. Accessing or evaluating on these events during E10 is categorically forbidden.

---

## 4. Synthesis and Methodological Decision

| Dataset Candidate | Modality | Bitemporal Pair | 3-Class Taxonomy | Resolution / Format | Compliance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sen1Floods11** | SAR + Optical | No (Single Post) | No (Binary 0/1) | 10 m ($512\times 512$) | **Incompatible** (No bitemporal pair, no 3-class label) |
| **FloodNet** | Optical UAV | No (Single Post) | No (Multi-class infra) | $\sim 2$ cm | **Incompatible** (Optical only, zero SAR) |
| **SpaceNet 8** | Optical VHR | Yes (Optical) | No (Roads/Buildings) | $0.5$ m | **Incompatible** (Optical only, zero SAR) |
| **CEMS Raw Activations** | Heterogeneous SAR | Ad-hoc / Unaligned | No (Vector polygons) | Disparate | **Incompatible** (Unstandardized, uncalibrated) |
| **Kuro Siwo Sealed-Test** | SAR (Sentinel-1) | Yes ($T_1, T_2$) | Yes (3-class) | 10 m ($224\times 224$) | **Quarantined** (Strictly reserved for E11) |

### Formal Conclusion
Because no external public dataset simultaneously provides bitemporal Sentinel-1 SAR inputs, identical 3-class flood change ground-truth, and standardized 10 m calibration without introducing major confounding variables, **the synthetic-shift family provides the sole scientifically sound, mathematically controlled, and physically grounded framework for evaluating out-of-distribution robustness in E10.**

By applying systematic, physically grounded radiometric backscatter shifts ($x' = (1 + s) \cdot x$) directly within the verified KuroSiwo preprocessing pipeline on the 6 development events, we rigorously evaluate model and adaptation robustness under known, calibrated covariate shifts while preserving exact ground-truth flood semantics and absolute research isolation from the sealed test set.

