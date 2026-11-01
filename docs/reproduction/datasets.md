# Reproduction Guide: Dataset Acquisition and Preprocessing

This document outlines official download sources, directory structures, and preprocessing contracts for both the primary Kuro Siwo benchmark and the external GEOID-Flood evaluation dataset.

---

## 1. Kuro Siwo Benchmark (Source & Historical Target Events)

Kuro Siwo is an open, global multi-temporal SAR flood change detection benchmark comprising Sentinel-1 GRD imagery pairs and high-confidence water masks.

### Acquisition
- **Official Repository:** [https://github.com/Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo)
- **Hugging Face / Zenodo:** Dataset cards and raw zip files are hosted via Hugging Face (`Orion-AI-Lab/KuroSiwo`) and Zenodo.
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).

### Directory Layout
Download and unpack the dataset into the repository root under `data/`:
```text
data/KuroSiwo/
├── raw/
│   ├── event_001/
│   │   ├── pre_event.tif
│   │   ├── post_event.tif
│   │   └── ground_truth.tif
│   └── ...
└── splits/
    ├── train_events.json
    └── test_events.json
```

### Preprocessing Specifications
- **Input Channels:** 4 channels total (VV pre, VH pre, VV post, VH post).
- **Physical Representation:** Sentinel-1 Ground Range Detected (GRD) linear backscatter amplitude.
- **Radiometric Clamping:** Values clamped to $[0.0, 0.15]$.
- **Invalid Value Fill:** Missing / NaN pixels filled with $0.15$ (reflecting high boundary reflectance rather than false permanent water).
- **Z-Score Normalization:** Standardized per-channel using pre-computed source statistics:
  - Mean: $\mu = (0.0953, 0.0264)$
  - Standard Deviation: $\sigma = (0.0427, 0.0215)$
- **Patch Dimension:** Non-overlapping or sliding $256 \times 256$ pixel crops.

---

## 2. GEOID-Flood Benchmark (External Zero-Shot Generalization)

GEOID-Flood is a large-scale Copernicus Emergency Management Service (EMS) flood change detection dataset spanning multiple European flood activations.

### Acquisition
- **Dataset Identifier:** `links-ads/geoid-flood` (Revision `v1.1`)
- **Modality:** `s1grd` (Sentinel-1 Ground Range Detected)
- **Split:** `heldout` split (1,301 paired $1024 \times 1024$ tiles across 9 EMS activations).
- **License:** Open Access under Copernicus EMS data terms.

### Directory Layout
```text
data/external_geoid/
└── geoid-flood-heldout/
    ├── EMSR857/
    │   ├── pre/
    │   ├── post/
    │   └── label/
    ├── EMSR858/
    └── ...
```

### Tile Metadata Catalog
The tile catalog mapping events, tile IDs, and bounding boxes is provided at:
```text
data/geoid_metadata/heldout_tile_catalog.json
```

### Automated Acquisition Helper
To download the S1-GRD heldout split using the included helper script:
```bash
python scripts/external_evaluation/download_geoid_s1grd.py \
    --output-dir data/external_geoid/geoid-flood-heldout \
    --revision v1.1
```

