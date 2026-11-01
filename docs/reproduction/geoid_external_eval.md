# Reproduction Guide: External Zero-Shot Evaluation on GEOID-Flood

This document details the external evaluation of frozen SNUNet-ECAM source models on the GEOID-Flood benchmark, including the input NaN-handling correction and statistical analysis.

---

## 1. Benchmark Scope

- **Dataset:** GEOID-Flood (`links-ads/geoid-flood`, Revision `v1.1`), heldout split.
- **Coverage:** 1,301 paired $1024 \times 1024$ Sentinel-1 GRD tiles partitioned into sixteen $256 \times 256$ non-overlapping chips per tile ($20,816$ inference forward passes per model).
- **Activations:** 9 Copernicus EMS activations (`EMSR857`, `EMSR858`, `EMSR861`, `EMSR864`, `EMSR865`, `EMSR866`, `EMSR868`, `EMSR870`, `EMSR871`).
- **Total Valid Pixels:** $873,090,528$ valid pixels ($45,048,201$ true flood pixels).
- **Models Evaluated:** 6 frozen SNUNet-ECAM models trained on all 27 Kuro Siwo source events:
  - R0 (Seeds 0, 1, 2)
  - R2 (Seeds 0, 1, 2)

---

## 2. Input Radiometric Transform & NaN Parity

Source training and Kuro Siwo inference clamp input amplitudes to $[0.0, 0.15]$ and replace missing/NaN values with $0.15$ (reflecting extreme border nodata values). The external evaluation adapter enforces exact numerical parity:
```python
# src/external_evaluation/geoid_adapter.py
sar = np.clip(sar, 0.0, sar_clamp_max)
sar = np.nan_to_num(sar, nan=0.15, posinf=sar_clamp_max, neginf=0.0)
sar = (sar - sar_mean) / sar_std
```

---

## 3. Running External Evaluation and Analysis

### Full Model Matrix Inference
```bash
python scripts/external_evaluation/run_geoid_nan_correction.py \
    --config configs/external_evaluation/geoid_eval_config_v1.json \
    --output-dir outputs/external_evaluation/geoid_nan_corrected_v1
```

### Statistical Synthesis & LaTeX Table Export
```bash
python scripts/external_evaluation/analyze_geoid_nan_corrected.py
```

### Key Statistical Results:
- **Canonical 9-Event Macro F1:**
  - R0: $0.5492 \pm 0.0326$
  - R2: $0.5475 \pm 0.0167$
  - Paired Difference: $\Delta = -0.0017$ (-0.17 pp)
  - 95% Event Bootstrap CI (9 events): `[-0.0238, +0.0217]` (the interval spans zero, indicating that a non-zero effect was not detected in this sample; this does not establish equivalence or statistical parity between recipes).
- **Precision / Recall Breakdown:**
  - Precision: R0 = $0.4972$, R2 = $0.5439$ ($\mathbf{+4.66\text{ pp}}$ gain in flood precision for R2).
  - Recall: R0 = $0.6519$, R2 = $0.5994$ ($-5.25\text{ pp}$).
- **Pooled Dataset F1 (873M pixels):** R0 = $0.7154$, R2 = $0.7156$ ($\Delta = +0.0002$).
