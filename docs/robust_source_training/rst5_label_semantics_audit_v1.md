# RST5 Label Semantics and Class-Index Alignment Audit

## 1. Executive Summary

This audit traces and establishes the canonical, end-to-end mapping between raw raster labels, PyTorch tensors, training loss objectives, and evaluation confusion matrices across the ChangeGuard robust source-training route.

### Audit Verdict:
1. **Perfect Semantic Alignment in Code**:
   The active training and evaluation pipeline uses **completely consistent class-index semantics**:
   - Class **0**: Dry ground / no water (background negative).
   - Class **1**: Permanent water (pre-existing water negative).
   - Class **2**: Newly inundated flood water (positive change target).
   - Class **3**: Invalid / sensor nodata / unannotated (ignore index).
2. **Identification of Documentation Prose Discrepancy**:
   In some RST3 and RST4 textual summaries, class names were casually labeled as `class_0_background`, `class_1_non_flooded`, and `class_2_flooded`, which created ambiguity as to whether Class 1 referred to all non-flooded land or specifically permanent water. Furthermore, in `configs/robust_source_training/rst4_selected_recipe_v1.json`, sample pixel counts from an older summary were inadvertently copied instead of the exact counts from `outputs/robust_source_training/rst4/rst4_r2_snunet_foldc_s0_v2/training_class_weights.json`.
3. **No Technical Changes to Code or Label Mappings Required**:
   Because all code (`dataset.py`, `loss.py`, `evaluation.py`, `runner.py`) already operates on the canonical KuroSiwo four-class scheme, no label remapping is altered. Only documentation and confirmatory config records are corrected.

---

## 2. Authoritative Class Semantics Table

The authoritative semantic definitions for all tensors and operations in ChangeGuard are defined below:

| Tensor Index | KuroSiwo Dataset Meaning | Training Loss Meaning | Evaluation Meaning | M2 Weight Position | Example Weight (Fold C Discovery) |
|---|---|---|---|---|---|
| **0** | `No Water` (dry ground, soil, vegetation, urban) | Negative Class 0 in Cross-Entropy | Non-flood negative (True Negative if pred 0/1; False Alarm if pred 2) | `weight[0]` | `0.492271` (downweighted; ~73.95% of valid pixels) |
| **1** | `Permanent Water` (pre-existing lakes, rivers, ocean) | Negative Class 1 in Cross-Entropy | Non-flood negative (True Negative if pred 0/1; False Alarm if pred 2) | `weight[1]` | `1.666056` (upweighted; ~5.30% of valid pixels) |
| **2** | `Flood Water` (newly inundated flood areas) | Positive Class 2 in Cross-Entropy | Flood change target (True Positive if pred 2; False Negative if pred 0/1) | `weight[2]` | `0.841673` (intermediate; ~20.75% of valid pixels) |
| **3** | `Invalid / NoData` (sensor nodata or unmapped) | Excluded via `ignore_index = 3` | Excluded via double-masking `(MK0_MNA == 1) & (MK0_MLU != 3)` | Excluded (`NA`) | N/A (gradient masked to zero) |

---

## 3. End-to-End Pipeline Verification Trace

### 3.1 Upstream KuroSiwo Rasters (`MK0_MLU` and `MK0_MNA`)
- `MK0_MLU`: Multi-class Land Use / inundation ground-truth mask. Pixels take values in `{0, 1, 2, 3}`.
- `MK0_MNA`: Physical sensor valid mask. `1` = valid radar observation, `0` = sensor nodata / missing line.

### 3.2 Dataset Loading (`src/robust_source_training/dataset.py`)
- Line 203-204:
  ```python
  mna = self._read_raster(valid_path).astype(np.uint8)
  mlu = self._read_raster(label_path).astype(np.int64)
  ```
- Line 228 (Double-Masking Enforcement):
  ```python
  mlu[mna == 0] = 3
  ```
  Any pixel where the radar sensor lacks data (`mna == 0`) is unconditionally set to label `3`.

### 3.3 M2 Class Weight Computation (`src/robust_source_training/loss.py`)
- Valid pixels are filtered as `valid = (mna == 1) & (lbl != 3)`.
- Valid pixels are binned into `class_0` (`lbl == 0`), `class_1` (`lbl == 1`), and `class_2` (`lbl == 2`).
- Weights are calculated via the authoritative formula:
  $$p_c = \frac{N_c}{\sum_{k} N_k}$$
  $$\text{raw}_c = \frac{1}{\sqrt{\max(p_c, 10^{-12})}}$$
  $$\text{norm}_c = \frac{\text{raw}_c}{\text{mean}(\text{raw})}$$
  $$\text{clipped}_c = \text{clip}(\text{norm}_c, 0.5, 2.0)$$
  $$\text{final}_c = \frac{\text{clipped}_c}{\text{mean}(\text{clipped})}$$
- Position 0 corresponds to Class 0, Position 1 to Class 1, Position 2 to Class 2.

### 3.4 Model Architecture & Loss Function
- The SNUNet model outputs logits of shape `(B, 3, H, W)` corresponding to classes `[0, 1, 2]`.
- In `ControlledSourceLoss`:
  ```python
  self.ce = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=3)
  ```
  PyTorch maps `weight[0]` to channel 0, `weight[1]` to channel 1, `weight[2]` to channel 2, and ignores pixels where `target == 3`.

### 3.5 Evaluation & Confusion Matrix (`src/robust_source_training/evaluation.py`)
- Line 16:
  ```python
  positive_class: int = 2
  ```
- Lines 36-37:
  ```python
  y_true = gt_labels[valid] == positive_class
  y_pred = pred_labels[valid] == positive_class
  ```
- Binary flood metrics (Precision, Recall, F1, IoU) are computed strictly with respect to Class 2.
- A prediction of Class 1 (permanent water) on a Class 1 ground-truth pixel evaluates as `y_true = False, y_pred = False`, correctly contributing to True Negatives ($TN$), preventing false alarms over permanent water bodies.

---

## 4. Rare-Class Emphasis Audit

Because permanent water is relatively rare in the training scenes (~5.3% of valid pixels across Groups A+B), its inverse square-root weight is the largest:
- Class 0 (Dry Ground, 73.95%): Weight `0.492271` (attenuated background penalties).
- Class 1 (Permanent Water, 5.30%): Weight `1.666056` (emphasized separation between permanent water and flood water).
- Class 2 (Flood Water, 20.75%): Weight `0.841673` (balanced change target penalty).

This weighting directly targets the core failure mode in SAR flood mapping: distinguishing temporary inundation (Class 2) from perennial water bodies (Class 1) and dry terrain (Class 0).

---

## 5. Audit Conclusion

The class-index semantic audit confirms:
- **No index collision**: Training and evaluation flood targets are identical (`index = 2`).
- **No background collision**: Dry land is consistently `index = 0` and permanent water is `index = 1`.
- **Ignore index invariant**: Label 3 is strictly excluded by both loss and evaluation masks.
- All documentation in RST5 and RST6 will use the explicit terms: `Class 0 (No Water / Dry Ground)`, `Class 1 (Permanent Water)`, `Class 2 (Flood Water)`, and `Class 3 (Invalid / Ignore)`.
