# Controlled Source-Model Training Protocol (v1.0 Frozen)

**Document Version:** 1.0.0-FROZEN  
**Status:** `APPROVED_FROZEN — READY FOR WEEK 1 SMOKE TESTING`  
**Project:** ChangeGuard FYP (Controlled Source-Model Phase)  
**Date:** 2026-10-06  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA RTX 3090 24GB)  
**Approved Split File:** [`configs/splits/kuro_source_inner_split_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/splits/kuro_source_inner_split_v1.json) (`7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3`)  
**Approved Model A Config:** [`configs/source/snunet_vvvh_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/snunet_vvvh_v1.json) (Week 0 Freeze: `ae18249e7c6ceac4db9ec27ae8797c95eb308c58070ade09b28367593a4cf310`; Week 1 Production Lock: `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc`)  
**Approved Model B Config:** [`configs/source/snunet_vvvh_slope_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/snunet_vvvh_slope_v1.json) (Week 0 Freeze: `8523f7448bfc6a4b153f285f769140978abdca670557eb0f9cdcd3e74f63e03f`; Week 1 Production Lock: `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec`)  

---

## 1. Executive Protocol Summary & Core Safeguards

This protocol specifies the frozen, reproducible experimental design for training and selecting controlled source models on the KuroSiwo benchmark. It eliminates reliance on the un-provenanced official checkpoint as the sole primary model while enforcing strict research integrity boundaries.

### Mandatory Research Integrity Safeguards
1. **Source-Inner Split Exclusivity:** All hyperparameter tuning, loss function evaluation, epoch selection, and model comparisons are performed **strictly on the frozen 22/5 source-inner split** (Alternative A1).
2. **Permanent TTA-Validation Quarantining:** The 6 usable TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) must **never** be accessed or monitored to choose source checkpoints, hyperparameters, loss functions, or architectures.
3. **Permanent Test Set Sealing & Manifest Ban:** All 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain untouched. Opening, unpickling, reading, or inspecting `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` or any other test manifest is **permanently prohibited**.
4. **Deterministic Median Retraining Rule:** The final production source models across all 27 events are retrained using a fixed epoch schedule determined by the median best epoch on inner-validation, completely eliminating post-hoc validation tuning.

---

## 2. Approved Source-Inner Split (Alternative A1)

- **Selection Rationale:** Alternative A1 replaces tiny Event `496` (21 samples) with medium Event `275` (742 samples, mid-flood tertile), covering the full flood prevalence spectrum while expanding the validation corpus to 2,284 patches (10.64%) and reducing the maximum multi-dimensional discrepancy between train and val to just **3.30%** (vs 8.60% in the proposed split). Evaluated strictly on source-label statistics without model-performance information.
- **Inner-Validation Events ($N=5$):**
  ```text
  [267, 275, 497, 518, 1111005]
  ```
  *(Total samples: 2,284 / 10.64%; valid pixels: 112,243,176; pooled flood: 16.88%; macro flood: 15.62%; pooled perm water: 6.85%; macro perm water: 7.62%)*
- **Inner-Training Events ($N=22$):**
  ```text
  [118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]
  ```
  *(Total samples: 19,178 / 89.36%; valid pixels: 950,602,654; pooled flood: 18.42%; macro flood: 14.36%; pooled perm water: 7.06%; macro perm water: 5.33%)*
- **Disjointness & Coverage:**
  - `inner_train` contains exactly 22 events;
  - `inner_val` contains exactly 5 events;
  - Disjoint: `inner_train ∩ inner_val = ∅`;
  - Union equals exactly all 27 official source-training events;
  - Zero TTA-validation events present;
  - Zero sealed test events present.

---

## 3. Approved Model Configurations

Both models utilize the Siamese Siam-NestedUNet (SNUNet) backbone with Ensemble Channel Attention Modules (ECAM):

### 3.1 Model A: Controlled 2-Channel Baseline (`[VV, VH]`)
- **Configuration ID:** `model_a_radar_only_v1`
- **Backbone Architecture:** `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
- **Temporal Pairing:**
  - Branch A ($x_A$): `[pre_event_2_VV, pre_event_2_VH]` ($B \times 2 \times 224 \times 224$)
  - Branch B ($x_B$): `[post_event_VV, post_event_VH]` ($B \times 2 \times 224 \times 224$)
- **Scientific Role:** Pure radar baseline without auxiliary topographic channels.

### 3.2 Model B: Controlled 3-Channel Ablation (`[VV, VH, slope]`)
- **Configuration ID:** `model_b_topographic_ablation_v1`
- **Backbone Architecture:** `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
- **Temporal Pairing:**
  - Branch A ($x_A$): `[pre_event_2_VV, pre_event_2_VH, slope]` ($B \times 3 \times 224 \times 224$)
  - Branch B ($x_B$): `[post_event_VV, post_event_VH, slope]` ($B \times 3 \times 224 \times 224$)
- **Static Topographic Sharing:** The identical static slope raster is provided to both temporal branches.
- **Explicit Provenance Clarification:** Model B is a **controlled scientific ablation study**, evaluating whether topographic slope aids flood discrimination and suppresses false positives on steep terrain. Model B does **not** claim that slope was the official checkpoint's third channel. The official checkpoint's third channel remains completely **unknown**.

---

## 4. Input Preprocessing & Slope Policy Specifications

### 4.1 Verified SAR Preprocessing Pipeline
Following verified loader execution order in `src/data/kuro_dataset.py` and upstream `Dataset.py`:
1. **Linear Amplitude Input:** Sentinel-1 GRD linear amplitude backscatter (square root of intensity).
2. **Clamping:** Raw values clamped to $[0.0, 0.15]$:
   $$\text{sar}_{\text{clamped}} = \text{clip}(\text{sar}_{\text{raw}}, 0.0, 0.15)$$
3. **Non-Finite Replacement:** Replace NaN, $+\infty$, and $-\infty$ with $0.15$ before normalization:
   $$\text{sar}_{\text{finite}} = \text{nan\_to\_num}(\text{sar}_{\text{clamped}}, \text{nan}=0.15, \text{posinf}=0.15, \text{neginf}=0.0)$$
4. **Z-Score Normalization:**
   $$\text{sar}_{\text{norm}} = \frac{\text{sar}_{\text{finite}} - \mu_{\text{sar}}}{\sigma_{\text{sar}}}$$
   with verified constants:
   - VV: $\mu = 0.0953, \sigma = 0.0427$
   - VH: $\mu = 0.0264, \sigma = 0.0215$

### 4.2 Approved Slope Policy (Policy S2: Physically Specified & Robust)
For controlled Model B, the slope generation pipeline must strictly implement the following 14 requirements:
1. **Dynamic Transform Reading:** Read pixel spacing $\Delta x, \Delta y$ directly from each DEM GeoTIFF transform matrix.
2. **CRS Verification:** Verify that the coordinate reference system (CRS) is projected with metric units.
3. **Empirical Verification:** If the transform confirms $10\text{ m}$ spacing, record this as observed data rather than a hard-coded assumption.
4. **Fail Loudly:** If a raster has geographic-degree spacing (e.g. EPSG:4326), raise an explicit runtime error rather than silently treating degrees as metres.
5. **NoData Identification:** Identify DEM nodata from raster metadata and verified sentinel values (e.g. values $< -1000\text{ m}$ via `kuro_dem_sanitize_v1`).
6. **No Boundary Cliffs:** Compute slope only over valid elevation support to prevent artificial steep cliffs at nodata edges.
7. **Physically Explicit Slope Angle:**
   $$\frac{\partial z}{\partial x} = \frac{z_{i, j+1} - z_{i, j-1}}{2 \Delta x}, \quad \frac{\partial z}{\partial y} = \frac{z_{i+1, j} - z_{i-1, j}}{2 \Delta y}$$
   $$\text{slope}_{\text{deg}} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \times \frac{180.0}{\pi}$$
8. **Spatial Alignment:** Verify spatial alignment with SAR rasters ($224 \times 224$).
9. **Branch Sharing:** Provide the identical static slope raster to both Siamese temporal branches.
10. **Invalid Slope Handling:** Replace invalid slope pixels with the training-set slope mean before normalization, producing $0.0$ after standardization.
11. **Strict Train-Only Normalization:** Calculate slope normalization statistics ($\mu_{\text{slope}}, \sigma_{\text{slope}}$) strictly using only the 22 approved inner-training events.
12. **Zero Leakage:** Never use TTA-validation or test data for slope statistics.
13. **Cryptographic Tracking:** Record the SHA-256 hash of the slope statistics artifact (`outputs/week1/source_train_slope_statistics.json`).
14. **Formal Launch Guard:** Formal multi-epoch Model B training remains **blocked** until the slope statistics artifact is generated and validated.

*Note on Upstream Slope Code:* The upstream KuroSiwo slope code omitted physical metric cell spacing and experienced outlier-distorted normalization statistics ($\sigma=67.50$); this is treated as a reproducibility and physical-spacing concern, handled cleanly by Policy S2.

---

## 5. Approved Sample Policy

- **All Records Retained:** All 19,178 patches belonging to the 22 approved inner-training events are retained.
- **Zero Filtering / Oversampling:**
  - Do NOT remove zero-flood patches.
  - Do NOT oversample flood patches.
  - Do NOT undersample background patches.
  - Do NOT generate filtered manifests.
- **Empirical Context:** The source manifest `KuroV2_grid_dict.gz` is already a curated dataset artifact (31,707 total patches). Exactly 15.57% of all source patches (and 15.22% of inner-train patches) contain zero flood pixels. Retaining them preserves the natural spatial prior distribution of non-flooded terrain, training the model to suppress false alarms on dry scenes.

---

## 6. Approved Loss and Optimization Protocol

### 6.1 Optimizer & Schedule
- **Optimizer:** Adam
- **Learning Rate:** $1.0 \times 10^{-3}$
- **Weight Decay:** $0.0$
- **Betas:** PyTorch defaults `(0.9, 0.999)`
- **Scheduler:** No learning rate scheduler in the primary protocol (constant learning rate).

### 6.2 Exact Loss Formulation (Unweighted Cross-Entropy + Soft Dice)
$$L_{\text{total}} = 0.5 \cdot L_{\text{CE}} + 0.5 \cdot L_{\text{Dice}}$$

1. **Cross-Entropy Loss ($L_{\text{CE}}$):**
   $$L_{\text{CE}} = \text{CrossEntropyLoss}(\text{logits}, y, \text{ignore\_index}=3)$$
   Masked invalid pixels ($y = 3$) generate zero loss and zero gradient.

2. **Soft Dice Loss ($L_{\text{Dice}}$):**
   Computed over the 3 valid classes $c \in \{0, 1, 2\}$ on the valid pixel mask $M = (y \ne 3)$:
   $$L_{\text{Dice}} = \frac{1}{3} \sum_{c=0}^2 L_{\text{Dice}}^{(c)}$$
   For each class $c$:
   $$p_{c, i} = \frac{\exp(\text{logits}_{c, i})}{\sum_{k=0}^2 \exp(\text{logits}_{k, i})}, \quad t_{c, i} = \mathbb{I}(y_i == c)$$
   $$L_{\text{Dice}}^{(c)} = 1 - \frac{2 \sum_{i \in M} p_{c, i} t_{c, i} + \epsilon}{\sum_{i \in M} p_{c, i}^2 + \sum_{i \in M} t_{c, i}^2 + \epsilon}$$
   with smoothing constant $\epsilon = 1.0$.

3. **Absent-Class Handling:**
   When a class $c$ is entirely absent from the ground truth of a batch ($\sum_{i \in M} t_{c, i} = 0$):
   - If the model correctly predicts zero probability ($p_{c, i} \approx 0$), numerator is $\epsilon$ and denominator is $\epsilon$, yielding Dice score $1.0$ and loss $0.0$.
   - If the model predicts false alarms ($p_{c, i} > 0$), the denominator increases while the numerator remains $\epsilon$, penalizing the false predictions smoothly.
   - The smoothing constant $\epsilon = 1.0$ mathematically prevents division by zero and guarantees stable, bounded gradients even on batches with absent classes.

---

## 7. Model Selection Metrics, Epoch Tie-Breaking & Final Retraining Rule

### 7.1 Selection Metrics on Inner-Validation
- **Primary Metric:** Event-Macro Flood F1 ($\text{Macro F1}_{\text{flood}}$) across the 5 inner-validation events.
- **Secondary Metrics:** Worst-Event Flood F1, Event-Macro Flood Recall, Event-Macro Flood IoU, Pooled Flood F1.

### 7.2 Deterministic Epoch Tie-Breaking
When choosing the best checkpoint epoch for a given training seed:
1. Higher event-macro Flood F1;
2. If tied to within $10^{-5}$, higher worst-event Flood F1;
3. If still tied, higher event-macro Flood Recall;
4. If still tied, the **earlier epoch** (smaller integer).

### 7.3 Training Budget & Early Stopping
- Maximum Budget: 150 epochs.
- Early Stopping: Patience 20 epochs based on inner-validation event-macro Flood F1.

### 7.4 Final Retraining Rule
For each model configuration (Model A and Model B) trained across seeds `[42, 43, 44]`:
1. Record optimal epochs $E_{42}^*, E_{43}^*, E_{44}^*$ from the 22/5 split.
2. Compute the locked retraining epoch:
   $$E_{\text{final}} = \text{median}(E_{42}^*, E_{43}^*, E_{44}^*)$$
3. Retrain the final controlled source models from scratch across **all 27 source events** (21,462 patches) for **exactly $E_{\text{final}}$ epochs**.
4. During this final retraining, **zero validation tuning** is permitted.

---

## 8. Batch-Size Policy & GPU Execution Strategy

- **Target Effective Batch Size:** 32.
- **Precision:** Automatic Mixed Precision (AMP FP16).
- **Physical Microbatch Status:** `TO_BE_CONFIRMED_BY_WEEK1_MEMORY_SMOKE_TEST`.
- **Gradient Accumulation Rule:**
  $$\text{accumulation\_steps} = \frac{32}{\text{physical\_microbatch\_size}}$$
  with mandatory exact divisibility (`32 % microbatch == 0`).
- **Week 1 Microbatch Test Progression:** Candidate microbatches `[2, 4, 8, 16]`.
- **Multi-Epoch Training Block:** Formal multi-epoch training remains strictly blocked until the physical microbatch size and gradient accumulation steps are validated and recorded in a versioned Week 1 protocol amendment.

---

## 9. Gate S: Controlled Source Model Acceptance Criteria

Before any controlled source model is accepted for downstream TTA benchmarking:
1. **Flood Detection Quality:** Model achieves inner-validation **Macro F1 $\ge 0.50$** and **Macro Recall $\ge 0.45$**.
2. **False Alarm Suppression:** Flood false-alarm rate on dry scenes (Event `518`) must not exceed 2.0% of valid pixels.
3. **Convergence Stability:** Training loss must monotonically decrease over the first 20 epochs without NaN/Inf gradients.
4. **Zero Leakage Certification:** Manifest verification must certify that zero TTA-validation or sealed test samples were ingested.

---

## 10. Week 1 Entry Conditions Checklist

- [x] Approved 22/5 split frozen in `configs/splits/kuro_source_inner_split_v1.json`.
- [x] Model A config frozen in `configs/source/snunet_vvvh_v1.json`.
- [x] Model B config frozen in `configs/source/snunet_vvvh_slope_v1.json`.
- [x] Protocol specification frozen in `docs/source_training_protocol_v1.md`.
- [x] All 32 baseline unit tests passing.
- [x] Week 1 memory smoke test to determine physical microbatch size (locked to 8).
- [x] Week 1 generation of train-only slope statistics artifact for Model B (locked in `artifacts/week1/`).
- [x] Week 1 launch and validation of controlled source training smoke tests on GPU 0.

---

## 11. Week 1 Implementation Amendments & Production Configuration Lock

During Week 1 execution, three technical amendments were calibrated, validated, and locked into the production configurations without altering scientific settings, model architectures, loss formulations, or optimizer hyperparameters:

### 11.1 Amendment W1-A: Batch Size & Microbatch Lock
- In accordance with Section 8, GPU memory smoke profiling on GPU 0 (`cuda:0`, 24 GB RTX 3090) evaluated candidate microbatches.
- Physical microbatch size is locked to `8` (`physical_microbatch_size = 8`).
- Gradient accumulation steps are locked to `4` (`gradient_accumulation_steps = 4`), achieving the target effective batch size of $32$ ($8 \times 4 = 32$).
- Peak VRAM observed: $2.28\text{ GB}$ (9.3% of 24 GB capacity).
- Gradient accumulation equivalence: Masked CE loss is bit-identical across microbatches; Soft Dice evaluated per microbatch differs by approximately $0.36\%$ ($< 5\times 10^{-5}$ per sample) from full-batch Dice, explicitly codified via the `accumulation_loss_semantics` key.

### 11.2 Amendment W1-B: Train-Only Topographic Slope Normalization
- In accordance with Section 4.2 (Items 11–13), topographic slope statistics were computed strictly over the 19,178 patches of the 22 inner-training events.
- Locked values: $\mu_{\text{slope}} = 1.492574^\circ$, $\sigma_{\text{slope}} = 2.372311^\circ$.
- Artifact locked in Git: `artifacts/week1/source_train_slope_statistics.json` (SHA-256: `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`).
- Zero TTA-validation or test events were accessed.

### 11.3 Amendment W1-C: Canonical Optimizer Naming & Runtime Binding
- The optimizer field in both production JSON configurations was standardized from `"adam"` to `"Adam"` (`"optimizer": "Adam"`).
- **Provenance Statement:** This change was purely semantic and capitalization-only. No changes were made to the optimizer class or hyperparameters. Both configurations and runtime code have always implemented the approved Adam optimizer.
- **Exact Effective Protocol:**
  - **Optimizer:** `torch.optim.Adam` (NOT `AdamW`)
  - **Learning Rate:** `0.001` ($1.0 \times 10^{-3}$)
  - **Weight Decay:** `0.0`
  - **Betas:** `(0.9, 0.999)` (PyTorch defaults)
  - **LR Scheduler:** `None` (`null`, constant learning rate)
  - **Precision:** FP16 Automatic Mixed Precision (`torch.cuda.amp.autocast()` + `GradScaler`)
  - **Early Stopping:** Patience 20 epochs evaluated on the 5-event inner-validation set
- Runtime code directly instantiates this protocol from config dictionaries via `SourceTrainer.from_config`, verified by automated regression test `test_approved_optimizer_protocol_and_runtime_agreement`.

### 11.4 Cryptographic Hashes Summary
| Configuration / Artifact | Week 0 Freeze (`af164ec`) | Week 1 Production Lock |
|---|---|---|
| `configs/source/snunet_vvvh_v1.json` | `ae18249e7c6ceac4db9ec27ae8797c95eb308c58070ade09b28367593a4cf310` | `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc` |
| `configs/source/snunet_vvvh_slope_v1.json` | `8523f7448bfc6a4b153f285f769140978abdca670557eb0f9cdcd3e74f63e03f` | `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec` |
| `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` (Unchanged) |
| `artifacts/week1/source_train_slope_statistics.json` | *To be generated* | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` |
