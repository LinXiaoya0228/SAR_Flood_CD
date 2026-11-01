# Controlled Source-Model Training Protocol (Draft)

- **Document Version:** 1.0.0-DRAFT  
- **Status:** `DRAFT — AWAITING USER APPROVAL`  
- **Project:** ChangeGuard FYP (Phase: Week 0 Protocol Establishment)  
- **Date:** 2026-10-06  
- **Primary Authors:** ChangeGuard Research Team  

---

## 1. Executive Protocol Objectives

The objective of the controlled source-model phase is to train, validate, and checkpoint fully reproducible baseline models on the KuroSiwo benchmark under strict version control, eliminating reliance on the un-provenanced official checkpoint as the sole primary model.

### Mandatory Research Integrity Safeguards
1. **Source-Inner Split Exclusivity:** All hyperparameter tuning, loss function selection, epoch optimization, and architecture decisions must be conducted **exclusively on the 22/5 source-inner split** (`configs/splits/kuro_source_inner_split_proposed.json`).
2. **Strict TTA-Validation Isolation:** The 6 TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) must **never** be accessed to select source model hyperparameters or choose checkpoints.
3. **Fixed Epoch Final Retraining:** After optimal training recipe selection on the 22/5 split, the final production source model is retrained across all 27 source events using a **fixed epoch schedule**, not by monitoring TTA-validation metrics.
4. **Permanent Test Set Sealing & Manifest Ban:** All 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain strictly sealed and unaccessed. Opening, unpickling, or inspecting `KuroV2_grid_dict_test_0_100.gz` or any other test manifest under `third_party/KuroSiwo/pickle/` is **permanently prohibited**.

---

## 2. Task and Label Semantics

### 2.1 Input Temporal Pairing
- **Primary Proposed Input Pair:** $(x_A, x_B) = (\text{pre\_event\_2}, \text{post\_event})$.
- Sentinel-1 GRD imagery closest in time prior to the flood disaster serves as the reference state ($x_A$), and the disaster observation serves as $x_B$.

### 2.2 Semantic Output Classes
- Class 0: `No Water` (dry ground background)
- Class 1: `Permanent Water` (pre-existing lakes/rivers; non-flood change background)
- Class 2: `Flood Water` (newly inundated flood areas — target change class)
- Class 3: `Invalid / NoData` (strictly excluded from loss computation and evaluation)

### 2.3 Evaluation Mask and Binary Target
- **Valid Evaluation Mask:**
  $$\text{valid} = (\text{valid\_mask} == 1) \land (\text{label} \ne 3)$$
- **Binary Target Extraction:**
  $$\text{target}_{\text{flood}} = (\text{label} == 2)$$
  Pixels with label 0 or 1 are treated as non-flood negative pixels.

### 2.4 Performance Reporting Policy
- **Dual Reporting Requirement:** Both equal-event macro metrics ($\text{Macro F1}$, $\text{Macro Recall}$, $\text{Macro IoU}$) and pooled micro metrics ($\text{Pooled F1}$, $\text{Pooled Precision}$) must be reported side-by-side to reflect both geographic event generalizability and global pixel precision.

---

## 3. Model Architecture Configurations

The protocol specifies two controlled model configurations using the same core SNUNet (Siam-NestedUNet, Hao et al.) backbone:

### 3.1 Model A: Controlled 2-Channel Baseline (`[VV, VH]`)
- **Backbone Architecture:** `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
- **Temporal Inputs:**
  - Branch A ($x_A$): `[pre_event_2_VV, pre_event_2_VH]` (shape: $B \times 2 \times 224 \times 224$)
  - Branch B ($x_B$): `[post_event_VV, post_event_VH]` (shape: $B \times 2 \times 224 \times 224$)
- **Outputs:** Logits tensor of shape $B \times 3 \times 224 \times 224$.
- **Scientific Role:** Serves as the pure radar-only baseline without auxiliary topography data.

### 3.2 Model B: Controlled 3-Channel Ablation (`[VV, VH, slope]`)
- **Backbone Architecture:** `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
- **Temporal Inputs:**
  - Branch A ($x_A$): `[pre_event_2_VV, pre_event_2_VH, slope]` (shape: $B \times 3 \times 224 \times 224$)
  - Branch B ($x_B$): `[post_event_VV, post_event_VH, slope]` (shape: $B \times 3 \times 224 \times 224$)
- **Outputs:** Logits tensor of shape $B \times 3 \times 224 \times 224$.
- **Scientific Role:** Controlled topographic ablation. Slope provides local terrain gradient to suppress false positives on steep terrain.
- **Explicit Provenance Clarification:** Model B is a principled scientific ablation, not an assertion that slope was the channel used in the official checkpoint. The official checkpoint's third channel remains `UNKNOWN` (first conv requires 3 channels `VERIFIED_FROM_DATA`, default GRD config has 2 channels `VERIFIED_FROM_CODE`, upstream commits contain conflicting DEM flags `OBSERVED`, and unmerged community PR suggested slope `COMMUNITY_HYPOTHESIS`).

---

## 4. Detailed Training Hyperparameter Specifications

Every field below is attributed to its verified upstream source or clearly labeled as a project decision:

| Hyperparameter / Dimension | Proposed Protocol Value | Provenance / Evidence Basis | Status |
| :--- | :--- | :--- | :---: |
| **Model Architecture** | SNUNet Siamese U-Net (`base_channel=32`) | `models/method/snunet/snunet.json:8` | `VERIFIED_FROM_CODE` |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.0$) | Checkpoint state dict (`checkpoint_provenance.md:17`) | `VERIFIED_FROM_DATA` |
| **Initial Learning Rate** | $1.0 \times 10^{-3}$ (with linear decay / warmup) | `models/method/snunet/snunet.json:4` | `VERIFIED_FROM_CODE` |
| **Loss Function** | Combined Cross-Entropy + Soft Dice Loss | Upstream uses CE (`train_config.json:11`); CE+Dice is standard CD practice | `CURRENT_PROJECT_DECISION` |
| **Ignore Index** | `ignore_index = 3` (masking invalid NoData) | `utilities.py:140` | `VERIFIED_FROM_CODE` |
| **Class Weighting** | Uniform $[1.0, 1.0, 1.0]$ vs Weighted $[0.37, 14.0, 8.2]$ | `train_config.json:9` uses unweighted; both evaluated on 22/5 split | `UNRESOLVED` |
| **Batch Size** | 32 (effective batch size across GPU) | Upstream config has 128; 32 fits comfortably in 24 GB VRAM with SNUNet | `CURRENT_PROJECT_DECISION` |
| **Patch Spatial Size** | $224 \times 224$ pixels | Verified uniform patch size across KuroSiwo dataset | `VERIFIED_FROM_DATA` |
| **Number of Epochs** | Proposed: 150 epochs (checkpoint-era official change detection duration) | Upstream change detection experiments train for ~150 epochs; 30 epochs is unverified | `SUPPORTED_BY_OFFICIAL_DOCUMENTATION` |
| **Early Stopping** | Patience = 20 epochs on 5 inner-val events | Standard stopping criterion on inner validation F1 | `CURRENT_PROJECT_DECISION` |
| **SAR Clamping** | $[0.0, 0.15]$ | `data_config.json:14`, `Dataset.py:969` | `VERIFIED_FROM_CODE` |
| **SAR Normalization** | $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$ | `data_config.json:16–17` | `VERIFIED_FROM_CODE` |
| **Slope Normalization** | Zero-mean, unit-variance standardized slope | Upstream stats (mean 2.12, std 67.5) contain outliers; requires robust recomputation | `UNRESOLVED` |
| **Data Augmentation** | Random horizontal/vertical flip, random 90° rotation | Implemented in `utilities/augmentations.py` | `CURRENT_PROJECT_DECISION` |
| **Random Seeds** | Seeds `[42, 43, 44]` | 3 distinct random seeds for statistical confidence | `CURRENT_PROJECT_DECISION` |
| **Checkpoint Selection** | Best Macro F1 on 5 source-inner validation events | Evaluated strictly out-of-sample on inner validation set | `CURRENT_PROJECT_DECISION` |
| **Completeness Checks** | Manifest sample verification via `CompletenessTracker` | `src/evaluation/completeness.py` | `VERIFIED_FROM_CODE` |

---

## 5. Phased Source-Training Workflow

```
[27 Source Events]
       │
       ├── Partition via deterministic rule
       ▼
┌─────────────────────────────────┐
│ 22 Inner-Train Events (19,899 s) │
└─────────────────────────────────┘
       │
       ▼ (train Model A & Model B recipes)
┌─────────────────────────────────┐
│ 5 Inner-Val Events (1,563 s)    │◄── Select optimal recipe, loss weighting,
└─────────────────────────────────┘    and convergence epoch schedule
       │
       ▼ (recipe and epoch locked)
┌─────────────────────────────────┐
│ Retrain on ALL 27 Source Events │
│ (Fixed epoch, zero val tuning)  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Final Controlled Source Model   │──► Ready for Week 1 TTA Evaluation
└─────────────────────────────────┘
```

---

## 6. Gate S: Controlled Source Model Acceptance Criteria

Before a newly trained controlled source model is accepted for downstream TTA benchmarking, it must satisfy all Gate S criteria on the 5 source-inner validation events:

1. **Non-Degenerate Flood Detection:** Model achieves inner-validation **Macro F1 $\ge 0.50$** and **Macro Recall $\ge 0.45$**.
2. **Balanced Background Calibration:** False positive flood rate on dry events must not exceed 2.0% of valid pixels.
3. **Training Reproducibility:** Training loss must monotonically decrease across the first 20 epochs; no NaN/Inf gradient events.
4. **Strict Isolation:** Manifest verification must certify that zero TTA-validation or sealed test samples were loaded during training.

---

## 7. Status and Approval Gate

- **Current Status:** `DRAFT — AWAITING USER APPROVAL`
- **Next Step:** Formal review and sign-off by the user on the 8 core training decisions.
