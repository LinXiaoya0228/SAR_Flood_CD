# RST1 Work Package Closeout Report: Source Audit and Event-Disjoint Folds

**Document version:** 1.0.0
**Closeout date:** 2026-10-26
**Status:** `RST1 COMPLETE — FROZEN SOURCE AUDIT AND EVENT-DISJOINT FOLDS VERIFIED`
**Governing proposal:** `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
**Transition commit:** `9c73e84e853458fcb765422e24a2599c70882ec9`

---

## 1. Executive Summary & Audit Scope

Work Package RST1 conducted a thorough, code-grounded audit of the source dataset, input domain, preprocessing pipeline, sampling mechanics, loss formulation, and augmentation state for the ChangeGuard robust source-training route. In addition, RST1 established the authoritative 27-event source inventory and constructed the frozen, deterministic event-disjoint folds (Groups A, B, and C).

Execution was strictly restricted to bounded, non-training, CPU-only inventory inspection and code analysis. Zero neural network forward passes, zero training updates, zero GPU jobs, and zero test-domain accesses occurred.

---

## 2. Key Technical Findings and Deliverables

### 2.1 Exact Source Event Population and Patch Accounting
- **Total Authorized Source Events:** Exactly **27 events**.
- **Total Eligible Training Patches:** Exactly **21,462 patches** (100% match with historical all-27 split).
- **Missing / Excluded Files:** **0**. All required rasters (`SL2_IVV`, `SL2_IVH`, `MS1_IVV`, `MS1_IVH`, `MK0_MNA`, `MK0_MLU`) are intact and valid.
- **Duplicate Sample IDs:** **0**.
- **Extreme Event Imbalance:**
  - Minimum event size: **21 patches** (Event 496, $0.098\%$ of corpus).
  - Median event size: **306 patches** (Event 518, $1.43\%$ of corpus).
  - Maximum event size: **7,556 patches** (Event 1111009, $35.21\%$ of corpus).
  - Disparity ratio: Event 1111009 contributes **$359.8\times$** more updates than Event 496 under patch-uniform sampling.
  - The top 4 events constitute **$71.01\%$** of all training patches.
- **Artifacts:**
  - `configs/robust_source_training/rst1_source_event_inventory_v1.csv`
  - `configs/robust_source_training/rst1_source_event_inventory_summary_v1.json`

### 2.2 Proven Input Domain and Preprocessing Pipeline
- **Physical Representation:** Sentinel-1 GRD **linear amplitude backscatter** ($\sqrt{I}$), range $[0.0, 0.15]$. Values are NOT in decibels (dB) and NOT calibrated power ($\sigma^0$).
- **Data Types:** Float32 GeoTIFF on disk $\to$ Float32 NumPy array $\to$ Float32 PyTorch tensor.
- **Preprocessing Sequence:**
  1. `np.clip(sar, 0.0, 0.15)`
  2. `np.nan_to_num(sar, nan=0.15, posinf=0.15, neginf=0.0)`
  3. Channel standardization: `(sar - mean) / std` using global constants ($\mu_{\text{VV}}=0.0953, \sigma_{\text{VV}}=0.0427, \mu_{\text{VH}}=0.0264, \sigma_{\text{VH}}=0.0215$).
- **Temporal Pairing:** Canonical `pre_event_2` (prefixes `SL2_IVV`, `SL2_IVH`) paired with `post_event` (`MS1_IVV`, `MS1_IVH`).
- **Masking & Labels:** Double masking enforces `label[valid_mask == 0] = 3` (`ignore_index = 3`).
- **Input Isolation:** Labels and valid masks never enter model inputs. Model receives paired tensors `xA` and `xB` of shape $(B, 2, 224, 224)$.
- **Artifact:** `docs/robust_source_training/rst1_input_domain_and_preprocessing_audit_v1.md`.

### 2.3 Existing Sampler, Loss, and Augmentation State
- **Sampler:** Uniform patch sampling without replacement (`RandomSampler(replacement=False)`), effective batch size 32 (micro-batch 8 $\times$ accumulation 4). Large events heavily dominate optimizer steps.
- **Loss:** `ControlledSourceLoss` combining unweighted $0.5 \times \text{CrossEntropyLoss}(\text{ignore}=3) + 0.5 \times \text{SoftDiceLoss}(\epsilon=1.0, \text{ignore}=3)$. Absent-class stable with smooth $\epsilon=1.0$.
- **Augmentation:** **Strictly none** (`data_augmentations: false`).
- **Artifact:** `docs/robust_source_training/rst1_sampler_loss_augmentation_audit_v1.md`.

### 2.4 Matched R0 Specification
- Reconstructed the exact historical source-training recipe (SNUNet-ECAM, Adam $\eta=10^{-3}$, FP32, 16 fixed epochs, no early stopping).
- Unavoidable differences for fold-specific R0: training restricted to 18 authorized fold events per fold (14,967 patches for Fold C discovery; 13,302 patches for Fold A confirmation; 14,655 patches for Fold B confirmation).
- **Artifact:** `configs/robust_source_training/rst1_matched_r0_spec_v1.json`.

### 2.5 Frozen Event-Disjoint Folds
- Partitioned the 27 source events into three outcome-blind groups of 9 events each using a deterministic constrained greedy algorithm:
  - **Group A (Confirmatory Target A):** 9 events (`[1111009, 273, 118, 498, 567, 554, 421, 174, 496]`), **8,160 patches** ($38.02\%$).
  - **Group B (Confirmatory Target B):** 9 events (`[1111004, 555, 502, 470, 1111010, 518, 267, 324, 417]`), **6,807 patches** ($31.72\%$).
  - **Group C (Discovery Target C):** 9 events (`[427, 1111011, 275, 1111005, 497, 147, 1111006, 130, 492]`), **6,495 patches** ($30.26\%$).
- Mutual disjointness verified. Zero overlap. Zero target events included.
- **Artifacts:**
  - `configs/robust_source_training/rst1_event_folds_v1.json`
  - `docs/robust_source_training/rst1_event_fold_design_v1.md`
  - `tools/robust_source_training/generate_rst1_event_folds.py`

### 2.6 Compute and Storage Resource Estimates
- One 16-epoch SNUNet run on 18 events requires **~2.5 to 2.8 GPU-hours** and **~4.9 GB VRAM** (measured historical rate: 889.3 s/epoch on 21,462 patches).
- Total compute budget:
  - Discovery Stage (4 runs): **11.04 GPU-hours** (~2.8 wall-clock hours across 4 RTX 3090s).
  - Confirmatory Stage (12 runs): **30.90 GPU-hours** (~8.2 wall-clock hours across 4 RTX 3090s).
  - Primary route total: **~42.4 GPU-hours** (~11.2 wall-clock hours).
- Total checkpoint/log storage: **~42 GB** (against 3.8 TB available disk space on `/usr1`).
- **Artifact:** `docs/robust_source_training/rst1_compute_storage_estimate_v1.md`.

---

## 3. Strict Boundary and Integrity Assertions

1. **Zero New Model Training:** No neural network parameters were updated, no models were trained, and no checkpoints were created.
2. **Zero Model Inference:** No evaluation scripts or model forward passes were executed.
3. **Zero Forbidden Event Access:** None of the 16 previously observed development (`[279, 437, 520, 559, 1111003, 1111008]`) or former test (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) events were opened, loaded, queried, or inspected.
4. **Historical Immutability:** Historical E0–E11 directories, configs, logs, and outputs remain completely unmodified.
5. **RST2 Status:** **RST2 has not started.**

---

## 4. Unresolved Issues

None. The input domain is definitively proven to be linear amplitude backscatter, the 27 source events are 100% complete and validated on disk, and the event-disjoint folds are mathematically frozen.
