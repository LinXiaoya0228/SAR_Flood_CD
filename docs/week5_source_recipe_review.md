# Week 5A Controlled Source-Training Recipe Review & Freeze Specification

**Date:** 2026-10-15  
**Project:** ChangeGuard FYP (Week 5 Phase A: Source Recipe Review & Freeze Planning)  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Git Branch & HEAD:** `main` at commit `84c801cf950f53458972496dde98b7891323f641` (`84c801c`)  
**Git Working Tree Status:** Tracked files clean; untracked/unstaged Week 4 report files (`docs/interim_report_final_v2.md`, `docs/interim_report_final_v3.md`, `exports/week4/*`, etc.) are preserved intact without modification.  
**Governing Standard:** IEEE / ChangeGuard Controlled Reproducibility Standard  

---

## 1. Epistemological Classification & Preflight Evidence Inventory

In compliance with the ChangeGuard Research Integrity Protocol, all statements and specifications in this document are tagged with epistemological categories:
- **[FACT]:** Directly verified from immutable on-disk artifacts (manifests, CSVs, logs, checkpoint binaries, Git tree).
- **[APPROVED RULE]:** Explicit procedural rule or formula already adopted and approved in pre-registered protocols.
- **[PROPOSAL]:** New proposal or engineering recommendation based on source-inner empirical findings.
- **[INTERPRETATION]:** Bounded scientific conclusions derived from verified quantitative evidence.

### 1.1 Targeted Evidence Registry & Cryptographic Verification [FACT]

| Document / Artifact Path | SHA-256 Digest | Status & Role |
| :--- | :--- | :--- |
| `docs/source_training_protocol_v1.md` | `bc7c2ea8b9a8e62553c200bc1a9f3de326f39edde0f9a0a3fa4843ec61934efa` | Frozen Parent Protocol v1.0 |
| `docs/source_training_protocol_v2.md` | `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144` | Approved FP32 Recovery Protocol v2.0 |
| `docs/week3_source_replication_protocol.md` | `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927` | Approved Week 3 Replication Protocol |
| `docs/week3_seed012_source_comparison.md` | `eec27eab34a937b504ccc024df10ab29c843cee969627e2de08e3c69c6926fc1` | Canonical 3-Seed Comparison Report |
| `configs/source/source_training_v2_lock.json` | `7139dbdd2031fd5654b4414299bf80e20295ebdba13bb507b64a59f10a50fbd3` | Protocol v2 Production Lock |
| `configs/source/source_training_week3_replication_lock.json` | `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677` | Week 3 Three-Seed Replication Lock |
| `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | Frozen 22/5 Data Split (Alternative A1) |
| `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | Frozen Train-Only Metric Slope Stats |
| `configs/source/snunet_vvvh_v2_fp32.json` | `528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3` | Model A Seed 0 Config |
| `configs/source/snunet_vvvh_seed1_v2_fp32.json` | `fdcad256e5710588b193dadd902ab2442630c1c8f494c43bf424c4087ae8dccf` | Model A Seed 1 Config |
| `configs/source/snunet_vvvh_seed2_v2_fp32.json` | `9cd06aae01319079bb597125324cef212a2f0ffa02cab71853b57dea0c35c852` | Model A Seed 2 Config |
| `outputs/week3_source_replication/comparison_seed0_seed1_seed2/source_seed012_comparison.csv` | `f02629ecea56ecf620dd5f5b15a7bfe26eb320679b01235c265ce91316e95398` | Master 6-Run Evaluation Metrics |

---

## 2. Actual Evidence-Backed Source Recipe

This section details the complete specification of the source-training recipe as implemented and verified across all completed Week 2 and Week 3 controlled runs.

### 2.1 Model Architecture & Input Channels [FACT]
- **Primary Model Candidate:** Model A (`[VV, VH]`).
  - Architecture: `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
  - Temporal Branches: Siamese architecture with branch $x_A$ (Pre-event SAR, specifically canonical acquisition `pre_event_2`) and branch $x_B$ (Post-event SAR, acquisition `post_event`).
  - Patch Geometry: $224 \times 224$ pixels at native $10\text{ m}$ pixel resolution ($2.24\text{ km} \times 2.24\text{ km}$ footprint).
  - Number of Parameters: 12,056,067 trainable parameters.
- **Retained Scientific Ablation:** Model B (`[VV, VH, slope]`).
  - Architecture: `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
  - Static Channel Sharing: The identical static metric slope raster ($1 \times 224 \times 224$) is concatenated to both temporal branches ($x_A$: `[pre_vv, pre_vh, slope]`, $x_B$: `[post_vv, post_vh, slope]`).

### 2.2 SAR Input Preprocessing Pipeline [FACT]
The SAR input preprocessing is strictly ordered and verified in `src/data/source_dataset.py` (lines 256–264) and `src/data/kuro_dataset.py`:
1. **Raw Backscatter:** Sentinel-1 GRD linear amplitude (square root of intensity).
2. **Clamping:** Values clamped to the interval $[0.0, 0.15]$:
   $$\text{sar}_{\text{clamped}} = \text{clip}(\text{sar}_{\text{raw}}, 0.0, 0.15)$$
3. **Non-Finite Replacement:** Replace `NaN`, $+\infty$ with $0.15$ and $-\infty$ with $0.0$ prior to standardization:
   $$\text{sar}_{\text{finite}} = \text{nan\_to\_num}(\text{sar}_{\text{clamped}}, \text{nan}=0.15, \text{posinf}=0.15, \text{neginf}=0.0)$$
4. **Z-Score Normalization:**
   $$\text{sar}_{\text{norm}} = \frac{\text{sar}_{\text{finite}} - \mu_{\text{sar}}}{\sigma_{\text{sar}}}$$

### 2.3 Normalization Values & Origin [FACT, UNCERTAINTY ACKNOWLEDGED]
- **SAR Normalization Parameters:**
  - Documented Constants: $\mu_{\text{VV}} = 0.0953$, $\sigma_{\text{VV}} = 0.0427$; $\mu_{\text{VH}} = 0.0264$, $\sigma_{\text{VH}} = 0.0215$.
  - Origin & Evidence: Directly documented in upstream KuroSiwo configuration (`configs/train/data_config.json:16–17`), alongside `train_acts` which lists the 27 source events.
  - Generation Scope Uncertainty: While the config lists the 27 source events under `train_acts`, the exact calculation code, sample filtering (e.g. whether all 27 events were unmasked, or zero-flood patches excluded), or pixel weighting that originally generated these numerical values is not recorded in upstream artifacts. Therefore, while these documented constants are retained for strict parity and reproducibility, their precise generation scope remains an acknowledged upstream uncertainty.
  - Policy: Retain existing constants without recomputation or scanning of dataset samples.
- **Topographic Slope Normalization Parameters (Model B Only):**
  - $\mu_{\text{slope}} = 1.492574^\circ$, $\sigma_{\text{slope}} = 2.372311^\circ$
  - **Origin:** Computed strictly over the 955,664,099 valid slope pixels across the 22 approved inner-training events under Policy S2 (physical metric cell spacing $\Delta x=\Delta y=10\text{ m}$, projected CRS verification, DEM nodata sanitization) in `artifacts/week1/source_train_slope_statistics.json`.

### 2.4 Loss Formulation [FACT]
Unweighted combination of Cross-Entropy and Soft Dice loss:
$$L_{\text{total}} = 0.5 \cdot L_{\text{CE}} + 0.5 \cdot L_{\text{Dice}}$$

1. **Cross-Entropy Loss:**
   $$L_{\text{CE}} = \text{CrossEntropyLoss}(\text{logits}, y, \text{ignore\_index}=3)$$
2. **Soft Dice Loss:**
   Evaluated across the 3 valid semantic classes $c \in \{0, 1, 2\}$ on the valid pixel mask $M = (y \ne 3)$:
   $$L_{\text{Dice}} = \frac{1}{3} \sum_{c=0}^2 \left( 1 - \frac{2 \sum_{i \in M} p_{c, i} t_{c, i} + \epsilon}{\sum_{i \in M} p_{c, i}^2 + \sum_{i \in M} t_{c, i}^2 + \epsilon} \right)$$
   where $p_{c, i} = \text{softmax}(\text{logits})_{c, i}$, $t_{c, i} = \mathbb{I}(y_i == c)$, and smoothing constant $\epsilon = 1.0$.
3. **Microbatch Dice Semantics:** The Soft Dice loss is computed over each physical microbatch of 8 patches, with gradient accumulation over 4 steps.

### 2.5 Semantic Class Mapping, Ignore Index & Double-Masking [FACT]
- **Class 0 (`no_water`):** Dry background land.
- **Class 1 (`permanent_water`):** Pre-existing permanent water bodies.
- **Class 2 (`flood_water`):** Newly inundated flood water (primary target class).
- **Class 3 (`invalid_no_data`):** Missing, corrupted, or unmapped pixels (`ignore_index = 3`).
- **Double-Masking Rule:** Verified in `src/data/source_dataset.py` (line 294):
  $$\text{label}[\text{valid\_mask} == 0] = 3$$
  Pixels with invalid data in raster mask `MK0_MNA` are strictly forced to class 3 and generate zero loss and zero gradient.

### 2.6 Augmentation Policy [FACT]
- **Policy:** **Strictly NO data augmentations** (`data_augmentations: false`).
- **Verification:** Verified in `src/data/source_dataset.py` (no random flips, rotations, cropping, or photometric jitter; patches are deterministically loaded at native coordinates). Corresponds to the default upstream KuroSiwo setting (`configs/train/data_config.json:13`).

### 2.7 Optimization Hyperparameters & Schedule [FACT]
- **Optimizer:** Standard PyTorch `Adam` (`torch.optim.Adam`).
- **Learning Rate:** $\eta = 1.0 \times 10^{-3}$ ($0.001$).
- **Weight Decay:** $0.0$.
- **Betas:** PyTorch defaults $(\beta_1 = 0.9, \beta_2 = 0.999)$.
- **Scheduler:** None (`lr_scheduler = None`, constant learning rate throughout training).
- **Prohibited Optimizers:** `AdamW` is explicitly prohibited by protocol.

### 2.8 Numerical Precision & FP32 Fail-Fast Safeguards [FACT]
- **Precision Mode:** Full 32-bit single precision (`torch.float32`).
- **Mixed Precision:** Completely disabled (`autocast = False`, `GradScaler = None`).
- **Fail-Fast Guards:** Immediate termination with status `FAILED_NUMERICAL` upon encountering any `NaN` or `Inf` in:
  1. Input tensors ($x_A, x_B$);
  2. Forward model logits;
  3. Pre-backward total loss;
  4. Post-backward gradients prior to optimizer step;
  5. Model parameters;
  6. BatchNorm running statistics (`running_mean`, `running_var`);
  7. Validation outputs and loss.
- **Atomic Checkpoint Commits:** Checkpoints are written to `.tmp` files and committed via `os.replace` only after passing strict parameter finiteness audits.

### 2.9 Batch Size & Gradient Accumulation [FACT]
- **Physical Microbatch Size:** 8 patches.
- **Gradient Accumulation Steps:** 4.
- **Effective Optimization Batch Size:** $8 \times 4 = 32$ patches.
- **Divisibility Constraint:** $32 \pmod 8 == 0$.

### 2.10 Random Seeds & Determinism [FACT]
- Replicated seeds: `seed = 0`, `seed = 1`, `seed = 2`.
- Deterministic seeding applied to PyTorch CPU, PyTorch CUDA (`manual_seed_all`), NumPy, and DataLoader worker initialization (`worker_seed = (seed + worker_id + 1) % (2**32)`).

### 2.11 27 Source Events & Data Exclusion Assertions [FACT]
- **22 Inner-Train Events:** `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]` (19,178 patches).
- **5 Inner-Validation Events:** `[267, 275, 497, 518, 1111005]` (2,284 patches).
- **All 27 Source Events:** Union of 22 inner-train and 5 inner-val events equals exactly 21,462 patches.
- **Exclusion Assertions:**
  - Quarantined TTA-Validation Events ($N=6$, strictly excluded): `[279, 437, 520, 559, 1111003, 1111008]`.
  - Sealed Test Events ($N=10$, strictly excluded and unopened): `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.
  - Disjointness: $\text{Source} \cap \text{TTA-Val} = \emptyset$, $\text{Source} \cap \text{Sealed-Test} = \emptyset$, $\text{TTA-Val} \cap \text{Sealed-Test} = \emptyset$.

---

## 3. Completed Model A Seeds 0/1/2 Summary

All three pre-registered replications of Model A (`[VV, VH]`) completed without human intervention, early stopping normally after 20 epochs of patience on inner-validation event-macro Flood F1 [FACT].

### 3.1 Model A Three-Seed Quantitative Results Table [FACT]

| Seed | Canonical Run ID | Best Val Epoch | Stopping Epoch | Event-Macro Flood F1 | Pooled Flood F1 | Worst-Event F1 (ID) | Macro Prec / Rec / IoU | Best Checkpoint SHA-256 | Best Checkpoint Output Path |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | `source_snunet_vvvh_seed0_v3_fp32` | 8 | 28 | 0.781741 | 0.791289 | 0.706420 (267) | 0.7568 / 0.8483 / 0.6461 | `750360fca8fdbef0ecfda8529bfdc3d815f93dd04686e41c10b928087b79ca9e` | `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/best_checkpoint.pt` |
| **1** | `source_snunet_vvvh_seed1_v3_fp32` | 32 | 52 | 0.793257 | 0.796832 | 0.733998 (267) | 0.7767 / 0.8447 / 0.6608 | `985a766f8ddcfd14fdaf5ade6bc6257f7e21f5ffac2483f4d8ea95536d02ed5d` | `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/best_checkpoint.pt` |
| **2** | `source_snunet_vvvh_seed2_v3_fp32` | 16 | 36 | 0.804616 | 0.818081 | 0.743370 (267) | 0.7801 / 0.8585 / 0.6763 | `0ab764ffd360efd4b9391a264702b3ba8cbb45e6e42817c0b8d384348c223066` | `outputs/week3_source_replication/source_snunet_vvvh_seed2_v3_fp32/best_checkpoint.pt` |

### 3.2 Three-Seed Statistical Aggregation ($N=3$) [CALC]
- **Event-Macro Flood F1:** $0.793205 \pm 0.011438$ (Range: $0.022875$, $[0.781741, 0.804616]$).
- **Pixel-Pooled Flood F1:** $0.802067 \pm 0.014142$ (Range: $0.026792$, $[0.791289, 0.818081]$).
- **Worst-Event Flood F1 (Event 267):** $0.727929 \pm 0.019208$ (Range: $0.036950$, $[0.706420, 0.743370]$).
- **Macro Precision:** $0.771198 \pm 0.012554$.
- **Macro Recall:** $0.850486 \pm 0.007134$.
- **Macro IoU:** $0.661050 \pm 0.015085$.

### 3.3 Comparison with Model B Ablation & Scientific Decision [INTERPRETATION]
- Model B (`[VV, VH, slope]`) achieved Mean Event-Macro F1 of $0.791344 \pm 0.006165$ and Mean Worst-Event F1 of $0.710277 \pm 0.011150$.
- The paired performance difference ($\Delta = \text{Model B} - \text{Model A}$) flipped sign across seeds for all 9 tracked evaluation metrics (Seed 0: $+0.0153$; Seed 1: $-0.0011$; Seed 2: $-0.0198$).
- In Seed 2, Model A outperformed Model B across 100% (5 of 5) of validation events, and on challenging Event 267, Model B's precision dropped to 0.5476, yielding F1 of 0.6976 (below 0.70).
- **Outcome:** The hypothesis that topographic slope provides a stable, generalizable benefit is not supported by the pre-registered three-seed evidence. Under the governing protocol rule, **Model A (`[VV, VH]`) is selected as the primary source-model architecture**. Model B is retained strictly as an exploratory ablation. No claim of asymptotic or statistically significant superiority is made from $N=3$.

---

## 4. Resolution of Transition to All-Source Training

This section resolves the key operational choices required to transition from inner-split exploration to canonical all-source training.

### 4.1 Epoch Budget Resolution [APPROVED RULE]
- **Governing Protocol Rule:** An explicit transition rule was approved in `docs/source_training_protocol_v1.md` (Section 1.1 Safeguard 4, Section 7.4) and recorded in `configs/source/snunet_vvvh_v2_fp32.json` (lines 115–120):
  $$E_{\text{final}} = \text{median}(E_0^*, E_1^*, E_2^*)$$
  *"Retrain the final controlled source models from scratch across all 27 source events (21,462 patches) for exactly $E_{\text{final}}$ epochs. During this final retraining, zero validation tuning is permitted."*
- **Application to Model A Completed Seeds:**
  - Seed 0 optimal inner-validation epoch: $E_0^* = 8$
  - Seed 1 optimal inner-validation epoch: $E_1^* = 32$
  - Seed 2 optimal inner-validation epoch: $E_2^* = 16$
  - Sorted best epochs: $[8, 16, 32]$
  - **Resulting Locked Epoch Budget:**
    $$E_{\text{final}} = \text{median}(8, 16, 32) = \mathbf{16\text{ epochs}}$$
- **Rejection of 150-Epoch Ceiling:** The 150-epoch parameter in Protocol v2 was strictly an upper ceiling for early stopping patience (20 epochs) on the inner-validation set. For fixed-budget full training on the 27-event union where no validation set exists and zero validation tuning is permitted, early stopping cannot be evaluated. Rather than arbitrarily running to the 150-epoch search ceiling (which was never intended as an unmonitored fixed schedule), the protocol explicitly pre-registers a fixed epoch schedule determined by the median of the optimal inner-validation epochs ($E_{\text{final}} = 16$).
- **Classification:** **Approved Protocol Rule**.

### 4.2 Normalization Statistics Policy [FACT, UNCERTAINTY ACKNOWLEDGED]
- **SAR Normalization Policy:**
  - Parameters: Documented constants $\mu_{\text{VV}} = 0.0953, \sigma_{\text{VV}} = 0.0427; \mu_{\text{VH}} = 0.0264, \sigma_{\text{VH}} = 0.0215$.
  - Origin: Upstream KuroSiwo benchmark `configs/train/data_config.json:16–17`, which lists 27 `train_acts`.
  - Evidence vs. Uncertainty: The numerical values are verified from configuration records, but the exact sample/pixel scope and script used to generate them are not preserved upstream.
  - Policy Decision: **Retain existing frozen statistics**. They are preserved across all ChangeGuard source and reference pipelines. No recomputation is permitted or required, and no dataset scanning will be performed.
- **Topographic Slope Normalization Policy (Model B Ablation Only):**
  - Existing statistics in `artifacts/week1/source_train_slope_statistics.json` ($\mu=1.492574^\circ, \sigma=2.372311^\circ$) were computed over the 22 inner-train events.
  - For Model A (`[VV, VH]`), slope is omitted entirely, making this decision irrelevant to the primary model.
  - For Model B ablation: Retaining the frozen 22-event statistics preserves strict cryptographic parity with the Week 2/3 ablation record. No additional full-source Model B runs will be prepared now.

### 4.3 Launcher Changes Required for 27-Event Fixed-Epoch Training [PROPOSAL]
Currently, `tools/train_source_model.py` and `src/data/source_dataset.py` are coupled to the 22/5 inner split. The following bounded modifications are required:
1. **Dataset Split Mode (`src/data/source_dataset.py`):**
   - Current: `split` only accepts `"train"` (22 events, 19,178 patches) or `"val"` (5 events, 2,284 patches).
   - Required: Add an all-source mode (e.g. `split="all_source"` or `split="all"`), which loads all 27 source events (exactly 21,462 patches) while enforcing strict runtime assertions of disjointness from the 6 quarantined TTA-val and 10 sealed test events.
2. **Training Launcher Validation Decoupling (`tools/train_source_model.py`):**
   - Current: The launcher unconditionally instantiates `val_dataset` and `val_loader`, runs a validation pass after every epoch, evaluates early stopping patience against 20 non-improvements, and saves `best_checkpoint.pt` based on validation Macro F1.
   - Required for Full-Source Mode:
     - Add flag `--all-source` or `--no-val`.
     - Disable validation loop (as all 27 source events are in the training set and zero validation tuning is permitted).
     - Disable early stopping patience counter.
     - Train for exactly $E_{\text{final}} = 16$ epochs.
     - Save checkpoint per epoch and save the final epoch 16 checkpoint as `final_checkpoint.pt`.

### 4.4 Checkpoint Selection Rule [APPROVED RULE]
- Under Section 7.4 of Protocol v1 and the research integrity protocol:
  - The model weights obtained at the conclusion of the locked final epoch (**Epoch 16**) are taken directly and unconditionally as the final canonical controlled source model.
  - **Zero selection on downstream sets:** Quarantined TTA-validation events ($N=6$) and sealed test events ($N=10$) must **never** be used to select checkpoints, filter runs, or stop training.
  - No post-hoc checkpoint searching across intermediate epochs is permitted.

---

## 5. Compact Readiness Table

The table below summarizes every core setting, its exact value, its evidence path, and its readiness status.

| Setting Name | Exact Value | Evidence Path | Status |
| :--- | :--- | :--- | :---: |
| **Model Architecture** | `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)` | `configs/source/snunet_vvvh_v2_fp32.json:14–19` | **FROZEN** |
| **Input Channels** | Dual-polarization Sentinel-1 SAR `[VV, VH]` | `configs/source/snunet_vvvh_v2_fp32.json:20–23` | **FROZEN** |
| **Input Patch Size** | $224 \times 224$ pixels ($10\text{ m}$ spacing) | `src/data/source_dataset.py:25` | **FROZEN** |
| **Temporal Branches** | Branch $x_A$: `pre_event_2`; Branch $x_B$: `post_event` | `configs/source/snunet_vvvh_v2_fp32.json:21` | **FROZEN** |
| **SAR Clamping** | Min: $0.0$, Max: $0.15$ | `configs/source/snunet_vvvh_v2_fp32.json:29–32` | **FROZEN** |
| **SAR Non-finite Value** | Replace `NaN` and $+\infty$ with $0.15$; $-\infty$ with $0.0$ | `configs/source/snunet_vvvh_v2_fp32.json:33–37` | **FROZEN** |
| **SAR Normalization** | $\mu_{\text{VV}}=0.0953, \sigma_{\text{VV}}=0.0427; \mu_{\text{VH}}=0.0264, \sigma_{\text{VH}}=0.0215$ | `configs/source/snunet_vvvh_v2_fp32.json:38–41` | **FROZEN** |
| **Semantic Classes** | 0: `no_water`, 1: `perm_water`, 2: `flood_water`, 3: `invalid` | `configs/source/snunet_vvvh_v2_fp32.json:43–58` | **FROZEN** |
| **Ignore Index** | `3` (`ignore_index = 3`) | `configs/source/snunet_vvvh_v2_fp32.json:51` | **FROZEN** |
| **Double-Masking Rule** | `label[valid_mask == 0] = 3` | `src/data/source_dataset.py:294` | **FROZEN** |
| **Data Augmentation** | None (`data_augmentations: false`) | `src/data/source_dataset.py` | **FROZEN** |
| **Loss Function** | $0.5 \times \text{CE}(\text{ignore}=3) + 0.5 \times \text{SoftDice}(\epsilon=1.0, \text{ignore}=3)$ | `configs/source/snunet_vvvh_v2_fp32.json:59–69` | **FROZEN** |
| **Optimizer** | PyTorch standard `Adam` | `configs/source/snunet_vvvh_v2_fp32.json:71` | **FROZEN** |
| **Learning Rate** | $\eta = 1.0 \times 10^{-3}$ ($0.001$) | `configs/source/snunet_vvvh_v2_fp32.json:72` | **FROZEN** |
| **Optimizer Betas** | $(0.9, 0.999)$ | `configs/source/snunet_vvvh_v2_fp32.json:74` | **FROZEN** |
| **Weight Decay** | $0.0$ | `configs/source/snunet_vvvh_v2_fp32.json:73` | **FROZEN** |
| **LR Scheduler** | None (constant learning rate) | `configs/source/snunet_vvvh_v2_fp32.json:75` | **FROZEN** |
| **Numerical Precision** | Full 32-bit float (`fp32`, autocast=False, scaler=None) | `configs/source/snunet_vvvh_v2_fp32.json:76–79` | **FROZEN** |
| **Physical Microbatch** | 8 patches | `configs/source/snunet_vvvh_v2_fp32.json:86` | **FROZEN** |
| **Accumulation Steps** | 4 steps | `configs/source/snunet_vvvh_v2_fp32.json:87` | **FROZEN** |
| **Effective Batch Size** | 32 patches | `configs/source/snunet_vvvh_v2_fp32.json:85` | **FROZEN** |
| **Source Events Corpus** | All 27 events ($22\text{ inner-train} \cup 5\text{ inner-val} = 21,462\text{ patches}$) | `configs/splits/kuro_source_inner_split_v1.json` | **FROZEN** |
| **Quarantined Events** | 6 TTA-Val events, 10 Sealed-Test events (100% excluded) | `configs/splits/kuro_source_inner_split_v1.json:19–24` | **FROZEN** |
| **Retraining Epochs** | Exactly $E_{\text{final}} = \text{median}(8, 32, 16) = \mathbf{16\text{ epochs}}$ | `docs/source_training_protocol_v1.md:173–178` | **FROZEN (FORMULA)** |
| **Retraining Validation**| Disabled (zero validation tuning permitted) | `docs/source_training_protocol_v1.md:179` | **FROZEN** |
| **Checkpoint Selection**| Final fixed Epoch 16 checkpoint taken unconditionally | `docs/source_training_protocol_v1.md:179` | **FROZEN** |
| **Retraining Seed(s)** | Seeds 0, 1, and 2 decided (all 3 trained & reported; no best-seed selection) | Week 5 Implementation Plan | **DECIDED / FROZEN** |
| **Model B Full Retrain** | Retain existing 3-seed inner comparison as ablation; no full-source runs | Week 5 Implementation Plan | **DECIDED / FROZEN** |

---

## 6. Minimal Files Needed for Freeze & Role of Model B Ablation

### 6.1 Required New Files for All-Source Training Freeze
To execute and lock the final 27-event source model training cleanly without disturbing existing inner-split assets:
1. **Full-Source Split / Manifest Config:**
   - File: `configs/splits/kuro_source_all27_v1.json`
   - Content: Formal declaration of all 27 source events (21,462 patches) as the training set, with empty validation set, and explicit assertions guaranteeing zero overlap with the 6 TTA-validation events and 10 sealed-test events.
2. **Production Training Configurations for Seeds 0, 1, 2:**
   - Files: `configs/source/snunet_vvvh_all27_seed0_v1_fp32.json`, `configs/source/snunet_vvvh_all27_seed1_v1_fp32.json`, `configs/source/snunet_vvvh_all27_seed2_v1_fp32.json`.
   - Content: Setting `max_epochs: 16`, `early_stopping_patience: null` (disabled), `validation: false`, `split_config: configs/splits/kuro_source_all27_v1.json`, targeting `outputs/week5_source_all27/source_snunet_vvvh_all27_seed{0,1,2}_v1_fp32/`.
3. **Week 5 Production Lock File:**
   - File: `configs/source/source_training_all27_lock.json`
   - Content: Authoritative SHA-256 digests of the governing protocol, split config, the three seed configs, and preflight checksums.
4. **Targeted Code Enhancements:**
   - Extend `src/data/source_dataset.py` to support `split="all_source"` loading exactly 21,462 patches.
   - Extend `tools/train_source_model.py` to support `--no-val` / fixed-epoch execution without early stopping.

### 6.2 Role of Model B Topographic Ablation
- **Primary Scientific Function:** Model B (`[VV, VH, slope]`) was formulated as a controlled scientific ablation to evaluate the hypothesis that ancillary topographic slope improves flood delineation and suppresses false positives in complex terrain.
- **Completed Empirical Finding:** Across the pre-registered three-seed replication ($N=3$), the paired difference flipped sign across all 9 metrics, worst-event F1 deteriorated by $-0.0177$, and mean event-macro F1 was lower than radar-only Model A ($0.7913$ vs $0.7932$).
- **Role in Subsequent Phases:**
  - Model A (`[VV, VH]`) is the **sole primary source model candidate** for downstream Test-Time Adaptation (TTA) benchmarking.
  - Model B remains an **exploratory ablation**. The existing three-seed inner-validation checkpoints are retained as the completed scientific ablation artifact. No additional full-source Model B runs will be prepared now, maintaining focus and computational discipline on the primary Model A pipeline.

---

## 7. Research Integrity Record

### 7.1 Files Inspected
- `git status`, `git log`
- `docs/source_training_protocol_v1.md`
- `docs/source_training_protocol_v2.md`
- `docs/week3_source_replication_protocol.md`
- `docs/week3_seed012_source_comparison.md`
- `configs/source/source_training_v2_lock.json`
- `configs/source/source_training_week3_replication_lock.json`
- `configs/splits/kuro_source_inner_split_v1.json`
- `artifacts/week1/source_train_slope_statistics.json`
- `configs/source/snunet_vvvh_v2_fp32.json`
- `configs/source/snunet_vvvh_seed1_v2_fp32.json`
- `configs/source/snunet_vvvh_seed2_v2_fp32.json`
- `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/run_manifest.json`
- `outputs/week3_source_replication/comparison_seed0_seed1_seed2/source_seed012_comparison.csv`
- `tools/train_source_model.py`
- `src/data/source_dataset.py`
- `docs/source_data_pipeline_audit.md`

### 7.2 Commands Run
- `git status --short`
- `git log -n 3 --oneline`
- `grep -rn "0.0953" configs/ src/ tools/ docs/`
- `grep -n -E "val_dataset|early_stopping|max_epochs|no_val|all_27|parse_args" tools/train_source_model.py`
- `grep -i -n -E "augment|flip|crop|rotate" configs/source/snunet_vvvh_v2_fp32.json src/data/source_dataset.py`
- `grep -i -n "augment" docs/*.md`
- `sha256sum docs/source_training_protocol_v1.md docs/source_training_protocol_v2.md docs/week3_source_replication_protocol.md docs/week3_seed012_source_comparison.md configs/source/source_training_v2_lock.json configs/source/source_training_week3_replication_lock.json configs/splits/kuro_source_inner_split_v1.json artifacts/week1/source_train_slope_statistics.json configs/source/snunet_vvvh_v2_fp32.json configs/source/snunet_vvvh_seed1_v2_fp32.json configs/source/snunet_vvvh_seed2_v2_fp32.json`
- `python -m unittest discover -s tests -p "test_*.py"`

### 7.3 Files Created / Changed
- Created: `docs/week5_source_recipe_review.md`
- Preserved: All Week 4 documents, drafts, audits, exports, and figures remain untouched.

### 7.4 Solemn Data Non-Access Declaration
In accordance with the ChangeGuard Research Integrity Protocol:
- **Zero access** occurred to the 6 quarantined TTA-validation events:
  ```text
  [279, 437, 520, 559, 1111003, 1111008]
  ```
- **Zero access** occurred to the 10 sealed test events:
  ```text
  [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
  ```
- Test manifest `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` was never opened, loaded, or inspected.
- All conclusions and recipe specifications are derived strictly and exclusively from verified source-domain artifacts.

