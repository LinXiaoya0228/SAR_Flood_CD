# Final All-Source Controlled Training Protocol (v1.0 Approved)

**Document Version:** 1.0.0-APPROVED-FINAL  
**Status:** `APPROVED — FROZEN FOR ALL-SOURCE CANONICAL SOURCE TRAINING`  
**Project:** ChangeGuard FYP (Final Source-Model Production Phase)  
**Approval Date:** 2026-10-15  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host Target:** `gpu22` (NVIDIA GeForce RTX 3090 24GB)  
**Governing Standard:** IEEE / ChangeGuard Controlled Reproducibility Standard  
**Parent Protocols:**  
- `docs/source_training_protocol_v1.md` (SHA-256: `bc7c2ea8b9a8e62553c200bc1a9f3de326f39edde0f9a0a3fa4843ec61934efa`)  
- `docs/source_training_protocol_v2.md` (SHA-256: `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144`)  
- `docs/week3_source_replication_protocol.md` (SHA-256: `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927`)  
**Parent Replication Report:** `docs/week3_seed012_source_comparison.md` (SHA-256: `eec27eab34a937b504ccc024df10ab29c843cee969627e2de08e3c69c6926fc1`)  
**Recipe Review Document:** `docs/week5_source_recipe_review.md`  
**Approved All-Source Split:** `configs/splits/kuro_source_all27_split_v1.json` (SHA-256: `601b21c81f7f1e26984fe6507a1532dbfa373d72b29346103fb5f0a1ac38d5e1`)  
**Immutable Final Production Lock:** `configs/source/source_training_all27_lock.json`  

---

## 1. Distinction from Model-Selection Protocols & Scientific Objective

This protocol governs the **final production source-model training** phase across all 27 KuroSiwo source events.

### 1.1 Formal Distinction from 22/5 Model-Selection Protocols
- **Phase 1 (Weeks 1–3, Protocols v1/v2):** The 22/5 inner-split protocols were designed strictly for **hypothesis testing and architectural selection** (evaluating whether adding topographic slope improves flood discrimination over dual-polarization radar) and determining the optimal stopping epoch distributions under early stopping.
- **Phase 2 (Week 5, Final Protocol):** This protocol governs the **full-corpus retraining** of the selected primary architecture (Model A `[VV, VH]`) from random initialization across all 27 source events. Because all 27 events are utilized for parameter updates, no disjoint validation set exists within the source partition, and zero validation tuning or early stopping is evaluated.
- **Empirical Hypothesis Resolution:** Across the completed three-seed inner-split replication ($N=3$), the paired difference flipped sign across seeds for all 9 metrics, and Model A achieved higher mean event-macro F1 ($0.7932 \pm 0.0114$) and higher worst-event F1 ($0.7279 \pm 0.0192$) compared to Model B ($0.7913 \pm 0.0062$ and $0.7103 \pm 0.0112$). Model A is therefore selected as the primary source model. The three-seed Model B runs are preserved intact as the completed scientific ablation; no additional full-source Model B runs will be prepared.

---

## 2. Derivation of the Fixed 16-Epoch Training Budget

### 2.1 The Pre-Registered Median Rule
Section 1.1 (Safeguard 4) and Section 7.4 of `docs/source_training_protocol_v1.md`, and `configs/source/snunet_vvvh_v2_fp32.json` (lines 115–120), pre-registered the deterministic final retraining rule:
$$E_{\text{final}} = \text{median}(E_0^*, E_1^*, E_2^*)$$
*"Retrain the final controlled source models from scratch across all 27 source events (21,462 patches) for exactly $E_{\text{final}}$ epochs. During this final retraining, zero validation tuning is permitted."*

### 2.2 Exact Mathematical Calculation
From the canonical Model A three-seed inner-validation artifacts:
1. **Seed 0 (`source_snunet_vvvh_seed0_v3_fp32`):** Best validation epoch $E_0^* = 8$ (best Event-Macro F1 = 0.781741);
2. **Seed 1 (`source_snunet_vvvh_seed1_v3_fp32`):** Best validation epoch $E_1^* = 32$ (best Event-Macro F1 = 0.793257);
3. **Seed 2 (`source_snunet_vvvh_seed2_v3_fp32`):** Best validation epoch $E_2^* = 16$ (best Event-Macro F1 = 0.804616).

Ordering the optimal epochs:
$$\{E_{(1)}^* = 8, \quad E_{(2)}^* = 16, \quad E_{(3)}^* = 32\}$$

The median value is exactly:
$$E_{\text{final}} = \mathbf{16\text{ epochs}}$$

All three production runs will train for **exactly 16 epochs** from random initialization.

---

## 3. Strict Research Integrity & Data Isolation Boundaries

### 3.1 All-Source Training Corpus (27 Events, 21,462 Samples) [FACT]
The all-source training corpus consists of the exact union of the 22 inner-train and 5 inner-validation events:
```text
[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]
```
- Total patches: exactly 21,462 patches.
- Retained patches: 100% of samples retained without filtering or oversampling.

### 3.2 Permanent Quarantine of TTA-Validation Events ($N=6$) [MANDATORY]
```text
[279, 437, 520, 559, 1111003, 1111008]
```
These 6 events are reserved exclusively for future Test-Time Adaptation development and validation. They must **never** be accessed, monitored, loaded, or evaluated during source training.

### 3.3 Permanent Sealing of Test Events ($N=10$) [MANDATORY]
```text
[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
```
Opening, unpickling, reading, or inspecting `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` or any test raster is **strictly prohibited**.

---

## 4. Inherited Frozen Scientific Recipe

All core scientific parameters established in Protocols v1 and v2 are strictly preserved:

1. **Backbone Architecture:** `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
2. **Input Polarizations:** Dual-polarization Sentinel-1 GRD SAR (`[VV, VH]`).
3. **Temporal Pairing:** Pre-event ($x_A$, canonical `pre_event_2`) and Post-event ($x_B$, `post_event`).
4. **Patch Geometry:** $224 \times 224$ pixels at $10\text{ m}$ pixel resolution.
5. **SAR Clamping & Non-finite Replacement:** Clamped to $[0.0, 0.15]$; `NaN` and $+\infty$ replaced with $0.15$, $-\infty$ with $0.0$.
6. **SAR Normalization Constants:**
   $$\mu_{\text{VV}} = 0.0953, \quad \sigma_{\text{VV}} = 0.0427; \qquad \mu_{\text{VH}} = 0.0264, \quad \sigma_{\text{VH}} = 0.0215$$
   - Documented Source: Upstream KuroSiwo benchmark `configs/train/data_config.json:16–17`.
   - Uncertainty Acknowledgment: The numerical values are verified from configuration records, but the exact sample/pixel scope and script used to generate them are not preserved upstream. The constants are retained without modification or recomputation.
7. **Semantic Classes:**
   - 0: `no_water` (background land)
   - 1: `permanent_water` (pre-existing water bodies)
   - 2: `flood_water` (newly inundated flood water — primary detection target)
   - 3: `invalid_no_data` (`ignore_index = 3`)
8. **Double-Masking Rule:** $\text{label}[\text{valid\_mask} == 0] = 3$.
9. **Data Augmentation:** Strictly none (`data_augmentations: false`).
10. **Loss Formulation:** Unweighted $0.5 \times \text{CrossEntropy}(\text{ignore}=3) + 0.5 \times \text{SoftDice}(\epsilon=1.0, \text{ignore}=3)$.
11. **Optimizer:** Standard PyTorch `Adam` ($\eta = 1.0 \times 10^{-3}$, weight decay $= 0.0$, $\beta_1 = 0.9, \beta_2 = 0.999$, constant learning rate, no scheduler).
12. **Numerical Precision:** Full 32-bit single precision (`fp32`, `autocast = False`, `scaler = None`).
13. **Batch Size Policy:** Physical microbatch 8, gradient accumulation 4, effective batch size 32.
14. **Fail-Fast Guards:** Immediate termination on any nonfinite input, logit, loss, gradient, parameter, or BatchNorm buffer.

---

## 5. Three-Seed Production Plan & Multi-Seed Reporting Mandate

To prevent outcome-dependent cherry-picking:
1. **Three Production Seeds:** Formal training will be executed across three independent seeds:
   - **Seed 0:** `source_snunet_vvvh_all27_seed0_v1_fp32`
   - **Seed 1:** `source_snunet_vvvh_all27_seed1_v1_fp32`
   - **Seed 2:** `source_snunet_vvvh_all27_seed2_v1_fp32`
2. **Reporting Mandate:** All three runs must be preserved, evaluated, and reported. Selecting a "best seed" post-hoc is strictly prohibited.
3. **Random Initialization:** Each seed initializes model weights from scratch using PyTorch default initializations under the respective random seed.

---

## 6. Execution & Checkpoint Management Rules

1. **Fixed Training Loop:**
   - Training commences at Epoch 1 and terminates unconditionally upon completing Epoch 16.
   - Validation passes are **completely disabled**; no validation dataset is constructed or iterated.
   - Early stopping counters are completely disabled.
2. **Checkpoint Schedule & Naming:**
   - Resumable checkpoint saved at every epoch: `epoch_{epoch:03d}_checkpoint.pt` and `last_checkpoint.pt`.
   - Final official checkpoint saved at the conclusion of Epoch 16: `final_checkpoint.pt`.
3. **Post-Completion Checksumming:**
   - Upon completion, SHA-256 digests of all 16 epoch checkpoints and `final_checkpoint.pt` must be computed and recorded in `checkpoint_hashes.json` and the execution manifest.
4. **Zero Target-Leakage Invariant:**
   - The Epoch 16 weights are taken directly as the final model. No checkpoint selection on downstream TTA-validation or test sets is permitted.

---

## 7. Versioned Configuration & Split Registry

| Component | File Path | Authoritative SHA-256 Digest |
| :--- | :--- | :--- |
| **All-Source Split Manifest** | `configs/splits/kuro_source_all27_split_v1.json` | `601b21c81f7f1e26984fe6507a1532dbfa373d72b29346103fb5f0a1ac38d5e1` |
| **Model A Seed 0 Config** | `configs/source/snunet_vvvh_all27_seed0_v1_fp32.json` | `6a01a3f8f979cf65746c306dba75d73bfac99a2fc41d60e80e04c679ec661b6f` |
| **Model A Seed 1 Config** | `configs/source/snunet_vvvh_all27_seed1_v1_fp32.json` | `910fbf0110a3e36b88958b3f909042c216e5e9ca12e5600e421ac4dcccc07a4f` |
| **Model A Seed 2 Config** | `configs/source/snunet_vvvh_all27_seed2_v1_fp32.json` | `440e54dd977a7d8b204a829a07cd1318e4b7ea3d16fd0c76670c118424e2c814` |
| **Production Lock File** | `configs/source/source_training_all27_lock.json` | One-way immutable lock anchor |
