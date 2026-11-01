# ChangeGuard Plan V3 Work Package E2.2: Preflight, Reuse & Input-Domain Audit

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `1a8b07c044ffd783552c266fe2b2a8986b2fbafc` (`main`)  
**Work Package:** `E2.2` (Augmentation Grid Screening)  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) Section 7.2  
**Measurement Contract:** [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  

---

## 1. Executive Summary & Purpose

Work Package E2.2 evaluates the strong augmentation grid for test-time consistency adaptation on Sentinel-1 SAR flood imagery.

Under Master Execution Plan V3 Section 7.2, the augmentation screen is executed under fixed parameters:
- **Confidence Reference:** Predefined primary reference $\tau = 0.90$ (locked prior to evaluation; not selected post-hoc from E2.1).
- **Factors:** Checkpoint Seed 0, Stream Order A, `reset_per_event`, batch size 16 across all six Development events ($N = 2,826$ samples).
- **One-Factor Comparison:** Augmentation strategy is the only intended experimental variable.
- **Sealed Test Quarantine:** Strictly zero access to the 10 reserved test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

This preflight document:
1. Verifies the clean starting repository commit;
2. Demonstrates exact equivalence for reusing the E2.1 primary reference cell as `A0`;
3. Formulates and verifies geometric transformations and coordinate realignment for `A1`, `A2`, and `A3`;
4. Conducts an end-to-end input-domain certification audit for `A4` (SAR intensity/noise perturbation) to determine whether physical execution is justified or must be formally blocked.

---

## 2. Git Status & Clean Tree Verification

- **Branch:** `main`
- **Base Commit:** `1a8b07c044ffd783552c266fe2b2a8986b2fbafc`
- **Subject:** `feat(e2.1): complete confidence grid screening and integrity verification`
- **Tracked Working Tree State:** Completely clean (verified via `git status --porcelain`).
- **Execution Target:** Logical GPU 0 (`cuda:0`, NVIDIA RTX 3090, 24 GiB VRAM), Python `vfm_env`.

---

## 3. Cell A0: Exact Reuse Equivalence Audit

The reference condition `A0` corresponds to Weak View = Identity and Strong View = Horizontal Flip under predefined threshold $\tau = 0.90$.

In Work Package E2.1, cell `E2_CONF_90` was executed and verified under the Plan V3 measurement contract (`CACHE_SCHEMA_V1`). We audit its exact equivalence to the required `A0` condition across all 13 governing dimensions:

| # | Dimension | E2.1 Primary Cell (`E2_CONF_90`) | E2.2 Required Cell `A0` | Equivalence Status |
| :-: | :--- | :--- | :--- | :---: |
| 1 | **Model & Checkpoint** | SNUNet_ECAM Seed 0 (`247d3584...`) | SNUNet_ECAM Seed 0 (`247d3584...`) | **EXACT MATCH** |
| 2 | **Stream Order** | Order A (`[279, 437, 520, 559, 1111003, 1111008]`) | Order A (`[279, 437, 520, 559, 1111003, 1111008]`) | **EXACT MATCH** |
| 3 | **Reset Mode** | `reset_per_event` | `reset_per_event` | **EXACT MATCH** |
| 4 | **Target Events & Samples** | Dev 6 events, $N = 2,826$ patches | Dev 6 events, $N = 2,826$ patches | **EXACT MATCH** |
| 5 | **Batch Sizing & Tail** | Batch size 16, tail processed as-is | Batch size 16, tail processed as-is | **EXACT MATCH** |
| 6 | **Confidence Threshold** | $\tau = 0.90$ (predefined reference) | $\tau = 0.90$ (predefined reference) | **EXACT MATCH** |
| 7 | **Weak View** | Identity ($x_A, x_B$) | Identity ($x_A, x_B$) | **EXACT MATCH** |
| 8 | **Strong View** | Paired Horizontal Flip (`torch.flip(..., dims=[-1])`) | Paired Horizontal Flip (`torch.flip(..., dims=[-1])`) | **EXACT MATCH** |
| 9 | **Consistency Loss** | One-way KL divergence from detached weak to strong | One-way KL divergence from detached weak to strong | **EXACT MATCH** |
| 10 | **Masking Semantics** | Adaptation: $(MK0\_MNA == 1) \land (P_{\max} \ge 0.90)$; Eval: Double-masking | Adaptation: $(MK0\_MNA == 1) \land (P_{\max} \ge 0.90)$; Eval: Double-masking | **EXACT MATCH** |
| 11 | **Optimization** | SGD ($\text{lr}=1.25\times 10^{-4}$, $\mu=0.9$, $\text{wd}=0.0$, BN affine) | SGD ($\text{lr}=1.25\times 10^{-4}$, $\mu=0.9$, $\text{wd}=0.0$, BN affine) | **EXACT MATCH** |
| 12 | **Telemetry & Schema** | Extended recorder `CACHE_SCHEMA_V1` (per-class counts, grad norms, BN disp, prob maps) | Extended recorder `CACHE_SCHEMA_V1` | **EXACT MATCH** |
| 13 | **Git Commit** | Commit `1a8b07c` | Commit `1a8b07c` | **EXACT MATCH** |

**Determination:**  
Cell `A0` is proven **100% equivalent** to `E2_CONF_90`. In accordance with Plan V3 Rule 4 and Rule 5, cell `A0` is reused from `outputs/e2_confidence_grid/v1/E2_CONF_90` without re-running redundant GPU compute, preserving full reproducibility.

---

## 4. Geometric Augmentation Specifications (A1, A2, A3)

### 4.1 Paired Synchronization Invariant
For all geometric augmentations, transformations $T$ must be applied identically and synchronously to both pre-event ($x_A$) and post-event ($x_B$) SAR channels, as well as the sensory validity mask ($V$):
$$x_{A,\text{strong}} = T(x_A), \quad x_{B,\text{strong}} = T(x_B), \quad V_{\text{strong}} = T(V)$$

### 4.2 Inversion into Original Coordinate Frame
To evaluate consistency loss against the weak view (which resides in the original coordinate frame), strong-branch logits $\mathbf{z}_{\text{strong}}$ must be inverted back into native coordinates:
$$\mathbf{z}_{\text{strong},\text{aligned}} = T^{-1}(\mathbf{z}_{\text{strong}})$$
All selection masks, probabilities, and losses are computed strictly in the native/original coordinate frame.

### 4.3 Cell Specifications:
1. **`A1_VFLIP` (Paired Vertical Flip):**
   - Forward: $T(X) = \text{torch.flip}(X, \text{dims}=[-2])$
   - Inverse: $T^{-1}(Z) = \text{torch.flip}(Z, \text{dims}=[-2])$
   - Overlap: 100% geometric overlap.

2. **`A2_TRANSLATION` (Small Integer Translation):**
   - Fixed displacement budget locked before execution: $(\Delta y, \Delta x) = (+4, +4)$ pixels ($\approx 1.5\%$ of $256\times 256$ patch).
   - Forward: $X_{\text{strong}}[:, :, dy:, dx:] = X[:, :, :H-dy, :W-dx]$ with zero padding on border strips $[0:dy, :]$ and $[:, 0:dx]$.
   - Inverse Realignment: $Z_{\text{aligned}}[:, :, :H-dy, :W-dx] = Z_{\text{strong}}[:, :, dy:, dx:]$.
   - **Declared Overlap Mask:** $M_{\text{overlap}}[:, :H-dy, :W-dx] = 1$, and $0$ elsewhere.
   - **Padding Exclusion Invariant:** Consistency is evaluated strictly on:
     $$V_{\text{active}} = V_{\text{native}} \land M_{\text{overlap}}$$
     Zero-padding pixels are strictly excluded from consistency loss, selection counts, and gradient steps.

3. **`A3_INDEPENDENT_WEAK` (Independent Weak Geometric Views):**
   - Standard weak-strong adaptation compares a weak view to a strong view. Comparing identity to identity is only a zero-loss sanity check ($L_{\text{KL}} \equiv 0$).
   - To make the weak-view comparison mathematically meaningful, A3 evaluates consistency between **two distinct, independent weak geometric views**:
     - Weak View 1 ($W_1$): Paired Horizontal Flip (`torch.flip(..., dims=[-1])`)
     - Weak View 2 ($W_2$): Paired Vertical Flip (`torch.flip(..., dims=[-2])`)
   - Both views are inverted back into the original native coordinate frame ($W_1^{-1}$ and $W_2^{-1}$).
   - Primary predictions are derived from the realigned $W_1$ predictions, ensuring all evaluation and cache export occur in the native coordinate frame.
   - Consistency loss is computed between realigned $W_1$ probabilities and realigned $W_2$ log-probabilities.

---

## 5. Cell A4: SAR Intensity / Noise Perturbation Audit & Certification

### 5.1 Preprocessing Pipeline Inspection
Inspection of `src/data/evaluation_dataset.py` (lines 174–179) and `configs/source/snunet_vvvh_v2_fp32.json` reveals the exact operational sequence:
1. Raw Sentinel-1 GRD TIFF rasters are read as `float32`.
2. Truncation: Values are clamped to $[0.0, 0.15]$ (`sar_clamp_max = 0.15`).
3. Non-finite handling: `NaN` / $+\infty \to 0.15$; $-\infty \to 0.0$.
4. Z-Score Standardization:
   $$\tilde{x}_{\text{VV}} = \frac{x_{\text{VV}} - 0.0953}{0.0427}, \quad \tilde{x}_{\text{VH}} = \frac{x_{\text{VH}} - 0.0264}{0.0215}$$
5. Standardized tensors $\tilde{x} \in \mathbb{R}^{2 \times 256 \times 256}$ (spanning approximately $[-2.5, +3.0]$) are delivered to the model.

### 5.2 Physical and Domain Inconsistencies & Inverse Transform Assessment
Applying synthetic intensity or noise perturbation to SAR inputs requires satisfying strict physical and numerical constraints:
1. **Multiplicative Speckle Physics:** Physical radar speckle is modeled as a multiplicative perturbation on power/intensity data ($\eta \sim \Gamma(L, L)$ with mean 1 and variance $1/L$) or amplitude ($\text{Nakagami-}m$). Multiplicative speckle is physically and semantically valid **only on physical non-negative intensity or amplitude values** ($I \ge 0$).
2. **Invalidity in Standardized Space:** In the adapter runtime, inputs are delivered as z-score standardized tensors $\tilde{x} = (x - \mu)/\sigma$, which have negative values (spanning approximately $[-2.5, +3.0]$). Applying multiplicative speckle directly in z-score-standardized space is semantically invalid because standardized features are zero-mean unit-variance statistical variables, not physical non-negative intensity values (e.g., scaling a negative standardized value by $\eta > 1$ moves it further negative, whereas on physical backscatter it should scale radiometric energy).
3. **Assessment of Inverse Standardization ($x = \tilde{x} \cdot \sigma + \mu$):**
   We explicitly evaluate whether applying the known inverse affine transform $x = \tilde{x} \cdot \sigma + \mu$ can recover a sufficiently justified linear-domain proxy for A4:
   - **Clamped Range vs. True Radiometry:** While $x = \tilde{x} \cdot \sigma + \mu$ maps $\tilde{x}$ back to the interval $[0.0, 0.15]$, this domain is an artificial truncated dynamic range (clamped at $0.15$ in `src/data/evaluation_dataset.py`), not raw calibrated radar backscatter.
   - **Missing Number of Looks ($L$) & Sensor Metadata:** Generating $\eta \sim \Gamma(L, L)$ requires knowing the effective number of looks ($L$) of the Sentinel-1 GRD product (or its equivalent noise equivalent sigma zero, NESZ). KuroSiwo provides only 8-bit normalized GRD patches without calibration metadata, incident angles, or sensor noise parameters, making any choice of $L$ uncalibrated and arbitrary.
   - **Absence of a Defensible Predeclared Perturbation Magnitude:** Source model pretraining (`docs/week5_source_recipe_review.md` line 95) explicitly froze `data_augmentations: false`. No prior study or validation set establishes a defensible, non-destructive noise magnitude or distribution for test-time adaptation on this frozen backbone. Introducing an arbitrary noise variance at test time risks unpredictable domain shift rather than meaningful consistency regularization.
   Consequently, even with the inverse transform, A4 cannot be formulated as a defensible, physically grounded perturbation.

### 5.3 Formal Decision on Cell A4
Plan V3 Section 7.2 explicitly states:
> *"If the input-domain/semantic assumption cannot be justified, keep A4 blocked with its exact reason; do not implement an invalid transform to fill a table."*

**Verdict:** Cell **`E2_AUG_A4`** is formally marked **`BLOCKED`**.  
**Reason:** Multiplicative speckle directly in standardized space is semantically invalid on zero-mean features; recovering a clamped linear-domain proxy via $x = \tilde{x} \cdot \sigma + \mu$ remains blocked due to missing calibration/look metadata ($L$), irreversible truncation at $0.15$, and the absence of a defensible predeclared perturbation magnitude.

---

## 6. Locked E2.2 Augmentation Grid Summary

| Cell ID | Augmentation Strategy | Weak View | Strong View | Threshold | Status | Execution / Reuse Plan |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`E2_AUG_A0`** | Horizontal Flip Reference | Identity | Horizontal Flip | $\tau=0.90$ | **`REUSED`** | Reused from E2.1 `E2_CONF_90` (100% equivalent). |
| **`E2_AUG_A1`** | Paired Vertical Flip | Identity | Vertical Flip | $\tau=0.90$ | **`EXECUTABLE`** | Fresh execution on Dev 6 events. |
| **`E2_AUG_A2`** | Paired Small Integer Translation | Identity | Shift $(+4, +4)$, overlap only | $\tau=0.90$ | **`EXECUTABLE`** | Fresh execution on Dev 6 events. |
| **`E2_AUG_A3`** | Independent Weak Views | Weak 1 (H-Flip) | Weak 2 (V-Flip) | $\tau=0.90$ | **`EXECUTABLE`** | Fresh execution on Dev 6 events. |
| **`E2_AUG_A4`** | SAR Intensity Perturbation | N/A | Multiplicative noise | $\tau=0.90$ | **`BLOCKED`** | Blocked due to input-domain invalidity. |

