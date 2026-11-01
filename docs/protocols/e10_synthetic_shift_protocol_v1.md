# ChangeGuard Work Package E10: Synthetic Domain-Shift Evaluation Protocol

**Protocol Version:** `e10_synthetic_shift_protocol_v1`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E10 (Section 15)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `FROZEN FOR EXECUTION`  

---

## 1. Physical Motivation & Domain Characterization

In Synthetic Aperture Radar (SAR) remote sensing, earth observation data are governed by radar backscatter physics rather than optical surface reflectance. The primary observed quantities are dual-polarimetric complex backscatter coefficients, converted to linear amplitude $\sqrt{\sigma^0}$ for VV and VH channels.

During real-world satellite operations, target streams frequently experience systematic radiometric shifts caused by:
1. **Orbital & Sensor Calibration Drift:** Minor variations in antenna temperature, transmit power, or receiver gain across acquisition passes.
2. **Atmospheric Attenuation:** Heavy convective precipitation or dense storm fronts causing two-way radar path attenuation.
3. **Soil Moisture & Baseline Surface Wetness Variations:** Regional antecedent moisture modifying the baseline non-flood backscatter across entire landscapes.

These physical processes act symmetrically across pre-event ($T_1$) and post-event ($T_2$) acquisitions as a **multiplicative gain shift** on linear backscatter amplitude.

---

## 2. Mathematical Shift Formulation

Given raw unnormalized linear SAR amplitude $x \in [0.0, 0.15]$ for polarizations $\text{VV}$ and $\text{VH}$:

$$x' = \text{clip}\left((1 + s) \cdot x,\; 0.0,\; 0.15\right)$$

where $s \ge 0$ is the severity gain factor.

The shifted observations are subsequently standardized into model input space using the frozen Kuro Siwo training normalization statistics:

$$z = \frac{x' - \mu}{\sigma}$$

with constants:
- $\mu_{\text{VV}} = 0.0953$, $\sigma_{\text{VV}} = 0.0427$
- $\mu_{\text{VH}} = 0.0264$, $\sigma_{\text{VH}} = 0.0215$

### Why This Shift Preserves Scientific Validity:
- **Semantic Integrity:** Because $s$ is applied uniformly and symmetrically to pre- and post-event images, permanent water and newly inundated water remain significantly darker (lower backscatter) than dry soils and urban structures. The physical definition of flood change remains intact.
- **Covariate Shift Isolation:** The shift shifts the marginal input distribution $p(x)$ away from the training distribution without corrupting or modifying the conditional ground-truth label distribution $p(y \mid \Delta x)$.
- **Controlled Parametrization:** Severity is controlled by a single scalar parameter $s$.

---

## 3. Evaluated Severities

The protocol freezes exactly 4 severity levels:

| Level | Severity Name | Factor $s$ | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| **0** | `clean` | $0.00$ | Unmodified baseline (exact bit-level reproducibility with E8/E9) |
| **1** | `severity_1` | $0.05$ | Mild calibration drift (+5% radiometric gain) |
| **2** | `severity_2` | $0.10$ | Moderate calibration drift (+10% radiometric gain) |
| **3** | `severity_3` | $0.20$ | Severe calibration drift (+20% radiometric gain) |

---

## 4. Test Matrix & Coverage Scope

The synthetic shift matrix evaluates:
1. **Architectures:**
   - Primary Backbone: `SNUNet_ECAM` (12M parameters).
   - Second Backbone: `SiamUnet_diff` (1.35M parameters).
2. **Checkpoints:** 3 source-trained random initializations (Seeds 0, 1, 2) per architecture.
3. **Adaptation Methods:**
   - `source_only`: Frozen baseline.
   - `bn_no_update`: Covariate-matched batch normalization running statistics ($m=1.0$, zero parameter updates).
   - `tent`: Unsupervised entropy minimization of affine BN parameters ($\text{lr}=10^{-4}$).
4. **Events:** Exactly the 6 development events (`[279, 437, 520, 559, 1111003, 1111008]`, 2,826 samples).
5. **Research Integrity Quarantine:**
   - Zero access to sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).
   - Clean baseline cells for SNUNet are matched to and verified against completed E8 confirmation results.

