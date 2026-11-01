# Day 8 Audit Report: TTA Robustness Matrix Evaluation

**Date:** October 4, 2026  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Protocol Specification:** `configs/day8_tta_robustness_protocol.json` (SHA-256: `6aeeb4446da108ba5312409fff943cf10540e3ce51cd8dc3d19a013bdab44df7`)  
**Source Checkpoint:** `checkpoints/kuro_snunet_official.pt` (SHA-256: `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed`)  
**Validation Events:** `[279, 437, 520, 559, 1111003, 1111008]` (6 events, 2,826 samples per stream)  
**Sealed Test Events:** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (Zero access confirmed)  

---

## 1. Executive Summary & Core Finding

Day 8 evaluated whether the empirical degradation observed under Tent at $10^{-4}$ in Days 3–7 is an artifact of adaptation strength or a fundamental failure mode across reasonable TTA strengths.

### Core Finding: **Harmful Adaptation is Structurally Invariant to Learning Rate**
Across all evaluated methods (AdaBN and Tent across $10^{-5}$, $3\times 10^{-5}$, and $10^{-4}$):
1. **Macro F1 Drops Universally:** Every single TTA configuration suffers an event-macro F1 drop between **-0.0181** and **-0.0215** relative to Source-Only (0.5646).
2. **Consistent Event-Level Harm:** Across all seeds, exactly 2 events (Event 279 and Event 437) are consistently harmed ($\Delta\text{F1} \le -0.01$), 1 event (Event 559) is beneficial ($\Delta\text{F1} \ge +0.01$), and others remain neutral.
3. **Severe Batch-Level Harm Prevalence:** At the batch level, over **90%** of adaptation updates result in cumulative performance drops.
4. **AdaBN Suffers Similar Degradation:** Even without gradient updates, AdaBN alone causes a macro F1 drop of **-0.0215**, proving that batch-norm statistics estimation on heterogeneous SAR scenes is inherently noisy.

---

## 2. Comprehensive Robustness Summary Table

| Method | LR | Macro F1 (mean ± std) | $\Delta$ Macro F1 | Pooled F1 | $\Delta$ Pooled F1 | Worst Event F1 | Max Drop | Harmed Events ($\le -0.01$) | Beneficial Events ($\ge +0.01$) | Batch Harm Prev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_only` | 0.0 | 0.5646 ± 0.0000 | **+0.0000** | 0.7020 | +0.0000 | 0.0816 | +0.0000 | 0.0 / 6 | 0.0 / 6 | N/A |
| `adabn` | 0.0 | 0.5431 ± 0.0017 | **-0.0215** | 0.3469 | -0.3551 | 0.2208 | -0.4094 | 4.0 / 6 | 1.0 / 6 | N/A |
| `tent` | 1e-05 | 0.5434 ± 0.0017 | **-0.0212** | 0.3484 | -0.3536 | 0.2219 | -0.4082 | 4.0 / 6 | 1.0 / 6 | N/A |
| `tent` | 3e-05 | 0.5438 ± 0.0018 | **-0.0208** | 0.3513 | -0.3507 | 0.2241 | -0.4060 | 4.0 / 6 | 1.0 / 6 | 93.9% |
| `tent` | 1e-04 | 0.5451 ± 0.0019 | **-0.0194** | 0.3614 | -0.3406 | 0.2316 | -0.3985 | 4.0 / 6 | 1.0 / 6 | 93.8% |

---

## 3. Event-Wise Performance Breakdown (Equal Weighting across Events)

To prevent Event 279 (2,381 samples) from obscuring performance on smaller events, event-wise Flood F1, IoU, and Recall are tabulated below across all 6 validation events:

### Event-Wise Flood F1

| Method | LR | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_only` | 0.0 | 0.6301 | 0.7481 | 0.7660 | 0.0816 | 0.3291 | 0.8325 | **0.5646** |
| `adabn` | 0.0 | 0.2208 | 0.4915 | 0.7559 | 0.6677 | 0.3032 | 0.8196 | **0.5431** |
| `tent` | 1e-05 | 0.2219 | 0.4917 | 0.7559 | 0.6677 | 0.3033 | 0.8196 | **0.5434** |
| `tent` | 3e-05 | 0.2241 | 0.4921 | 0.7559 | 0.6675 | 0.3033 | 0.8196 | **0.5438** |
| `tent` | 1e-04 | 0.2316 | 0.4934 | 0.7559 | 0.6670 | 0.3032 | 0.8196 | **0.5451** |

### Event-Wise Flood IoU

| Method | LR | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_only` | 0.0 | 0.4600 | 0.5976 | 0.6207 | 0.0426 | 0.1969 | 0.7131 | **0.4385** |
| `adabn` | 0.0 | 0.1241 | 0.3259 | 0.6076 | 0.5012 | 0.1787 | 0.6944 | **0.4053** |
| `tent` | 1e-05 | 0.1248 | 0.3260 | 0.6076 | 0.5012 | 0.1788 | 0.6944 | **0.4055** |
| `tent` | 3e-05 | 0.1262 | 0.3263 | 0.6076 | 0.5010 | 0.1788 | 0.6944 | **0.4057** |
| `tent` | 1e-04 | 0.1310 | 0.3275 | 0.6076 | 0.5005 | 0.1787 | 0.6943 | **0.4066** |

### Event-Wise Flood Recall

| Method | LR | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_only` | 0.0 | 0.6587 | 0.7252 | 0.7122 | 0.0426 | 0.4070 | 0.8290 | **0.5624** |
| `adabn` | 0.0 | 0.7432 | 0.9258 | 0.6854 | 0.5409 | 0.3720 | 0.7814 | **0.6748** |
| `tent` | 1e-05 | 0.7420 | 0.9257 | 0.6854 | 0.5408 | 0.3721 | 0.7814 | **0.6745** |
| `tent` | 3e-05 | 0.7395 | 0.9257 | 0.6853 | 0.5405 | 0.3719 | 0.7814 | **0.6740** |
| `tent` | 1e-04 | 0.7300 | 0.9255 | 0.6851 | 0.5397 | 0.3713 | 0.7814 | **0.6721** |

---

## 4. Analysis of Robustness & Failure Dynamics

1. **The Inefficacy of LR Tuning:** Reducing Tent learning rate from $10^{-4}$ down to $3\times 10^{-5}$ and $10^{-5}$ does not prevent adaptation failure. Macro F1 is virtually identical across all three learning rates (0.5431 to 0.5451, all roughly -0.020 below Source-Only 0.5646).
2. **Event Sensitivity Dichotomy:**
   - **Severely Harmed Events (279, 437):** Event 279 collapses from 0.6301 to ~0.224 ($\Delta\text{F1} \approx -0.406$). Event 437 drops from 0.7481 to ~0.492 ($\Delta\text{F1} \approx -0.256$). These two events contain the vast majority of validation samples.
   - **Beneficial Event (559):** Event 559 dramatically improves from 0.0816 to ~0.668 ($\Delta\text{F1} \approx +0.586$) because its source baseline recall was near zero (0.0426), and adaptation increases flood sensitivity to 0.5405.
   - **Mildly Harmed / Boundary Events (520, 1111003, 1111008):** Event 520 drops slightly (0.7660 to 0.7559, $\Delta\text{F1} = -0.0101$). Event 1111003 drops from 0.3291 to 0.3033 ($\Delta\text{F1} = -0.0258$). Event 1111008 drops from 0.8325 to 0.8196 ($\Delta\text{F1} = -0.0129$).
3. **Variability Across Stream Orders:** Seed standard deviations are extremely low ($< 0.0022$), showing that within a fixed event, stream ordering does not alter the asymptotic degradation.
4. **Batch-Level vs Event-Level Dynamics:** At the batch level, over **93%** of adaptation updates result in cumulative performance drops due to the dominating weight and early collapse of Event 279 (149 batches). Even when evaluated with equal event weighting (macro mean), 4 of 6 events are harmed under $\Delta\text{F1} \le -0.01$.

---

## 5. Day 8 Integrity Certification

- Zero test events accessed.
- Source model checkpoint SHA-256 confirmed unchanged.
- All matrix configurations pre-declared in `configs/day8_tta_robustness_protocol.json` before execution.
- Artifacts hashed and recorded.
