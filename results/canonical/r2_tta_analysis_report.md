# Supplementary Evaluation Report: Test-Time Adaptation on Robust Source Model (R2 + TTA)

**Research Question:** Does Tent or CoTTA provide additional benefit when initialized from R2, compared with deploying the same R2 checkpoint without adaptation?

---

## 1. Six-Condition Consolidated Benchmark Table

| Condition | Macro F1 | Macro IoU | Macro Prec | Macro Rec | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Seed Mean ± SD | Worst Event | Bottom-3 F1 | Pooled F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed R0** | 0.7453 | 0.6035 | 0.8063 | 0.7249 | 0.7645 | 0.7055 | 0.7659 | 0.7453 ± 0.0345 | 0.6299 | 0.6546 | 0.7461 |
| **R0 + Tent** | 0.7628 | 0.6221 | 0.8158 | 0.7390 | 0.7539 | 0.7798 | 0.7549 | 0.7628 ± 0.0147 | 0.6335 | 0.6911 | 0.7543 |
| **R0 + CoTTA** | 0.7669 | 0.6276 | 0.8162 | 0.7462 | 0.7600 | 0.7821 | 0.7586 | 0.7669 ± 0.0132 | 0.6387 | 0.6891 | 0.7464 |
| **Fixed R2** | 0.8086 | 0.6824 | 0.8430 | 0.7908 | 0.8102 | 0.8009 | 0.8149 | 0.8086 ± 0.0071 | 0.7209 | 0.7312 | 0.7757 |
| **R2 + Tent** | 0.8012 | 0.6715 | 0.8526 | 0.7710 | 0.8009 | 0.7995 | 0.8033 | 0.8012 ± 0.0019 | 0.7170 | 0.7346 | 0.7790 |
| **R2 + CoTTA** | 0.8049 | 0.6768 | 0.8524 | 0.7779 | 0.8065 | 0.8023 | 0.8058 | 0.8049 ± 0.0022 | 0.7197 | 0.7366 | 0.7764 |

---

## 2. Statistical Contrasts and Bootstrap Intervals

Pointwise paired event-bootstrap intervals (10,000 resamples, seed 42) across the 10 historical target events:

| Contrast Type | Comparison | Observed $\Delta$ F1 | 95% Bootstrap CI | Benefit ($\ge +0.010$) | Neutral ($|\Delta| < 0.010$) | Harm ($\le -0.010$) | Direction (Pos / Neg / Zero) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary** | R2 + Tent vs. Fixed R2 | **-0.0074** | **[-0.0165, +0.0005]** | 0 | 7 | 3 | 3 / 7 / 0 |
| **Primary** | R2 + CoTTA vs. Fixed R2 | **-0.0038** | **[-0.0136, +0.0044]** | 1 | 7 | 2 | 5 / 5 / 0 |
| **Secondary** | R2 + Tent vs. R0 + Tent | **+0.0384** | **[+0.0236, +0.0542]** | 9 | 1 | 0 | 10 / 0 / 0 |
| **Secondary** | R2 + CoTTA vs. R0 + CoTTA | **+0.0380** | **[+0.0229, +0.0532]** | 8 | 2 | 0 | 9 / 1 / 0 |
| **Contextual** | Fixed R2 vs. Fixed R0 | **+0.0633** | **[+0.0244, +0.1134]** | 9 | 1 | 0 | 9 / 1 / 0 |
| **Contextual** | R0 + Tent vs. Fixed R0 | **+0.0175** | **[-0.0147, +0.0597]** | 3 | 4 | 3 | 4 / 6 / 0 |
| **Contextual** | R0 + CoTTA vs. Fixed R0 | **+0.0216** | **[-0.0121, +0.0667]** | 2 | 6 | 2 | 4 / 6 / 0 |

*Statistical Note: Pointwise 95% confidence intervals are computed using 10,000 paired event-bootstrap resamples conditional on the trained checkpoints. An interval crossing zero indicates that a non-zero effect was not detected in this sample at the 95% level; it does not demonstrate equivalence or statistical parity.*

---

## 3. Ten-Event Detail Breakdown (Seed-Averaged F1 Scores)

| Event ID | Samples | Valid Pixels | Fixed R0 | R0 + Tent | R0 + CoTTA | Fixed R2 | R2 + Tent | R2 + CoTTA | $\Delta$(R2+Tent - R2) | $\Delta$(R2+CoTTA - R2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **205** | 130 | 5,800,497 | 0.6745 | 0.6335 | 0.6387 | 0.7234 | 0.7170 | 0.7197 | -0.0064 | -0.0037 |
| **277** | 1,834 | 91,628,873 | 0.8476 | 0.8586 | 0.8562 | 0.8754 | 0.8761 | 0.8771 | +0.0007 | +0.0017 |
| **321** | 432 | 20,189,261 | 0.7271 | 0.7223 | 0.7264 | 0.7493 | 0.7566 | 0.7640 | +0.0074 | +0.0148 |
| **411** | 149 | 7,178,589 | 0.7819 | 0.7744 | 0.7840 | 0.8418 | 0.8382 | 0.8491 | -0.0036 | +0.0073 |
| **445** | 361 | 16,763,207 | 0.8137 | 0.7798 | 0.7855 | 0.8369 | 0.8236 | 0.8328 | -0.0133 | -0.0041 |
| **561** | 69 | 3,097,419 | 0.8610 | 0.8473 | 0.8551 | 0.8565 | 0.8562 | 0.8616 | -0.0003 | +0.0051 |
| **562** | 77 | 3,444,987 | 0.6299 | 0.8005 | 0.8138 | 0.8490 | 0.8135 | 0.8126 | -0.0355 | -0.0364 |
| **1111002** | 32 | 1,421,262 | 0.6593 | 0.7500 | 0.7621 | 0.8436 | 0.8154 | 0.8195 | -0.0282 | -0.0241 |
| **1111007** | 4,134 | 205,248,884 | 0.7496 | 0.7445 | 0.7449 | 0.7896 | 0.7856 | 0.7862 | -0.0040 | -0.0034 |
| **1111013** | 8,678 | 432,565,804 | 0.7085 | 0.7175 | 0.7021 | 0.7209 | 0.7301 | 0.7261 | +0.0093 | +0.0052 |
| **Mean** | **15,896** | **787,338,783** | **0.7453** | **0.7628** | **0.7669** | **0.8086** | **0.8012** | **0.8049** | **-0.0074** | **-0.0038** |

---

## 4. Execution Telemetry & Compute Overhead

| Stream | Updates Applied | Updates Skipped | Updates Failed | Runtime (s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `r2_tent_s0` | 998 | 0 | 0 | 2561.2s | 9681.2MB |
| `r2_cotta_s0` | 998 | 0 | 0 | 3024.3s | 9899.0MB |
| `r2_tent_s1` | 998 | 0 | 0 | 2561.3s | 9681.2MB |
| `r2_cotta_s1` | 998 | 0 | 0 | 3022.9s | 9899.0MB |
| `r2_tent_s2` | 998 | 0 | 0 | 2561.3s | 9681.2MB |
| `r2_cotta_s2` | 998 | 0 | 0 | 3023.0s | 9899.0MB |

*Note: Runtime reflects total job elapsed wall-clock duration under the evaluation harness (including dataset raster reads, patch extraction, prediction shard persistence, and metric evaluation), not isolated incremental adaptation latency.*
