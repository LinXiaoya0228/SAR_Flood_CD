# Reproduction Guide: Historical Target Evaluation & TTA Comparison

This document describes the evaluation of source-trained models against test-time adaptation (TTA) algorithms on 10 historical Kuro Siwo target events.

---

## 1. Context and Baseline Overview

Test-Time Adaptation (TTA) methods adapt a pre-trained model during deployment on unlabeled test data:
- **Tent (Wang et al., 2021):** Minimizes prediction entropy by updating BatchNorm affine parameters.
- **CoTTA (Wang et al., 2022):** Continual test-time adaptation with weight-averaged teacher, stochastic augmentation, and threshold filtering.
- **EATA (Niu et al., 2022):** Active sample selection based on prediction confidence and Fisher importance regularizer.
- **Source Anchor:** Evaluation of the frozen source-trained baseline without test-time updates.

Under severe SAR flood class imbalance (where foreground flood pixels often account for $< 2\%$ of the image), entropy minimization causes catastrophic error collapse—suppressing positive flood predictions entirely to achieve low entropy on the dominant background class.

---

## 2. Running Offline R2 vs TTA Synthesis

ChangeGuard provides an offline analysis script that reconstructs the comparative performance across all 10 target events and 3 seeds using verified archived evaluation runs:

```bash
# Run comparison tool
python tools/robust_source_training/run_r2_vs_tta_offline_comparison.py
```

### Outputs Generated:
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_summary.json`
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_event_averaged_paired.csv`
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_comparison_figure.png`

### Summary Findings:
- Grand Macro F1:
  - Source Anchor (R0 Baseline): $0.7453$
  - Tent: $0.7628$
  - CoTTA: $0.7669$
  - R2 (Target-Free Source Training): $\mathbf{0.8086}$
- R2 outperforms Tent by $+4.58$ pp ($p < 0.001$, 95% CI `[+0.0274, +0.0648]`).
- R2 outperforms CoTTA by $+4.17$ pp ($p < 0.001$, 95% CI `[+0.0260, +0.0585]`).
- Operational Overhead: R2 requires zero test-time backpropagation, zero additional memory, and zero adaptation latency.

