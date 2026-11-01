# Reproduction Guide: Historical Target Evaluation & TTA Comparison

This document describes the evaluation of source-trained models against test-time adaptation (TTA) algorithms on 10 historical Kuro Siwo target events, as well as the full 16-event historical target extension.

---

## 1. Context and Baseline Overview

Test-Time Adaptation (TTA) methods adapt a pre-trained model during deployment on unlabeled test data:
- **Tent (Wang et al., 2021):** Minimizes prediction entropy by updating BatchNorm affine parameters.
- **CoTTA (Wang et al., 2022):** Continual test-time adaptation with weight-averaged teacher, stochastic augmentation, and threshold filtering.
- **EATA (Niu et al., 2022):** Active sample selection based on prediction confidence and Fisher importance regularizer.
- **Source Anchor (R0 Baseline):** Standard feed-forward evaluation of the source baseline without test-time parameter updates.

In our empirical benchmark on multi-temporal SAR flood change detection, standard test-time adaptation strategies exhibited severe instability under the extreme foreground class imbalance characteristic of SAR scenes (where flood pixels often constitute $< 2\%$ of the image area), frequently suppressing positive flood predictions over the dominant non-flood background.

---

## 2. Running Offline R2 vs TTA Synthesis

ChangeGuard provides an offline analysis script that reconstructs the comparative performance across all 10 target events and 3 seeds using verified archived evaluation runs:

```bash
# Run comparison tool (reads from results/canonical/ or outputs/)
python tools/robust_source_training/run_r2_vs_tta_offline_comparison.py
```

### Outputs Generated:
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_summary.json`
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_event_averaged_paired.csv`
- `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_comparison_figure.png`

### Summary Findings (10-Event TTA Benchmark Subset):
- Grand Macro F1:
  - Source Anchor (R0 Baseline): $0.7453$
  - Tent: $0.7628$
  - CoTTA: $0.7669$
  - R2 (Target-Free Source Training): $\mathbf{0.8086}$ ($\Delta = +0.0633$ vs R0)
- R2 outperforms Tent by $+4.58$ pp ($p < 0.001$, 95% bootstrap CI `[+0.0274, +0.0648]`).
- R2 outperforms CoTTA by $+4.17$ pp ($p < 0.001$, 95% bootstrap CI `[+0.0260, +0.0585]`).
- Operational Characteristics: R2 operates in standard feed-forward evaluation mode without test-time backpropagation passes, online parameter updates, or adaptation memory buffers.

---

## 3. Full 16-Event Historical Target Extension

In addition to the 10-event TTA subset, full-source deployable models trained on all 27 events were evaluated across all 16 historical target events:
- Grand Macro F1 across all 16 events:
  - R0 Baseline: $0.6873 \pm 0.0223$
  - R2 ChangeGuard: $0.7380 \pm 0.0059$
  - Paired Difference: $\mathbf{+0.0507}$ (+5.07 pp, 95% event bootstrap CI `[+0.0234, +0.0845]`)
- Historical Tail (Bottom-3 Events):
  - R0: $0.3889$
  - R2: $0.4607$ ($\Delta = +0.0719$, +7.19 pp)
