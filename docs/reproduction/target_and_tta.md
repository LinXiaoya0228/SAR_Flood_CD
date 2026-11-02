# Reproduction Guide: Historical Target Evaluation & TTA Comparison

This document describes the evaluation of source-trained models against test-time adaptation (TTA) algorithms on 10 historical Kuro Siwo target events, as well as the full 16-event historical target extension and the supplementary evaluation of TTA applied directly to robust R2 source checkpoints.

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

---

## 4. Supplementary Evaluation: TTA Applied to R2 Source Models (R2 + Tent / CoTTA)

To determine whether test-time adaptation provides complementary gains when initialized from a robust source model, we conducted a matched supplementary evaluation applying Tent and CoTTA directly to the frozen Recipe R2 source checkpoints across the 10 Kuro Siwo target events.

### Reproducing the R2 + TTA Evaluation:
```bash
# Execute the 6 formal streams in parallel (Seeds 0, 1, 2 on dedicated GPUs)
./scripts/run_r2_tta_parallel_evaluation.sh

# Or run the statistical analysis engine independently from archived outputs:
python tools/supplementary_r2_tta/analyze_r2_tta_evaluation.py \
    --results_dir outputs/supplementary_r2_tta \
    --output_dir outputs/supplementary_r2_tta/analysis
```

### Canonical Six-Condition Benchmark Results:
| Condition | Macro F1 | Macro IoU | Macro Prec | Macro Rec | Seed Mean ± SD | Worst Event | Bottom-3 F1 | Pooled F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed R0** | 0.7453 | 0.6035 | 0.8063 | 0.7249 | 0.7453 ± 0.0345 | 0.6299 | 0.6546 | 0.7461 |
| **R0 + Tent** | 0.7628 | 0.6221 | 0.8158 | 0.7390 | 0.7628 ± 0.0147 | 0.6335 | 0.6911 | 0.7543 |
| **R0 + CoTTA** | 0.7669 | 0.6276 | 0.8162 | 0.7462 | 0.7669 ± 0.0132 | 0.6387 | 0.6891 | 0.7464 |
| **Fixed R2** | **0.8086** | **0.6824** | 0.8430 | **0.7908** | **0.8086 ± 0.0071** | **0.7209** | 0.7312 | 0.7757 |
| **R2 + Tent** | 0.8012 | 0.6715 | **0.8526** | 0.7710 | 0.8012 ± 0.0019 | 0.7170 | 0.7346 | **0.7790** |
| **R2 + CoTTA** | 0.8049 | 0.6768 | 0.8524 | 0.7779 | 0.8049 ± 0.0022 | 0.7197 | **0.7366** | 0.7764 |

### Primary Paired Contrasts (10,000 Paired Event-Bootstrap Resamples):
- **R2 + Tent vs. Fixed R2:** Observed $\Delta = \mathbf{-0.0074}$ (95% CI: `[-0.0165, +0.0005]`), Benefit: 0, Neutral: 7, Harm: 3.
- **R2 + CoTTA vs. Fixed R2:** Observed $\Delta = \mathbf{-0.0038}$ (95% CI: `[-0.0136, +0.0044]`), Benefit: 1, Neutral: 7, Harm: 2.

### Definitive Conclusion:
Neither Tent nor CoTTA provides additional performance benefit over deploying the unadapted frozen R2 checkpoint. Deploying the frozen R2 model without test-time adaptation delivers higher macro accuracy, superior recall, zero risk of test-time divergence, and avoids the substantial computational cost of test-time backward passes and EMA teacher maintenance.

Canonical artifacts:
- Report: `results/canonical/r2_tta_analysis_report.md`
- Summary JSON: `results/canonical/r2_tta_analysis_summary_v1.json`
- Event Results CSV: `results/canonical/r2_tta_canonical_event_results_v1.csv`
- Paired Deltas CSV: `results/canonical/r2_tta_paired_event_deltas_v1.csv`
- Protocol: `docs/supplementary_r2_tta/r2_tta_evaluation_protocol_v1.md`
- Detailed Closeout: `docs/supplementary_r2_tta/r2_tta_evaluation_closeout_v1.md`
