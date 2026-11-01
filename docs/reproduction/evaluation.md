# Reproduction Guide: Source-Event Evaluation and Statistical Inference

This document details the evaluation protocol for measuring cross-event generalization on held-out source events, paired bootstrap confidence intervals, and tail performance.

---

## 1. Metric Formulations

### Macro-Averaged F1 Score
To ensure equal importance across flood disasters regardless of geographic extent or tile count, primary performance is evaluated as an unweighted macro average over held-out events:

$$\text{Macro F1} = \frac{1}{|E|} \sum_{e \in E} \text{F1}_e$$

where $\text{F1}_e = \frac{2 \cdot \text{TP}_e}{2 \cdot \text{TP}_e + \text{FP}_e + \text{FN}_e}$ computed on event $e$.

### Tail Reliability (Bottom-3 Tail F1)
Captures performance on the most challenging, catastrophic failure events:

$$\text{Bottom-3 F1} = \frac{1}{3} \sum_{k=1}^3 \text{F1}_{(k)}$$

where $\text{F1}_{(1)} \le \text{F1}_{(2)} \le \text{F1}_{(3)}$ are the lowest three event F1 scores in the evaluation set.

---

## 2. Paired Event-Level Bootstrap

Statistical significance is evaluated using non-parametric event-level paired bootstrapping:
- Resample $N=18$ unique events with replacement for $B = 10,000$ iterations (`np.random.RandomState(42)`).
- For each bootstrap replicate $b$, calculate the paired difference $\Delta^{(b)} = \text{Macro F1}_{\text{R2}}^{(b)} - \text{Macro F1}_{\text{R0}}^{(b)}$.
- The 95% two-sided empirical confidence interval is given by the 2.5th and 97.5th percentiles of $\{\Delta^{(b)}\}$.
- One-sided $p$-value: $p = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{(b)} \le 0)$.

---

## 3. Running Confirmatory Analysis

To execute the full statistical evaluation across Folds A and B from saved run outputs:

```bash
python tools/robust_source_training/run_rst7_analysis.py \
    --folda-dir outputs/robust_source_training/rst6 \
    --foldb-dir outputs/robust_source_training/rst6 \
    --output-dir outputs/robust_source_training/rst7
```

