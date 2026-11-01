# RST5 Confirmatory Decision Rule and Statistical Evaluation Protocol

## 1. Executive Summary

This document specifies the pre-registered statistical decision criteria for the confirmatory evaluation of **Recipe R2** versus **Baseline R0** across Confirmation Folds A and B and random seeds 0, 1, and 2.

To guarantee scientific integrity and prevent post-hoc hypothesis shifting, all criteria (P1, P2, P3), aggregation formulas, variance definitions, and outlier conditions are **frozen prior to launching any confirmatory run in RST6**.

---

## 2. Mathematical Aggregation Definitions

### 2.1 Event-Macro Metrics
For a given recipe $r \in \{\text{R0}, \text{R2}\}$, fold $f \in \{\text{A}, \text{B}\}$, and seed $s \in \{0, 1, 2\}$, evaluation produces binary confusion metrics for each held-out event $e \in \mathcal{E}_f$ ($|\mathcal{E}_f| = 9$):
$$\text{F1}_{r, f, s, e} = \frac{2 \cdot \text{TP}_{r, f, s, e}}{2 \cdot \text{TP}_{r, f, s, e} + \text{FP}_{r, f, s, e} + \text{FN}_{r, f, s, e}}$$

The fold event-macro F1 is the unweighted arithmetic mean across the 9 events:
$$\overline{\text{F1}}_{r, f, s} = \frac{1}{9} \sum_{e \in \mathcal{E}_f} \text{F1}_{r, f, s, e}$$

### 2.2 Seed-Mean & Grand Mean
The seed-mean within a fold is:
$$\overline{\text{F1}}_{r, f} = \frac{1}{3} \sum_{s=0}^2 \overline{\text{F1}}_{r, f, s}$$

The grand mean across both confirmatory folds (representing all 18 unique confirmatory events) is:
$$\overline{\text{F1}}_r = \frac{1}{2} \left( \overline{\text{F1}}_{r, \text{A}} + \overline{\text{F1}}_{r, \text{B}} \right)$$

The overall effect size of Recipe R2 over Baseline R0 is:
$$\Delta \overline{\text{F1}} = \overline{\text{F1}}_{\text{R2}} - \overline{\text{F1}}_{\text{R0}}$$

### 2.3 Across-Seed Standard Deviation
For each recipe $r$ and fold $f$, the sample standard deviation across the three independent initialization seeds is:
$$s_{r, f} = \sqrt{\frac{1}{3 - 1} \sum_{s=0}^2 \left( \overline{\text{F1}}_{r, f, s} - \overline{\text{F1}}_{r, f} \right)^2}$$

The average seed standard deviation across folds is:
$$\overline{s}_r = \frac{s_{r, \text{A}} + s_{r, \text{B}}}{2}$$

The relative reduction in across-seed variance is:
$$\text{RelRed}_s = \frac{\overline{s}_{\text{R0}} - \overline{s}_{\text{R2}}}{\overline{s}_{\text{R0}}}$$

### 2.4 Tail Reliability (Worst & Bottom-3 Events)
For each seed and fold, events in $\mathcal{E}_f$ are sorted in ascending order of F1:
$$\text{F1}_{(1)} \le \text{F1}_{(2)} \le \dots \le \text{F1}_{(9)}$$
- Worst-Event F1: $\text{F1}_{(1)}$
- Bottom-3 Mean F1: $\frac{1}{3} \sum_{k=1}^3 \text{F1}_{(k)}$

Tail improvement is evaluated on the grand mean of the bottom-3 F1 across seeds and folds:
$$\Delta \text{B3} = \overline{\text{B3}}_{\text{R2}} - \overline{\text{B3}}_{\text{R0}}$$

---

## 3. Confirmatory Decision Criteria

### 3.1 Criterion P1: Average Improvement
Recipe R2 satisfies **P1** if:
1. **Magnitude**: $\Delta \overline{\text{F1}} \ge +0.010$ (at least 1.0 percentage point improvement in grand mean event-macro F1).
2. **Event Breadth ("Not driven by only one event")**:
   Let $\Delta_e = \frac{1}{3} \sum_s (\text{F1}_{\text{R2}, e, s} - \text{F1}_{\text{R0}, e, s})$ be the seed-averaged delta for event $e$.
   Let $e^* = \arg\max_e \Delta_e$ be the single event with the largest positive gain.
   Excluding $e^*$, the remaining 17 events must still exhibit a positive mean delta:
   $$\frac{1}{17} \sum_{e \ne e^*} \Delta_e > 0.000$$
3. **No Widespread Precision/Recall Collapse**:
   - Grand mean event-macro precision must not decrease by $\ge 0.050$ relative to R0 ($\overline{\text{Prec}}_{\text{R2}} - \overline{\text{Prec}}_{\text{R0}} \ge -0.050$).
   - Grand mean event-macro recall must not decrease by $\ge 0.050$ relative to R0 ($\overline{\text{Rec}}_{\text{R2}} - \overline{\text{Rec}}_{\text{R0}} \ge -0.050$).
   - No more than 4 of the 18 confirmatory events may experience simultaneous drops in both precision and recall exceeding $0.030$.

### 3.2 Criterion P2: Tail Reliability Improvement
Recipe R2 satisfies **P2** if:
1. **Tail Gain**: $\Delta \text{B3} \ge +0.020$ (at least 2.0 percentage points improvement on bottom-3 event mean F1, or worst-event delta $\Delta \text{Worst} \ge +0.020$).
2. **Mean Preservation**: $\Delta \overline{\text{F1}} \ge -0.005$ (grand mean event-macro F1 does not degrade by more than 0.5 percentage points).

### 3.3 Criterion P3: Seed Stability
Recipe R2 satisfies **P3** if:
1. **Variance Reduction**: Relative across-seed standard deviation reduction $\text{RelRed}_s \ge 20\%$ ($\text{RelRed}_s \ge 0.20$).
2. **Consistency Across Folds**: The standard deviation of R2 must be lower than R0 in **both** folds independently ($s_{\text{R2}, \text{A}} < s_{\text{R0}, \text{A}}$ and $s_{\text{R2}, \text{B}} < s_{\text{R0}, \text{B}}$).
3. **Mean Preservation**: $\Delta \overline{\text{F1}} \ge -0.005$.

---

## 4. Sensitivity & Supplementary Analyses

In addition to primary macro metrics, RST7 will report:
1. **Pooled F1 and IoU**: Validating that aggregate pixel-level accuracy across the entire confirmatory corpus is preserved.
2. **Per-Event Delta Distribution**: Number of improved ($\Delta_e \ge +0.01$), harmed ($\Delta_e \le -0.01$), and neutral ($|\Delta_e| < 0.01$) events.
3. **Paired Permutation / Bootstrap Tests**: Reporting non-parametric 95% confidence intervals for $\Delta \overline{\text{F1}}$ across the 18 events.

---

## 5. Frozen Immutability Guarantee

These decision rules, thresholds, and formulas are **immutable**. They are recorded in `configs/robust_source_training/rst5_confirmation_decision_rule_v1.json` and will not be adjusted after viewing RST6 results.
