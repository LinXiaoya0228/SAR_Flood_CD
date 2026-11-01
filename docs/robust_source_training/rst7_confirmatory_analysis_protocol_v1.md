# RST7 Confirmatory Analysis Protocol

## 1. Scope and Governing Principles

This protocol specifies the statistical analysis plan, aggregation formulas, bootstrap resampling procedure, and decision criteria for **RST7 -- Confirmatory Analysis and Route Decision** of the ChangeGuard robust source-training route (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`).

This protocol is permanently frozen before inspecting any confirmatory evaluation results across Groups A or B. It strictly operationalizes the confirmatory decision contract established in `configs/robust_source_training/rst5_confirmation_decision_rule_v1.json`.

---

## 2. Input Data and Provenance

The confirmatory analysis ingests the 12 verified runs produced during RST6:
- **Baseline Recipe R0** (matched source-training baseline):
  - Fold A: seeds 0, 1, 2 (`rst6_r0_snunet_folda_s0_v1`, `s1_v1`, `s2_v1`)
  - Fold B: seeds 0, 1, 2 (`rst6_r0_snunet_foldb_s0_v1`, `s1_v1`, `s2_v1`)
- **Selected Method Recipe R2** (M1 event-balanced sampling + M2 training-only inverse-square-root class-balanced cross-entropy):
  - Fold A: seeds 0, 1, 2 (`rst6_r2_snunet_folda_s0_v2`, `s1_v2`, `s2_v2`)
  - Fold B: seeds 0, 1, 2 (`rst6_r2_snunet_foldb_s0_v2`, `s1_v2`, `s2_v2`)

Each run evaluated exactly 9 held-out events, producing a canonical matrix of 108 event-level result records.
The 16 historical target events `[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]` remain strictly untouched and excluded from all inputs.

---

## 3. Statistical Unit of Replication and Aggregation Hierarchy

### 3.1 Unit of Scientific Replication
The scientific unit of statistical independence is the **held-out event**, not individual sample tiles or training seeds.
There are 18 unique held-out events across the two confirmatory folds:
- **Fold A Evaluated Events (9)**: `118, 174, 273, 421, 496, 498, 554, 567, 1111009`
- **Fold B Evaluated Events (9)**: `267, 324, 417, 470, 502, 518, 555, 1111004, 1111010`

### 3.2 Metric Definitions
The primary evaluation metric is the **binary F1 score for newly inundated flood water (Class 2)**:
```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
IoU = TP / (TP + FP + FN)
```
If denominator is 0: Precision = 0.0, Recall = 0.0, F1 = 0.0, IoU = 0.0 (except if TP = FP = FN = 0, where score = 1.0).

### 3.3 Multi-Seed Event Mean
For each unique event e in {1..18} and recipe R in {R0, R2}, the multi-seed estimate is the unweighted arithmetic mean across the three seeds:
```text
F1_bar(e, R) = (1/3) * sum_{s in {0, 1, 2}} F1(e, s, R)
P_bar(e, R)  = (1/3) * sum_{s in {0, 1, 2}} Precision(e, s, R)
R_bar(e, R)  = (1/3) * sum_{s in {0, 1, 2}} Recall(e, s, R)
```

### 3.4 Paired Per-Event Deltas
For each event e, the paired effect of Recipe R2 over R0 is:
```text
Delta_F1(e) = F1_bar(e, R2) - F1_bar(e, R0)
Delta_P(e)  = P_bar(e, R2) - P_bar(e, R0)
Delta_R(e)  = R_bar(e, R2) - R_bar(e, R0)
```

### 3.5 Fold-Level and Grand Mean Aggregation
- **Fold Event-Macro F1 for seed s**:
  ```text
  FoldMacro(F, s, R) = (1/9) * sum_{e in F} F1(e, s, R)
  ```
- **Seed Mean Fold Macro F1**:
  ```text
  FoldMacro_bar(F, R) = (1/3) * sum_{s=0}^2 FoldMacro(F, s, R)
  ```
- **Across-Seed Standard Deviation**:
  ```text
  SD(F, R) = sqrt( (1/2) * sum_{s=0}^2 (FoldMacro(F, s, R) - FoldMacro_bar(F, R))^2 )
  ```
- **Grand Mean Event-Macro F1**:
  ```text
  GrandMean(R) = (1/2) * [FoldMacro_bar(FoldA, R) + FoldMacro_bar(FoldB, R)]
               = (1/18) * sum_{e in A U B} F1_bar(e, R)
  ```
- **Grand Mean Delta**:
  ```text
  Delta_GrandMean = GrandMean(R2) - GrandMean(R0)
  ```

---

## 4. Tail and Sensitivity Metrics

### 4.1 Worst-Event and Bottom-3 Tail Analysis
- **Worst-Event F1**:
  ```text
  WorstEvent(R) = min_{e in A U B} F1_bar(e, R)
  Delta_WorstEvent = WorstEvent(R2) - WorstEvent(R0)
  ```
- **Bottom-3 Mean F1**:
  Let bottom3(R) be the set of 3 events with lowest F1_bar(e, R):
  ```text
  Bottom3Mean(R) = (1/3) * sum_{e in bottom3(R)} F1_bar(e, R)
  Delta_Bottom3Mean = Bottom3Mean(R2) - Bottom3Mean(R0)
  ```

### 4.2 Single-Event Driver Test
A gain is considered driven by a single outlier event if excluding the event with the largest positive delta reduces the remaining grand mean delta to 0.000 or below:
```text
e_star = argmax_{e} Delta_F1(e)
Delta_GrandMean_minus_e_star = (1/17) * sum_{e != e_star} Delta_F1(e)
gain_is_distributed = (Delta_GrandMean_minus_e_star > 0.000)
```

### 4.3 Precision/Recall Collapse Test
A recipe suffers precision/recall collapse if any of the following occur:
1. Macro precision drop: `GrandPrecision(R2) - GrandPrecision(R0) <= -0.050`
2. Macro recall drop: `GrandRecall(R2) - GrandRecall(R0) <= -0.050`
3. Dual-drop events: More than 4 out of 18 events experience simultaneous precision drop >= 0.030 AND recall drop >= 0.030:
   `count(e such that Delta_P(e) <= -0.030 and Delta_R(e) <= -0.030) > 4`

---

## 5. Paired Event-Bootstrap Resampling

To establish non-parametric confidence intervals:
1. Resampling unit: The 18 unique held-out events.
2. Number of bootstrap resamples: B = 10,000.
3. Random seed: 42 (fixed for determinism).
4. Algorithm:
   - For b = 1 .. B:
     - Sample 18 event indices with replacement from {1..18}: (e_1^(b), ..., e_18^(b)).
     - Compute the bootstrap grand mean delta:
       ```text
       Delta^(b) = (1/18) * sum_{i=1}^18 Delta_F1(e_i^(b))
       ```
   - Sort the 10,000 bootstrap deltas: Delta_(1) <= ... <= Delta_(10000).
   - 95% Percentile Confidence Interval:
     ```text
     [CI_lower, CI_upper] = [Delta_(250), Delta_(9750)]
     ```
   - Bootstrap p-value (one-sided against H0: Delta <= 0):
     ```text
     p_bootstrap = (1/B) * sum_{b=1}^B I(Delta^(b) <= 0.0)
     ```

---

## 6. Frozen Decision Rules

The route decision follows `rst5_confirmation_decision_rule_v1.json`:

### 6.1 Criterion P1: Average Improvement
- **Condition 1**: `Delta_GrandMean >= +0.010`
- **Condition 2**: Gain is not driven by one event (`Delta_GrandMean_minus_e_star > 0.000`)
- **Condition 3**: No widespread precision or recall collapse (`Delta_GrandPrecision > -0.050`, `Delta_GrandRecall > -0.050`, `dual_drop_events <= 4`)
- **Verdict**: If all 3 conditions hold, P1 is SATISFIED (`P1_SUPPORTED_AVERAGE_IMPROVEMENT`).

### 6.2 Criterion P2: Tail Reliability Improvement
- **Condition 1**: `Delta_Bottom3Mean >= +0.020` OR `Delta_WorstEvent >= +0.020`
- **Condition 2**: `Delta_GrandMean >= -0.005` (no unacceptable degradation of average performance)
- **Verdict**: If both conditions hold, P2 is SATISFIED (`P2_SUPPORTED_TAIL_RELIABILITY_IMPROVEMENT`).

### 6.3 Criterion P3: Seed Stability
- **Condition 1**: `Delta_GrandMean >= -0.005`
- **Condition 2**: Relative reduction in across-seed SD >= 20%:
  ```text
  (SD_grand(R0) - SD_grand(R2)) / SD_grand(R0) >= 0.20
  ```
- **Condition 3**: Stability direction holds in both folds:
  `SD(FoldA, R2) < SD(FoldA, R0)` AND `SD(FoldB, R2) < SD(FoldB, R0)`
- **Verdict**: If all 3 conditions hold, P3 is SATISFIED (`P3_SUPPORTED_SEED_STABILITY`).

### 6.4 Route Verdict Synthesis
- **RST7 COMPLETE -- POSITIVE**: If ANY of P1, P2, or P3 is satisfied.
- **RST7 COMPLETE -- NEGATIVE**: If R2 degrades average performance (`Delta_GrandMean <= -0.010`) AND degrades tail performance (`Delta_Bottom3Mean < 0.000`).
- **RST7 COMPLETE -- INCONCLUSIVE**: If neither P1, P2, nor P3 is satisfied, and the negative criteria are not met (null hypothesis retained).

---

## 7. Artifact Deliverables

The offline analysis script `tools/robust_source_training/run_rst7_analysis.py` generates:
1. `canonical_event_results_v1.csv` & `.json`: Long-form table of all 108 event evaluations.
2. `paired_event_deltas_v1.csv` & `.json`: 18 unique event paired summaries (R0 vs R2).
3. `bootstrap_distribution_v1.json`: 10,000 bootstrap draws and confidence intervals.
4. `confirmatory_summary_v1.json`: Machine-readable audit and criteria evaluation summary.
5. `docs/robust_source_training/rst7_confirmatory_closeout_v1.md`: Narrative scientific closeout.
