# RST7 Confirmatory Analysis and Decision Template

> Note: This template must be populated upon completion of all 12 confirmatory runs in RST6.

## 1. Confirmatory Matrix Summary

- **Total Planned Runs**: 12 (6 on Fold A, 6 on Fold B)
- **Completed Runs**: 12 / 12
- **Evaluated Events**: 18 unique source events (Group A: 9 events; Group B: 9 events)
- **Total Event-Result Rows**: 108 (12 runs * 9 events)

---

## 2. Comprehensive 12-Run Results Table

| Fold | Recipe | Seed | Macro F1 | Macro IoU | Macro Prec | Macro Rec | Pooled F1 | Worst Ev F1 | Bottom-3 F1 | Checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold A** | R0 | s0 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold A** | R0 | s1 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold A** | R0 | s2 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold A** | R2 | s0 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold A** | R2 | s1 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold A** | R2 | s2 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R0 | s0 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R0 | s1 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R0 | s2 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R2 | s0 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R2 | s1 | ... | ... | ... | ... | ... | ... | ... | ... |
| **Fold B** | R2 | s2 | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 3. Statistical Summary Across Seeds and Folds

### 3.1 Fold A (Evaluated on Group A, 8,160 patches):
- **R0 Mean Macro F1**: `<MEAN_F1_R0_A>` $\pm$ `<SD_R0_A>`
- **R2 Mean Macro F1**: `<MEAN_F1_R2_A>` $\pm$ `<SD_R2_A>`
- **Fold A Delta (R2 - R0)**: `<DELTA_F1_A>`
- **Fold A Bottom-3 Delta**: `<DELTA_B3_A>`

### 3.2 Fold B (Evaluated on Group B, 6,807 patches):
- **R0 Mean Macro F1**: `<MEAN_F1_R0_B>` $\pm$ `<SD_R0_B>`
- **R2 Mean Macro F1**: `<MEAN_F1_R2_B>` $\pm$ `<SD_R2_B>`
- **Fold B Delta (R2 - R0)**: `<DELTA_F1_B>`
- **Fold B Bottom-3 Delta**: `<DELTA_B3_B>`

### 3.3 Grand Mean (Across All 18 Held-Out Events):
- **Grand Mean R0 F1**: `<GRAND_MEAN_R0>`
- **Grand Mean R2 F1**: `<GRAND_MEAN_R2>`
- **Overall Delta F1**: `<OVERALL_DELTA>`
- **Grand Mean Bottom-3 Delta**: `<OVERALL_B3_DELTA>`
- **Average Seed SD (R0)**: `<AVG_SD_R0>`
- **Average Seed SD (R2)**: `<AVG_SD_R2>`
- **Relative Variance Reduction**: `<REL_RED_SD>` %

---

## 4. Evaluation of Frozen Confirmatory Criteria

### 4.1 Criterion P1: Average Improvement
- **Condition 1 (Delta >= +0.010)**: `<TRUE/FALSE>` (Observed: `<OVERALL_DELTA>`)
- **Condition 2 (Not driven by 1 event)**: `<TRUE/FALSE>` (Delta excluding top event: `<DELTA_EX_TOP>`)
- **Condition 3 (No precision/recall collapse)**: `<TRUE/FALSE>` (Prec delta: `<PREC_DELTA>`, Rec delta: `<REC_DELTA>`)
- **P1 Status**: `<SATISFIED / NOT_SATISFIED>`

### 4.2 Criterion P2: Tail Reliability Improvement
- **Condition 1 (Bottom-3 delta >= +0.020 or Worst-event delta >= +0.020)**: `<TRUE/FALSE>` (Observed B3: `<OVERALL_B3_DELTA>`, Worst: `<WORST_DELTA>`)
- **Condition 2 (Mean delta >= -0.005)**: `<TRUE/FALSE>` (Observed: `<OVERALL_DELTA>`)
- **P2 Status**: `<SATISFIED / NOT_SATISFIED>`

### 4.3 Criterion P3: Seed Stability
- **Condition 1 (Mean delta >= -0.005)**: `<TRUE/FALSE>` (Observed: `<OVERALL_DELTA>`)
- **Condition 2 (Across-seed SD reduction >= 20%)**: `<TRUE/FALSE>` (Observed: `<REL_RED_SD>` %)
- **Condition 3 (Consistency across both folds)**: `<TRUE/FALSE>` (Fold A: `<SD_R2_A> < <SD_R0_A>`, Fold B: `<SD_R2_B> < <SD_R0_B>`)
- **P3 Status**: `<SATISFIED / NOT_SATISFIED>`

---

## 5. Final Confirmatory Verdict

**`<P1_SUPPORTED_AVERAGE_IMPROVEMENT | P2_SUPPORTED_TAIL_RELIABILITY_IMPROVEMENT | P3_SUPPORTED_SEED_STABILITY | P0_NULL_HYPOTHESIS_RETAINED>`**
