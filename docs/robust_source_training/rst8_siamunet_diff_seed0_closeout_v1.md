# ChangeGuard RST8 Stage-A Closeout & Route Decision (SiamUnet_diff)

## Executive Summary & Terminal Verdict

Work Package **RST8 Stage A** evaluated whether the empirical benefits of Recipe R2 (M1 event-balanced sampling + M2 training-only inverse-square-root class-balanced cross-entropy), confirmed on SNUNet-ECAM during RST6 and RST7, transfer to a distinct second architecture: **SiamUnet_diff** (FC-Siam-diff).

Four formal training runs were executed across the two frozen confirmatory folds using seed 0:
- Baseline Recipe R0 on Fold A and Fold B;
- Recipe R2 on Fold A and Fold B.

Across all 18 unique held-out confirmatory source events, Recipe R2 did **not** improve performance on SiamUnet_diff. Instead, R2 resulted in a slight reduction in average event-macro F1 and a marked degradation in tail reliability:
- **Grand Mean Event-Macro F1**: R0 `0.7238` vs R2 `0.7139` ($\Delta = -0.0099$ / $-0.010$ rounded);
- **Tail Reliability (Bottom-3 Mean F1)**: R0 `0.4350` vs R2 `0.3902` ($\Delta = -0.0448$);
- **Worst-Event F1**: R0 `0.3565` (Event 1111010) vs R2 `0.2660` (Event 118) ($\Delta = -0.0905$);
- **Paired Bootstrap 95% CI**: `[-0.0478, +0.0229]`, one-sided $p$-value = `0.6744`;
- **Criterion P1 (Average Improvement)**: **NOT SATISFIED**;
- **Criterion P2 (Tail Reliability Improvement)**: **NOT SATISFIED**;
- **Criterion P3 (Seed Stability)**: **NOT EVALUABLE AT STAGE A** (seed 0 only).

Under the frozen Stage-A continuation rule (`docs/robust_source_training/rst8_siamunet_diff_protocol_v1.md`, Section 4.3) and the governing proposal (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`, RST8 Exit Condition), multi-seed expansion (seeds 1 and 2) is **not authorized**. The transfer direction is unsupported, compute is preserved, and the claim of Recipe R2 is scoped strictly to SNUNet-ECAM with architectural dependence transparently documented.

```text
RST8 STAGE A COMPLETE — STOP AT SEED 0
```

---

## 1. Architecture Verification & Invariants: SiamUnet_diff

1. **Source & Implementation**:
   - Implemented in `src/models/siam_diff_wrapper.py` following Daudt et al. (2018) FC-Siam-diff.
   - Dual-stream Siamese encoder with shared weights processing dual-polarization pre-event (`xA`) and post-event (`xB`) SAR patches `(B, 2, 256, 256)`.
   - Four-stage difference decoder utilizing transposed convolution upsampling concatenated with absolute encoder feature differences:
     $$\text{skip\_diff} = |\mathbf{z}_{\text{pre}} - \mathbf{z}_{\text{post}}|$$
2. **Parameter Count**:
   - Exactly **1,350,147 parameters** (~1.35M parameters, compared to 12.03M parameters in SNUNet-ECAM).
3. **Semantic Labels & Evaluation Positive Class**:
   - Class 0: Dry ground / unflooded land
   - Class 1: Permanent water body
   - Class 2: Newly inundated flood water (**evaluation positive class**)
   - Class 3: Ignore / invalid data mask (strictly excluded from loss and metric computation)
4. **Initial Weight Determinism**:
   - Seed 0 initialization state hash:
     `6ee419e4f06ab18dd7cb7b68e7fae74b639d573203e2fb274e8ee673f68e1b13` (matched identically across all seed-0 initializations).

---

## 2. Experimental Matrix & Execution Ledger

All four planned training runs completed with `COMPLETED_SCIENTIFIC_RUN` and exit code 0:

| Run ID | Recipe | Fold | Seed | Steps | Runtime | Peak VRAM | Checkpoint SHA-256 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rst8_r0_siamunet_diff_folda_s0_v1` | R0 | Fold A | 0 | 6,656 | 2,499.75s | 1,165.6 MiB | `5a3c33f8000cb7b4...` | `COMPLETED_SCIENTIFIC_RUN` |
| `rst8_r2_siamunet_diff_folda_s0_v2` | R2 | Fold A | 0 | 6,656 | 2,117.22s | 1,165.6 MiB | `e0f8efb8d3e6968a...` | `COMPLETED_SCIENTIFIC_RUN` |
| `rst8_r0_siamunet_diff_foldb_s0_v2` | R0 | Fold B | 0 | 7,328 | 2,866.52s | 1,165.6 MiB | `61705d0095af63c3...` | `COMPLETED_SCIENTIFIC_RUN` |
| `rst8_r2_siamunet_diff_foldb_s0_v2` | R2 | Fold B | 0 | 7,328 | 2,265.73s | 1,165.6 MiB | `70d469617fd5f5b6...` | `COMPLETED_SCIENTIFIC_RUN` |

### Invalidation Record for Preflight Attempts
Three initial v1 attempts (`rst8_r2_folda_s0_v1`, `rst8_r0_foldb_s0_v1`, `rst8_r2_foldb_s0_v1`) failed during preflight due to a device ordinal mismatch (`CUDA_VISIBLE_DEVICES` remapping physical devices 1–3 to virtual ordinal `cuda:0`). Per the confirmatory recovery policy:
- No weights or results were produced;
- Terminal status `FAILED_PREFLIGHT_DEVICE_ORDINAL` was preserved permanently in their respective run directories;
- Corrected `_v2` configs specifying `"device": "cuda:0"` were launched and completed formally without altering any scientific parameter.

### Integrity & Event Partition Verification
- **Total Event Rows**: Exactly 36 canonical evaluation rows produced (9 per run).
- **Unique Confirmatory Events**: Exactly 18 held-out source events (9 in Fold A: `118, 174, 273, 421, 496, 498, 554, 567, 1111009`; 9 in Fold B: `267, 324, 417, 470, 502, 518, 555, 1111004, 1111010`).
- **Forbidden Target Events**: Zero access to any of the 16 forbidden target events (`205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013`).
- **Historical E11 Preservation**: `docs/e11_final_scientific_conclusions.md` and `docs/e11_final_test_closeout.md` remain byte-identically untouched and unstaged.

---

## 3. Comprehensive Evaluation Metrics

### 3.1 Grand Aggregate Metrics (18 Held-Out Events)

| Metric | R0 (Seed 0) | R2 (Seed 0) | Delta ($\text{R2} - \text{R0}$) | Interpretation |
| :--- | :---: | :---: | :---: | :---: |
| **Event-Macro F1** | **0.7238** | **0.7139** | **-0.0099** | Slight average degradation (-1.0% F1) |
| **Event-Macro Precision** | 0.8441 | 0.8140 | -0.0301 | Mild precision loss (-3.0%) |
| **Event-Macro Recall** | 0.6757 | 0.6872 | +0.0115 | Small recall gain (+1.1%) |
| **Event-Macro IoU** | 0.5894 | 0.5805 | -0.0089 | Slight IoU decrease (-0.9%) |
| **Bottom-3 Mean F1** | **0.4350** | **0.3902** | **-0.0448** | **Substantial tail degradation (-4.5% F1)** |
| **Worst-Event F1** | **0.3565** | **0.2660** | **-0.0905** | **Severe worst-case collapse (-9.1% F1)** |

### 3.2 Fold-Level Macro Metrics

| Fold | Metric | R0 (Seed 0) | R2 (Seed 0) | Delta ($\text{R2} - \text{R0}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Fold A** (9 events) | Macro F1 | 0.7355 | 0.7134 | -0.0221 |
| | Macro Precision | 0.8777 | 0.7996 | -0.0780 |
| | Macro Recall | 0.6672 | 0.7079 | +0.0408 |
| | Macro IoU | 0.6016 | 0.5889 | -0.0126 |
| **Fold B** (9 events) | Macro F1 | 0.7121 | 0.7145 | +0.0024 |
| | Macro Precision | 0.8105 | 0.8283 | +0.0178 |
| | Macro Recall | 0.6842 | 0.6664 | -0.0178 |
| | Macro IoU | 0.5772 | 0.5721 | -0.0051 |

---

## 4. Per-Event Paired Deltas (18 Events)

| Event ID | Fold | R0 F1 | R2 F1 | $\Delta \text{F1}$ | R0 Precision | R2 Precision | $\Delta \text{Precision}$ | R0 Recall | R2 Recall | $\Delta \text{Recall}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **118** | Fold A | 0.4073 | 0.2660 | **-0.1413** | 0.7965 | 0.7028 | -0.0937 | 0.2736 | 0.1640 | -0.1096 |
| **174** | Fold A | 0.6554 | 0.7304 | +0.0750 | 0.9810 | 0.9621 | -0.0189 | 0.4920 | 0.5886 | +0.0965 |
| **273** | Fold A | 0.7222 | 0.4802 | **-0.2421** | 0.7465 | 0.3508 | -0.3957 | 0.6995 | 0.7604 | +0.0610 |
| **421** | Fold A | 0.6211 | 0.6436 | +0.0224 | 0.9682 | 0.9343 | -0.0340 | 0.4572 | 0.4908 | +0.0336 |
| **496** | Fold A | 0.8048 | 0.8583 | +0.0535 | 0.7278 | 0.8320 | +0.1042 | 0.9000 | 0.8862 | -0.0137 |
| **498** | Fold A | 0.8358 | 0.8287 | -0.0071 | 0.8952 | 0.8055 | -0.0897 | 0.7837 | 0.8533 | +0.0696 |
| **554** | Fold A | 0.8155 | 0.8247 | +0.0092 | 0.9352 | 0.8717 | -0.0635 | 0.7230 | 0.7825 | +0.0596 |
| **567** | Fold A | 0.9326 | 0.9252 | -0.0074 | 0.9656 | 0.9211 | -0.0444 | 0.9018 | 0.9293 | +0.0275 |
| **1111009** | Fold A | 0.8247 | 0.8633 | +0.0386 | 0.8830 | 0.8162 | -0.0668 | 0.7737 | 0.9162 | +0.1425 |
| **267** | Fold B | 0.7441 | 0.7676 | +0.0235 | 0.6175 | 0.6560 | +0.0384 | 0.9359 | 0.9249 | -0.0110 |
| **324** | Fold B | 0.7825 | 0.7767 | -0.0058 | 0.7897 | 0.8238 | +0.0342 | 0.7754 | 0.7346 | -0.0408 |
| **417** | Fold B | 0.9474 | 0.9463 | -0.0011 | 0.9518 | 0.9405 | -0.0112 | 0.9431 | 0.9521 | +0.0090 |
| **470** | Fold B | 0.5412 | 0.6203 | +0.0792 | 0.9800 | 0.9627 | -0.0172 | 0.3738 | 0.4576 | +0.0838 |
| **502** | Fold B | 0.7975 | 0.7659 | -0.0315 | 0.8622 | 0.8627 | +0.0005 | 0.7417 | 0.6887 | -0.0531 |
| **518** | Fold B | 0.7434 | 0.7280 | -0.0154 | 0.6826 | 0.6804 | -0.0022 | 0.8161 | 0.7828 | -0.0333 |
| **555** | Fold B | 0.6462 | 0.6609 | +0.0148 | 0.7715 | 0.8564 | +0.0848 | 0.5559 | 0.5381 | -0.0177 |
| **1111004** | Fold B | 0.8504 | 0.7401 | **-0.1103** | 0.9393 | 0.9060 | -0.0333 | 0.7769 | 0.6256 | -0.1513 |
| **1111010** | Fold B | 0.3565 | 0.4245 | +0.0680 | 0.7000 | 0.7662 | +0.0662 | 0.2392 | 0.2936 | +0.0544 |

- **Win / Loss Ratio**: 7 events improved ($\Delta \text{F1} > 0$), 11 events degraded ($\Delta \text{F1} < 0$).
- **Severe Degradations**: Three events suffered drops $> 0.10$ F1:
  - Event 273: $-0.2421$ F1 (precision collapsed from 0.7465 to 0.3508);
  - Event 118: $-0.1413$ F1 (recall dropped from 0.2736 to 0.1640);
  - Event 1111004: $-0.1103$ F1 (recall dropped from 0.7769 to 0.6256).

---

## 5. Decision Criteria Evaluation

### 5.1 Criterion P1: Average Improvement
- **Condition 1**: $\Delta \text{GrandMean} \ge +0.010$.
  - Observed: $\Delta = -0.0099$. **NOT SATISFIED**.
- **Condition 2**: Gain is not driven by single event.
  - Max gain: Event 470 ($\Delta = +0.0792$).
  - Mean delta excluding Event 470: $-0.0151 < 0.0$. **NOT SATISFIED**.
- **Condition 3**: No widespread precision/recall collapse.
  - Macro precision delta: $-0.0301 > -0.050$ (SATISFIED).
  - Macro recall delta: $+0.0115 > -0.050$ (SATISFIED).
  - Dual-drop events ($\Delta \text{P} \le -0.030$ AND $\Delta \text{R} \le -0.030$): 2 events (Events 118 and 1111004) $\le 4$ (SATISFIED).
- **Criterion P1 Verdict**: **NOT SATISFIED**.

### 5.2 Criterion P2: Tail Reliability Improvement
- **Condition 1**: $\Delta \text{Bottom3Mean} \ge +0.020$ OR $\Delta \text{WorstEvent} \ge +0.020$.
  - $\Delta \text{Bottom3Mean} = -0.0448$ (Threshold: $\ge +0.020$).
  - $\Delta \text{WorstEvent} = -0.0905$ (Threshold: $\ge +0.020$). **NOT SATISFIED**.
- **Condition 2**: Average performance not degraded ($\Delta \text{GrandMean} \ge -0.005$).
  - Observed: $\Delta = -0.0099 < -0.005$. **NOT SATISFIED**.
- **Criterion P2 Verdict**: **NOT SATISFIED**.

### 5.3 Criterion P3: Seed Stability
- **Condition**: Across-seed standard deviation reduction $\ge 20\%$ in both folds.
- **Status**: **NOT EVALUABLE AT STAGE A** (Stage A executes seed 0 only).

### 5.4 Paired Event-Bootstrap Analysis (10,000 Resamples, Seed 42)
- Resampling unit: 18 unique held-out confirmatory events
- Observed mean delta: $-0.0099$
- **95% Bootstrap Confidence Interval**: `[-0.0478, +0.0229]`
- One-sided $p$-value ($H_0: \Delta \le 0$): `0.6744`
- Percentile distribution:
  - 1st: $-0.0566$
  - 5th: $-0.0413$
  - 25th: $-0.0213$
  - 50th (median): $-0.0084$
  - 75th: $+0.0038$
  - 95th: $+0.0188$
  - 99th: $+0.0280$

---

## 6. Scientific Implications & Route Decision

### 6.1 Architectural Scope and Limitation of Findings
On **SNUNet-ECAM**, Recipe R2 produced statistically supported improvements on confirmatory source folds:
- Grand mean F1: $+0.0136$;
- Bottom-3 tail F1: $+0.0511$;
- Fulfilling both P1 and P2.

On **SiamUnet_diff** (the tested FC-Siam-diff implementation at seed 0):
- Recipe R2 failed to reproduce either average or tail gains;
- Average F1 degraded by $-0.0099$;
- Tail F1 degraded substantially (bottom-3 dropped by $-0.0448$, worst event dropped by $-0.0905$);
- 11 of 18 events degraded.

Importantly, the mechanism behind the architecture-dependent outcome was not experimentally tested. The RST8 empirical evidence shows only that Recipe R2 did not transfer to the tested SiamUnet_diff implementation at seed 0. Consequently:
- We do not claim that the absence of multi-scale attention caused the negative result.
- We do not claim that R2 is generally effective for all attention-augmented architectures.
- The positive claim for Recipe R2 is currently limited specifically to SNUNet-ECAM.

### 6.2 Continuation Decision: Stop at Seed 0
Per `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`:
> *"Start with seed 0. Expand to three seeds only if the direction is meaningful. If the method does not transfer, limit the claim to SNUNet-ECAM and report the architectural dependence. Exit condition: the architecture scope of the claim is known; unnecessary full-factor runs are avoided."*

Because neither P1 nor P2 is satisfied and the observed delta is non-positive across average and tail metrics:
1. **Multi-seed expansion (seeds 1 and 2) is NOT authorized.**
2. **Work Package RST8 is permanently closed at Stage A.** The seed-0 stopping decision remains unchanged.
3. The ChangeGuard contribution of Recipe R2 is scoped strictly to SNUNet-ECAM, and the empirical lack of transfer to the tested SiamUnet_diff model at seed 0 is transparently established.

---

## 7. Artifact Manifest

The complete analysis artifacts have been generated and archived:
- Canonical 36-row table: `outputs/robust_source_training/rst8/canonical_event_results_v1.csv` & `.json`
- 18-row paired delta table: `outputs/robust_source_training/rst8/paired_event_deltas_v1.csv` & `.json`
- 10,000-resample bootstrap: `outputs/robust_source_training/rst8/bootstrap_distribution_v1.json`
- Stage-A summary: `outputs/robust_source_training/rst8/stage_a_summary_v1.json`
- Matrix definition: `configs/robust_source_training/rst8_matrix_v1.json`
- Execution ledger: `configs/robust_source_training/rst8_ledger_v1.json`
- Frozen protocol: `docs/robust_source_training/rst8_siamunet_diff_protocol_v1.md`
- Regression test suite: `tests/test_rst8_siamunet_diff.py` (11/11 passing)

