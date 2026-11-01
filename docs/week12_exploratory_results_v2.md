# Week 12 Exploratory Ablation Analysis Report (Revision v2)

**Document Version:** `v2.0`  
**Supersedes:** [`docs/week12_exploratory_results_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week12_exploratory_results_v1.md)  
**Governing Protocol:** [`docs/protocols/week12_exploratory_ablation_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/protocols/week12_exploratory_ablation_protocol_v1.md)  
**Governing Config:** [`configs/changeguard/week12_ablation_dev6_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week12_ablation_dev6_v1.json)  
**Raw Outputs Directory:** [`outputs/week12_ablation/approved_dev6_s0_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/approved_dev6_s0_v1/)  
**Analysis Directory:** [`outputs/week12_ablation/analysis_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/analysis_v1/)  
**Execution Log:** [`logs/week12_ablation/approved_dev6_s0_v1/ablation_execution.log`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/logs/week12_ablation/approved_dev6_s0_v1/ablation_execution.log)  
**Evidence Level:** `EXPLORATORY` (Development events only, single seed 0, single order `order_a`, reset-per-event).

---

## 1. Executive Summary & Verification Verdict

### Completion & Integrity Verdict: **PASS (100% COMPLETE)**
1. **Full Condition-Event Matrix Completed:** All 4 conditions across all 6 development events (24 condition-event streams) executed to completion without unhandled exceptions or early termination.
2. **Identical Sample Coverage, Ordering, and Batching:**
   - Total samples processed per condition: **2,826 samples** across 6 events.
   - Total batches processed per condition: **179 batches** (Batch size 16 with natural tail).
   - Sample sequence verified bit-for-bit identical across all 4 conditions (`source_only`, `matched_tent_pre_update_v1`, `C1_consistency_no_conf`, `C2_consistency_conf90`).
3. **Confusion Matrix Conservation Invariant:**
   - $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels}$ holds exactly without discrepancy across all 24 individual events and pooled summaries.
4. **Honest Update / Skip / Failure Totals:**
   - `source_only`: 0 updates applied, 0 skipped, 0 failed (179 forward-only evaluation batches).
   - `matched_tent_pre_update_v1`: 179 updates applied, 0 skipped, 0 failed.
   - `C1_consistency_no_conf`: 179 updates applied, 0 skipped, 0 failed.
   - `C2_consistency_conf90`: 179 updates applied, 0 skipped, 0 failed.
5. **Finite Metrics & Diagnostics:** Zero `NaN` or `Inf` values across all metrics, loss values, update times, and diagnostic records.
6. **Provenance Verified:**
   - Checkpoint SHA-256 (`final_checkpoint.pt`): `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` (Bit-identical match to expected hash).
   - KuroSiwo Manifest SHA-256: `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`.
   - All 103 generated output artifacts have been hashed and cataloged in [`outputs/week12_ablation/analysis_v1/artifact_sha256.txt`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/analysis_v1/artifact_sha256.txt).

---

## 2. Summary Metric Definitions & Results

### Metric Definitions
- **Event-Macro $F_1$:** Unweighted arithmetic mean of the per-event $F_1$ scores across the $N = 6$ development events:
  $$\text{Macro } F_1 = \frac{1}{6} \sum_{e \in \mathcal{E}} F_{1, e}$$
- **Pooled (Micro) $F_1$:** Aggregate $F_1$ computed from the pooled sum of confusion matrix cells across all 6 events:
  $$\text{Pooled } F_1 = \frac{2 \sum_{e} \text{TP}_e}{2 \sum_{e} \text{TP}_e + \sum_{e} \text{FP}_e + \sum_{e} \text{FN}_e}$$
- **Worst-Event $F_1$:** Minimum $F_1$ score attained by a condition across any single event:
  $$\text{Worst-Event } F_1 = \min_{e \in \mathcal{E}} F_{1, e}$$

### Summary Comparison Table (Full Precision)

| Condition ID | Macro $F_1$ | Pooled $F_1$ | Worst-Event $F_1$ (Event ID) | Pooled IoU | Pooled Precision | Pooled Recall | Updates Applied | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **`source_only`** | **0.594880** | **0.519331** | 0.256835 (Event 1111003) | **0.350741** | **0.401615** | 0.734667 | 0 / 179 | 192.9 s |
| **`matched_tent_pre_update_v1`** | 0.574088 | 0.446873 | **0.265045** (Event 1111003) | 0.287725 | 0.319989 | 0.740502 | 179 / 179 | 235.5 s |
| **`C1_consistency_no_conf`** | 0.573270 | 0.441606 | **0.265045** (Event 1111003) | 0.283373 | 0.314398 | **0.741712** | 179 / 179 | 490.5 s |
| **`C2_consistency_conf90`** | 0.573318 | 0.441912 | **0.265045** (Event 1111003) | 0.283624 | 0.314720 | 0.741642 | 179 / 179 | 480.4 s |

---

## 3. Per-Event Full-Precision Results & Differences

The following table presents per-event evaluation metrics, ground truth and predicted flood counts, and differences against `source_only`, against `matched_tent_pre_update_v1`, and between C2 and C1 ($C_2 - C_1$).

| Event ID | Condition | $F_1$ | IoU | Precision | Recall | Predicted Flood ($\text{TP}+\text{FP}$) | GT Flood ($\text{TP}+\text{FN}$) | $\Delta F_1$ vs Source | $\Delta F_1$ vs Tent | $C_2 - C_1$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | `source_only` | 0.376893 | 0.232204 | 0.256801 | 0.707970 | 2,394,889 | 868,697 | 0.000000 | +0.071915 | — |
| | `matched_tent` | 0.304978 | 0.179925 | 0.193249 | 0.722960 | 3,249,856 | 868,697 | -0.071915 | 0.000000 | — |
| | `C1_no_conf` | 0.300132 | 0.176562 | 0.189204 | 0.725458 | 3,330,811 | 868,697 | -0.076761 | -0.004846 | [ref] |
| | `C2_conf90` | 0.300410 | 0.176754 | 0.189435 | 0.725310 | 3,326,077 | 868,697 | -0.076483 | -0.004568 | **+0.000278** |
| **437** | `source_only` | 0.699365 | 0.537711 | 0.645254 | 0.763382 | 75,719 | 64,002 | 0.000000 | +0.064905 | — |
| | `matched_tent` | 0.634460 | 0.464623 | 0.541161 | 0.766632 | 90,668 | 64,002 | -0.064905 | 0.000000 | — |
| | `C1_no_conf` | 0.634416 | 0.464575 | 0.541089 | 0.766648 | 90,682 | 64,002 | -0.064949 | -0.000044 | [ref] |
| | `C2_conf90` | 0.634424 | 0.464583 | 0.541107 | 0.766632 | 90,677 | 64,002 | -0.064942 | -0.000037 | **+0.000008** |
| **520** | `source_only` | 0.713750 | 0.554908 | 0.730456 | 0.697792 | 314,520 | 329,243 | 0.000000 | +0.014160 | — |
| | `matched_tent` | 0.699590 | 0.537976 | 0.709597 | 0.689861 | 320,086 | 329,243 | -0.014160 | 0.000000 | — |
| | `C1_no_conf` | 0.699566 | 0.537948 | 0.709567 | 0.689843 | 320,091 | 329,243 | -0.014184 | -0.000024 | [ref] |
| | `C2_conf90` | 0.699574 | 0.537958 | 0.709574 | 0.689852 | 320,092 | 329,243 | -0.014176 | -0.000016 | **+0.000008** |
| **559** | `source_only` | 0.672577 | 0.506679 | 0.872522 | 0.547186 | 31,778 | 50,672 | 0.000000 | -0.019856 | — |
| | `matched_tent` | 0.692433 | 0.529559 | 0.874079 | 0.573295 | 33,235 | 50,672 | +0.019856 | 0.000000 | — |
| | `C1_no_conf` | 0.692433 | 0.529559 | 0.874079 | 0.573295 | 33,235 | 50,672 | +0.019856 | 0.000000 | [ref] |
| | `C2_conf90` | 0.692425 | 0.529549 | 0.874052 | 0.573295 | 33,236 | 50,672 | +0.019848 | -0.000008 | **-0.000008** |
| **1111003** | `source_only` | 0.256835 | 0.147338 | 0.171849 | 0.508117 | 20,582 | 6,961 | 0.000000 | -0.008210 | — |
| | `matched_tent` | 0.265045 | 0.152768 | 0.187429 | 0.452378 | 16,801 | 6,961 | +0.008210 | 0.000000 | — |
| | `C1_no_conf` | 0.265045 | 0.152768 | 0.187429 | 0.452378 | 16,801 | 6,961 | +0.008210 | 0.000000 | [ref] |
| | `C2_conf90` | 0.265045 | 0.152768 | 0.187429 | 0.452378 | 16,801 | 6,961 | +0.008210 | 0.000000 | **0.000000** |
| **1111008** | `source_only` | 0.849858 | 0.738916 | 0.873493 | 0.827469 | 454,994 | 480,301 | 0.000000 | +0.001835 | — |
| | `matched_tent` | 0.848023 | 0.736146 | 0.872060 | 0.825276 | 454,534 | 480,301 | -0.001835 | 0.000000 | — |
| | `C1_no_conf` | 0.848027 | 0.736151 | 0.872037 | 0.825303 | 454,561 | 480,301 | -0.001831 | +0.000003 | [ref] |
| | `C2_conf90` | 0.848033 | 0.736162 | 0.872049 | 0.825305 | 454,556 | 480,301 | -0.001825 | +0.000010 | **+0.000007** |

---

## 4. Pixel Selection Analysis: C1 vs C2

A primary hypothesis of Week 12 was whether confidence thresholding ($\tau \ge 0.90$) altered the adaptation process.

Across the 6 development events, confidence filtering excluded **15,763,945 pixels (11.21% of all sensory-valid pixels)**:

| Event ID | Total Batches | Sensory-Valid Pixels | C1 Selected Pixels ($\tau = 0.0$) | C1 Selection % | C2 Selected Pixels ($\tau \ge 0.90$) | C2 Selection % | Pixels Filtered Out | Filtered % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 149 | 118,983,019 | 118,983,019 | 100.00% | 104,833,479 | 88.11% | 14,149,540 | 11.89% |
| **437** | 11 | 8,479,744 | 8,479,744 | 100.00% | 8,223,010 | 96.97% | 256,734 | 3.03% |
| **520** | 6 | 4,554,128 | 4,554,128 | 100.00% | 4,083,581 | 89.67% | 470,547 | 10.33% |
| **559** | 3 | 2,107,392 | 2,107,392 | 100.00% | 2,033,353 | 96.49% | 74,039 | 3.51% |
| **1111003** | 2 | 815,490 | 815,490 | 100.00% | 701,465 | 86.02% | 114,025 | 13.98% |
| **1111008** | 8 | 5,686,145 | 5,686,145 | 100.00% | 4,987,085 | 87.71% | 699,060 | 12.29% |
| **POOLED** | **179** | **140,625,918** | **140,625,918** | **100.00%** | **124,861,973** | **88.79%** | **15,763,945** | **11.21%** |

### Correct Interpretation of Selection vs Metric Differences
1. **Pixel Selection is Distinct from Update Counts:** Both C1 and C2 updated on every batch (179/179 updates, 0 skips). However, C1 and C2 did not select the same pixels; 15.76 million pixels were removed from gradient computation in C2.
2. **Negligible Metric Difference:** Despite discarding 11.21% of pixels, the resulting metric difference is practically negligible:
   - Pooled $F_1$: $0.441912$ (C2) vs $0.441606$ (C1) ($\Delta = +0.000305$).
   - Event-Macro $F_1$: $0.573318$ (C2) vs $0.573270$ (C1) ($\Delta = +0.000049$).
   - Maximum per-event difference was on Event 279 ($+0.000278$). On four other events, the absolute difference was $\le 0.000008$. On Event 1111003, it was exactly zero.
3. **Runtime Observation Does Not Establish Reliable Acceleration:** C2 ran in 480.4 s versus 490.5 s for C1. A single run with a 2% difference (~10 seconds across 23 minutes) reflects normal GPU kernel/scheduler variance and does not establish systematic computational acceleration from confidence masking. Both consistency variants took roughly $2\times$ the compute of Tent (235.5 s).

---

## 5. Corrected Scientific Interpretation & Methodological Clarifications

The exploratory findings must be interpreted strictly within their validated scope, correcting several non-rigorous assumptions:

### 1. Current C1/C2 Settings Did Not Improve Aggregate Performance
In this seed 0 exploratory ablation, neither C1 nor C2 improved overall performance over the frozen source baseline:
- **Macro $F_1$:** Source-Only ($0.5949$) > Tent ($0.5741$) > C2 ($0.5733$) $\ge$ C1 ($0.5733$).
- **Pooled $F_1$:** Source-Only ($0.5193$) > Tent ($0.4469$) > C2 ($0.4419$) > C1 ($0.4416$).
Source-only outperformed all adapted variants on 4 out of 6 events (Events 279, 437, 520, 1111008), including the two largest events by sample volume. Continuous unconstrained consistency adaptation suffered from severe false positive inflation on Event 279, exactly like Tent.

### 2. Loss Magnitude Alone Does Not Establish Gradient Magnitude or Parameter Drift
- The recorded loss values show:
  - Tent Entropy: mean $= 0.1074$ (range $[0.0374, 0.1862]$).
  - C1 Consistency: mean $= 0.0125$ (range $[0.0033, 0.0299]$).
  - C2 Consistency: mean $= 0.0046$ (range $[0.0016, 0.0120]$).
- While the consistency loss is $10–25\times$ smaller than entropy loss, **loss magnitude does not directly establish gradient magnitude or parameter drift**. Gradient magnitudes depend on loss landscape curvature, softmax derivatives, and backpropagation scaling through the architecture. Parameter drift was not measured directly via parameter norm tracking in this run.

### 3. Initial Source Weights with Candidate Batch-Stat BN Do NOT Imply Source-Only Predictions
- It is a methodological error to assume that on Batch 1, candidate predictions equal `source_only` predictions simply because weights have not yet updated.
- `source_only` evaluates with `model.eval()`, normalizing features using frozen training running statistics ($\mu_{\text{source}}, \sigma^2_{\text{source}}$).
- In contrast, all candidate adapters (Tent, C1, C2) configure BN modules with `track_running_stats = False`, computing feature normalization from the current batch statistics ($\mu_B, \sigma^2_B$).
- Therefore, even on the very first batch before any gradient step, **the candidate BN context diverges from the frozen source context**. This explains why immediate prediction shifts appear even before optimization steps.

### 4. Small-Event Adapted Predictions are Not Uniformly Close to Source
- While small-event predictions are identical *between* Tent, C1, and C2 on Event 1111003, they are **not uniformly close to source**:
  - Event 437 (11 batches): Source $F_1 = 0.6994$ vs Tent/C1/C2 $F_1 = 0.6344$ ($\Delta = -0.0649$, a substantial drop).
  - Event 520 (6 batches): Source $F_1 = 0.7138$ vs Tent/C1/C2 $F_1 = 0.6996$ ($\Delta = -0.0142$).
  - Event 559 (3 batches): Source $F_1 = 0.6726$ vs Tent/C1/C2 $F_1 = 0.6924$ ($\Delta = +0.0199$).
- Thus, substantial differences between adapted models and source occur even on events with only 3 to 11 batches.

### 5. Selective Gating Remains a Hypothesis
- These empirical results do **not** establish the necessity or effectiveness of selective gating.
- While the observation that continuous adaptation degrades performance on 4/6 events motivates the *desire* for a gate, proving that an unsupervised indicator can successfully gate adaptation without oracle knowledge remains an unproven research hypothesis.

### 6. Role of C2 in Future Work
- **C2 is NOT a proven winning method.**
- C2 (`C2_consistency_conf90`) is retained strictly as the **predefined reference configuration** for the planned Week 13 source-anchor comparison, providing a controlled baseline with explicit confidence filtering.

---

## 6. Disposition & Next Actions

1. **Retain v2 Report:** This v2 report serves as the authoritative, corrected interpretation of the Week 12 exploratory ablation.
2. **Execute Diagnostic Stream (`matched_bn_no_update_v1`):** To isolate the effect of batch-statistics normalization from gradient optimization, execute exactly one diagnostic comparison with candidate BN context and zero parameter updates.
3. **Prepare Week 13 Draft:** Formulate the frozen-source probability anchor protocol (`C2` vs `C3`) without initiating experiments.
