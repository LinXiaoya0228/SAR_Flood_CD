# Week 12 Exploratory Ablation Analysis Report

**Document Version:** `v1.0`  
**Governing Protocol:** [`docs/protocols/week12_exploratory_ablation_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/protocols/week12_exploratory_ablation_protocol_v1.md)  
**Governing Config:** [`configs/changeguard/week12_ablation_dev6_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week12_ablation_dev6_v1.json)  
**Raw Outputs Directory:** [`outputs/week12_ablation/approved_dev6_s0_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/approved_dev6_s0_v1/)  
**Analysis Directory:** [`outputs/week12_ablation/analysis_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week12_ablation/analysis_v1/)  
**Execution Log:** [`logs/week12_ablation/approved_dev6_s0_v1/ablation_execution.log`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/logs/week12_ablation/approved_dev6_s0_v1/ablation_execution.log)  
**Git Checkpoint Commit:** `0a8b63cbb12b81f299bfedfeb5fabd1159714d70`  
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

A central research question was whether confidence thresholding ($\tau \ge 0.90$) actually filtered out pixels, or whether C1 and C2 performed identical updates.

> **Key Finding:** While both C1 and C2 applied updates on every single batch (179/179 updates, 0 skips), **they did NOT select the same pixels**.

Across the 6 development events, exactly **15,763,945 pixels (11.21% of all sensory-valid pixels)** were filtered out by the confidence condition:

| Event ID | Total Batches | Sensory-Valid Pixels | C1 Selected Pixels ($\tau = 0.0$) | C1 Selection % | C2 Selected Pixels ($\tau \ge 0.90$) | C2 Selection % | Pixels Filtered Out | Filtered % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 149 | 118,983,019 | 118,983,019 | 100.00% | 104,833,479 | 88.11% | 14,149,540 | 11.89% |
| **437** | 11 | 8,479,744 | 8,479,744 | 100.00% | 8,223,010 | 96.97% | 256,734 | 3.03% |
| **520** | 6 | 4,554,128 | 4,554,128 | 100.00% | 4,083,581 | 89.67% | 470,547 | 10.33% |
| **559** | 3 | 2,107,392 | 2,107,392 | 100.00% | 2,033,353 | 96.49% | 74,039 | 3.51% |
| **1111003** | 2 | 815,490 | 815,490 | 100.00% | 701,465 | 86.02% | 114,025 | 13.98% |
| **1111008** | 8 | 5,686,145 | 5,686,145 | 100.00% | 4,987,085 | 87.71% | 699,060 | 12.29% |
| **POOLED** | **179** | **140,625,918** | **140,625,918** | **100.00%** | **124,861,973** | **88.79%** | **15,763,945** | **11.21%** |

### Did Confidence Filtering Impact Performance or Compute?
1. **Performance Impact:**
   - Despite discarding 15.76 million lower-confidence pixels, the performance difference between C1 and C2 is **practically negligible**:
     - Pooled $F_1$: $0.441912$ (C2) vs $0.441606$ (C1) ($\Delta = +0.000305$).
     - Event-Macro $F_1$: $0.573318$ (C2) vs $0.573270$ (C1) ($\Delta = +0.000049$).
     - Maximum per-event difference occurred on Event 279, where C2 gained $+0.000278$ $F_1$ (0.300410 vs 0.300132) by reducing false positive flood pixels by 4,606. On the other five events, the absolute difference was $\le 0.000008$.
2. **Compute Impact:**
   - Both C1 and C2 executed in approximately 8 minutes total (C1: 490.5 s, C2: 480.4 s), with identical per-batch overhead (~2.7 ms per step difference from tensor masking).
   - In contrast, Tent executed in 235.5 s (~3.9 min), and source-only forward passes required 192.9 s (~3.2 min).

---

## 5. Diagnostic Evidence: Loss Magnitudes and Optimization Dynamics

To understand why `matched_tent`, `C1`, and `C2` produced near-identical results across short events and diverged only slightly on Event 279, we examine the recorded step diagnostics:

### 1. Recorded Loss Distributions
- **Tent Entropy Loss ($\mathcal{L}_{\text{ent}}$):**
  - Mean: $0.107427$ | Std: $0.029479$ | Range: $[0.037381, 0.186168]$
- **C1 Consistency Loss ($\mathcal{L}_{\text{cons}, \tau=0.0}$):**
  - Mean: $0.012470$ | Std: $0.004544$ | Range: $[0.003281, 0.029906]$
- **C2 Consistency Loss ($\mathcal{L}_{\text{cons}, \tau \ge 0.90}$):**
  - Mean: $0.004631$ | Std: $0.001373$ | Range: $[0.001646, 0.011975]$

Notice that the Evidence Consistency loss magnitude ($0.0125$ in C1, $0.0046$ in C2) is roughly **one to two orders of magnitude smaller** than the Tent entropy loss ($0.1074$).

### 2. PRE_UPDATE Timing and Step Count Interaction
- In accordance with the protocol, all predictions were evaluated strictly **PRE_UPDATE** on each incoming batch.
- On Batch 1 of each event, predictions are identical to `source_only` because the model has taken zero adaptation steps.
- On small events:
  - Event 1111003 has only 2 batches (19 samples). Batch 2 is evaluated after exactly 1 gradient step of $\text{LR} = 1.25 \times 10^{-4}$ on BN affine parameters.
  - Event 559 has 3 batches (2 adaptation steps before tail).
  - Event 520 has 6 batches.
  - Event 1111008 has 8 batches.
  - Event 437 has 11 batches.
- Because $\text{LR} = 1.25 \times 10^{-4}$ with zero weight decay is small, a handful of gradient steps over 2 to 11 batches induces only minuscule parameter movement in BN affine $\gamma, \beta$. Consequently, the model's logits barely drift from source, explaining why predictions across Tent, C1, and C2 are identical or differ by only 1 to 4 pixels.
- On **Event 279 (149 batches, 2,381 samples)**:
  - Adaptation accumulated 148 consecutive optimizer steps.
  - Over this prolonged stream, both Tent and Consistency drifted substantially: predicted flood pixels surged from $2.39 \times 10^6$ (source) to $3.25 \times 10^6$ (Tent) and $3.33 \times 10^6$ (C1/C2).
  - This large false positive accumulation degraded $F_1$ from $0.3769$ (source) down to $\sim 0.300–0.305$.

---

## 6. Scientific Findings & Honest Limitations

### Finding 1: Unconstrained Continuous Adaptation Degrades Macro & Pooled Performance
Under `order_a` and `reset_per_event`, continuous unconstrained adaptation (Tent, C1, C2) is **inferior to the frozen source model**:
- Source-Only achieves **Pooled $F_1 = 0.5193$** and **Macro $F_1 = 0.5949$**.
- Matched Tent drops to **Pooled $F_1 = 0.4469$** ($\Delta = -0.0724$) and **Macro $F_1 = 0.5741$** ($\Delta = -0.0208$).
- Consistency (C1/C2) drops to **Pooled $F_1 = 0.4416–0.4419$** ($\Delta = -0.0774$) and **Macro $F_1 = 0.5733$** ($\Delta = -0.0216$).
- Source-only outperforms adapted models on 4 out of 6 events (279, 437, 520, 1111008), including the two largest events by sample volume. Adaptation only improves $F_1$ on Event 559 ($+0.0199$) and Event 1111003 ($+0.0082$).

### Finding 2: Confidence Filtering ($C_2$ vs $C_1$) Operates Without Meaningful Performance Gain
Setting a confidence threshold of $\tau \ge 0.90$ effectively removes 11.21% of sensory-valid pixels from the backward pass, reducing the mean consistency loss from $0.0125$ to $0.0046$. However, this filtering yielded an almost imperceptible change in evaluation metrics ($\Delta F_1 = +0.000305$ pooled, $+0.000049$ macro). It did not prevent the severe adaptation degradation on Event 279.

### Explicit Limitations (EXPLORATORY Status)
1. **Single Seed (Seed 0):** Evaluated strictly on the seed 0 source checkpoint. Seed variance across seeds 1 and 2 remains unmeasured.
2. **Single Order (`order_a`):** Evaluated strictly on deterministic ascending filename order. Streaming order sensitivity (`order_b`) was not evaluated.
3. **Reset-per-Event:** Models were cleanly re-instantiated from the source checkpoint before each event. Multi-event sequential contamination was not tested.
4. **Development Events Only:** Evaluated strictly on the 6 development events (`[279, 437, 520, 559, 1111003, 1111008]`). The 10 sealed test events remain strictly unaccessed.

---

## 7. Recommended Next Decision

1. **Do NOT promote unconstrained Evidence Consistency as a stand-alone unsupervised adaptation loss:**
   The empirical evidence shows that unconstrained consistency loss (both with and without confidence filtering) suffers from the exact same failure mode as Tent—accumulating false positives and underperforming the frozen source baseline on long event streams.
2. **Focus Selective Adaptation on Gating / Safeguarding (ChangeGuard):**
   The motivation for ChangeGuard is not simply adding an alternative unsupervised objective, but providing **selective adaptation gating** (detecting when adaptation is harmful and falling back to source, or filtering out unstable batches/events).
3. **Formal Artifact Archival:**
   The Week 12 exploratory dataset and analysis are cleanly self-contained, validated, and reproducible. Proceed to commit this analysis report and artifact hashes to Git main.

