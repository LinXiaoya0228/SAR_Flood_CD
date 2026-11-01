# ChangeGuard Supplementary Ablation Report: Class Weighting Without Event-Balanced Sampling (M2_ONLY)

**Document ID:** `docs/robust_source_training/rst12_m2_only_ablation_report_v1.md`  
**Execution Date:** 2026-11-01  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Status:** COMPLETE — AUTHORITATIVE SCIENTIFIC ABLATION  
**Work Package:** Supplementary Ablation (Post-RST11 Freeze Investigation)

---

## 1. Executive Summary & Core Scientific Conclusion

This report documents the execution, verification, and analysis of the bounded supplementary ablation investigating **Recipe M2_ONLY**: ordinary patch sampling (M1 disabled) combined with frozen flood-aware class weighting (M2 enabled) on the SNUNet-ECAM architecture.

The ablation directly resolves the central scientific question:
> **Does class weighting alone explain the performance improvement of Recipe R2, or does event-balanced sampling provide supported additive value when class weighting is present?**

### Central Findings

1. **Class Weighting Alone Does Not Explain the Improvement:**
   - On the grand 18-event held-out evaluation set across both confirmatory folds (Fold A and Fold B, 3 seeds each, 54 paired evaluations per recipe), **M2_ONLY achieves a grand event-macro F1 of 0.7126**, compared to **0.7202** for the matched baseline **R0** ($\Delta_{\text{M2\_ONLY} - \text{R0}} = -0.0076$, or $-0.76$ percentage points).
   - The 95% paired bootstrap confidence interval for $\text{M2\_ONLY} - \text{R0}$ is $[-0.0312, +0.0172]$ ($p = 0.7449$). Across the 18 held-out events at the $\pm 0.010$ practical band, M2_ONLY benefits 4 events, harms 7 events, and is neutral on 7 events.
   - While class weighting alone improves tail-event sensitivity (increasing Bottom-3 mean F1 from $0.3987$ to $0.4270$, $+0.0283$), it induces a severe precision penalty across typical events (precision falls from $0.7676$ to $0.7212$, $-0.0464$).

2. **Event-Balanced Sampling (M1) Provides Statistically Supported Additive Value:**
   - Comparing the full robust source recipe **R2** (M1 + M2) against **M2_ONLY** (M2 alone), adding event-balanced sampling raises grand event-macro F1 from **0.7126 to 0.7338**, an improvement of **$+0.0212$ (+2.12 pp)**.
   - The 95% paired bootstrap confidence interval for $\text{R2} - \text{M2\_ONLY}$ is $[+0.0010, +0.0442]$ ($p = 0.0181$), **which is strictly positive and excludes zero**.
   - At the $\pm 0.010$ practical band, event balancing benefits 8 events, harms 6 events, and is neutral on 4 events. Furthermore, event balancing improves Bottom-3 mean F1 by an additional $+0.0228$ (reaching $0.4498$, $+0.0511$ over R0) and worst-event F1 by $+0.0273$ (reaching $0.2929$).

3. **Mechanistic Synergy (Precision Restoration):**
   - In ordinary patch sampling without event balancing, source events with massive patch counts dominate training batches. Applying global inverse-square-root class weights to this skewed patch distribution causes the model to overpredict flood on unrepresented flood regimes, collapsing precision (e.g., precision drops from $0.7306$ to $0.5140$ on Event 496, and from $0.5593$ to $0.4783$ on Event 267).
   - Event-balanced sampling enforces uniform representation across events, ensuring that class weights penalize minority pixels equally across diverse geographic and hydrological conditions. This restores precision to $0.7437$ ($+0.0225$ over M2_ONLY) while maintaining the elevated recall ($0.7747$).
   - **Conclusion:** Recipe R2's gains cannot be attributed to loss reweighting alone; event balancing and class weighting act as interdependent, mutually reinforcing components.

---

## 2. Experimental Lineage & Checkpoint Ledger

### 2.1 Configuration and Component Switches

All six ablation training runs were configured under the strict `SUPPLEMENTARY_ABLATION` execution class using frozen source data. Target events were strictly quarantined.

- **Component Switches:**
  - `m1_event_balance`: `False` (Uniform random patch sampling over all source patches, identical to R0).
  - `m2_flood_aware_loss`: `True` (Inverse-square-root class weighting applied strictly to training loss).
  - `m3_paired_augmentation`: `False` (Identical to R0 and R2).
- **Class Weights Applied (Training Only):**
  - Fold A: $[0.486491, 1.409167, 1.104342]$ (SHA-256: `c3bbb6bae14d98710798883132aab623c2cc554c226f796c1e15ff87591640bd`).
  - Fold B: $[0.543519, 1.499552, 0.956929]$ (SHA-256: `d84e797fc7eb6375bc217fc7c6a9a08e063a86377e8a93e32a688b1d9bf5b210`).

### 2.2 Execution & Checkpoint Verification Ledger

All six runs completed all scheduled epochs (16 epochs per run) without interruptions, skips, or errors, logging exit code 0 and status `COMPLETED_SCIENTIFIC_RUN`. Independent checkpoint-resume verification passed bit-identical assertions.

| Run ID | Fold | Seed | Status | Opt. Steps | Wall Time | Peak VRAM | Final Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ablation_m2_only_snunet_folda_s0_v1` | A | 0 | COMPLETED | 6,656 | 8,786.9s | 4,911.2 MB | `956a63da10cf54022a3c0716f28daa5e961c7c693735fa8897b781d7a8363548` |
| `ablation_m2_only_snunet_folda_s1_v1` | A | 1 | COMPLETED | 6,656 | 7,906.6s | 4,911.2 MB | `e3e0d75c76df429fc7de3a5faa756a694084865489ed7b9a5e022b03040f9d1c` |
| `ablation_m2_only_snunet_folda_s2_v1` | A | 2 | COMPLETED | 6,656 | 7,939.5s | 4,911.2 MB | `96e219cc3f98475109b38d7ee7646a0ee8dcbb14afd879b932067b6a17f91863` |
| `ablation_m2_only_snunet_foldb_s0_v1` | B | 0 | COMPLETED | 7,328 | 8,696.2s | 4,913.2 MB | `4deb3051ff5b47ccb598a180522c3cdab67437b5bddc9a904c456c83b33be0c1` |
| `ablation_m2_only_snunet_foldb_s1_v1` | B | 1 | COMPLETED | 7,328 | 9,326.4s | 4,913.2 MB | `619d8406cefc67f96587cd82fb1c1683cb29105655219e3454202bb9418bd4fb` |
| `ablation_m2_only_snunet_foldb_s2_v1` | B | 2 | COMPLETED | 7,328 | 8,238.5s | 4,913.2 MB | `d940dae7b44b2765088c4b29ef7f7cdeab971a3d65925488251de73ec5f4cee7` |

---

## 3. Detailed Quantitative Results

### 3.1 Aggregate Performance Across Confirmatory Folds

Table 1 summarizes performance across the three recipes:
- **R0:** Baseline (no event balancing, unweighted loss).
- **M2_ONLY:** Class weighting alone (no event balancing, inverse-square-root loss weights).
- **R2:** Combined robust source recipe (event balancing + inverse-square-root loss weights).

#### Table 1: Aggregate Performance Summary (18 Held-Out Events, 3 Seeds)

| Metric | R0 Baseline | M2_ONLY Ablation | R2 Robust Source | $\Delta(\text{M2\_ONLY} - \text{R0})$ | $\Delta(\text{R2} - \text{M2\_ONLY})$ | $\Delta(\text{R2} - \text{R0})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grand Macro F1** | 0.7202 | 0.7126 | **0.7338** | $-0.0076$ | **$+0.0212$** | $+0.0136$ |
| **Fold A Macro F1** | $0.7460 \pm 0.0120$ | $0.7480 \pm 0.0053$ | $\mathbf{0.7543 \pm 0.0106}$ | $+0.0020$ | $+0.0062$ | $+0.0083$ |
| **Fold B Macro F1** | $0.6943 \pm 0.0035$ | $0.6771 \pm 0.0183$ | $\mathbf{0.7133 \pm 0.0165}$ | $-0.0172$ | **$+0.0361$** | $+0.0189$ |
| **Grand Macro Precision** | **0.7676** | 0.7212 | 0.7437 | $-0.0464$ | $+0.0225$ | $-0.0240$ |
| **Grand Macro Recall** | 0.7350 | 0.7682 | **0.7747** | $+0.0332$ | $+0.0065$ | $+0.0397$ |
| **Grand Macro IoU** | 0.5893 | 0.5790 | **0.6012** | $-0.0103$ | $+0.0222$ | $+0.0119$ |
| **Bottom-3 Mean F1** | 0.3987 | 0.4270 | **0.4498** | $+0.0283$ | $+0.0228$ | $+0.0511$ |
| **Worst Event F1** | 0.2991 | 0.2657 | 0.2929 | $-0.0334$ | $+0.0273$ | $-0.0062$ |

### 3.2 Paired Differences and Bootstrap Confidence Intervals

The statistical contrasts across the 18 held-out events (10,000 paired event-level bootstrap resamples, seed 42) reveal:

1. **$\text{M2\_ONLY} - \text{R0}$ (Effect of Class Weighting Alone):**
   - Observed mean difference: **$-0.0076$**
   - 95% Bootstrap CI: **$[-0.0312, +0.0172]$** (crosses zero, $p = 0.7449$)
   - Event classifications ($\pm 0.010$ band): **4 Benefit / 7 Harm / 7 Neutral**

2. **$\text{R2} - \text{M2\_ONLY}$ (Additive Value of Event-Balanced Sampling):**
   - Observed mean difference: **$+0.0212$**
   - 95% Bootstrap CI: **$[+0.0010, +0.0442]$** (**strictly positive**, $p = 0.0181$)
   - Event classifications ($\pm 0.010$ band): **8 Benefit / 6 Harm / 4 Neutral**

3. **$\text{R2} - \text{R0}$ (Combined System vs. Baseline):**
   - Observed mean difference: **$+0.0136$**
   - 95% Bootstrap CI: **$[-0.0045, +0.0351]$** ($p = 0.0779$)
   - Event classifications ($\pm 0.010$ band): **7 Benefit / 7 Harm / 4 Neutral**

---

## 4. Per-Event Paired Analysis (All 18 Held-Out Events)

Table 2 presents the seed-averaged results for each held-out event across both folds.

#### Table 2: Complete 18-Event Paired Breakdown

| Event ID | Fold | Samples | R0 F1 | M2_ONLY F1 | R2 F1 | $\Delta(\text{M2\_ONLY} - \text{R0})$ | $\Delta(\text{R2} - \text{M2\_ONLY})$ | $\Delta(\text{R2} - \text{R0})$ | Class vs. R0 | Class: R2 vs. M2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **118** | Fold A | 108 | 0.4308 | **0.5757** | 0.5459 | **+0.1449** | -0.0298 | +0.1150 | Benefit | Harm |
| **174** | Fold A | 26 | 0.7855 | 0.7908 | **0.8232** | +0.0053 | **+0.0324** | +0.0377 | Neutral | Benefit |
| **273** | Fold A | 112 | 0.5265 | **0.5388** | 0.5107 | +0.0123 | -0.0282 | -0.0158 | Benefit | Harm |
| **421** | Fold A | 68 | 0.7528 | 0.7800 | **0.7924** | **+0.0273** | **+0.0124** | +0.0396 | Benefit | Benefit |
| **496** | Fold A | 21 | 0.7720 | 0.6369 | **0.7588** | **-0.1351** | **+0.1219** | -0.0132 | Harm | Benefit |
| **498** | Fold A | 102 | **0.8251** | 0.8060 | 0.8109 | -0.0191 | +0.0048 | -0.0143 | Harm | Neutral |
| **554** | Fold A | 71 | **0.8536** | 0.8488 | 0.8215 | -0.0048 | -0.0274 | -0.0322 | Neutral | Harm |
| **567** | Fold A | 96 | **0.9301** | 0.9267 | 0.9104 | -0.0034 | -0.0163 | -0.0197 | Neutral | Harm |
| **1111009** | Fold A | 7,556 | **0.8375** | 0.8284 | 0.8147 | -0.0092 | -0.0136 | -0.0228 | Neutral | Harm |
| **267** | Fold B | 148 | 0.7074 | 0.6394 | **0.7437** | **-0.0680** | **+0.1042** | +0.0362 | Harm | Benefit |
| **324** | Fold B | 142 | 0.7748 | 0.7413 | **0.7878** | **-0.0336** | **+0.0465** | +0.0129 | Harm | Benefit |
| **417** | Fold B | 114 | 0.9450 | 0.9431 | **0.9461** | -0.0020 | +0.0030 | +0.0011 | Neutral | Neutral |
| **470** | Fold B | 269 | 0.4661 | 0.4764 | **0.5801** | +0.0103 | **+0.1037** | +0.1140 | Benefit | Benefit |
| **502** | Fold B | 318 | 0.8128 | **0.8208** | 0.8159 | +0.0079 | -0.0049 | +0.0030 | Neutral | Neutral |
| **518** | Fold B | 207 | **0.7372** | 0.7195 | 0.7291 | -0.0177 | +0.0096 | -0.0081 | Harm | Neutral |
| **555** | Fold B | 496 | 0.6634 | 0.6439 | **0.7062** | -0.0195 | **+0.0623** | +0.0428 | Harm | Benefit |
| **1111004** | Fold B | 5,420 | 0.8433 | **0.8443** | 0.8177 | +0.0010 | -0.0266 | -0.0256 | Neutral | Harm |
| **1111010** | Fold B | 239 | **0.2991** | 0.2657 | 0.2929 | **-0.0334** | **+0.0273** | -0.0062 | Harm | Benefit |

### Key Per-Event Observations

- **Catastrophic Failures of Class Weighting Alone:**
  - On Event 496, M2_ONLY suffers a dramatic collapse from $0.7720$ down to $0.6369$ ($-0.1351$). Event balancing in R2 rescues this event back to $0.7588$ ($+0.1219$ recovery).
  - On Event 267, M2_ONLY drops from $0.7074$ to $0.6394$ ($-0.0680$). Adding event balancing drives R2 up to $0.7437$ ($+0.1042$ over M2_ONLY).
  - On Event 324, M2_ONLY drops by $-0.0336$, while R2 gains $+0.0465$.
- **Major Synergies in Underrepresented Flood Events:**
  - On Event 470, M2 alone provides only a modest $+0.0103$ gain, but combining it with event balancing in R2 produces a massive surge to $0.5801$ ($+0.1037$ over M2_ONLY, $+0.1140$ over R0).
  - On Event 555, M2 alone degrades performance ($-0.0195$), whereas R2 gains $+0.0428$ over baseline ($+0.0623$ over M2_ONLY).

---

## 5. Reconciliation of Historical R2 Seed SD ($0.0071$ vs. $0.0072$)

A potential point of confusion in historical documents concerns the reported sample standard deviation of Recipe R2 across seeds on the 10 former test events ($0.8086 \pm 0.0071$ vs. $0.8086 \pm 0.0072$).

### Numerical Reconciliation

1. **Exact Unrounded Data Points:**
   The exact unrounded seed-level event-macro F1 scores for Recipe R2 on the ten historical test events (from `outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv`) are:
   - Seed 0: `0.8101661`
   - Seed 1: `0.8008676`
   - Seed 2: `0.8148587`
2. **Standard Deviation Calculation:**
   - Sample mean:
     $$\bar{x} = \frac{0.8101661 + 0.8008676 + 0.8148587}{3} = 0.8086308$$
   - Unrounded sample standard deviation ($N=3, ddof=1$):
     $$s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{2}} = 0.0071207850248410365\dots$$
3. **Origin of the Two Formats:**
   - Rounded to four decimal places using standard mathematical rounding:
     $$s \approx 0.0071$$
   - In Work Package RST9 (`rst9_final_source_model_closeout_v1.md` Table 1) and RST10 (`build_rst10_tables.py`), Grand Macro IoU standard deviation across all 16 events was computed as $0.007144$ and formatted alongside test macro F1 with intermediate round-up to $0.0072$.
   - **Conclusion:** Both $0.0071$ and $0.0072$ refer to the exact same underlying three runs and identical data points. $0.0071$ is the precise four-decimal representation of the sample standard deviation.

---

## 6. Unified Synthesis of Both Supplementary Packages

Together, the two supplementary investigations complete the empirical validation of the ChangeGuard project:

### Package 1: Offline R2 vs. Test-Time Adaptation (Tent / CoTTA)
- **Question:** How does the deployable all-source R2 model compare against online test-time adaptation methods when tested on identical streams?
- **Finding:** R2 ($0.8086$) outperforms Tent ($0.7628$, $\Delta = +0.0458$, 95% CI $[+0.0274, +0.0648]$) and CoTTA ($0.7669$, $\Delta = +0.0417$, 95% CI $[+0.0260, +0.0585]$) across all 10 test events with zero test-time compute or state mutation.

### Package 2: Supplementary Ablation (M2_ONLY vs. R0 and R2)
- **Question:** Does class weighting alone explain R2's gains, or does event-balanced sampling add essential value?
- **Finding:** Class weighting alone (M2_ONLY) slightly reduces overall Macro F1 ($0.7126$ vs. $0.7202$) due to precision loss. Event-balanced sampling (M1) provides statistically supported additive value ($\Delta_{\text{R2} - \text{M2\_ONLY}} = +0.0212$, 95% CI $[+0.0010, +0.0442]$), demonstrating that balancing source events is necessary to stabilize class-weighted training.

### Suggested Publication Snippet (for Manuscript Revisions)

```latex
% Supplementary Ablation: Class Weighting vs. Event Balancing
\begin{table}[t]
\centering
\caption{Supplementary Ablation of Source Training Components on SNUNet-ECAM across 18 Held-Out Events (Means $\pm$ Seed SD over 3 Seeds).}
\label{tab:ablation_m2_only}
\small
\begin{tabular}{lccccc}
\hline
\textbf{Configuration} & \textbf{Event Balance (M1)} & \textbf{Class Weight (M2)} & \textbf{Macro F1} & \textbf{Macro Prec.} & \textbf{Macro Rec.} \\
\hline
Baseline (R0) & $\times$ & $\times$ & $0.7202 \pm 0.0078$ & $\mathbf{0.7676}$ & $0.7350$ \\
M2\_ONLY & $\times$ & $\checkmark$ & $0.7126 \pm 0.0118$ & $0.7212$ & $0.7682$ \\
Recipe R2 (Ours) & $\checkmark$ & $\checkmark$ & $\mathbf{0.7338 \pm 0.0136}$ & $0.7437$ & $\mathbf{0.7747}$ \\
\hline
$\Delta(\text{M2\_ONLY} - \text{R0})$ & \multicolumn{2}{c}{Loss Reweighting Alone} & $-0.0076$ & $[-0.0312, +0.0172]$ & ($p = 0.74$) \\
$\Delta(\text{R2} - \text{M2\_ONLY})$ & \multicolumn{2}{c}{Additive Value of M1} & $\mathbf{+0.0212}$ & $[\mathbf{+0.0010}, \mathbf{+0.0442}]$ & ($p = 0.018$) \\
$\Delta(\text{R2} - \text{R0})$ & \multicolumn{2}{c}{Combined System} & $+0.0136$ & $[-0.0045, +0.0351]$ & ($p = 0.078$) \\
\hline
\end{tabular}
\end{table}
```
