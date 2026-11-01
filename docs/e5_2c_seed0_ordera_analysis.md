# ChangeGuard Work Package E5.2C: Offline Temporal-Risk Analysis Report (Seed 0, Stream Order A)

**Governing Plans:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) & [`docs/e5_2c_analysis_contract.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_2c_analysis_contract.md)  
**Configuration File:** [`configs/changeguard/e5_temporal_risk_analysis_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/e5_temporal_risk_analysis_v1.json) (`aeb8854a765e0bcf04e9ad744ef461942add0074613f7f8bb29bc4f39eba2219`)  
**Source Collection Run:** [`outputs/e5_temporal_evidence/formal_s0_oa_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/formal_s0_oa_v1/) (`e5_temporal_risk_s0_oa_v1`)  
**Completed v1 Analysis Directory:** [`outputs/e5_temporal_evidence/analysis_s0_oa_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v1/)  
**Execution Environment:** CPU only (`vfm_env`, Python 3.10.20, x86_64 Linux)  
**Status:** COMPLETE — WORDING & REPRODUCIBILITY AUDITED (READY FOR COMMITTED V2 REGENERATION)  

---

## 1. Executive Summary & Epistemic Boundaries

This document reports the deterministic offline temporal-risk analysis for the formal screening cell **Seed 0, Stream Order A (`s0_oa_v1`)**, evaluating 2,826 samples across 179 authoritative batch shards and six development events (`[279, 437, 520, 559, 1111003, 1111008]`).

### 1.1 Strict Epistemic Discipline
1. **Descriptive Screening Only**: With $N = 6$ development events, this evaluation provides an exploratory and descriptive screening of pre-update risk indicators against observed adaptation outcomes. Statistical claims of general population superiority or asymptotic convergence are explicitly out of scope.
2. **Zero In-Sample Fitting / No Post-Hoc Composites**: The multivariate generic-only and generic-plus-change combinations have **no predeclared calibration rule**. In strict compliance with the analysis contract, they are designated **`NOT EVALUATED — CALIBRATION RULE NOT PREDECLARED`**. No post-hoc weights or high-capacity classifiers were trained or tuned on these data.
3. **Quarantine Assurance**: Zero sealed-test samples or events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) were accessed or referenced.
4. **Grouped Statistical Discipline**: The 179 batches are evaluated within their event clusters; they are never treated as 179 independent events.
5. **Restrained Terminology**: No causal, predictive, or advance-warning claims are made. Observed associations describe empirical correlations under the frozen protocol.

---

## 2. Provenance & Preflight Verification

| Item | Recorded Value | Integrity Verification |
| :--- | :--- | :--- |
| **Analysis Git Commit** | `0af9c004e11952574e1b0f723f1979787176327b` | Baseline commit prior to offline execution refinement |
| **Source Run ID** | `e5_temporal_risk_s0_oa_v1` | Matches formal collection manifest |
| **Source Git Commit** | `7ae4535361e04a91728b9105f3cb03f9444cb2be` | Match |
| **Source Status** | `COMPLETED_RUN` | Verified via `completion_status.json` |
| **Evaluated Events** | `[279, 437, 520, 559, 1111003, 1111008]` | Exact match with approved development partition |
| **Total Samples** | 2,826 | Verified bit-exact with zero duplicates |
| **Total Evidence Shards** | 179 | All 179 `.npz` SHA-256 hashes verified bit-exact |
| **Failure Records** | 0 | `failure.json` absent across all events |
| **Analysis Tmux Session** | `cg_e5_2c2_analysis_s0_oa_v1` | Detached session; completed with exit code 0 |
| **Analysis Log** | `logs/e5_temporal_evidence/analysis_s0_oa_v1.log` | Clean execution; no warnings or exceptions |
| **Peak Analysis Memory** | ~1.1 GB RAM | Bounded execution; shards loaded incrementally |
| **Analysis Runtime** | ~35.2 seconds | Rapid deterministic CPU execution |

---

## 3. Direct Observations: Primary Outcome Evaluation (Event Level)

Event Class-2 (flood) F1 scores were reconstructed offline from the frozen prediction arrays and development ground-truth masks ($M_{\text{valid}} \land y \ne 3$):
$$\text{F1} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}, \quad \Delta \text{F1} = \text{F1}_{\text{cand}} - \text{F1}_{\text{src}}$$

Regime thresholds:
- **Harmful**: $\Delta \text{F1} \le -0.010$ (loss of $\ge 1.0$ percentage points)
- **Beneficial**: $\Delta \text{F1} \ge +0.010$ (gain of $\ge 1.0$ percentage points)
- **Neutral**: $|\Delta \text{F1}| < 0.010$

### 3.1 Event Outcomes Table
*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v1/event_outcomes.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v1/event_outcomes.csv)

| Event ID | Samples | Shards | Valid Pixels | Source F1 | BN Control F1 | Candidate F1 | $\Delta \text{F1}_{\text{Cand-Src}}$ | $\Delta \text{F1}_{\text{BN-Src}}$ | $\Delta \text{F1}_{\text{Cand-BN}}$ | Candidate Outcome | BN Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | 2,381 | 149 | 116,199,247 | 0.3769 | 0.3758 | 0.3224 | **-0.0545** | -0.0011 | -0.0533 | **Harmful** | Neutral |
| **437** | 169 | 11 | 7,859,430 | 0.6994 | 0.6925 | 0.6352 | **-0.0642** | -0.0069 | -0.0573 | **Harmful** | Neutral |
| **520** | 92 | 6 | 3,455,248 | 0.7138 | 0.7106 | 0.6997 | **-0.0140** | -0.0032 | -0.0108 | **Harmful** | Neutral |
| **559** | 42 | 3 | 1,843,520 | 0.6726 | 0.6841 | 0.6923 | **+0.0197** | +0.0115 | +0.0082 | **Beneficial** | Beneficial |
| **1111003** | 19 | 2 | 815,490 | 0.2568 | 0.2665 | 0.2650 | **+0.0082** | +0.0096 | -0.0014 | **Neutral** | Neutral |
| **1111008** | 123 | 8 | 5,686,145 | 0.8499 | 0.8496 | 0.8479 | **-0.0019** | -0.0003 | -0.0016 | **Neutral** | Neutral |

### 3.2 Direct Observations & Bounded Context Interpretation
1. **Empirical Event Distribution**:
   - **Harmful Events**: 3 out of 6 (50.0%) — Events 279, 437, 520.
   - **Beneficial Events**: 1 out of 6 (16.7%) — Event 559.
   - **Neutral Events**: 2 out of 6 (33.3%) — Events 1111003, 1111008.
   - **Harm Prevalence**: $\pi_{\text{harm}} = 3 / 6 = 0.500$ (50.0%).
2. **Degeneracy Guard Status**: Because both harmful ($N=3$) and non-harmful ($N=3$) cases exist, event-level AUROC and AUPRC are mathematically non-degenerate and evaluated directly against the 0.500 empirical prevalence benchmark.
3. **Bounded Interpretation of Normalization Control**:
   - Under the declared BN-no-update control, normalization-context change alone did not reproduce the full degradation observed for the adaptive candidate in Events 279, 437, and 520.
   - The remaining candidate–control difference is consistent with an effect associated with the adaptation procedure, but this comparison does not uniquely identify weight updates as the sole causal mechanism.
   - In Event 559, both the BN-no-update control (+0.0115) and Candidate (+0.0197) show positive signed shifts relative to the frozen source reference, indicating that normalization adjustments to current test statistics are associated with performance gains in this event.

---

## 4. Descriptive Associations: Signal Discrimination & Statistical Summary

### 4.1 Event-Level Descriptive Ranking
*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v1/signal_event_summary.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v1/signal_event_summary.csv)

All signals were extracted strictly **`PRE_UPDATE`** before candidate adaptation updates. Risk directions are frozen: higher denotes higher risk (with `mean_max_confidence` negated).

| Indicator Key | Family | Declared Risk Direction | Available Events | Spearman $\rho$ (vs $\Delta \text{F1}$) | AUROC (Harm) | AUPRC (Harm) | No-Skill Benchmark |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`source_to_candidate_kl`** | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.8857** | **1.0000** | **1.0000** | 0.5000 |
| **`bn_weight_displacement`** | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.8286** | **0.8889** | **0.9167** | 0.5000 |
| **`s_flood_expansion`** | Change-Specific | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.7714** | **1.0000** | **1.0000** | 0.5000 |
| **`s_null`** | Change-Specific | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.6000** | **0.6667** | **0.6389** | 0.5000 |
| **`abs_delta_flood_pixels`** | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.5429** | **0.7778** | **0.8056** | 0.5000 |
| **`hard_disagreement_rate`** | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | **-0.4286** | **0.7778** | **0.8667** | 0.5000 |
| `s_flood_disappearance` | Change-Specific | Higher $\rightarrow$ Higher Risk | 6 / 6 | +0.2571 | 0.4444 | 0.5556 | 0.5000 |
| `mean_entropy` | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | +0.0857 | 0.4444 | 0.5556 | 0.5000 |
| `aug_inconsistency` | Generic | Higher $\rightarrow$ Higher Risk | 6 / 6 | +0.0857 | 0.4444 | 0.5556 | 0.5000 |
| `mean_max_confidence` | Generic | Lower $\rightarrow$ Higher Risk | 6 / 6 | +0.0286 | 0.4444 | 0.5556 | 0.5000 |

#### Event-Level Findings:
- **Strongest Negative Monotonic Association**: `source_to_candidate_kl` ($\rho = -0.886$) and `bn_weight_displacement` ($\rho = -0.829$) show strong negative rank correlation with signed performance change $\Delta \text{F1}$.
- **Top Descriptive Event Indicators**: `source_to_candidate_kl` and `s_flood_expansion` achieve descriptive separation ($\text{AUROC} = 1.000$, $\text{AUPRC} = 1.000$) between harmful and non-harmful events on this sample.
- **Uninformative Confidence Signals**: Raw model predictive confidence metrics (`mean_max_confidence`, `mean_entropy`, `aug_inconsistency`) exhibit near-zero rank correlation and near-random AUROC (~0.444) on these six events.

---

### 4.2 Grouped Batch-Level Analysis ($N = 179$ Batches)
*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v1/signal_batch_grouped_summary.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v1/signal_batch_grouped_summary.csv)

Across 179 batches, 73 produced harmful candidate batch outcomes ($\Delta \text{F1}_{\text{batch}} \le -0.010$), 18 produced beneficial outcomes, and 88 produced neutral outcomes. Batch outcome harm prevalence is **40.78%** ($\pi = 0.4078$). Grouped Spearman $\rho$ is averaged across individual event clusters to maintain grouped discipline.

| Indicator Key | Family | Available Batches | Mean Grouped Spearman $\rho$ | Pooled AUROC (Harm) | Pooled AUPRC (Harm) | No-Skill Benchmark |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`abs_delta_flood_pixels`** | Generic | 179 / 179 | -0.1736 | **0.7190** | **0.6457** | 0.4078 |
| **`s_flood_expansion`** | Change-Specific | 179 / 179 | -0.1360 | **0.6929** | **0.6221** | 0.4078 |
| **`source_to_candidate_kl`** | Generic | 179 / 179 | -0.2329 | **0.6530** | **0.5765** | 0.4078 |
| **`hard_disagreement_rate`** | Generic | 179 / 179 | -0.2423 | **0.6215** | **0.5870** | 0.4078 |
| **`s_null`** | Change-Specific | 179 / 179 | -0.1840 | **0.5861** | **0.4725** | 0.4078 |
| **`aug_inconsistency`** | Generic | 179 / 179 | -0.1608 | **0.5691** | **0.5300** | 0.4078 |
| **`mean_max_confidence`** | Generic | 179 / 179 | -0.2030 | **0.5682** | **0.5055** | 0.4078 |
| **`mean_entropy`** | Generic | 179 / 179 | -0.1559 | **0.5584** | **0.4964** | 0.4078 |
| `s_flood_disappearance` | Change-Specific | 179 / 179 | +0.1413 | 0.4497 | 0.4254 | 0.4078 |
| `bn_weight_displacement` | Generic | 179 / 179 | +0.2293 | 0.4360 | 0.3726 | 0.4078 |

#### Grouped Batch Findings:
- At batch resolution, **spatial shift indicators** (`abs_delta_flood_pixels` $\text{AUROC} = 0.7190$, `s_flood_expansion` $\text{AUROC} = 0.6929$) provide fine-grained discrimination over purely parameter-level metrics.
- `bn_weight_displacement` is uninformative at step 0 within individual batches before adaptation steps have accumulated.

---

## 5. Metric-Semantics Audit & Rejection Budget Analysis

*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v1/rejection_budget_summary.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v1/rejection_budget_summary.csv)

### 5.1 Aggregation Definitions & Mathematical Semantics
1. **Definition of Baseline Candidate F1 (0.4578)**:
   - This metric represents the **pooled pixel Class-2 F1 score** across all valid pixels in all 179 batches across the six development events:
     $$\text{F1}_{\text{pooled}} = \frac{2 \sum_{b=1}^{179} \text{TP}_{\text{cand}, b}}{2 \sum_{b=1}^{179} \text{TP}_{\text{cand}, b} + \sum_{b=1}^{179} \text{FP}_{\text{cand}, b} + \sum_{b=1}^{179} \text{FN}_{\text{cand}, b}} = 0.4578$$
   - It is **not** the unweighted macro event F1 (which is $\frac{1}{6} \sum_{e} \text{F1}_e = 0.5771$).
   - The pooled pixel F1 is lower because Event 279 comprises 116,199,247 valid pixels (85.5% of the total 135,858,070 valid pixels across all six events) and has a low Class-2 F1 of 0.3224.
2. **Computation of Net Realized F1**:
   - For a given rejection budget $b$, threshold $\tau_b$ is the $(1 - b)$-quantile of the risk score.
   - Candidate batches with score $> \tau_b$ are rejected and replaced by the **exactly matched source confusion counts** ($\text{TP}_{\text{src}, b}, \text{FP}_{\text{src}, b}, \text{FN}_{\text{src}, b}$).
   - The realized F1 is computed as the pooled pixel F1 over all 179 batches after substitution.
3. **Threshold Selection & Boundary Tie Handling**:
   - Scores strictly follow the declared risk direction (higher score = higher risk; `mean_max_confidence` is negated).
   - Batches are rejected if $\text{score} > \tau_b$ (strict inequality). Tied scores exactly equal to $\tau_b$ are not rejected.
4. **Unit of Harm Counting**:
   - The 73 counted harmful units represent **harmful candidate batch outcomes** ($\Delta \text{F1}_{\text{batch}} \le -0.010$), evaluated on test data regardless of whether an optimizer parameter update occurred on that batch.
   - The 18 beneficial units represent **beneficial candidate batch outcomes** ($\Delta \text{F1}_{\text{batch}} \ge +0.010$) using the identical batch-level evaluation unit.

### 5.2 Rejection Performance Across Budgets

#### Signal: `s_flood_expansion` (Top Change-Specific Indicator)
| Budget $b$ | Threshold $\tau_b$ | Rejected Batches | Harmful Batch Outcomes Blocked | Harm Averted Rate | Beneficial Batch Outcomes Rejected | Lost Benefits Rate | Net Realized Pooled Pixel F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline ($b=0$) | — | 0 | 0 / 73 | 0.0% | 0 / 18 | 0.0% | 0.4578 |
| **0.05** | 0.02185 | 9 | 7 / 73 | 9.6% | 0 / 18 | 0.0% | 0.4809 |
| **0.10** | 0.01576 | 18 | 12 / 73 | 16.4% | 0 / 18 | 0.0% | 0.4882 |
| **0.15** | 0.01248 | 27 | 18 / 73 | 24.7% | 0 / 18 | 0.0% | 0.4962 |
| **0.20** | 0.01107 | 36 | 24 / 73 | **32.9%** | 1 / 18 | **5.6%** | **0.5000** |
| **0.25** | 0.00923 | 45 | 30 / 73 | **41.1%** | 1 / 18 | **5.6%** | **0.5032** |
| **0.30** | 0.00835 | 54 | 36 / 73 | **49.3%** | 1 / 18 | **5.6%** | **0.5064** |

#### Signal: `source_to_candidate_kl` (Top Generic Indicator)
| Budget $b$ | Threshold $\tau_b$ | Rejected Batches | Harmful Batch Outcomes Blocked | Harm Averted Rate | Beneficial Batch Outcomes Rejected | Lost Benefits Rate | Net Realized Pooled Pixel F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline ($b=0$) | — | 0 | 0 / 73 | 0.0% | 0 / 18 | 0.0% | 0.4578 |
| **0.05** | 0.03976 | 9 | 7 / 73 | 9.6% | 0 / 18 | 0.0% | 0.4803 |
| **0.10** | 0.02798 | 18 | 11 / 73 | 15.1% | 1 / 18 | 5.6% | 0.4865 |
| **0.15** | 0.02441 | 27 | 17 / 73 | 23.3% | 1 / 18 | 5.6% | 0.4932 |
| **0.20** | 0.01951 | 36 | 22 / 73 | **30.1%** | 1 / 18 | **5.6%** | **0.4959** |
| **0.25** | 0.01635 | 45 | 26 / 73 | **35.6%** | 2 / 18 | **11.1%** | **0.4992** |
| **0.30** | 0.01516 | 54 | 32 / 73 | **43.8%** | 2 / 18 | **11.1%** | **0.5024** |

---

## 6. Artifact Accounting & Reproducibility Assessment

### 6.1 Artifact Accounting
In the completed v1 directory (`outputs/e5_temporal_evidence/analysis_s0_oa_v1/`), two summary JSON files existed with identical SHA-256 hashes:
- `analysis_summary.json`
- `e5_risk_analysis_summary.json`

**Reason for Duplication in v1**: The JSON configuration file specified `"summary_json": "e5_risk_analysis_summary.json"`, while the prompt's minimum required file list specified `analysis_summary.json`. The v1 analyzer satisfied both requirements by writing the identical summary payload to both filenames.

**Protocol for v2 Regeneration**: To eliminate duplication while maintaining contractual precision, the analyzer for v2 has been refactored to emit only the single canonical `analysis_summary.json`. The completed v1 directory remains unaltered.

### 6.2 Code Chronology & Reproducibility Status
- The v1 analysis was executed from working-tree modifications on top of commit `0af9c004e11952574e1b0f723f1979787176327b`.
- The modifications were required because commit `0af9c00` contained only an initial skeleton that did not perform offline label matching from rasters or generate the required CSV tables.
- Because the analyzer source was not committed in HEAD prior to running v1, **the v1 analysis is computationally complete but not formally frozen in Git**.
- Therefore, a clean committed regeneration into `analysis_s0_oa_v2/` is required to establish strict cryptographic reproducibility.

---

## 7. Scientific Checkpoint Conclusion

The empirical findings from the Seed 0, Stream Order A screening cell warrant the following bounded conclusions:

1. **Observed Adaptation Behavior**: Three of the six development events (279, 437, 520) exhibited harmful adaptation outcomes ($\Delta \text{F1} \le -0.010$); one event (559) was beneficial ($\Delta \text{F1} \ge +0.010$); two events (1111003, 1111008) were neutral ($|\Delta \text{F1}| < 0.010$).
2. **Promising Descriptive Risk Indicators**: Generic source-to-candidate KL divergence (`source_to_candidate_kl`) and change-specific source flood expansion (`s_flood_expansion`) exhibited the strongest descriptive discrimination against harmful adaptation outcomes.
3. **Sample Size Limitations**: Six development events are insufficient to support confirmed population-level conclusions or causal assertions.
4. **No Composite Evaluated**: No multivariate composite rule was evaluated, as no calibration rule was predeclared.
5. **Replication Requirement**: Repeat seed and arrival order evidence (e.g. Seed 1/2, Order B) is required to evaluate indicator stability across stream conditions.
6. **No Deployment Claim**: These results represent an offline descriptive screening and do not establish a production deployment gate or final ChangeGuard efficacy claim.
