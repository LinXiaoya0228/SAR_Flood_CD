# ChangeGuard Plan V3: Work Package E4.2 Screening Closeout Report

**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 9: Block E4)  
**Contract Document:** [`docs/e4_preflight_and_contracts.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e4_preflight_and_contracts.md)  
**Execution Environment:** Miniconda `vfm_env` (Python 3.10.19, PyTorch 2.4.1+cu121) on 4x NVIDIA RTX 3090 GPUs  
**Base Terminal Commit:** `4277ee8d224f5aa18a81b30812389b4bf7a9d006`  
**Date of Execution:** 2026-10-20  
**Status:** WORK PACKAGE E4.2 COMPLETE — PROMOTION VERDICT: `NONE ELIGIBLE` (All 4 K-Interventions Failed Plan V3 Promotion Rules)

---

## 1. Executive Summary & Core Scientific Findings

In Work Package **`E4.2`**, we conducted the full six-event screening grid evaluating four isolated class-aware test-time adaptation interventions (K1–K4) against the fixed internal consistency baseline (`E4_BASE` $\equiv$ `E2_CONF_90`). All experiments adhered strictly to the locked Plan V3 protocol: Seed 0, Order A, reset per event, batch size 16, confidence threshold $\tau = 0.90$, over all 2,826 patch pairs (179 batches) across the six development events.

### Key Empirical Findings:
1. **Mechanical Promotion Verdict — `NONE ELIGIBLE`:**
   None of the four class-aware interventions met either of the two pre-frozen Plan V3 method-promotion criteria:
   - **Substantial Aggregate Improvement ($\Delta\text{Macro F1} \ge +0.010000$):**
     - K1 (Class-Balanced Selection): $+0.000628$ (`FAIL`)
     - K2 (Flood Loss Weighting): $+0.001278$ (`FAIL`)
     - K3 (Permanent-Water Exclusion): $+0.002200$ (`FAIL`)
     - K4 (Flood Change Preservation): $-0.001641$ (`FAIL`)
   - **Tail Event Safety ($\Delta\text{WorstEvent F1} \ge 0.000000$):**
     All four K interventions degraded performance on the tail event (Event 1111003) from $0.265045$ (`E4_BASE`) down to $0.256616$ ($\Delta\text{WorstEvent F1} = -0.008429$). Every intervention cell failed the tail safety veto.
2. **K1 Bounded Majority Land Selection but Could Not Halt Shift:**
   K1 successfully enforced the uniform per-class quota cap $C_{\text{batch}} = \lfloor N_{\text{eligible}} / 3 \rfloor$, reducing total selected pseudo-labels from $88.79\%$ to $32.16\%$ of valid pixels. However, limiting Land selection alone did not prevent false-alarm surges ($c_{02}$ Land $\to$ Flood false positives increased from 2.72M to 2.88M pixels), resulting in a negligible $+0.0006$ Macro F1 gain and lower Pooled F1 ($0.4344$ vs $0.4419$).
3. **K3 Confirmed the Non-Invariance Caveat of Global BN Updates:**
   Excluding source-predicted permanent water ($\hat{y}_{\text{src}} == 1$) from the adaptation loss mask prevented direct gradient backpropagation on permanent water pixels. However, because batch normalization affine parameters $(\gamma, \beta)$ adapt globally across the image, permanent-water-to-flood false positives were NOT eliminated ($c_{12} = 185,329$ pixels in K3 vs $185,282$ in Base). This directly validates our pre-declared theoretical boundary: loss masking does not guarantee prediction invariance under global affine adaptation.
4. **K4 Caused Severe Flood Over-Expansion:**
   Enforcing the auxiliary source flood preservation loss ($\lambda_{\text{pres}} = 0.50$) generated massive gradient norms (mean $0.4641$ vs $0.0097$ in Base) and severe flood over-expansion ($1,522,770$ non-flood pixels converted to flood, an expansion rate of $1.15\%$). This caused a catastrophic surge in Land $\to$ Flood false alarms ($c_{02} = 3,129,209$ pixels, $+15.0\%$ vs Base), degrading Macro F1 to $0.571677$.
5. **Legitimate Negative Outcome — E4.3 Repeats Not Triggered:**
   Because all four cells failed the pre-frozen promotion criteria, Work Package E4.3 (Seed 1 / Order B repeat matrix) is formally **NOT REQUIRED**. Block E4 exits via the pre-declared negative-result terminal closeout route, exactly as occurred in Block E2 and Block E3.

---

## 2. Primary Screening Grid Results

The formal screening results across all development events are summarized below, benchmarked against matched and contextual controls:

| Cell ID | Condition Name | Evidence Level | Macro F1 | $\Delta$ Macro | Pooled F1 | Worst Event | Worst F1 | $\Delta$ Worst | Promotion Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Source-only* | Source unadapted (Seed 0, Order A) | `REUSED_CONTROL` | 0.594880 | +0.021562 | 0.519331 | Evt 1111003 | 0.256835 | -0.008210 | `CONTROL_ONLY` |
| *BN-no-update*| BN statistics only (bs=16, S0, OA) | `REUSED_CONTROL` | 0.573298 | -0.000020 | 0.441744 | Evt 1111003 | 0.265045 | -0.000000 | `CONTROL_ONLY` |
| *E3_LOGIT_10* | E3 Generic Protection Comparator | `GENERIC_COMPARATOR`| 0.573800 | +0.000482 | 0.444967 | Evt 1111003 | 0.265045 | -0.000000 | `GENERIC_COMPARATOR` |
| **`E4_BASE`** | Fixed Consistency Base (E2_CONF_90)| `REUSED_CONTROL` | **0.573318** | *ref (0.00)* | **0.441912** | Evt 1111003 | **0.265045** | *ref (0.00)* | `REUSED_CONTROL` |
| **`E4_K1_CLASS_BALANCED`** | K1: Class-Balanced Selection | `EXPERIMENTAL` | **0.573947** | +0.000628 | 0.434422 | Evt 1111003 | 0.256616 | -0.008429 | **`NOT_PROMOTED`** |
| **`E4_K2_FLOOD_WEIGHTED`** | K2: Flood Loss Weighting ($w_2=4.0$) | `EXPERIMENTAL` | **0.574597** | +0.001278 | 0.438544 | Evt 1111003 | 0.256616 | -0.008429 | **`NOT_PROMOTED`** |
| **`E4_K3_PERM_PROTECTED`** | K3: Source Perm-Water Exclusion | `EXPERIMENTAL` | **0.575518** | +0.002200 | 0.444536 | Evt 1111003 | 0.256616 | -0.008429 | **`NOT_PROMOTED`** |
| **`E4_K4_FLOOD_PRESERVED`**| K4: Flood Preservation ($\lambda=0.50$) | `EXPERIMENTAL` | **0.571677** | -0.001641 | 0.419146 | Evt 1111003 | 0.256616 | -0.008429 | **`NOT_PROMOTED`** |

---

## 3. Event-by-Event Performance Breakdown

Performance was evaluated across all 6 development events ($N=2,826$ patch pairs, 179 total batches):

| Cell ID | Evt 279 ($N=2381$) | Evt 437 ($N=169$) | Evt 520 ($N=92$) | Evt 559 ($N=42$) | Evt 1111003 ($N=19$) | Evt 1111008 ($N=123$) | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E4_BASE`** | 0.300410 | 0.634424 | 0.699574 | 0.692425 | **0.265045** | 0.848033 | 0.573318 |
| **`E4_K1_CLASS_BALANCED`** | 0.292809 | 0.661835 | 0.705916 | 0.678037 | 0.256616 | 0.848467 | 0.573947 |
| **`E4_K2_FLOOD_WEIGHTED`** | 0.296645 | 0.661869 | 0.705962 | 0.678045 | 0.256616 | 0.848443 | 0.574597 |
| **`E4_K3_PERM_PROTECTED`** | 0.302116 | 0.661952 | 0.705982 | 0.678006 | 0.256616 | 0.848437 | 0.575518 |
| **`E4_K4_FLOOD_PRESERVED`**| 0.279394 | 0.661558 | 0.705986 | 0.678053 | 0.256616 | 0.848459 | 0.571677 |

### Event Analysis:
- **Event 279 (Land-Dominated False Alarm Surge):**
  Event 279 contains $85.5\%$ of all valid development pixels ($116.2\text{M}$ pixels). `E4_BASE` achieves an F1 of $0.3004$. K1 ($0.2928$), K2 ($0.2966$), and K4 ($0.2794$) all underperform Base on Event 279 due to elevated Land $\to$ Flood misclassifications. Only K3 marginally improves on Event 279 ($0.3021$, $+0.0017$).
- **Event 1111003 (Tail Risk Event):**
  Event 1111003 is the recognized tail-risk event ($N=19$ samples, $815\text{K}$ valid pixels). `E4_BASE` achieves an F1 of $0.2650$. In contrast, all four K interventions converge to $0.2566$, suffering a $-0.0084$ F1 degradation.

---

## 4. Class-Specific Error & Confusion Breakdown

To understand the failure modes of class-aware interventions, all errors were separated with explicit true-class denominators across the $135,859,080$ valid pixels:

| Cell ID | Class 0 (Land) F1 | Class 1 (Water) F1 | Class 2 (Flood) F1 | $c_{02}$ Land $\to$ Flood FP (Rate) | $c_{12}$ Water $\to$ Flood FP (Rate) | $c_{20}$ Flood $\to$ Land FN (Rate) | $c_{21}$ Flood $\to$ Water FN (Rate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E4_BASE`** | 0.980682 | 0.603450 | 0.441912 | 2,721,293 (2.0615%) | 185,282 (9.0331%) | 367,110 (20.3964%) | 97,902 (5.4394%) |
| **`E4_K1_CLASS_BALANCED`** | 0.980925 | 0.627218 | 0.434422 | 2,882,118 (2.1833%) | 185,303 (9.0341%) | 352,587 (19.5895%) | 96,695 (5.3723%) |
| **`E4_K2_FLOOD_WEIGHTED`** | 0.981476 | 0.635925 | 0.438544 | 2,818,810 (2.1353%) | 186,755 (9.1049%) | 354,745 (19.7094%) | 95,495 (5.3056%) |
| **`E4_K3_PERM_PROTECTED`** | 0.982025 | 0.642577 | 0.444536 | 2,728,591 (2.0670%) | 185,329 (9.0354%) | 357,573 (19.8665%) | 95,147 (5.2863%) |
| **`E4_K4_FLOOD_PRESERVED`**| 0.979749 | 0.618399 | 0.419146 | 3,129,209 (2.3705%) | 193,114 (9.4150%) | 345,743 (19.2093%) | 96,040 (5.3359%) |

### Diagnostic Error Dynamics:
1. **$c_{02}$ Land Mistaken for Flood:**
   In all four interventions, Land $\to$ Flood false alarms remain the single largest error category, comprising $>93\%$ of all false positive pixels. Even when Land pseudo-labels are capped (K1), $c_{02}$ increases from 2.72M to 2.88M pixels. Under K4, $c_{02}$ explodes to 3.13M pixels ($+407,916$ false alarm pixels).
2. **$c_{12}$ Permanent Water Mistaken for Flood:**
   Under K3 (permanent water exclusion), $c_{12}$ false alarms total $185,329$ pixels ($9.035\%$), effectively identical to `E4_BASE` ($185,282$ pixels, $9.033\%$). Excluding water pixels from the loss mask did not insulate permanent water representations from affine-induced drift.
3. **$c_{20}$ Flood Missed as Land:**
   While K1, K2, and K4 marginally reduce missed flood pixels ($c_{20}$ drops from $367\text{K}$ to $\approx 345\text{K} - 354\text{K}$), this small sensitivity improvement is completely overwhelmed by the large surge in false positives ($c_{02}$ surges by $97\text{K}$ to $408\text{K}$ pixels).

---

## 5. Source-to-Candidate Transition Dynamics

We evaluate the transitions between the frozen source model $\hat{y}_{\text{src}}$ and candidate adapted model $\hat{y}_{\text{cand}}$:

| Cell ID | Harmful Adaptations ($\text{Src}_{\text{corr}} \to \text{Cand}_{\text{wrong}}$) | Beneficial Adaptations ($\text{Src}_{\text{wrong}} \to \text{Cand}_{\text{corr}}$) | Net Accuracy Impact | Preserved Source Errors | Base Error Recoveries | Lost Base Corrections |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E4_BASE`** | 1,697,543 | 551,869 | -1,145,674 | 3,560,287 | *ref* | *ref* |
| **`E4_K1_CLASS_BALANCED`** | 1,605,978 | 524,076 | -1,081,902 | 3,595,409 | 312,251 (26.53%) | 115,727 (45.73%) |
| **`E4_K2_FLOOD_WEIGHTED`** | 1,503,743 | 562,170 | -941,573 | 3,557,733 | 329,759 (28.02%) | 112,903 (44.62%) |
| **`E4_K3_PERM_PROTECTED`** | 1,391,967 | 592,326 | -799,641 | 3,528,655 | 365,015 (31.02%) | 109,278 (43.19%) |
| **`E4_K4_FLOOD_PRESERVED`**| 1,886,617 | 497,957 | -1,388,660 | 3,617,455 | 248,048 (21.08%) | 124,801 (49.32%) |

### Transition Observations:
- **Consistent Negative Net Adaptation Impact:**
  Across every single intervention, harmful adaptations vastly outnumber beneficial adaptations by a factor of 2.3x to 3.8x. Every adapted model actively degrades more correct source pixels than it repairs.
- **Base Error Recovery vs. Beneficial Loss:**
  While K1, K2, and K3 recover $\approx 26\% - 31\%$ of errors introduced by Base consistency, they simultaneously discard $43\% - 46\%$ of the beneficial corrections that Base achieved. Net trade-offs remain neutral to negative.

---

## 6. Telemetry & Hardware Resource Accounting

| Cell ID | Selected Fraction | Mean Loss | Mean Grad Norm | Mean BN Displacement (W / B) | Peak VRAM | Adaptation Time | Updates Applied |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E4_BASE`** | 88.79% | 0.004631 | 0.009741 | 0.000235 / 0.000349 | Reused | Reused | 179 / 179 |
| **`E4_K1_CLASS_BALANCED`** | **32.16%** | 0.017778 | 0.005887 | 0.067657 / 0.071722 | 20,022.0 MB | 768.3s | 179 / 179 |
| **`E4_K2_FLOOD_WEIGHTED`** | 89.28% | 0.064359 | 0.011850 | 0.077667 / 0.079195 | 20,020.0 MB | 733.8s | 179 / 179 |
| **`E4_K3_PERM_PROTECTED`** | 88.20% | 0.057454 | 0.010112 | 0.081577 / 0.082098 | 20,024.0 MB | 736.0s | 179 / 179 |
| **`E4_K4_FLOOD_PRESERVED`**| 88.53% | 0.228133 | **0.464077** | 0.138536 / 0.141177 | 20,024.0 MB | 723.8s | 179 / 179 |

- **Execution Safety:** Zero non-finite losses, zero NaNs, and zero CUDA out-of-memory errors across all 716 GPU streaming batch steps.
- **Hardware Isolation:** All four cells ran in parallel across dedicated RTX 3090 GPUs (one process per GPU) with peak VRAM $\approx 20,024\text{ MB}$, completing all 179 batches per cell in under 13 minutes.

---

## 7. Research Integrity & Sealing Verification

All 5 cells passed 11 out of 11 formal integrity checks:
1. **Zero Sealed-Test Access:** Strictly disjoint from forbidden test events; no sealed test data opened.
2. **Double-Masking Enforcement:** Binary sensory valid mask `MK0_MNA` combined with valid bounds mask across all batches.
3. **Confusion Conservation:** 2-class confusion $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv N_{\text{valid}}$ and 3x3 confusion $\sum_{i,j} C_{ij} \equiv N_{\text{valid}}$ verified down to the single pixel ($135,859,080$ pixels).
4. **Transition Conservation:** $\sum_{i,j} T_{ij} \equiv N_{\text{valid}}$ verified across all events.
5. **Bit-Exact Source Immutability:** Parameter SHA-256 hashes of the frozen source reference model verified identical before and after every event.
6. **Air-Tight Ground-Truth Sealing:** All streaming adaptation steps operated strictly on unlabeled sensory rasters. `batch.get_labels()` was inaccessible until predictions were frozen on disk.

---

## 8. Provenance Reconciliation & Methodological Clarifications

Before finalizing the E4 terminal closeout, an exhaustive provenance and contract audit was conducted to resolve discrepancies between draft records and authoritative source evidence:

### 8.1 Source-Only Baseline Provenance Reconciliation
- **Discordant Early Values:** Draft tables transcribed unadapted Source-only performance as Macro F1 = `0.588498`, Pooled F1 = `0.507850`, Worst-Event F1 = `0.265045`.
- **Traced Root Cause:** Audit confirmed these values matched the historical **Seed 2** unadapted Source-only run from [`outputs/week7_source_baseline/source_baseline_seed2_summary.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week7_source_baseline/source_baseline_seed2_summary.json) (Macro F1 = `0.588527`, Pooled F1 = `0.507778`), while the worst-event F1 of `0.265045` was transcribed from the `E4_BASE` worst event.
- **Authoritative Seed 0 Values:** The true unadapted Seed 0 / Order A baseline from [`outputs/week7_source_baseline/source_baseline_seed0_summary.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week7_source_baseline/source_baseline_seed0_summary.json) (re-verified in E3 reconciliation) is:
  - **Macro F1:** **`0.594880`** (precisely `0.594878`)
  - **Pooled F1:** **`0.519331`**
  - **Worst-Event F1:** **`0.256835`** (Event 1111003)
- **Evaluation Independence:** Source-only is strictly a contextual external reference (`CONTROL_ONLY`). The decision-bearing reference for all K-cells is `E4_BASE` (`0.573318 / 0.441912 / 0.265045`). Reconciling the contextual Source-only values does not alter any K-versus-base deltas or the promotion outcome.

### 8.2 Checkpoint SHA-256 Provenance Audit
- **Physical File Verification:** The physical file on disk at `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` was hashed via `sha256sum`, yielding:
  `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
- **Root Cause of Discordance:** The string `247d35848529...` appeared in draft contract text as a manual transcription copy/paste error from early draft notes.
- **Model Weight Identity:** Inspection of execution logs and run manifests confirms that all runs (`E2_CONF_90`, `E4_BASE`, `E4_K1`, `E4_K2`, `E4_K3`, `E4_K4`) loaded this exact physical file. Zero model checkpoint mismatch exists; all models adapted from identical initial weights.

### 8.3 Promotion Rule Provenance
- **Numeric Threshold Provenance:** The numeric promotion criteria ($\Delta\text{Macro F1} \ge +0.010000$ with $\Delta\text{WorstEvent F1} \ge 0.0$, or $\Delta\text{WorstEvent F1} > 0.0$ with $\Delta\text{Macro F1} \ge 0.0$) derive directly from the project-standard method-promotion principle in **Plan V3 Section 1.3** ("approximately 1–2 percentage points of macro-F1 improvement, or meaningful worst-event improvement without an average loss"), formalized as an operator-authorized pre-run decision rule for Block E4 screening.
- **Section 9 Role:** Plan V3 Section 9 mandates isolated component screening, multi-seed repeats, and terminal negative closeout, but does not itself contain the numeric thresholds.
- **Verdict Integrity:** The pre-frozen decision rule was evaluated deterministically without post-hoc modification. The promotion verdict remains unambiguously **`NONE ELIGIBLE`**.

### 8.4 K3 Naming, Legacy Identifier & Theoretical Caveat
- **Scientific Definition:** **`Source-Predicted Permanent-Water Exclusion`**.
- **Legacy Identifier:** `E4_K3_PERM_PROTECTED` is retained strictly as a filesystem and artifact identifier for backward compatibility.
- **Implemented Mechanism & Non-Invariance Caveat:** The intervention excludes pixels where the frozen source model predicts permanent water ($\hat{y}_{\text{src}} == 1$) from the consistency loss mask. Crucially, this loss-mask exclusion does **not** guarantee prediction invariance. Because batch normalization parameters $(\gamma, \beta)$ are shared and updated globally across the image, permanent-water representations still drift under affine shifts, as empirically proven by $c_{12} = 185,329$ false alarms (vs $185,282$ in Base).

---

## 9. Final Decision & Work Package Exit

Under Plan V3 Section 9.3:
> *"Promote only supported components. If all fail, retain the diagnosis and negative comparisons; do not add a fifth class-aware method simply to obtain a positive result."*

- **Promotion Verdict:** **`NONE ELIGIBLE`** / **`NONE RETAINED`**
- **E4.3 Repeat Requirement:** **`NOT REQUIRED`** (no intervention cell satisfied promotion criteria).
- **Block E4 Status:** **`COMPLETE`** (All 4 isolated interventions screened, audited, and closed via the negative-result terminal closeout route).
- **Next Work Package:** **`Block E5: Temporal Evidence & Generic-Risk Comparison`** is now **`READY`**.

