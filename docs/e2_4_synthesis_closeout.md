# ChangeGuard Plan V3 Work Package E2.4: Block E2 Terminal Synthesis & Closeout Report

**Status:** COMPLETE (Implementer Self-Verification)  
**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `d93efcf2400e28f3ea7e2034958ec78a48ef0e96`  
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md` (Block E2: Sections 1.3, 7.1, 7.2, 7.3, 7.4)
- `docs/CACHE_AND_MEASUREMENT_CONTRACT.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/CURRENT_STATUS.md`
- `docs/e2_1_confidence_grid_closeout.md`
- `docs/e2_2_augmentation_grid_closeout.md`
- `docs/e2_3_promotion_and_repeat_closeout.md`

---

## 1. Executive Summary & Terminal Block E2 Status

Work Package **E2.4** (*Block E2 Synthesis, Audit and Terminal Closeout*) provides the definitive, deterministic, CPU-only audit and synthesis of all experimental investigations conducted under Block E2 (**Confidence and Augmentation Comparison**, subtasks E2.1, E2.2, and E2.3).

With this closeout, **Block E2 is terminally closed**. All 11 planned logical cells have been executed, reused with verified equivalence, or formally blocked with physical/mathematical justification.

### 1.1 Key Verdicts
1. **Standalone Method Promotion:** **`NONE RETAINED`**.
   - No consistency adaptation candidate achieved the method-promotion target ($+1.00$ to $+2.00$ percentage points Macro F1 improvement over matched controls, or meaningful worst-event improvement without an average loss).
   - Unadapted Source-only (on Seed 0) and generic entropy-minimization controls (Tent/SAR on both seeds) remain superior in Macro F1.
   - On the worst event (Event 1111003), single-view consistency updates produced identically $0.000000$ improvement over AdaBN.
   - This legitimate negative outcome rigorously supports the core research thesis of ChangeGuard: consistency adaptation alone cannot prevent representation collapse on non-stationary SAR flood streams without explicit source anchoring or risk gating.

2. **Internal Fixed Base/Reference Recipe for Block E3:**
   - Confidence threshold **$\tau = 0.90$** with **Horizontal Flip Strong Augmentation** is retained strictly as the fixed internal base/reference recipe for subsequent component experiments (Block E3 *Bounded Source-Anchor Comparison*).
   - This configuration is **not** promoted as a successful standalone method, but serves as the baseline consistency platform upon which probability, logit, and feature anchors are evaluated in Block E3.

---

## 2. Complete Inventory: Executed, Reused, and Blocked Cells

| Cell ID | Subtask | Role / Strategy | Provenance | Seed / Order | Thresh $\tau$ | Augmentation | Physical Execution Status | Macro F1 | Pooled F1 |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **`E2_CONF_NONE`** | E2.1 | C1: Unfiltered Consistency | `EXECUTED_NEW` | S0 / OA | 0.00 | Horizontal Flip | Unique Physical Stream | 0.5733 | 0.4416 |
| **`E2_CONF_70`** | E2.1 | Confidence Filter 0.70 | `EXECUTED_NEW` | S0 / OA | 0.70 | Horizontal Flip | Unique Physical Stream | 0.5733 | 0.4418 |
| **`E2_CONF_90`** | E2.1 | C2: Primary Reference 0.90 | `EXECUTED_NEW` | S0 / OA | 0.90 | Horizontal Flip | Unique Physical Stream | 0.5733 | 0.4419 |
| **`E2_CONF_95`** | E2.1 | Confidence Filter 0.95 | `EXECUTED_NEW` | S0 / OA | 0.95 | Horizontal Flip | Unique Physical Stream | 0.5733 | 0.4419 |
| **`E2_AUG_A0`** | E2.2 | A0: Horizontal Flip Ref | `REUSED` | S0 / OA | 0.90 | Horizontal Flip | Reused from `E2_CONF_90` | 0.5733 | 0.4419 |
| **`E2_AUG_A1`** | E2.2 | A1: Vertical Flip Strong | `EXECUTED_NEW` | S0 / OA | 0.90 | Vertical Flip | Unique Physical Stream | 0.5733 | 0.4419 |
| **`E2_AUG_A2`** | E2.2 | A2: Integer Translation (4, 4) | `EXECUTED_NEW` | S0 / OA | 0.90 | Translation (Overlap) | Unique Physical Stream | 0.5733 | 0.4419 |
| **`E2_AUG_A3`** | E2.2 | A3: Dual Weak Views ($W_1/W_2$)| `EXECUTED_NEW` | S0 / OA | 0.90 | Dual Weak Geometric | Unique Physical Stream | 0.5718 | 0.4501 |
| **`E2_AUG_A4`** | E2.2 | A4: SAR Noise Perturbation | `BLOCKED` | S0 / OA | 0.90 | SAR Noise / Speckle | Formally Blocked | N/A | N/A |
| **`E2_REP_CONF_90`** | E2.3 | Retained Base Repeat | `EXECUTED_NEW` | S1 / OB | 0.90 | Horizontal Flip | Unique Physical Stream | 0.5977 | 0.5859 |
| **`E2_REP_CONF_NONE`**| E2.3 | Decisive Contrast Repeat (C1) | `EXECUTED_NEW` | S1 / OB | 0.00 | Horizontal Flip | Unique Physical Stream | 0.5976 | 0.5857 |

### Inventory Accounting
- **Total Logical Cells:** 11 cells.
- **Unique Physical Streams Executed:** Exactly **9 streams** ($4 \text{ in E2.1} + 3 \text{ in E2.2} + 2 \text{ in E2.3}$).
- **Reused Logical Streams:** Exactly **1 stream** (`E2_AUG_A0` reused from `E2_CONF_90` with full cryptographic equivalence). No redundant GPU execution occurred.
- **Formally Blocked Questions:** Exactly **1 question** (`E2_AUG_A4` blocked due to physical and input-domain invalidity).
- **Total Logical Event Evaluations:** $10 \text{ cells} \times 6 \text{ events} = 60 \text{ evaluations}$ ($54$ unique physical evaluations).

---

## 3. Invariant Audits & Quarantine Enforcement

The deterministic synthesis auditor (`tools/verify_e2_synthesis.py`) re-verified all mathematical and protocol invariants offline:
1. **Confusion Conservation:** For all 60 logical event evaluations:
   $$TP + FP + FN + TN == \text{valid\_pixels}$$
   conserved at 100.0% ($135,859,080$ total valid pixels across the 6 development events).
2. **Sample Count Conservation:** Every event in every stream evaluated exactly the expected patch count ($N = 2,826$ development patches total: Event 279: 2381; Event 437: 169; Event 520: 86; Event 559: 44; Event 1111003: 31; Event 1111008: 115).
3. **Telemetry Completeness:** All selection masks satisfied $c_0 + c_1 + c_2 = \text{total\_selected}$; all loss values, gradient norms, and parameter hashes are finite and deterministic.
4. **Strict Sealed-Test Quarantine:** Zero access was made to the 10 quarantined sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 4. Synthesis of Scientific Findings

### 4.1 Confidence Filtering Sensitivity & Low-Confidence Pixel Exclusion
- Under the Seed 0 / Order A screen, confidence thresholds from $\tau = 0.00$ through $\tau = 0.95$ produced negligible performance differences (Macro F1 range: $0.57327$ to $0.57332$, a variation of $0.005$ percentage points).
- Threshold $\tau = 0.90$ consistently excluded approximately 10–11% of low-confidence pixels and reduced gradient magnitude by approximately 50% relative to no filtering directionally and numerically consistently across the two tested seed/order settings:
  - On Seed 0 / Order A: $\Delta_{\text{macro}} = +0.00004864$, $\Delta_{\text{pooled}} = +0.00030525$, gradient norm reduced from $0.01166$ to $0.00628$ ($-46.1\%$).
  - On Seed 1 / Order B: $\Delta_{\text{macro}} = +0.00006074$, $\Delta_{\text{pooled}} = +0.00029527$, gradient norm reduced from $0.01016$ to $0.00455$ ($-55.2\%$).
- Despite stabilizing optimization, confidence filtering did **not** produce a meaningful Macro F1 or Worst-Event F1 improvement over matched controls, and did not resolve adaptation collapse.

### 4.2 Bounded Geometric Stability (A0–A2)
- Paired discrete geometric transformations (horizontal flip A0, vertical flip A1, small integer translation A2 with overlap masking) produced nearly identical aggregate metrics under the tested discrete transformations ($\text{Macro F1} \approx 0.5733$, $\text{Pooled F1} \approx 0.4419$).
- We report this strictly as a bounded empirical observation across the tested discrete transformations, without claiming general rotational or translational equivariance for the underlying network.

### 4.3 Rejection of Dual Weak Views (A3)
- Cell `E2_AUG_A3` achieved a higher Pooled F1 ($0.450124$ vs $0.441912$, $+0.008212$) by capturing more true positives across large flood events.
- However, it degraded Macro F1 ($0.571816$ vs $0.573318$, $-0.001503$), dropped Event 559 by $-1.20$ percentage points, and degraded the worst event ($0.260483$ vs $0.265045$, $-0.004562$).
- Under the explicit promotion rule, candidates that improve Pooled F1 at the expense of Macro F1 and worst-event stability are rejected.

### 4.4 Formal Blocking of Cell A4
- Cell `E2_AUG_A4` remained `BLOCKED` because applying multiplicative speckle directly in standardized space is semantically invalid on zero-mean features.
- Inverting standardization via $x = \tilde{x} \cdot \sigma + \mu$ was explicitly evaluated but remains blocked because KuroSiwo metadata lacks effective look counts ($L$) and calibration parameters, dynamic range is irreversibly truncated at $0.15$, and no predeclared, non-destructive noise variance exists on the frozen source model.
- A4 is recorded as a formally blocked question, not as a negative-performance experimental cell.

---

## 5. Matched Control Comparison & Promotion Audit

| Evaluation Setting | Stream / Candidate | Macro F1 | Pooled F1 | Worst Ev F1 | Delta Macro vs AdaBN | Delta Macro vs Source | Target Promotion Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0 / Order A** | Source-Only (Frozen Stats) | **0.5949** | 0.5193 | 0.2628 | +0.0216 | 0.0000 | Baseline |
| **Seed 0 / Order A** | SAR (Matched Pre-Update) | **0.5741** | 0.4472 | 0.2650 | +0.0008 | -0.0207 | Generic TTA Control |
| **Seed 0 / Order A** | Tent (Matched Pre-Update) | **0.5741** | 0.4469 | 0.2650 | +0.0008 | -0.0208 | Generic TTA Control |
| **Seed 0 / Order A** | AdaBN (Matched Pre-Update) | **0.5733** | 0.4417 | 0.2650 | 0.0000 | -0.0216 | Normalization Control |
| **Seed 0 / Order A** | E2 Base Reference ($\tau=0.90$) | **0.5733** | 0.4419 | 0.2650 | +0.00002 | -0.0216 | **FAIL (None Retained)** |
| **Seed 0 / Order A** | E2 Decisive Contrast ($\tau=0.00$) | **0.5733** | 0.4416 | 0.2650 | -0.00003 | -0.0216 | **FAIL (None Retained)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 1 / Order B** | Tent (Matched Pre-Update) | **0.6035** | 0.6129 | 0.2740 | +0.0058 | +0.0151 | Generic TTA Control |
| **Seed 1 / Order B** | SAR (Matched Pre-Update) | **0.5996** | 0.5954 | 0.2740 | +0.0020 | +0.0113 | Generic TTA Control |
| **Seed 1 / Order B** | AdaBN (Matched Pre-Update) | **0.5977** | 0.5858 | 0.2741 | 0.0000 | +0.0093 | Normalization Control |
| **Seed 1 / Order B** | E2 Base Reference ($\tau=0.90$) | **0.5977** | 0.5859 | 0.2741 | +0.00003 | +0.0093 | **FAIL (None Retained)** |
| **Seed 1 / Order B** | E2 Decisive Contrast ($\tau=0.00$) | **0.5976** | 0.5857 | 0.2741 | -0.00003 | +0.0093 | **FAIL (None Retained)** |
| **Seed 1 / Order B** | Source-Only (Frozen Stats) | **0.5884** | 0.4766 | 0.2628 | -0.0093 | 0.0000 | Baseline |

---

## 6. Acceptance Criteria Scorecard

| Criterion | Target Requirement | Audit Finding | Status |
| :---: | :--- | :--- | :---: |
| **1** | CPU-only, offline execution | No GPU, no model runs, no raster re-reads | Verified CPU-only in 1.8s | **PASSED** |
| **2** | All cells accounted for | 11 logical cells (4 E2.1, 5 E2.2, 2 E2.3) | All 11 audited without omission | **PASSED** |
| **3** | Reused provenance explicit | A0 provenance verified from E2.1 `E2_CONF_90` | Provenance pinned; no double counting | **PASSED** |
| **4** | A4 accurately represented | Blocked status with physical domain reason | Formally BLOCKED; zero execution | **PASSED** |
| **5** | All invariants re-audited | Confusion conservation, sample counts, telemetry | 60/60 evaluations passed 100% | **PASSED** |
| **6** | Zero sealed-test access | Strictly quarantine 10 test events | Verified clean (0 test events accessed) | **PASSED** |
| **7** | Promotion rule recomputed | Unrounded metrics evaluated against Plan V3 | Recomputed mechanically; zero errors | **PASSED** |
| **8** | Final promotion verdict | Standalone method promotion verdict | Verified `NONE RETAINED` | **PASSED** |
| **9** | Negative/neutral cells preserved | Full grid and negative cells documented | All cells preserved in master tables | **PASSED** |
| **10** | Terminology guardrails enforced | Low-confidence exclusion, bounded stability | Enforced across all documents | **PASSED** |
| **11** | Artifact checksums pinned | SHA-256 digests pinned in closeout | Checksums pinned in Section 7 | **PASSED** |
| **12** | Dashboards updated | Update `EXPERIMENT_REGISTRY` & `CURRENT_STATUS` | Updated upon verifier pass | **PASSED** |
| **13** | Block E2 marked complete | Terminal closeout of Block E2 | E2 terminally closed | **PASSED** |
| **14** | Block E3 marked ready | Block E3 ready; no E3 runs launched | E3 marked `READY (NOT STARTED)` | **PASSED** |
| **15** | Clean working tree | Staged and committed cleanly | Verified clean tracked working tree | **PASSED** |

---

## 7. Pinned Final Synthesis Artifact Checksums

```
c12fd1eca18a105f4d2bdc46286ca31e3d0e15461c7661d304e5bfb9ef338948  tools/verify_e2_synthesis.py
ae7fe5bf1b52a6c830a4f477f830f2d21d9e0cdf2899cab4f3dbc4e3328c4926  outputs/e2_synthesis/v1/master_e2_synthesis.json
8c1a2612d239dabb5ae2f2a3a37e7620b1ccdbe3add23b28432a65be6838ca98  outputs/e2_synthesis/v1/master_e2_synthesis.md
```

---

## 8. Terminal Transition: Block E2 Complete, Block E3 Ready

- **Block E2 is terminally COMPLETE.** All exit conditions specified in Plan V3 Section 7.4 are fulfilled.
- **Fixed Reference for Block E3:** In accordance with Plan V3 Section 8, the base consistency recipe for Block E3 anchor comparisons is fixed as `tau=0.90` with horizontal-flip strong augmentation.
- **Block E3 Status:** **`READY (NOT STARTED)`**. Zero E3 code or experiments have been executed.
