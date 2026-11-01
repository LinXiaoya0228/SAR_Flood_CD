# ChangeGuard Plan V3 Work Package E2.3: Promotion Decision Audit and Locked Repeat Matrix

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `c94ed77` (`docs(e2.2): correct A4 physical domain justification and bound equivariance claims`)  
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md` (Sections 1.3, 7.1, 7.2, 7.3, 7.4)
- `docs/CACHE_AND_MEASUREMENT_CONTRACT.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/CURRENT_STATUS.md`
- `docs/e2_1_confidence_grid_closeout.md`
- `docs/e2_2_augmentation_grid_closeout.md`

---

## 1. Executive Preflight & Git State Verification

Prior to executing any E2.3 runtime code or model evaluations:
1. **Clean Commit Base:** Working tree is clean on tracked files; Git HEAD is at commit `c94ed77`.
2. **Prior Subtasks Complete:** E2.1 and E2.2 are complete with deterministic verifiers passing 100%. Documentation hotfix for E2.2 is committed.
3. **Quarantine:** The 10 sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain strictly quarantined with zero code access.
4. **Environment:** Miniconda Python `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`, PyTorch 2.5.1+cu121).

---

## 2. Governing Promotion Rule in Plan V3

Plan V3 governs method promotion and component retention through three explicit clauses:

1. **Section 1.3 (Method-Promotion Target, line 72):**
   > *"The original aspiration of approximately 1-2 percentage points of macro-F1 improvement, or meaningful worst-event improvement without an average loss, remains a method-promotion target. It is not a condition under which negative results may be hidden or a guarantee that an anchor must improve F1."*

2. **Section 7.3 (Screening and Repeat, lines 429–431):**
   > *"Screen seed0/order_a/reset-per-event across all six development events. This is a one-factor comparison, not the full Cartesian product of thresholds and transforms. Repeat the retained configuration and its decisive component contrast on seed1 and the other declared order before promotion. The final main comparison later covers all three seeds."*

3. **Section 7.4 (Exit Criteria, lines 437–438):**
   > *"All confidence cells and valid augmentation questions are accounted for. Choose at most one main consistency recipe through the recorded rule, or record that none improves the controls. Preserve the full grid and all negative cells."*

---

## 3. Unrounded Metrics for Every Candidate Cell

All candidate cells evaluated in E2.1 and E2.2 on Seed 0, Order A, `reset_per_event`, Batch Size 16, Dev 6 events ($N = 2,826$ patches, $135,859,080$ valid pixels):

| Candidate Cell ID | Strategy Description | Confidence $\tau$ | Macro F1 | Pooled F1 | Worst-Event F1 | Worst Event ID | Updates | Mean Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E2_CONF_NONE`** | C1: No Confidence Filter | $\tau = 0.00$ | 0.57326982 | 0.44160649 | 0.26504503 | 1111003 | 179 / 179 | 0.00412 |
| **`E2_CONF_70`** | Confidence Filter 0.70 | $\tau = 0.70$ | 0.57330364 | 0.44180585 | 0.26504503 | 1111003 | 179 / 179 | 0.00405 |
| **`E2_CONF_90`** / **`A0`**| C2 / A0: Horizontal Flip Ref | $\tau = 0.90$ | 0.57331846 | 0.44191174 | 0.26504503 | 1111003 | 179 / 179 | 0.00392 |
| **`E2_CONF_95`** | Confidence Filter 0.95 | $\tau = 0.95$ | 0.57331658 | 0.44189431 | 0.26504503 | 1111003 | 179 / 179 | 0.00388 |
| **`E2_AUG_A1`** | Vertical Flip Strong | $\tau = 0.90$ | 0.57332183 | 0.44194808 | 0.26504503 | 1111003 | 179 / 179 | 0.00448 |
| **`E2_AUG_A2`** | Translation $(\Delta y=4, \Delta x=4)$ | $\tau = 0.90$ | 0.57332084 | 0.44191645 | 0.26504503 | 1111003 | 179 / 179 | 0.00149 |
| **`E2_AUG_A3`** | Dual Weak Views ($W_1$ vs $W_2$) | $\tau = 0.90$ | 0.57181579 | 0.45012429 | 0.26048327 | 1111003 | 179 / 179 | 0.00563 |
| **`E2_AUG_A4`** | SAR Intensity Perturbation | $\tau = 0.90$ | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` | N/A | N/A |

### Matched External Baseline Controls (Seed 0, Order A, `reset_per_event`, Batch Size 16)
- **Source-Only Baseline:** Macro F1 = `0.59487972` | Pooled F1 = `0.51933112` | Worst-Event F1 = `0.26279899` (#1111003)
- **AdaBN Matched Control:** Macro F1 = `0.57329774` | Pooled F1 = `0.44174396` | Worst-Event F1 = `0.26504503` (#1111003)
- **Tent Matched Control:** Macro F1 = `0.57408785` | Pooled F1 = `0.44687251` | Worst-Event F1 = `0.26504503` (#1111003)
- **SAR Matched Control:** Macro F1 = `0.57413598` | Pooled F1 = `0.44718599` | Worst-Event F1 = `0.26504503` (#1111003)

---

## 4. Promotion Criteria and Mechanical Evaluation

We evaluate each candidate under four explicit decision criteria derived from Plan V3:

- **Criterion 1 (Macro Improvement Target):** Approximately $+1.00$ to $+2.00$ percentage points ($\Delta_{\text{macro}} \ge +0.0100$) over matched baseline controls (Source-only or AdaBN).
- **Criterion 2 (Worst-Event Improvement Target):** Meaningful worst-event improvement ($\Delta_{\text{worst}} > 0.0000$) without an average macro loss ($\Delta_{\text{macro}} \ge 0.0000$).
- **Criterion 3 (Generic TTA Superiority):** Macro F1 and Pooled F1 superior to matched generic TTA controls (Tent: $0.574088$, SAR: $0.574136$).
- **Criterion 4 (Internal Consistency Component Superiority):** Superiority over the unfiltered consistency control C1 (`E2_CONF_NONE`: $0.573270$) without introducing severe per-event degradation or worst-event drop.

### Mechanical Candidate Evaluation Matrix

| Candidate Cell | Criterion 1 (Macro +1-2 pp) | Criterion 2 (Worst-Event Impr) | Criterion 3 (Beats Tent/SAR) | Criterion 4 (Beats C1 / Stable) | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`E2_CONF_NONE`** (C1) | **FAIL** ($-2.16$ pp vs Source, $-0.003$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | Baseline Control (C1) | Control |
| **`E2_CONF_70`** | **FAIL** ($-2.16$ pp vs Source, $+0.0006$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | **PASS** ($+0.003$ pp vs C1) | Not Best |
| **`E2_CONF_90`** / **`A0`**| **FAIL** ($-2.16$ pp vs Source, $+0.0021$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | **PASS** ($+0.005$ pp vs C1, $+0.03$ pp pooled) | **TOP REFERENCE** |
| **`E2_CONF_95`** | **FAIL** ($-2.16$ pp vs Source, $+0.0019$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | **PASS** (< $\tau=0.90$) | Inferior to 0.90 |
| **`E2_AUG_A1`** (V-Flip) | **FAIL** ($-2.16$ pp vs Source, $+0.0024$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | **PASS** ($+0.0003$ pp vs A0, 20px on Ev279) | Redundant with A0 |
| **`E2_AUG_A2`** (Trans) | **FAIL** ($-2.16$ pp vs Source, $+0.0023$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = 0.0000$) | **FAIL** (< Tent/SAR) | **PASS** ($+0.0002$ pp vs A0) | Redundant with A0 |
| **`E2_AUG_A3`** (Dual-Weak) | **FAIL** ($-2.31$ pp vs Source, $-0.15$ pp vs AdaBN) | **FAIL** ($\Delta_{\text{worst}} = -0.0046$, worse!)| **FAIL** (< Tent/SAR) | **FAIL** (Degrades Macro F1 & Worst Ev) | **REJECTED** |
| **`E2_AUG_A4`** (Noise) | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` | Formally Blocked |

---

## 5. Promotion Decision & Justification

### 5.1 Rejection of Cell `A3` Despite Higher Pooled F1
Cell `E2_AUG_A3` achieved a higher Pooled F1 ($0.450124$ vs $0.441912$, $+0.008212$). However, under the explicit instructions:
1. **Macro F1 Degradation:** `A3` suffers an overall Macro F1 drop of $-0.001503$ ($0.571816$ vs $0.573318$).
2. **Severe Event Drop:** On Event 559, F1 drops by $-0.012020$ ($0.680373$ vs $0.692393$).
3. **Worst-Event Deterioration:** On worst-event 1111003, F1 drops by $-0.004562$ ($0.260483$ vs $0.265045$).
4. **Mandate:** The plan strictly forbids promoting a candidate merely because it has a higher Pooled F1 when it degrades Macro F1 and worst-event stability. Cell `E2_AUG_A3` is definitively **REJECTED**.

### 5.2 Equivalence of `A1` and `A2` to Reference `A0`
- `E2_AUG_A1` (Vertical Flip) yields Macro F1 = $0.57332183$ ($+0.00000337$ over A0). Across 5 of the 6 events (Events 437, 520, 559, 1111003, 1111008), performance is **identically equal** to A0 down to 8 decimal places. On Event 279, it differs by only 20 true-positive pixels out of 630,075.
- `E2_AUG_A2` (Translation) yields Macro F1 = $0.57332084$ ($+0.00000238$ over A0).
- Neither transformation provides a statistically or practically meaningful improvement over the predefined primary reference `A0` (Horizontal Flip Strong). Therefore, `A0` remains the canonical geometric reference.

### 5.3 Promotion Decision Verdict: Two-Level Distinction
1. **Method-Level Promotion Target (External Claim):**
   - **`NONE RETAINED` as a standalone successful TTA method.**
   - No consistency adaptation configuration in E2 improves upon the unadapted Source-only baseline (which achieves $0.594880$ Macro F1, $>2.15$ percentage points higher than any E2 cell).
   - Furthermore, consistency adaptation fails to beat generic TTA baselines (Tent $0.574088$, SAR $0.574136$) and fails to improve upon AdaBN on the worst event ($0.26504503$).
   - This validates the central hypothesis of ChangeGuard: unsupervised consistency adaptation alone is insufficient on SAR change detection without anchoring or risk control.

2. **Internal Component Selection (Base Consistency Recipe for Block E3):**
   - For the sequential progression of ChangeGuard to Block E3 (*Source Anchoring*), the framework requires a locked base consistency recipe upon which source anchors (probability, logit, feature) are evaluated.
   - Within Block E2, `E2_CONF_90` (Confidence threshold $\tau = 0.90$, Horizontal Flip strong) is retained as the internal base consistency recipe.
   - Its decisive component control/contrast is `E2_CONF_NONE` (Unfiltered consistency / C1, $\tau = 0.00$).

---

## 6. Locked E2.3 Repeat Matrix

To satisfy Section 7.3 and test the repeatability of this decisive component contrast across random seeds and presentation sequences before Block E3, the E2.3 repeat matrix is locked as follows:

| Stream ID | Condition ID | Role / Strategy | Threshold $\tau$ | Augmentation | Seed | Order | Reset Mode | Batch Size | Target Events |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`E2_REP_CONF_90_s1_ob_re`** | `E2_REP_CONF_90` | Retained Recipe Repeat | $0.90$ | Horizontal Flip Strong | **1** | **`order_b`** | `reset_per_event` | 16 | Dev 6 events ($N=2,826$) |
| **`E2_REP_CONF_NONE_s1_ob_re`**| `E2_REP_CONF_NONE`| Decisive Component Contrast (C1) | $0.00$ | Horizontal Flip Strong | **1** | **`order_b`** | `reset_per_event` | 16 | Dev 6 events ($N=2,826$) |

### Invariants Locked for Execution:
- **Seed:** Checkpoint Seed 1 (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt`, SHA-256: `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a`).
- **Order:** `order_b` (Reverse event sequence `[1111008, 1111003, 559, 520, 437, 279]`, descending sample IDs within each event).
- **Reset Mode:** `reset_per_event` (source model parameters, buffers, optimizer, and EMA reloaded before every event).
- **Evaluation:** Strict Plan V3 double masking, offline evaluation, confusion conservation, complete step records, and prediction caching.
- **Zero Sealed-Test Access:** Strictly restricted to the 6 development events.

