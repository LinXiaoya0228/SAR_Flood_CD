# ChangeGuard Work Package E6.1: Gate E Decision

**Status:** COMPLETE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**HEAD at Decision:** `1311bad` (`fix(e5): enforce multiset validation and finalize temporal-risk closeout`)
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.1 (Gate E, previously defined in commit `24ffcc9`)
**Decision Contract:** `configs/changeguard/e6_decision_contract_v1.json` (retrospective formalization created 2026-10-21)

> [!IMPORTANT]
> **E6.1 Terminal Verdict: Gate E FAIL → Branch C**
> The Gate E criteria defined in Section 11.1 of `docs/MASTER_EXECUTION_PLAN_V3.md` (committed at `24ffcc9`) and formalized in `configs/changeguard/e6_decision_contract_v1.json` are applied directly to the audited E5 evidence. Criteria 2, 3, and 6 FAIL; Criterion 4 is INCONCLUSIVE. No criterion status is relabeled.

---

## 1. Decision Contract Provenance and Evidence Registry

The Gate E evaluation criteria applied herein were defined in `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.1 (committed at `24ffcc9`). The companion configuration `configs/changeguard/e6_decision_contract_v1.json` was created retrospectively on 2026-10-21 following the completion of Work Package E5 to provide a machine-readable specification of those governing plan criteria. It does not introduce new thresholds post-hoc and does not constitute prospective evidence.

All empirical data are derived directly from frozen, committed E5 analysis artifacts.

| Cell | Seed | Order | Event Outcomes SHA-256 | Signal Summary SHA-256 |
| :--- | :---: | :---: | :--- | :--- |
| `s0_oa` (canonical) | 0 | A | `db100f7b...` | `7c405304...` |
| `s0_ob` (order repeat) | 0 | B | `0b16d371...` | `9d8e98f9...` |
| `s1_oa` (seed repeat) | 1 | A | `d47dd4b2...` | `ca737b06...` |

**Development events evaluated:** `[279, 437, 520, 559, 1111003, 1111008]` ($N = 6$).
**Total formal samples:** 2,826 per cell × 3 cells = 8,478 sample records.
**Sealed-test events:** not accessed.

---

## 2. Canonical Cell Event Outcomes (s0_oa)

| Event | Geography | Source F1 | BN-Control F1 | Candidate F1 | ΔF1 (Cand vs Src) | Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 279 | Spain/Ebro | 0.37689 | 0.37576 | 0.32244 | **−0.05446** | Harmful |
| 437 | UK/Severn | 0.69937 | 0.69246 | 0.63517 | **−0.06420** | Harmful |
| 520 | USA/Florence | 0.71375 | 0.71059 | 0.69975 | **−0.01400** | Harmful |
| 559 | USA/Midwest | 0.67258 | 0.68412 | 0.69229 | **+0.01971** | Beneficial |
| 1111003 | Bangladesh | 0.25683 | 0.26647 | 0.26505 | **+0.00821** | Neutral |
| 1111008 | Myanmar | 0.84986 | 0.84957 | 0.84795 | **−0.00191** | Neutral |
| **Macro (unweighted)** | | **0.5949** | | **0.5771** | **−0.0178** | |

**Batch-level:** 73/179 batches harmful (40.78%), 18 beneficial (10.06%), 88 neutral (49.16%).

---

## 3. Order Repeat Cell Event Outcomes (s0_ob)

| Event | Source F1 | Candidate F1 | ΔF1 | Classification |
| :--- | :---: | :---: | :---: | :---: |
| 279 | 0.37689 | 0.32737 | **−0.04953** | Harmful |
| 437 | 0.69937 | 0.63927 | **−0.06010** | Harmful |
| 520 | 0.71375 | 0.69947 | **−0.01428** | Harmful |
| 559 | 0.67258 | 0.68898 | **+0.01640** | Beneficial |
| 1111003 | 0.25683 | 0.27234 | **+0.01551** | Beneficial |
| 1111008 | 0.84986 | 0.84646 | **−0.00339** | Neutral |

**Batch-level:** 78/179 harmful (43.58%), 103 beneficial (57.54% — driven by smaller events), 88 neutral in canonical vs redistributed in reordered stream.

---

## 4. Seed Repeat Cell Event Outcomes (s1_oa)

| Event | Source F1 | Candidate F1 | ΔF1 | Classification | Direction vs s0_oa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 279 | 0.33385 | 0.49263 | **+0.15877** | **Beneficial** | **INVERTED** |
| 437 | 0.71264 | 0.65897 | **−0.05366** | Harmful | Consistent |
| 520 | 0.70101 | 0.72133 | **+0.02032** | **Beneficial** | **INVERTED** |
| 559 | 0.66398 | 0.63767 | **−0.02631** | **Harmful** | **INVERTED** |
| 1111003 | 0.27897 | 0.28028 | **+0.00131** | Neutral | Consistent (direction) |
| 1111008 | 0.83984 | 0.82986 | **−0.00998** | Neutral | Consistent |

**Batch-level:** 20/179 harmful (11.17%), 103 beneficial (57.54%), 56 neutral (31.28%).

> [!WARNING]
> **Empirical finding:** Three of six events inverted classification direction between Seed 0 and Seed 1. Only Event 437 remained harmful in all three cells.

---

## 5. Signal Performance Table

### 5.1 Event-Level AUROC (6-event ranking)

| Signal | Family | s0_oa AUROC | s0_ob AUROC | s1_oa AUROC | s0_oa ρ | s0_ob ρ | s1_oa ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `source_to_candidate_kl` | generic | **1.0000** | **0.8889** | **0.0000** | −0.886 | −0.771 | **+0.886** |
| `s_flood_expansion` | change | **1.0000** | **1.0000** | 0.2500 | −0.771 | −0.771 | +0.009 |
| `bn_weight_displacement` | generic | 0.8889 | 0.8889 | 0.5000 | −0.829 | −0.829 | +0.143 |
| `abs_delta_flood_pixels` | generic | 0.7778 | 0.7778 | 0.2500 | −0.543 | −0.543 | +0.543 |
| `hard_disagreement_rate` | generic | 0.7778 | 0.6667 | **0.0000** | −0.429 | −0.257 | **+0.886** |
| `s_null` | change | 0.6667 | 0.7778 | 0.4375 | −0.600 | −0.771 | −0.232 |
| `mean_entropy` | generic | 0.4444 | 0.2222 | **0.0000** | +0.086 | +0.429 | +0.829 |
| `mean_max_confidence` | generic | 0.4444 | 0.2222 | **0.0000** | +0.029 | +0.429 | +0.829 |
| `aug_inconsistency` | generic | 0.4444 | 0.2222 | **0.0000** | +0.086 | +0.429 | +0.829 |
| `s_flood_disappearance` | change | 0.4444 | 0.4444 | 0.1250 | +0.257 | +0.257 | +0.886 |

### 5.2 Batch-Level Pooled AUROC (179 batches per cell)

| Signal | s0_oa pooled AUROC | s0_ob pooled AUROC | s1_oa pooled AUROC |
| :--- | :---: | :---: | :---: |
| `abs_delta_flood_pixels` | **0.7190** | **0.7153** | 0.2145 |
| `s_flood_expansion` | 0.6929 | 0.6862 | 0.5038 |
| `source_to_candidate_kl` | 0.6530 | 0.6102 | 0.1610 |
| `hard_disagreement_rate` | 0.6215 | 0.5805 | 0.2138 |
| `s_null` | 0.5861 | 0.6108 | **0.6624** |
| `s_flood_disappearance` | 0.4497 | 0.3987 | 0.1497 |
| `bn_weight_displacement` | 0.4360 | 0.4056 | 0.1643 |

---

## 6. Gate E Criterion Assessment

Evaluation follows the criteria defined in Section 11.1 of `docs/MASTER_EXECUTION_PLAN_V3.md`.

---

### Criterion 1: Harm and benefit coexist in relevant conditions

**Requirement:** Harm and benefit coexist across events under relevant conditions.

**Evidence:**
- Canonical cell (s0_oa): Events 279, 437, 520 → Harmful; Event 559 → Beneficial.
- Both conditions occur within the canonical evaluation cell.

**Qualifications:**
- $N = 6$ development events is a limited sample. Harm prevalence is 3/6 in s0_oa, 3/6 in s0_ob, and 1/6 in s1_oa. The coexistence pattern is seed-dependent.
- BN-no-update (parameter-frozen control) is near-neutral on all events ($|\Delta \text{F1}| \le 0.012$), indicating the observed harm is adaptation-induced rather than measurement artifact.

**Verdict: PASS**

---

### Criterion 2: A usable score improves on no-skill and reasonable generic evidence

**Requirement:** A usable score improves on no-skill (AUROC > 0.50, target threshold > 0.70) and reasonable generic evidence across evaluated conditions.

**Evidence:**
- `source_to_candidate_kl`: AUROC = 1.000 (s0_oa), 0.889 (s0_ob), **0.000** (s1_oa). Spearman $\rho$ inverts from −0.886 to +0.886.
- `s_flood_expansion`: AUROC = 1.000 (s0_oa), 1.000 (s0_ob), **0.250** (s1_oa). Collapses on seed repeat; $\rho$ collapses from −0.771 to +0.009.
- `abs_delta_flood_pixels`: AUROC = 0.778 (s0_oa), 0.778 (s0_ob), **0.250** (s1_oa).
- `bn_weight_displacement`: AUROC = 0.889 (s0_oa), 0.889 (s0_ob), 0.500 (s1_oa).
- `hard_disagreement_rate`: AUROC = 0.778 (s0_oa), 0.667 (s0_ob), **0.000** (s1_oa).
- No evaluated score maintains above-no-skill performance across all three cells.

**Interpretation:** High apparent AUROC on Seed 0 reflects that KL divergence and flood expansion track adaptation displacement magnitude. On Seed 0, Event 279 exhibits large displacement that happens to be harmful. On Seed 1, Event 279 exhibits large displacement that is beneficial (+0.1588 $\Delta \text{F1}$). The displacement signals measure adaptation magnitude, not direction.

**Verdict: FAIL**

---

### Criterion 3: Risk direction survives at least one second seed/order

**Requirement:** Risk direction survives at least a second seed/order evaluation.

**Evidence — event classification consistency:**

| Event | s0_oa | s0_ob | s1_oa | Consistent across all 3? |
| :--- | :---: | :---: | :---: | :---: |
| 279 | Harmful | Harmful | **Beneficial** | **No** |
| 437 | Harmful | Harmful | Harmful | **Yes** |
| 520 | Harmful | Harmful | **Beneficial** | **No** |
| 559 | Beneficial | Beneficial | **Harmful** | **No** |
| 1111003 | Neutral | Beneficial | Neutral | Partial |
| 1111008 | Neutral | Neutral | Neutral | Yes |

- Across all three cells, only 2 of 6 events (437 and 1111008) maintain direction.
- Between orders (s0_oa vs s0_ob), 5 of 6 events maintain classification direction.
- Between source seeds (s0_oa vs s1_oa), 3 of 6 events invert classification direction.
- Predictive signal directions (`source_to_candidate_kl`, `hard_disagreement_rate`) completely invert sign on the seed repeat.

**Verdict: FAIL**

---

### Criterion 4: Incremental change-specific value remains under comparable calibration

**Requirement:** Incremental change-specific value remains under comparable calibration/acceptance.

**Evidence:**

| Cell | Best generic AUROC | Best change-specific AUROC | Change > generic? |
| :--- | :---: | :---: | :---: |
| s0_oa | `source_to_candidate_kl` = 1.0000 | `s_flood_expansion` = 1.0000 | Tied |
| s0_ob | `source_to_candidate_kl` = 0.8889 | `s_flood_expansion` = 1.0000 | Yes (+0.111) |
| s1_oa | `abs_delta_flood_pixels` = 0.2500 | `s_null` = 0.4375 | Both below no-skill |

`s_flood_expansion` shows advantage over generic signals in s0_ob, but collapses on s1_oa (0.2500). On s1_oa, `s_null` is 0.4375 (below no-skill reference 0.50). Multivariate composite policy calibration was not evaluated in the analysis protocol. Incremental value is not demonstrated across seeds.

**Verdict: INCONCLUSIVE**

---

### Criterion 5: Signal available before declared action

**Requirement:** The signal is available before the declared action.

**Evidence:**
- Formal collection protocol enforced `prediction_timing = "PRE_UPDATE"` across all cells (`configs/changeguard/e5_temporal_risk_v1.json`).
- Atomic shard writing wrote `.npz` records before ground-truth label access.
- Verified across 537 shards (3 cells × 179 shards); zero sealed-test access.
- All candidate and source predictions used to compute signals are generated before applying batch updates.

**Verdict: PASS**

---

### Criterion 6: Compute overhead and false rejection acceptable

**Requirement:** Compute overhead and false rejection are acceptable for the proposed claim.

**Evidence (rejection budget analysis at budget = 0.20):**
- On s0_oa, rejecting top 20% by `source_to_candidate_kl` averts 30.1% harmful batches at the cost of 5.6% beneficial batches.
- On s1_oa, the same rejection rule averts 0.0% of harmful batches (0/20) and discards 29.1% of beneficial batches (30/103), dropping realized candidate F1 to 0.5362 (below source-only macro F1 of 0.5884).
- False rejection is severe when applying a Seed-0-derived threshold to Seed-1 data because the signal direction inverts.

**Verdict: FAIL**

---

## 7. Gate E Summary

| Criterion | Plan Reference (Section 11.1) | Verdict |
| :--- | :--- | :---: |
| 1 | Harm and benefit coexist in relevant conditions | **PASS** |
| 2 | Usable score improves on no-skill and generic evidence | **FAIL** |
| 3 | Direction survives at least a second seed/order | **FAIL** |
| 4 | Incremental value remains with comparable calibration | **INCONCLUSIVE** |
| 5 | Signal available before declared action | **PASS** |
| 6 | Overhead and false rejection acceptable | **FAIL** |
| **Overall Gate E** | | **FAIL** |

---

## 8. Branch Assignment

Per `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.2:

- **Branch A** requires Gate E support for stable incremental change-specific value. Gate E FAIL disqualifies Branch A. Signal inversion across seeds demonstrates that pre-update displacement scores do not provide a stable basis for selective adaptation. **Branch A: NOT SUPPORTED.**

- **Branch B** requires a stable adaptation/normalization/anchor component that improves fair controls across adequate seed/order repeat.
  - BN-no-update (AdaBN): $\Delta \text{F1} \approx 0.000$ across all events in all cells; neutral, not an improvement over source-only.
  - E2 consistency candidates: none achieved the $+1.0$ pp Macro F1 promotion threshold on Seed 0; Seed 1 inversion reflects event-contingent variance, not a stable component.
  - E3 source anchor: best anchor variant achieved $+0.05$ pp vs AdaBN (threshold: $+1.0$ pp). None promoted.
  - E4 loss-weighting: all variants neutral-to-negative vs E4_BASE on Seed 0.
  - E5 candidate TTA: Macro $\Delta \text{F1} = -0.0178$ (s0_oa), $-0.0159$ (s0_ob), $+0.0151$ (s1_oa) — seed-unstable.
  - No component in E2–E5 demonstrates stable, positive improvement across seeds. **Branch B: NOT SUPPORTED.**

- **Branch C** is selected: no method component improves reasonable controls stably.

**Branch Assignment: Branch C**

---

## 9. What Branch C Entails

Per `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.2:
> "No method improves reasonable controls. Complete an empirical study with repeated normalization/seed/event evidence, strong baselines, robustness and final reserved-event evaluation."

Branch C formalizes the research as a controlled empirical study:
1. Adaptation harm is real on Seed 0 (macro $\Delta \text{F1} = -0.0178$) but seed-contingent (reversed to $+0.0151$ on Seed 1).
2. Pre-update displacement signals measure adaptation magnitude, not direction, inverting between source checkpoints.
3. Event 437 (UK/Severn) is the only event that is consistently degraded across all evaluated cells.
4. BN-no-update is near-neutral ($|\Delta \text{F1}| \le 0.012$), serving as a negative control that isolates consistency parameter updates as the operative factor.

---

## 10. Cross-Reference

| Document | Role |
| :--- | :--- |
| `configs/changeguard/e6_decision_contract_v1.json` | Formalized decision rules |
| `docs/e5_final_closeout.md` | Source E5 evidence closeout |
| `docs/e5_3_repeat_results.md` | Cross-cell repeat results |
| `docs/e6_primary_study_definition.md` | Branch C empirical study definition |
| `docs/e6_claim_to_experiment_map.md` | Claim-to-experiment map |
| `docs/e6_nearest_method_applicability_matrix.md` | Nearest-method applicability matrix |
| `docs/protocols/e6_cotta_compatibility_protocol_v1.md` | CoTTA compatibility protocol |
| `docs/e6_final_route_decision.md` | Final integrated route decision |
