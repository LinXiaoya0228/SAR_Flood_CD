# ChangeGuard Work Package E6.2: Claim-to-Experiment Map

**Status:** COMPLETE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**HEAD at Decision:** `1311bad`
**Branch:** C — Empirical Study
**Source:** `docs/e6_primary_study_definition.md`, `docs/e6_gate_e_decision.md`

> [!NOTE]
> This map traces every claim in the Branch C empirical study to the specific committed evidence that supports it. No claim is included unless it is directly evidenced by committed E2–E5 experiment artifacts.

---

## 1. Structure

Each claim entry specifies:
- The textual claim (as it would appear in a thesis or paper)
- The supporting experiment(s)
- The specific artifact and metric
- The claim strength (Supported / Partial / Excluded)

---

## 2. Claim-to-Experiment Map

### Claim C1: Adaptation harm and benefit coexist on Seed 0

**Claim text:** Online test-time consistency adaptation degrades flood-class F1 on 3 of 6 development events on Seed 0 and improves it on 1 of 6, demonstrating that adaptation outcomes are event-dependent.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E5.2 (canonical, s0_oa), E5.3 order-repeat (s0_ob) |
| **Artifacts** | `outputs/e5_temporal_evidence/analysis_s0_oa_v2/event_outcomes.csv`, `analysis_s0_ob_v1/event_outcomes.csv` |
| **Key metrics** | Event 279: ΔF1 = −0.0545 (s0_oa); Event 437: ΔF1 = −0.0642; Event 559: ΔF1 = +0.0197 |
| **Strength** | **Supported** — directly evidenced in committed E5 outputs |
| **Scope limitation** | N = 6 development events; N = 2 source seeds |

---

### Claim C2: Adaptation outcome is strongly seed-contingent

**Claim text:** Adaptation outcome sign inverts between source checkpoints on 3 of 6 events. Event 437 (UK/Severn) is the only event with consistently harmful adaptation in all evaluated seed/order cells.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E5.3 seed-repeat (s1_oa) vs. canonical (s0_oa) |
| **Artifacts** | `analysis_s1_oa_v1/event_outcomes.csv` vs. `analysis_s0_oa_v2/event_outcomes.csv` |
| **Key metrics** | Ev 279: s0_oa = −0.0545, s1_oa = +0.1588; Ev 520: s0_oa = −0.0140, s1_oa = +0.0203; Ev 559: s0_oa = +0.0197, s1_oa = −0.0263 |
| **Strength** | **Supported** — cross-seed inversions are directly observable in committed evidence |
| **Scope limitation** | Only Seed 0 and Seed 1 evaluated; Seed 2+ outcomes unknown |

---

### Claim C3: Stream order has limited effect on adaptation outcomes on Seed 0

**Claim text:** Reversing the event stream order (Order B) replicates the Seed 0 adaptation outcomes with high fidelity; macro ΔF1 differs by less than 0.002 between the two order conditions.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E5.3 order-repeat (s0_ob) vs. canonical (s0_oa) |
| **Artifacts** | `analysis_s0_ob_v1/event_outcomes.csv` |
| **Key metrics** | s0_oa macro ΔF1 = −0.0178; s0_ob macro ΔF1 = −0.0159; difference = 0.0019 |
| **Strength** | **Supported** |
| **Scope limitation** | Order sensitivity tested on Seed 0 only; generality to Seed 1 not evaluated |

---

### Claim C4: Pre-update displacement signals are magnitude-informative but direction-unstable across seeds

**Claim text:** Source-to-candidate KL divergence and flood-expansion delta achieve event-level AUROC = 1.000 on Seed 0 / Order A, but collapse or invert on Seed 1 / Order A (KL AUROC = 0.000, flood expansion AUROC = 0.250), demonstrating that these signals measure adaptation magnitude rather than direction.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E5.2C2 analysis (s0_oa), E5.3 repeat analyses (s0_ob, s1_oa) |
| **Artifacts** | `analysis_s0_oa_v2/signal_event_summary.csv`, `analysis_s1_oa_v1/signal_event_summary.csv` |
| **Key metrics** | `source_to_candidate_kl`: ρ = −0.886 (s0_oa) vs. +0.886 (s1_oa); AUROC = 1.000 vs. 0.000 |
| **Strength** | **Supported** |
| **Scope limitation** | Event-level analysis (N = 6); direction instability may not generalise |

---

### Claim C5: No evaluated adaptation component provides stable net improvement over source-only

**Claim text:** Across E2 (confidence/augmentation), E3 (source anchor), E4 (loss weighting), and E5 (full TTA with risk scoring), no configuration consistently improved macro flood-class F1 over unadapted source-only inference across both seeds.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E2.4 synthesis; E3.3 terminal closeout; E4.2 screening closeout; E5.3 cross-cell analysis |
| **Artifacts** | `docs/e2_4_synthesis_closeout.md` (no candidate promoted), `docs/e3_3_terminal_closeout.md` (+0.05 pp below threshold), `docs/e4_2_screening_closeout.md` (all variants neutral/negative vs. E4_BASE), `docs/e5_3_repeat_results.md` (seed instability) |
| **Key metrics** | Best E3 anchor: +0.05 pp vs. AdaBN (threshold: +1.0 pp). Candidate macro ΔF1: −0.0178 (s0_oa), +0.0151 (s1_oa). |
| **Strength** | **Supported** — evidence spans E2–E5 |
| **Scope limitation** | Evaluation restricted to development events only; reserved-event results pending (E7) |

---

### Claim C6: BN-no-update is consistently near-neutral

**Claim text:** Updating only batch normalisation running statistics (without updating any learned parameters) produces near-zero adaptation effect (ΔF1 ≈ 0.000 on all evaluated events), serving as a useful negative control that isolates the contribution of the full consistency-training update.

| Element | Value |
| :--- | :--- |
| **Supporting experiments** | E5.2 and E5.3 (BN-control column in event_outcomes.csv for all three cells) |
| **Artifacts** | `event_outcomes.csv` for all three cells — column `delta_f1_bn_vs_src` |
| **Key metrics** | s0_oa: max |ΔF1_BN| = 0.0116 (Ev 559); all others < 0.004. s1_oa: max |ΔF1_BN| = 0.020 (Ev 559) |
| **Strength** | **Supported** |

---

### Claim C7 [EXCLUDED]: Temporal-risk gate improves performance

**Claim text (excluded):** Applying a temporal-risk gate based on pre-update signals prevents adaptation harm and improves net performance.

**Reason for exclusion:** Gate E criteria 2, 3, and 6 FAIL. No signal achieves reliable above-no-skill AUROC across all three cells. On Seed 1, applying a Seed-0-calibrated gate causes severe false rejection of beneficial adaptations (net realized F1 drops to 0.5362 vs. source-only macro F1 of 0.5884). No multivariate calibration rule was evaluated in the analysis protocol. This claim is not supported by committed evidence and is excluded from the Branch C study.

---

### Claim C8 [EXCLUDED]: Change-specific signals provide incremental value over generic signals

**Claim text (excluded):** Change-specific signals (s_flood_expansion, s_null) outperform generic signals (KL divergence, disagreement rate) for predicting adaptation harm.

**Reason for exclusion:** On Seed 0 Order B, `s_flood_expansion` (AUROC = 1.000) outperforms `source_to_candidate_kl` (AUROC = 0.889), providing some evidence. However, on Seed 1, both collapse (0.250 and 0.000 respectively), eliminating any stable incremental advantage. Criterion 4 is INCONCLUSIVE. This claim requires more evidence and is excluded.

---

## 3. Experiment Coverage Summary

| Claim | Status | Supporting Work Package |
| :--- | :---: | :---: |
| C1: Harm/benefit coexist on Seed 0 | Supported | E5.2 |
| C2: Seed-contingent outcomes | Supported | E5.3 |
| C3: Stream order has limited effect (Seed 0) | Supported | E5.3 |
| C4: Displacement signals are magnitude-informative only | Supported | E5.2–E5.3 |
| C5: No component shows stable net improvement | Supported | E2–E5 |
| C6: BN-no-update is near-neutral | Supported | E5.2–E5.3 |
| C7: Temporal-risk gate improves performance | **Excluded** | Gate E FAIL |
| C8: Change-specific > generic signals across seeds | **Excluded** | Criterion 4 INCONCLUSIVE |

---

## 4. Remaining Work (E7)

The following evidence gaps will be addressed in E7:

| Gap | Plan |
| :--- | :--- |
| Reserved-event performance (sealed test set) | Apply source-only, always-adapt, and BN-no-update to sealed test events |
| Seed 1 reserved-event performance | Apply Seed 1 always-adapt to reserved events for cross-seed comparison |
| CoTTA comparison (if implemented) | Apply bounded CoTTA variant to development events (see `docs/e6_nearest_method_applicability_matrix.md` and `docs/protocols/e6_cotta_compatibility_protocol_v1.md`) |

No new adaptation components are introduced in E7. Evaluation is restricted to comparators already defined in the E6.2 empirical study scope.
