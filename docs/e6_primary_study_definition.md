# ChangeGuard Work Package E6.2: Primary Study Definition — Branch C

**Status:** COMPLETE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**HEAD at Decision:** `1311bad`
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.2 and 11.4
**Branch Selected:** **C** — Empirical Study
**Gate E Verdict Source:** `docs/e6_gate_e_decision.md`

> [!IMPORTANT]
> **Branch C is evidence-driven.** Gate E criteria 2, 3, and 6 FAIL. No method component shows stable cross-seed improvement. This document freezes the empirical study scope for E7 onward.

---

## 1. Study Identity

| Field | Value |
| :--- | :--- |
| **Study type** | Controlled empirical study |
| **Primary question** | Under what conditions does online test-time adaptation degrade, maintain, or improve flood-class segmentation performance on paired multi-temporal SAR streams? |
| **Primary claim** | The study characterises adaptation outcome variability across source checkpoints, stream orders, and event geographies; it does not claim that any evaluated adaptation component provides stable net improvement over source-only inference. |
| **Architecture** | SNUNet-ECAM (3-class dense segmentation, bi-temporal SAR input) |
| **Adaptation protocol** | Per-event reset; confidence-filtered consistency self-training; τ = 0.90; horizontal flip augmentation |
| **Evaluation scope** | 6 development events (canonical E5 evidence); ≥ 1 reserved sealed test set to be evaluated in E7 |
| **Seed/order cells** | 3 cells executed: Seed 0 / Order A, Seed 0 / Order B, Seed 1 / Order A |

---

## 2. Central Empirical Findings to Report

The study reports the following findings, all supported by the committed E5 evidence base:

### 2.1 Adaptation outcome is seed-contingent

| Event | s0_oa outcome | s1_oa outcome | Seed-stable? |
| :--- | :---: | :---: | :---: |
| 279 (Spain/Ebro, large) | Harmful (−0.0545) | Beneficial (+0.1588) | No |
| 437 (UK/Severn, medium) | Harmful (−0.0642) | Harmful (−0.0537) | **Yes** |
| 520 (USA/Florence, medium) | Harmful (−0.0140) | Beneficial (+0.0203) | No |
| 559 (USA/Midwest, small) | Beneficial (+0.0197) | Harmful (−0.0263) | No |
| 1111003 (Bangladesh, small) | Neutral (+0.0082) | Neutral (+0.0013) | Yes (neutral) |
| 1111008 (Myanmar, medium) | Neutral (−0.0019) | Neutral (−0.0100) | Yes (neutral) |

Only Event 437 is consistently harmful across all three cells.

### 2.2 Stream order is not a primary driver of adaptation outcome

The order repeat (Seed 0 / Order B) replicates the canonical Seed 0 findings with high fidelity:
- All event classifications are preserved (within resolution of the neutral boundary).
- Macro ΔF1: s0_oa = −0.0178, s0_ob = −0.0159. Difference = 0.0019 (negligible).
- Signal AUROC values are stable across orders (e.g., `source_to_candidate_kl`: 1.0000 → 0.8889).

### 2.3 Pre-update displacement signals measure magnitude, not direction

- Signals such as `source_to_candidate_kl` and `s_flood_expansion` achieve AUROC = 1.000 on Seed 0 because Event 279 produces both the largest displacement and the worst outcome on that seed.
- On Seed 1, Event 279 produces the largest displacement but the best outcome (+0.1588 ΔF1). The ranking inverts; AUROC collapses to 0.000.
- This demonstrates that uncalibrated displacement signals are confounded by adaptation magnitude, not predictive of adaptation direction across seeds.

### 2.4 No evaluated adaptation component provides stable net improvement

- E2: No consistency adaptation candidate met the +1.0 pp Macro F1 improvement threshold.
- E3: Best source anchor +0.05 pp vs. AdaBN (far below threshold). Not promoted.
- E4: All loss-weighting variants neutral-to-negative vs. E4_BASE.
- E5: Candidate TTA: Macro ΔF1 = −0.0178 (s0_oa), −0.0159 (s0_ob), +0.0151 (s1_oa). Instability precludes a positive claim.
- BN-no-update (batch normalization statistics updated, parameters frozen): consistently near-neutral (ΔF1 ≈ 0) — a useful negative control.

### 2.5 Batch Normalization update alone is near-neutral

BN-no-update (update BN running statistics only, freeze all other parameters) was evaluated as a control in E5:
- s0_oa: BN ΔF1 vs. source ranges from −0.007 (Event 437) to +0.012 (Event 559). Macro difference ≈ 0.
- This shows that BN adaptation alone does not drive the candidate's harmful outcomes; the full consistency self-training update is the operative mechanism.

---

## 3. Study Scope — What Is Included and Excluded

### Included in Branch C empirical study:
1. **Full characterisation of adaptation outcome variability** across seed, order, and event type.
2. **Ablation of BN-no-update** as the most conservative adaptation step (already completed as E5 control).
3. **Signal applicability report** — characterising why pre-update signals are magnitude-informative but direction-unstable across seeds.
4. **Reserved-event evaluation (E7)** — applying the frozen E5 protocol to sealed test events with source-only and always-adapt as primary comparators. No thresholds are calibrated on reserved events.
5. **Nearest-method comparison (E6.3)** — applicability audit for CoTTA and related methods.

### Excluded from Branch C empirical study:
- A temporal-risk policy controller (requires Gate E PASS; not supported).
- Any claim that a specific method component provides stable, beneficial, generalised improvement.
- Post-hoc threshold calibration to make harmful outcomes appear beneficial.
- New adaptation components not evaluated in E2–E5.

---

## 4. Reserved-Event Evaluation Protocol (E7 Preview)

The reserved sealed test events are:
`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`

These events have not been accessed at any point during E2–E5. In E7, the following comparators will be reported:
1. Source-only (frozen seed 0 checkpoint, zero updates).
2. Always-adapt (full E5 candidate protocol, seed 0).
3. BN-no-update (BN statistics updated only, seed 0).

No temporal-risk gate will be applied to reserved events, as no gate has demonstrated cross-seed reliability. Source-only will serve as the primary reference.

---

## 5. FYP Contribution Statement

The empirical contribution of this study is:

> ChangeGuard provides a controlled, multi-seed, multi-order investigation of online test-time adaptation applied to paired SAR flood segmentation. The study finds that adaptation harm and benefit coexist and are strongly seed-contingent, that no evaluated adaptation component provides stable improvement over source-only inference, and that pre-update displacement signals — while discriminative within a single seed — do not reliably predict adaptation direction across seeds. These findings motivate caution in applying uncalibrated consistency self-training to non-stationary SAR flood streams.

This statement is supported by the committed E5 evidence base and does not make any unsupported causal, universal, or generalisation claim.

---

## 6. Cross-Reference

| Document | Role |
| :--- | :--- |
| `docs/e6_gate_e_decision.md` | Gate E evidence ledger (preceding document) |
| `docs/e6_claim_to_experiment_map.md` | Claim-to-experiment mapping (next document) |
| `docs/e6_nearest_method_applicability_matrix.md` | Closest-method audit |
| `docs/protocols/e6_cotta_compatibility_protocol_v1.md` | CoTTA compatibility protocol |
| `docs/e6_final_route_decision.md` | Final integrated decision |
