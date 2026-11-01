# ChangeGuard Work Package E6.4: Final Route Decision

**Status:** COMPLETE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**HEAD at Decision:** `1311bad`
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Sections 11.1–11.4

> [!IMPORTANT]
> **Terminal E6 Verdict: `E6 COMPLETE — BRANCH C`**
>
> Gate E FAIL. No method component demonstrates stable net improvement across seeds. ChangeGuard proceeds as a controlled empirical study characterising TTA outcome variability on paired SAR flood segmentation streams. CoTTA is scheduled as a bounded secondary formal baseline for E7.

---

## 1. Decision Lineage

| Step | Document | Verdict |
| :--- | :--- | :--- |
| E6.1 — Gate E evidence ledger | `docs/e6_gate_e_decision.md` | Gate E: **FAIL** (Criteria 2, 3, 6 FAIL; Criterion 4 INCONCLUSIVE) |
| E6.2 — Branch selection | `docs/e6_primary_study_definition.md`, `docs/e6_claim_to_experiment_map.md` | Branch: **C** |
| E6.3 — Nearest-method audit | `docs/e6_nearest_method_applicability_matrix.md` | CoTTA: SCHEDULE; EATA & SAR: IMPLEMENTED & EVALUATED; Others: DISCUSS ONLY |
| E6.4 — Integrated final decision | This document | **E6 COMPLETE — BRANCH C** |

---

## 2. Gate E Summary (reproduced from E6.1)

| # | Criterion | Verdict |
|---|---|---|
| 1 | Harm and benefit coexist | PASS |
| 2 | Usable score above no-skill in all cells | **FAIL** |
| 3 | Risk direction survives both repeats | **FAIL** |
| 4 | Incremental change-specific value | **INCONCLUSIVE** |
| 5 | Signal available before declared action | PASS |
| 6 | Compute overhead and false rejection acceptable | **FAIL** |
| | **Overall Gate E** | **FAIL** |

---

## 3. Branch Decision

**Branch A — REJECTED.** Gate E fails criteria required for a selective policy. The best-performing signal (`source_to_candidate_kl`, AUROC = 1.000 on s0_oa) fully inverts on the seed repeat (AUROC = 0.000 on s1_oa). No signal is reliably directional across seeds.

**Branch B — REJECTED.** No adaptation/normalisation/anchor component from E2–E5 shows stable positive delta versus source-only across both evaluated seeds:
- E2 consistency candidates: no candidate met the +1.0 pp Macro F1 promotion threshold.
- E3 source anchor: best result +0.05 pp vs. AdaBN (threshold +1.0 pp).
- E4 loss weighting: all variants neutral-to-negative vs. E4_BASE.
- E5 candidate TTA: Macro ΔF1 = −0.0178 (s0_oa), −0.0159 (s0_ob), +0.0151 (s1_oa) — unstable across seeds.
- BN-no-update: consistently near-zero (neither improvement nor harm) — a valid negative control but not a beneficial component.

**Branch C — SELECTED.** No method component improves reasonable controls stably. The study characterises adaptation outcome variability across seeds, orders, and event geographies.

---

## 4. Empirical Study Scope (Branch C)

### 4.1 Primary comparators for E7 development-event implementation and smoke verification

| Comparator | Description | Seed(s) |
| :--- | :--- | :---: |
| Source-only | Frozen checkpoint, zero parameter updates | 0, (1 if resources permit) |
| Always-adapt | Full E5 candidate TTA (τ = 0.90, horizontal flip, per-event reset) | 0 |
| BN-no-update | BN running statistics updated, all other parameters frozen | 0 |
| CoTTA variant | Teacher-student EMA + augmentation-averaged pseudo-labels; per-event reset | 0 |

### 4.2 Primary claims (supported by E2–E5 evidence)

1. Adaptation harm and benefit coexist and are seed-contingent (C1, C2 from claim map).
2. Stream order has limited effect on Seed 0 outcomes (C3).
3. Pre-update displacement signals measure magnitude, not direction, and invert across seeds (C4).
4. No evaluated adaptation component provides stable net improvement over source-only across seeds (C5).
5. BN-no-update is near-neutral, isolating the full consistency update as the operative mechanism (C6).

### 4.3 Excluded claims

- Temporal-risk gate improves performance (Claim C7 — excluded, Gate E FAIL).
- Change-specific signals provide stable incremental value over generic signals (Claim C8 — excluded, Criterion 4 INCONCLUSIVE).

---

## 5. CoTTA Implementation Decision

Per `docs/e6_nearest_method_applicability_matrix.md` and the compatibility protocol `docs/protocols/e6_cotta_compatibility_protocol_v1.md`:
- **Action: SCHEDULE AS FORMAL BASELINE**
- **Rationale:** CoTTA is the nearest structurally similar method. Its deviations are bounded and documentable (per-event reset and continual modes both supported; bi-temporal input requires joint spatial augmentation). Code is available (MIT License).
- **Implementation window:** E7.
- **Scope:** EMA teacher + augmentation-averaged pseudo-labels; optional intra-event stochastic restoration. Protocol fully defined in `docs/protocols/e6_cotta_compatibility_protocol_v1.md`.
- **Constraint:** Must be implemented from committed code, not dirty working tree. Must not alter E2–E5 formal evidence. Zero GPU execution during E6.

No other new method is scheduled for implementation.

---

## 6. E7 Preview

E7 (window: Dec 7–13 per master plan) will:

1. Implement and smoke-test bounded CoTTA using development Event 520 only.
2. Preserve the source-only, always-adapt and BN-no-update comparators for the later development-event confirmation matrix.
3. Apply no temporal-risk gate or newly fitted calibration threshold.
4. Use committed code for real-data execution; no dirty working tree.

The earlier wording in this section that prospectively mentioned applying E7 comparators to
"sealed reserved test events" was stale and unauthorized. It conflicted with the permanent
quarantine in `docs/research_integrity_protocol.md`. Actual E7 and E8 v1 execution accessed
development events only; sealed-test access remained zero and no reserved-test result exists.

---

## 7. Research Integrity Commitments

All decisions in E6 are evidence-driven and grounded in the governing plan criteria defined in `docs/MASTER_EXECUTION_PLAN_V3.md` (Section 11.1, commit `24ffcc9`) and formalized in `configs/changeguard/e6_decision_contract_v1.json`:
- Criteria and thresholds formalize the governing plan directly; no rule was altered post-hoc to manipulate verdicts.
- `INCONCLUSIVE` labels are not relabeled as `PASS`.
- Branch C selection is not a fallback due to implementation difficulty; it reflects the empirical evidence from E2–E5.
- No unsupported causal claim is included in Branch C scope.
- No superlatives, universal claims, or prescriptive technology recommendations for future work are included.
- The FYP contribution is framed as a controlled empirical investigation with explicit scope limitations.

---

## 8. Full Document Manifest (E6)

| File | Role | Status |
| :--- | :--- | :---: |
| `configs/changeguard/e6_decision_contract_v1.json` | Formalized evaluation rules (retrospective contract) | Created |
| `docs/e6_gate_e_decision.md` | Six-criterion Gate E evidence ledger | Created |
| `docs/e6_primary_study_definition.md` | Branch C study definition | Created |
| `docs/e6_claim_to_experiment_map.md` | Claim-to-evidence traceability | Created |
| `docs/e6_nearest_method_applicability_matrix.md` | Nearest-method applicability matrix | Created |
| `docs/protocols/e6_cotta_compatibility_protocol_v1.md` | CoTTA compatibility protocol | Created |
| `docs/e6_final_route_decision.md` | This integrated decision | Created |

---

## 9. Terminal Verdict

```
E6 COMPLETE — BRANCH C
```

Gate E FAIL. No method component shows stable cross-seed improvement. ChangeGuard proceeds as a controlled empirical study. CoTTA scheduled as bounded secondary formal baseline. No temporal-risk policy. All E6 documents committed; E7 window begins Dec 7.
