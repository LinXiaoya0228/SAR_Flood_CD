# ChangeGuard Work Package E8: Final Route Decision

**Status:** COMPLETE
**Date:** 2026-10-25
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E8 (Sections 13.1–13.4) & Block E9 (Section 14)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/e7_final_closeout.md` (`E7 COMPLETE — READY FOR E8`)
- `docs/protocols/e8_cross_event_confirmation_protocol_v1.md`
- `docs/e8_closest_method_transfer_audit.md`
- `docs/e8_v1_label_io_integrity_incident.md`
- `docs/e8_v2_launch_authorization.md`

---

> [!IMPORTANT]
> **Terminal E8 Verdict: `E8 CLOSED — BRANCH C CONFIRMATION COMPLETE`**
>
> The full 69-stream, 414-event-row E8 confirmation matrix has been executed, audited, and certified.
> All three criteria for promoting to Branch B have definitively **FAILED**.
> CoTTA causes net degradation versus `source_only` (mean delta: -0.0086 Macro F1; 95% CI: `[-0.0202, +0.0010]`)
> and produces statistically negligible gains over unadapted streaming `bn_no_update` control (mean delta: +0.0028 Macro F1; 95% CI: `[-0.0010, +0.0065]`),
> with complete adaptation stall on Seed 2 (100% neutral event outcomes).
>
> **ChangeGuard remains definitively on Branch C: Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams.**

---

## 1. Decision Lineage

| Milestone | Artifact | Decision / Verdict |
| :--- | :--- | :--- |
| **E6 Final Decision** | `docs/e6_final_route_decision.md` | `E6 COMPLETE — BRANCH C`. Rejects Branch A/B; schedules CoTTA baseline. |
| **E7 Implementation** | `docs/e7_final_closeout.md` | `E7 COMPLETE — READY FOR E8`. CoTTA adapter and smoke tests complete. |
| **E8 Literature Transfer Audit** | `docs/e8_closest_method_transfer_audit.md` | Audits PA-TTA (IEEE TGRS 2026) & CIM-TTDA; both classified as `INCOMPATIBLE_FOR_E8_EXECUTION`. |
| **E8 v1 Incident & Repair** | `docs/e8_v1_label_io_integrity_incident.md` | Identifies early GT disk read in batch plan; repairs deferred-label loader. v1 preserved. |
| **E8 v2 Authorization** | `docs/e8_v2_launch_authorization.md` | Authorizes smoke v3 gate and formal v2 execution (67 execute-new + 2 certified reused). |
| **E8 Final Route Decision** | This document | **`E8 CLOSED — BRANCH C CONFIRMATION COMPLETE`** |

---

## 2. Lineage Selection and Provenance

In accordance with Plan V3 research integrity rules, both execution lineages remain preserved:

1. **Lineage 2 (Canonical Evidence):**
   - Output root: `outputs/e8_confirmation_v2`
   - Log: `logs/e8_confirmation/e8_confirmation_v2.log` (all 67 execute-new streams completed, launcher exit code 0)
   - Reused streams: exactly 2 certified Week-12 physical streams (`source_only_frozen_running_stats_s0`, `tent_matched_pre_update_s0_oa_re`)
   - Data I/O status: **FORMALLY CERTIFIED — POST-PERSISTENCE DEFERRED GT DISK READ**
   - Sealed-test access: Exactly 0 samples across all streams.
2. **Lineage 1 (Historical Preserved Record):**
   - Output root: `outputs/e8_confirmation`
   - Log: `logs/e8_confirmation/e8_confirmation_v1.log` (completed with exit code 0)
   - Status: `COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ`.
   - Preserved without overwrite, deletion, or renaming.

---

## 3. Evaluation of Branch B Promotion Criteria

Plan V3 Section 13.1 specifies that a transition from Branch C to Branch B requires satisfying three formal criteria across the confirmation matrix:

### Rule 1: Statistically Significant Net Improvement over Source-Only
- **Criterion:** Adaptation must demonstrate consistent, statistically significant positive delta versus the frozen `source_only` baseline across seeds and event clusters (95% bootstrap CI lower bound > 0, benefit ratio >= 80%).
- **Empirical Evidence (Lineage 2):**
  - CoTTA Macro F1: `0.5820 ± 0.0151` vs `source_only`: `0.5906 ± 0.0037` (mean delta: `-0.0086`).
  - CoTTA Pooled F1: `0.4903 ± 0.0783` vs `source_only`: `0.5012 ± 0.0221` (mean delta: `-0.0110`).
  - Paired Event-Cluster Bootstrap 95% CI (vs `source_only`): `[-0.0202, +0.0010]` (negative mean; upper bound barely touches zero).
  - Pairwise Event Outcomes (72 CoTTA event rows): Harms = `26` (36.1%), Benefits = `14` (19.4%), Neutrals = `32` (44.4%).
  - Harms outnumber benefits nearly 2:1.
- **Verdict: FAILED.**

### Rule 2: Consistent Advantage over Matched BN-No-Update Control
- **Criterion:** Adaptation must demonstrate positive delta versus the unadapted streaming batch-norm control (`bn_no_update`) across all seeds and orders, without severe degradation or failure modes on individual events.
- **Empirical Evidence (Lineage 2):**
  - CoTTA Macro F1: `0.5820 ± 0.0151` vs `bn_no_update`: `0.5793 ± 0.0147` (mean delta: `+0.0028`).
  - Paired Event-Cluster Bootstrap 95% CI (vs `bn_no_update`): `[-0.0010, +0.0065]` (crosses zero).
  - Pairwise Event Outcomes (vs `bn_no_update`): Harms = `0` (0.0%), Benefits = `10` (13.9%), Neutrals = `62` (86.1%).
  - **Seed 2 Collapse:** On Seed 2, CoTTA achieves exactly 0 benefits across 24 event evaluations (100% neutral, delta +0.0005 to +0.0015). Adaptation mechanisms provide zero benefit.
  - **Worst-Event Degradation:** CoTTA worst-event minimum F1 is `0.2584` (on Event 1111003), essentially matching unadapted controls (`0.2529`).
- **Verdict: FAILED.**

### Rule 3: Feasibility of an Unsupervised Adaptive Controller
- **Criterion:** Existence of a validated pre-update unsupervised risk indicator capable of reliably routing or gating adaptation decisions without domain-specific inversion.
- **Empirical Evidence:**
  - Gate E (E6.1) established that pre-update unsupervised risk indicators (entropy, KL divergence, teacher-student discrepancy) invert directionally across seeds (e.g. AUROC drops from 1.000 on Seed 0 to 0.000 on Seed 1).
  - In E8, no domain-invariant risk indicator was found to predict catastrophic degradation on hard events (such as Event 279 or Event 1111003).
  - Policy LOEO is marked as `"NA under Branch C"` because no policy gate was approved or claimed. Fabricating a controller gate is strictly prohibited by Plan V3 Section 13.1.
- **Verdict: FAILED.**

---

## 4. Literature-Method Transfer Audit Status

The E8 transfer audit (`docs/e8_closest_method_transfer_audit.md`) audited the nearest published TTA methods in remote sensing and vision:

1. **PA-TTA (Huang et al., IEEE Transactions on Geoscience and Remote Sensing, 2026):**
   - **Upstream Repository:** `https://github.com/DonMuv/PA-TTA`
   - **Audit Finding:** Specifically formulated for optical bitemporal photovoltaic growth assessment. Consumes SAM foundation model masks (`sam_mask`) for spatial pseudo-label merging, outputs binary change, and executes multi-epoch target dataset iteration.
   - **Protocol Incompatibility:** Incompatible with ChangeGuard's single-pass online PRE_UPDATE SAR architecture.
   - **Classification:** `INCOMPATIBLE_FOR_E8_EXECUTION — DISCUSS ONLY IN CHAPTER 4/5`.
2. **CIM-TTDA:**
   - **Audit Finding:** Formulated for multi-epoch domain adaptation using optical spatial priors and offline target batches.
   - **Protocol Incompatibility:** Incompatible with online temporal streaming SAR flood segmentation.
   - **Classification:** `INCOMPATIBLE_FOR_E8_EXECUTION — DISCUSS ONLY IN CHAPTER 4/5`.

Neither method can be executed as a valid online SAR baseline without compromising research integrity or inventing arbitrary ad-hoc adapters.

---

## 5. Formal Route Decision

```
================================================================================
FINAL CHANGEGUARD WORK PACKAGE E8 ROUTE DECISION
================================================================================
Terminal Status:   E8 CLOSED — BRANCH C CONFIRMATION COMPLETE
Branch Decision:   BRANCH_C_CONFIRMED
Branch B Promotion: REJECTED (Criteria 1, 2, 3 FAILED)
Canonical Lineage: Lineage 2 (outputs/e8_confirmation_v2, 69 streams, 414 rows)
Sealed-Test Access: ZERO (0 events, 0 samples)
Next Work Package: E9 (Minimal Empirical Ablations for Branch C)
================================================================================
```

---

## 6. Scope Boundaries for Work Package E9

Under the confirmed Branch C:
1. **No Policy or Controller Work:** No acceptance threshold tuning, no policy search, no score calibration.
2. **No Model Re-training:** Checkpoint weights remain frozen.
3. **No Sealed-Test Event Access:** Sealed test events remain strictly quarantined until all development work is frozen.
4. **Scope of E9:** Focus exclusively on minimal targeted empirical ablations to explain the empirical findings of Branch C:
   - Sample-size weighting and Event 279 dominance on pooled statistics;
   - Batch normalization momentum and tracking sensitivity under severe SAR backscatter distribution shifts;
   - Teacher momentum and augmentation view sensitivity in CoTTA.

