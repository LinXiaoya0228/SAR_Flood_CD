# ChangeGuard Work Package E11: Gate T Authorization Governance Amendment

**Document Version:** `e11_gate_t_authorization_amendment_v1`  
**Date:** 2026-10-26  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E11 (Section 16.3: Gate T)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `AUTHORIZED_BY_USER — GOVERNANCE_AMENDED`  

---

## 1. Governance Context & Purpose of Amendment

In the original Gate T checklist (`docs/e11_gate_t_checklist.md`), two separate external authorization criteria were specified:
- **Criterion 10A:** Explicit User Authorization.
- **Criterion 10B:** Explicit Supervisor Approval.

This amendment formally records the institutional clarification regarding Criterion 10B and authorizes the one-time final test execution on the 10 sealed Kuro Siwo test events.

### Formal Declarations:
1. **Self-Imposed Governance Condition:** Supervisor approval was established as a self-imposed project-governance safeguard within the ChangeGuard engineering plan, and is **not an NTU institutional execution requirement** for running computational research scripts on allocated workstation hardware.
2. **Explicit User Authorization:** In the authoritative instruction for this task, the student investigator / user has explicitly authorized the frozen final sealed-test evaluation.
3. **No Supervisor Endorsement Claimed:** In strict accordance with academic honesty, **no claim is made that supervisor approval or endorsement was received**. The evaluation proceeds under explicit user authorization with supervisor approval clarified as a non-institutional requirement.

---

## 2. Invariance Guarantees: What Does NOT Change

This amendment applies **strictly and exclusively to authorization governance**. It introduces **zero changes** to the scientific or technical protocol:
1. **Research Route:** Unchanged; strictly Branch C (Controlled Empirical Study of TTA Instability). Zero controller, heuristic gate, or threshold mechanism is retained or evaluated.
2. **Target Test Events:** Unchanged; strictly the 10 frozen sealed Kuro Siwo test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`). Zero development events are included.
3. **Model Architectures:** Unchanged; SNUNet-ECAM (12.06M params) and SiamUnet_diff (1.35M params).
4. **Source Checkpoints & Provenance:** Unchanged; the exact 6 source checkpoints verified by SHA-256 in Block E10/E11.
5. **Evaluated Methods:** Unchanged; `source_only`, `bn_no_update`, `tent`, and `cotta` for SNUNet-ECAM; `source_only`, `bn_no_update`, and `tent` for SiamUnet_diff.
6. **Random Seeds:** Unchanged; Seeds 0, 1, and 2.
7. **Prediction Timing:** Unchanged; strictly `PRE_UPDATE` primary prediction protocol.
8. **Deployment Semantics:** Unchanged; strictly `reset_per_event` with canonical Order A event sequencing.
9. **Label Access Timing:** Unchanged; strict deferred post-persistence disk read (`mark_persisted()`).
10. **Metrics & Bootstrap Plan:** Unchanged; Event-Macro Flood F1 primary, Worst-Event F1, Macro IoU, Pooled F1, and 10,000-draw paired event-cluster bootstrap for 95% BCa confidence intervals.
11. **Matrix Accounting:** Unchanged; exactly 21 physical streams producing exactly 210 event-result rows.
12. **Scientific Hypotheses:** Unchanged; pre-registered hypotheses $H_1$, $H_2$, $H_3$ as recorded in `docs/e11_paper_manuscript_draft.md`.

---

## 3. One-Time Execution Commitment

The sealed final test remains a **one-time research decision**:
- All 21 streams will be executed once under strict failure preservation.
- **No tuning, threshold adjustment, hyperparameter tweaking, seed cherry-picking, or method modification is permitted after viewing test outcomes.**
- If test results confirm development findings, they will be reported as confirmation. If test results contradict development findings, the contradiction will be reported transparently as a scientific finding.

---

## 4. Official Authorization Record

> **"The user explicitly authorized execution of the frozen final-test matrix and confirmed that supervisor approval is not an institutional requirement. No supervisor endorsement is claimed."**

With this amendment recorded, Gate T is formally updated from `AWAITING USER AND SUPERVISOR APPROVAL` to `AUTHORIZED_FOR_ONE_TIME_EXECUTION`.
