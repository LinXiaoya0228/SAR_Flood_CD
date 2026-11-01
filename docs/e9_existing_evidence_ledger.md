# ChangeGuard Work Package E9: Existing Evidence and Claim-to-Ablation Ledger

**Status:** COMPLETE — PHASE 1 AUDIT
**Date:** 2026-10-25
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E9 (Section 14.1)
- `docs/e8_confirmation_closeout.md`
- `docs/e8_final_route_decision.md`
- `docs/e6_final_route_decision.md`
- `configs/changeguard/e9_existing_evidence_ledger.json`

---

## 1. Executive Summary & Audit Methodology

Plan V3 Section 14.1 establishes that **the ablation matrix must follow the actual retained claims of the project**. Because ChangeGuard definitively operates under **Branch C: Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams**, no deployable controller, threshold tuning, or policy gate was approved or retained. Therefore:
1. Components rejected in earlier work packages (E2–E6) cite their complete negative experiments rather than being artificially re-inserted solely to be removed again.
2. Unresolved scientific questions regarding normalization context vs. parameter updates and CoTTA component contributions are identified for bounded diagnostic execution.
3. Unsupported literature methods (PA-TTA and CIM-TTDA) remain classified as incompatible.

---

## 2. Claim-to-Ablation Evidence Matrix

| # | Plan V3 Item | Classification | Existing Artifact Path / Reference | Evaluated Factors (Seeds, Orders, Resets) | Empirical Finding / Conclusion | Compatibility Limitations & Governance Rationale |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | **No confidence filtering** | `COMPLETE_EXISTING_EVIDENCE` | `outputs/e2_confidence_grid/v1/`<br>`docs/e2_1_confidence_grid_closeout.md`<br>`outputs/e8_confirmation_v2` | Seed 0, Order A, Event 520 & Dev-6;<br>Thresholds $C \in \{0.0, 0.70, 0.85, 0.90, 0.95\}$;<br>E8: Tent ($C=0.0$) vs CoTTA ($C=0.90$) | Confidence thresholding yielded Macro F1 deltas of only $+0.01$ to $+0.08$ pp vs unfiltered baseline ($C=0.0$), failing the $+1.0$ pp promotion threshold. In E8, unfiltered Tent ($0.5824$) matches thresholded CoTTA ($0.5820$). | Confidence filtering does not prevent adaptation failure under noisy SAR pseudo-labels. No further confidence sweep required. |
| **2** | **No consistency** | `COMPLETE_EXISTING_EVIDENCE` | `outputs/e2_augmentation_grid/v1/`<br>`docs/e2_2_augmentation_grid_closeout.md`<br>`outputs/e8_confirmation_v2` | Seed 0, Order A, Dev-6;<br>Augmentations: Clean, Flips, Rotations, Multi-view;<br>E8: Single-view Tent vs 10-view CoTTA | Multi-view augmentation consistency produced Macro F1 deltas of $-0.05$ to $+0.02$ pp vs single clean view (`E2_AUG_A0`). In E8, 10-view CoTTA ($0.5820$) yielded no gain over single-view Tent ($0.5824$). | Multi-view augmentation increases compute $10\times$ with zero measurable benefit on SAR flood segmentation. Complete existing evidence. |
| **3** | **No anchor** | `COMPLETE_EXISTING_EVIDENCE` | `outputs/e3_source_anchor/v1/`<br>`docs/e3_3_terminal_closeout.md`<br>`configs/changeguard/e3_source_anchor_v1.json` | Seeds 0, 1; Order A; Dev-6;<br>Anchor weights $\lambda \in \{0.0, 0.01, 0.05, 0.10\}$ across Feature, Logit, and Prob metrics | Source anchor regularization yielded at best $+0.05$ pp Macro F1 over unanchored AdaBN baseline ($\lambda=0.0$), far below the $+1.0$ pp promotion gate. Formally rejected in E3.3. | Anchoring to source parameters fails to prevent catastrophic forgetting on hard events. Fully evaluated and rejected. |
| **4** | **No temporal / null evidence** | `COMPLETE_EXISTING_EVIDENCE` | `outputs/e5_temporal_evidence/`<br>`docs/e5_final_closeout.md`<br>`docs/e5_3_repeat_results.md` | Seeds 0, 1; Orders A, B; Dev-6;<br>Temporal difference rasters, pre/post null pseudo-labels | Change-detection temporal risk scores inverted across seeds (AUROC $1.000$ on Seed 0 $\to 0.000$ on Seed 1). Offered zero stable benefit over generic signals. | Null temporal evidence fails domain-invariance across seeds. Fully evaluated and rejected. |
| **5** | **No controller** | `COMPLETE_EXISTING_EVIDENCE` | `docs/e6_gate_e_decision.md`<br>`docs/e6_final_route_decision.md`<br>`docs/e8_final_route_decision.md` | All 69 streams; Seeds 0, 1, 2; Orders A, B;<br>Unconditional streaming evaluation | Gate E failed Criteria 2, 3, 6. Unsupervised risk indicators invert across seeds. Branch C operates unconditionally with NO controller. | Branch C is itself the empirical characterization of unconditional TTA. Fabricating a controller is prohibited. |
| **6** | **Generic protection substituted** | `COMPLETE_EXISTING_EVIDENCE` | `outputs/e8_confirmation_v2/`<br>`docs/e8_confirmation_closeout.md` | 69 physical streams, 414 event rows;<br>Seeds 0, 1, 2; Orders A, B; Reset & Continual;<br>Methods: AdaBN, Tent, EATA, SAR, CoTTA | All generic protection baselines fail to beat `source_only` ($0.5906$). CoTTA achieves $0.5820$, Tent $0.5824$, SAR $0.5801$, AdaBN $0.5793$. 95% bootstrap CIs cross zero vs BN control. | Generic vision TTA protection mechanisms (entropy, Fisher, SAM, EMA) do not prevent degradation under SAR distribution shift. |
| **7** | **Normalization control** | `REQUIRES_BOUNDED_CONFIRMATION` | `outputs/e8_confirmation_v2/`<br>`docs/e1_5_bn_normalization_closeout.md`<br>`docs/e8_confirmation_closeout.md` | Existing: `source_only` ($m=0.0$), `bn_no_update` ($m=1.0$), `adabn` ($m=0.10$);<br>Diagnostic: $m=0.01$ and $m=0.05$ | Switching from frozen source stats ($0.5906$) to batch stats ($0.5793$) accounts for the entire degradation observed in TTA baselines. | A bounded momentum diagnostic ($m=0.01, 0.05$) is required to characterize the exact sensitivity of BN statistics tracking without parameter updates. |
| **8** | **Reject-update-only vs source-fallback** | `NOT_APPLICABLE_UNDER_BRANCH_C` | `docs/MASTER_EXECUTION_PLAN_V3.md`, Sec 14.1 | N/A (No controller retained) | N/A | Because Gate E failed and no adaptive controller exists, comparing fallback policies is not applicable. Inventing a policy is prohibited. |
| **9** | **Matched acceptance / calibration budget** | `NOT_APPLICABLE_UNDER_BRANCH_C` | `docs/e5_final_closeout.md` | N/A (No policy retained) | N/A | Acceptance budget calibration requires a selective gating policy. Under Branch C, no policy exists. E5 raw signal evidence is cited. |
| **10** | **CoTTA component diagnosis** | `REQUIRES_BOUNDED_CONFIRMATION` | `src/changeguard/cotta_adapter.py`<br>`outputs/e8_confirmation_v2` (Full CoTTA reference) | Seeds 0, 1; Orders A, B; Reset mode; Dev-6;<br>Removals: (1) no stochastic restoration (`p=0.0`); (2) no teacher EMA (`alpha=0.0`) | Full CoTTA ($0.5820$) yields only $+0.0028$ over unadapted BN control ($0.5793$). Specific component contributions remain unisolated. | Bounded 8-stream diagnostic required to isolate whether SSR or teacher EMA materially alters adaptation outcomes beyond normalization. |

---

## 3. Scope of Unresolved Claims for Bounded Confirmation

From the audit above, exactly two targeted scientific questions remain unresolved:

1. **Question A (Normalization Dynamics):**
   - *Question:* Does running-statistics momentum ($m \in \{0.01, 0.05\}$) mitigate the degradation observed under current-batch statistics ($m=1.0$) and default AdaBN ($m=0.10$) without any parameter updates?
   - *Scope:* Exactly 4 physical streams (Seeds 0 and 1, Order A, batch size 16, zero parameter updates).
2. **Question B (CoTTA Component Diagnosis):**
   - *Question:* Which additional CoTTA mechanisms (stochastic source restoration vs teacher EMA) materially alter performance beyond the baseline normalization context?
   - *Scope:* Exactly 8 physical streams (Seeds 0 and 1, Orders A and B, reset-per-event mode, 6 development events):
     - 4 streams for CoTTA without Stochastic Source Restoration (`cotta_no_ssr`, `restoration_prob = 0.0`).
     - 4 streams for CoTTA without Teacher EMA (`cotta_no_ema`, `ema_alpha = 0.0`).

All other items (1–6, 8, 9) are fully closed by existing evidence or are formally not applicable under Branch C.

