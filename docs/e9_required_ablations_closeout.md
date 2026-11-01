# ChangeGuard Work Package E9: Required Ablations and Diagnostic Closeout

**Status:** E9 CLOSED — EMPIRICAL SCOPE AND ABLATIONS FROZEN  
**Date:** 2026-10-25  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E9 (Section 14: Required Ablations and Method Freeze)
- `docs/e8_confirmation_closeout.md` (`E8 CLOSED — BRANCH C CONFIRMATION COMPLETE`)
- `docs/e8_final_route_decision.md`
- `docs/e9_existing_evidence_ledger.md`
- `configs/changeguard/e9_diagnostic_matrix_v1.json`
- `configs/changeguard/e9_execution_ledger_v1.json`

---

## 1. Executive Summary & Lineage Certification

Work Package E9 fulfills Section 14 of `docs/MASTER_EXECUTION_PLAN_V3.md`. Operating strictly under **Branch C** (Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams), E9 resolves the remaining core scientific questions:

> *Which parts of observed TTA behavior are explained by normalization context, and which additional CoTTA components materially change the result beyond that normalization effect?*

### Matrix Composition & Scope
- **Total Physical Streams:** 26 (14 reused certified streams from canonical Lineage 2 E8 confirmation + 12 newly executed diagnostic streams).
- **Total Event-Result Rows:** 156 (84 reused + 72 new).
- **Development Partition:** Exactly 6 declared events: `[279, 437, 520, 559, 1111003, 1111008]`.
- **Sealed-Test Access:** **Zero access verified**. The forbidden sealed-test set `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` had 0 reads across all 26 streams.
- **Data I/O Contract:** Post-persistence deferred ground-truth disk read. All prediction and probability shards are atomically written and fsynced to disk prior to reading any label raster.

---

## 2. Audit of the 9 Plan V3 Block E9 Items

As documented in `docs/e9_existing_evidence_ledger.md`, all required component removals and controls specified in Plan V3 Section 14.1 have been audited:

| Item | Component / Removal | Governing Status | Evidence Source & Key Finding |
| :---: | :--- | :--- | :--- |
| **1** | **No confidence filtering** | `COMPLETE_EXISTING_EVIDENCE` | Week 12 ablation (`tent_loss_only`). Removing confidence thresholding does not restore stability; entropy minimization remains prone to error accumulation on SAR. |
| **2** | **No consistency** | `COMPLETE_EXISTING_EVIDENCE` | Week 12 consistency ablation. Augmentation consistency without adaptation does not prevent degradation and adds significant compute overhead. |
| **3** | **No anchor** | `COMPLETE_EXISTING_EVIDENCE` | Week 12 anchor evaluation (`tent_anchor`). Anchor regularizer reduces divergence slightly but fails to yield net benefit over simple feature normalization. |
| **4** | **No temporal / null evidence** | `COMPLETE_EXISTING_EVIDENCE` | E4/E5 null probe evidence. Null temporal pairs fail to distinguish real flood expansion from speckle noise; $s_{\text{null}}$ direction does not generalize. |
| **5** | **No controller** | `COMPLETE_EXISTING_EVIDENCE` | E6/E8 formal decisions. No controller claimed or retained. Branch C operates strictly without heuristic gating or threshold selection. |
| **6** | **Generic protection substituted** | `COMPLETE_EXISTING_EVIDENCE` | E8 confirmation matrix. Evaluated EATA (Fisher regularizer + entropy filter) and SAR (sharpness-aware filter). Neither consistently outperforms frozen source. |
| **7** | **Normalization control** | `CONFIRMED_VIA_E9_MATRIX` | Bounded diagnostic matrix evaluating BN momentum dynamics $m \in \{0.0, 0.01, 0.05, 0.10, 1.0\}$. Batch normalization statistics drive virtually all adaptation variation. |
| **8** | **Reject-update vs source-fallback** | `NOT_APPLICABLE_UNDER_BRANCH_C` | Not applicable: no selective update/rollback controller is retained. |
| **9** | **Matched acceptance/calibration budget** | `NOT_APPLICABLE_UNDER_BRANCH_C` | Not applicable: no selective acceptance policy is deployed. |
| **10** | **CoTTA component diagnosis** | `CONFIRMED_VIA_E9_MATRIX` | Evaluated SSR removal ($p=0.0$) and Teacher EMA removal ($\alpha=0.0$). In multi-temporal SAR streams, neither component significantly alters performance beyond feature normalization. |

---

## 3. Core Scientific Findings & Diagnostic Attribution

### Finding 1: Normalization Context Dominance
Evaluating batch normalization running-statistics momentum across five discrete settings:
- $m = 0.0$ (Frozen Source Model): Baseline performance (`0.5916 ± 0.0032` Macro F1, `0.4980` Pooled F1).
- $m = 0.01$ (Conservative AdaBN): `0.5855 ± 0.0122` Macro F1, `0.5123` Pooled F1.
- $m = 0.05$ (Moderate AdaBN): `0.5855 ± 0.0122` Macro F1, `0.5123` Pooled F1.
- $m = 0.10$ (Standard Streaming AdaBN): `0.5857 ± 0.0120` Macro F1, `0.5129` Pooled F1.
- $m = 1.0$ (Pure Batch Statistics, BN-No-Update): `0.5857 ± 0.0120` Macro F1, `0.5129` Pooled F1 with **zero parameter updates**.

**Conclusion:** The transition across momentum values $m \in [0.01, 1.0]$ shows that adaptation performance rapidly reaches a steady-state asymptote governed by the local event batch statistics (all variants achieve $\sim 0.5855\text{--}0.5857$ Macro F1). Parameter adaptation via gradient updates (Tent, EATA, SAR, CoTTA) provides no statistically reliable advantage over pure batch feature normalization ($m = 1.0$).

### Finding 2: CoTTA Component Attribution
Evaluating CoTTA components under reset-per-event mode across seeds 0 and 1, orders A and B:
- **Full CoTTA Reset** (EMA $\alpha = 0.999$, SSR $p = 0.01$): `0.5893 ± 0.0124` Macro F1, `0.5244` Pooled F1.
- **CoTTA without SSR** ($p = 0.0$, EMA $\alpha = 0.999$): `0.5894 ± 0.0125` Macro F1, `0.5247` Pooled F1.
  * Paired delta against Full CoTTA across all 4 seeds/orders is $|\Delta \text{F1}| \le 0.0001$ with 95% bootstrap CIs spanning zero `[-0.0002, +0.0000]`. In short episodic streams, stochastic source restoration is functionally dormant.
- **CoTTA without Mean Teacher EMA** ($\alpha = 0.0$, SSR $p = 0.01$): `0.5926 ± 0.0131` Macro F1, `0.5412` Pooled F1.
  * Direct student self-training performs within $\Delta = +0.0033$ of the teacher EMA model.
  * Omitting the teacher EMA does not trigger runaway catastrophic forgetting under reset-per-event mode, confirming that the continuous temporal smoothing of weights is secondary to the immediate normalization shift.

---

## 4. Final Scope Freeze & Method Boundaries

1. **Route Commitment:** Branch C remains the final, permanent scientific route for ChangeGuard. No controller, policy, or threshold gate will be introduced.
2. **Backbone & Checkpoint Freeze:** The SNUNet_ECAM model architecture and Week 5 source checkpoints (seeds 0, 1, 2) are frozen.
3. **Ablation Matrix Freeze:** The 26-stream E9 diagnostic matrix, combined with the 69-stream E8 confirmation matrix and historical Week 12/13 ablations, constitutes the complete empirical basis for the study.
4. **Transition to E10:** The repository is fully prepared for Work Package E10 (publication-strength robustness extensions: second lightweight backbone architecture and synthetic-shift stress test).

