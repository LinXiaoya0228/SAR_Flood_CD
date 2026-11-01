# ChangeGuard Work Package E10: Execution Contract and Governance Specification

**Contract Version:** `e10_execution_contract_v1`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E10 (Section 15)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `AUTHORIZED — FROZEN FOR EXECUTION`  

---

## 1. Intended Second Backbone
- **Selected Architecture:** `SiamUnet_diff` (FC-Siam-diff; Daudt et al., IEEE ICIP 2018).
- **Architectural Paradigm:** Fully Convolutional Siamese Difference Network with shared encoder weights and multi-scale difference skip connections.
- **Channels:** In-channels = 2 (dual-pol VV/VH SAR at pre-event and post-event); Out-channels = 3 (dense logits for `[No Water, Permanent Water, Flood]`).
- **Parameter Scale:** $\sim 1.35\text{M}$ parameters (lightweight, distinct from dense nested SNUNet-ECAM).
- **Target Precision:** FP32.

---

## 2. Training-Selection Data (Source-Inner Only)
- **Source-Inner Split:** `configs/splits/kuro_source_inner_split_v1.json`.
- **Inner-Train Corpus:** Exactly 22 events (19,178 samples).
- **Inner-Validation Corpus:** Exactly 5 events (`[267, 275, 497, 518, 1111005]`, 2,284 samples).
- **Selection Metric:** Event-Macro Flood F1 on inner-validation partition at Seed 0.
- **Tie-Breaking Rule:** (1) Higher Worst-Event Flood F1; (2) Higher Event-Macro Flood IoU; (3) Lower validation loss.
- **Candidate Recipe Screen:** Maximum 3 candidate budgets/recipes screened strictly at Seed 0.
- **Isolation Rule:** Zero access to the 6 TTA development events or 10 sealed-test events during recipe selection.

---

## 3. Fixed Source-Training Plan
- **Training Corpus:** All 27 source events (21,462 patches) from `configs/splits/kuro_source_all27_split_v1.json`.
- **Checkpoint Seeds:** 3 independent random initializations: Seeds 0, 1, 2.
- **Budget:** Fixed selected epoch budget $E^*$ derived from Phase 2 source-inner screening.
- **Loss Formulation:** ControlledSourceLoss (unweighted Cross-Entropy + Soft Dice, class 3 ignored).
- **Deterministic Recording:** SHA-256 hashes of code, split, and checkpoints recorded upon completion.

---

## 4. Development Evaluation Conditions
- **Events:** 6 development events (`[279, 437, 520, 559, 1111003, 1111008]`, 2,826 samples).
- **Evaluated Methods:**
  1. `source_only`: Frozen source model with frozen batch normalization running statistics.
  2. `bn_no_update`: Matched batch normalization statistics ($m=1.0$) with zero parameter updates.
  3. `tent`: Entropy minimization adapting affine BN parameters ($\text{lr}=10^{-4}$).
- **Seeds:** 3 source checkpoint seeds (Seeds 0, 1, 2).
- **Deployment Mode:** `reset_per_event` (primary operational condition).
- **Prediction Timing:** PRE_UPDATE primary predictions.
- **I/O Integrity:** Post-persistence deferred ground-truth disk read. Every prediction shard atomically written and fsynced before label access.

---

## 5. Synthetic-Shift Definition
- **Numerical Domain:** Investigated and verified from KuroSiwo preprocessing (linear amplitude/power vs dB, clamping at 0.15, standard normalization).
- **Shift Family:** Physically interpretable multiplicative radar backscatter calibration/attenuation shift (or additive dB equivalent) applied symmetrically to pre- and post-event observations to preserve ground-truth flood semantics.
- **Severities:** Clean ($s=0.0$) plus three ordered nonzero severities ($s \in \{0.05, 0.10, 0.20\}$).
- **Primary Stress Test:** Evaluated on SNUNet (Seeds 0, 1, 2) across `source_only`, `bn_no_update`, and `tent`.
- **Robustness Replication:** Evaluated on Second Backbone (`SiamUnet_diff`) under identical conditions.

---

## 6. Run Counts & Maximum Compute Budget
- **Phase 2 (Source-Inner Screen):** 3 candidate runs on 22 inner-train events $\times$ 1 seed $\le 1.5$ GPU-hours.
- **Phase 3 (Source All-27 Training):** 3 seeds on 27 events $\approx 2.5$ GPU-hours total.
- **Phase 4 (Second Backbone Dev Eval):** 3 seeds $\times$ 3 methods = 9 physical streams (54 event rows) $\approx 45$ GPU-minutes.
- **Phase 6 (Synthetic Shift Stress Test):** 
  - SNUNet: 3 seeds $\times$ 3 methods $\times$ 3 nonzero severities = 27 streams (clean reused from E8).
  - Second Backbone: 3 seeds $\times$ 3 methods $\times$ 4 severities = 36 streams.
  - Total stress test streams = 63 streams $\approx 3.5$ GPU-hours.
- **Total Compute Budget:** $\le 8$ GPU-hours across available RTX 3090 GPUs.

---

## 7. Prohibited Test Access & Quarantine Affirmation
- **Quarantined Events:** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.
- **Quarantine Guarantee:** Zero reads, queries, metadata extraction, or preprocessing on the 10 sealed test events. Fail-closed assertion present in all runners.

