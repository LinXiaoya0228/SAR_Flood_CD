# RST0 Route Transition Record: Robust Source Training

**Document version:** 1.0.0
**Transition date:** 2026-10-26
**Status:** Approved for RST0 route transition and historical freeze
**Repository root:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Current Git branch:** `main`
**Transition Git HEAD:** `904f3559513638c97cc28ea62c32be75b8a3f300`
**Authoritative proposal:** `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`

---

## 1. Executive Summary and Scope Transition

This document formally records the transition of the primary ChangeGuard research route from test-time adaptation (TTA) to **Event-Balanced Robust Source Training for Paired SAR Flood Change Detection**.

The historical research phases E0 through E11 conducted a controlled, reproducible empirical study of test-time adaptation for multi-temporal synthetic aperture radar (SAR) flood mapping. Work Package E11 completed the frozen sealed-test evaluation, finalizing the Branch C investigation.

All E0–E11 code, configurations, execution ledgers, outputs, and scientific conclusions remain immutable historical assets. The new route is additive, strictly versioned under a new namespace, and leaves the historical evidence intact.

---

## 2. Status of Completed Historical Work (E0–E11)

Work Packages E0 through E11 constitute completed historical work. The empirical findings of the controlled TTA study are:

1. **Inconsistent adaptation across events and backbones:** Standard unsupervised test-time adaptation methods (including batch normalization statistic adaptation, entropy minimization, and continual teacher-student adaptation) produced marginal shifts whose 95% paired event-cluster bootstrap confidence intervals contain zero on the primary SNUNet-ECAM backbone. On the lightweight Siamese difference backbone (`SiamUnet_diff`), adaptation induced severe performance drops (-5.69 points in macro F1).
2. **Dominant role of normalization context:** Spatial aggregation of batch statistics across small batch sizes under extreme flood class imbalance corrupts batch normalization statistics during inference, explaining a substantial portion of apparent adaptation degradation.
3. **Unreliability of unsupervised risk signals:** Prediction entropy, maximum confidence, and feature drift did not provide a dependable basis for selective adaptation controllers across stream orderings and random seeds.
4. **Sensor drift sensitivity:** Unsupervised adaptation doubled vulnerability to synthetic radiometric gain drift compared to frozen source checkpoints.

The new robust source-training route **does not invalidate** the controlled TTA study. The empirical characterization of TTA instability and normalization dynamics remains a valid, rigorous, and valuable scientific contribution of this Final Year Project (FYP).

---

## 3. Rationale for Robust Source Training

In operational disaster response, remote sensing models are deployed across geographically and seasonally disparate flood events without target-domain ground truth. Because test-time parameter adaptation introduces risks of catastrophic minority class collapse and architectural instability, the project shifts the intervention upstream: **from test-time adaptation to supervised source training**.

The core premise is to determine whether source training itself can be redesigned to produce models that generalize more reliably across unseen flood events, without requiring online parameter updates or target-domain inference-time modification.

### 3.1 Primary Research Question

> Can event-balanced supervised training, a flood-aware loss, and physically justified paired SAR augmentation improve cross-event flood change detection without using target-event data or test-time model updates?

### 3.2 Intended Contribution

The intended contribution is a reproducible, target-free source-training recipe for paired SAR flood change detection that demonstrates:
- improved average cross-event flood F1 under event-disjoint evaluation;
- improved worst-event reliability or bottom-event average performance; or
- reduced variance across source-training random seeds.

If only a subset of the proposed components (e.g., event-balanced sampling alone) delivers repeatable gains, the simpler recipe will be adopted as the final contribution.

### 3.3 Target-Free Deployment Boundary

All proposed interventions operate strictly during source-model training. Deployment remains completely **target-free**:
- No test-time gradient updates.
- No test-time batch normalization statistic recalculation from target batches.
- No test-time unsupervised loss minimization.
- No reliance on target-domain labels or unlabeled adaptation streams.

### 3.4 No Presumed Positive Improvement

No positive improvement is assumed prior to experimental execution. The research questions are empirical hypotheses. If the proposed recipe fails to satisfy predefined promotion criteria (P1, P2, or P3 in the proposal), that negative outcome will be preserved and reported with full scientific transparency.

---

## 4. Relationship to Historical Branch C Results

The historical Branch C findings and the new robust source-training route form a coherent, unified thesis:
1. Part I establishes that generic unsupervised test-time adaptation is not reliably beneficial across events and architectures in paired SAR flood change detection.
2. Part II investigates whether supervised training improvements—specifically event balancing, flood-aware loss formulation, and paired SAR augmentation—provide a simpler, safer, and more stable alternative.

The thesis does not claim that adaptation is universally flawed, nor that robust source training is guaranteed to eliminate cross-event domain shift. Both parts represent controlled scientific inquiries under rigorous evaluation protocols.

---

## 5. Previously Observed Event Limitation

The following sixteen Kuro Siwo flood events were accessed and evaluated during historical phases E0–E11:

### 5.1 Historical Development Events (6 events)
```text
[279, 437, 520, 559, 1111003, 1111008]
```

### 5.2 Historical Former Test Events (10 events)
```text
[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
```

### 5.3 Mandatory Integrity Constraints
1. **Status:** All 16 events are permanently designated as `PREVIOUSLY_OBSERVED`.
2. **Prohibition:** None of these 16 events may be used for method development, component selection, loss weighting, augmentation design, hyperparameter tuning, epoch budget selection, or checkpoint promotion during RST0 through RST8.
3. **No Access in RST0–RST8:** Their image rasters, masks, and ground-truth labels are strictly prohibited from being read, loaded, or inspected during RST0–RST8.
4. **Conditional Future Boundary (RST9):** These 16 events may only be evaluated as an optional, secondary cross-domain extension under RST9 after the source-training recipe is frozen and confirmed on source data. In all reports, they must be explicitly disclosed as previously observed, non-pristine evaluation domains.

---

## 6. Source Event Population for New Method Development

The 27 labeled source events from the authoritative frozen split (`configs/splits/kuro_source_all27_split_v1.json`) constitute the sole authorized population for new method development and confirmation:
```text
[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]
```
These 27 events will be partitioned into three disjoint groups (A, B, and C) during RST1. No grouping or fold assignment is performed in RST0.

---

## 7. RST0 Completion Requirements and Boundary

RST0 is restricted to route transition, historical protection, namespace definition, and data boundary freezing:
- Historical files and directories are audited and protected.
- Machine-readable contracts are established.
- No raster or label files were accessed.
- No training, evaluation, smoke, or test runs were launched.

**RST1 has not started.** Execution will proceed to RST1 only upon explicit instruction and full verification of the RST0 milestone.
