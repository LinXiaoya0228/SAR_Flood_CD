# Reproduction Guide: Event Splits and Label Semantics

This document details the event-level disjoint splits, cross-validation folds, discovery splits, and label definitions used across all ChangeGuard evaluations.

---

## 1. Label Semantics

All evaluations strictly follow binary change detection semantics:
- **Class 0 (No Flood / Unchanged):** Background land, permanent water bodies, or unflooded areas.
- **Class 1 (Flood / Changed):** Newly inundated areas between pre-event and post-event acquisitions.
- **Ignore / Invalid Mask (255):** Out-of-bounds, invalid sensor swath context, or nodata pixels. These pixels are masked out from all loss computations and evaluation confusion matrices.

In the external GEOID-Flood dataset, source labels denote Class 2 as Flood and Class 3 as Ignore; the included adapter `src/external_evaluation/geoid_adapter.py` maps these dynamically to Class 1 (Flood) and 255 (Ignore) to maintain semantic equivalence.

---

## 2. Event Partitioning & Independence

To prevent spatial data leakage, no geographic region or flood event is ever shared between training and evaluation splits.

### Confirmatory Folds (RST6, RST7)
The primary confirmatory matrix divides 18 source events into two equal, balanced 9-event folds:

- **Fold A (9 Evaluation Events):**
  - Evaluated on: `[event_01, event_03, event_05, event_07, event_09, event_11, event_13, event_15, event_17]`
  - Trained on: Remaining 18 events (Fold B + Group C).
- **Fold B (9 Evaluation Events):**
  - Evaluated on: `[event_02, event_04, event_06, event_08, event_10, event_12, event_14, event_16, event_18]`
  - Trained on: Remaining 18 events (Fold A + Group C).
- **Grand 18-Event Evaluation:** Concatenates Fold A and Fold B held-out evaluation sets, providing complete, disjoint coverage of all 18 events.

### Discovery Split (RST4)
- **Group C (9 Events):** Dedicated exploratory split used to screen candidate components (M1, M2, M3) without contaminating the confirmatory folds.

### All-27 Source Deployment (RST9)
- Combines all 27 source events (Fold A + Fold B + Group C) for training full-source checkpoints evaluated on external or historical target events.

### Historical Target Events (Week 1–14 Kuro Siwo Targets)
- 10 distinct, held-out test events used in historical literature (e.g., Event IDs: `101`, `102`, `104`, `202`, `205`, `561`, `641`, `1111002`, `1111006`, `1111013`).

### External GEOID-Flood Held-out Activations
- 9 European flood activations (Copernicus Emergency Management Service):
  - `EMSR857`, `EMSR858`, `EMSR861`, `EMSR864`, `EMSR865`, `EMSR866`, `EMSR868`, `EMSR870`, `EMSR871`
  - Grouped 8-event sensitivity merges `EMSR861` and `EMSR864` (both covering the October 2024 Valencia/Iberia flash flood event).

