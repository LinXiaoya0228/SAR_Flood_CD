# Protocol: Week 7A Controlled Source-Only Baseline Evaluation

**Protocol Version:** `week7_source_baseline_v1`
**Date:** 2026-10-16
**Status:** Frozen Protocol Specification
**Governing Checkpoints:** `outputs/week5_source_all27/source_snunet_vvvh_all27_seed{0,1,2}_v1_fp32/final_checkpoint.pt`
**Training Commit:** `6c63a5f8aa7ae2c608b145f0badc84b422b62ef1`

---

## 1. Executive Summary & Evaluation Boundary

This protocol establishes the rigorous, controlled evaluation methodology for the three frozen 16-epoch all-source trained Model A (`SNUNet_ECAM [VV, VH]`) models on the six designated TTA development events.

### 1.1 Scope and Inviolable Boundaries
1. **Source-Only Inference:**
   - Evaluated exclusively with `model.eval()` and under `torch.no_grad()`.
   - Zero backward passes, zero optimizer steps, zero learning-rate scheduling.
   - Zero running statistics updates to batch normalization layers (`track_running_stats` active with `training=False`, leaving running mean and variance untouched).
   - Cryptographic parameter and buffer hashing (`SHA-256`) before and after inference to verify zero in-memory weight/buffer mutation.
2. **Development Events Boundary (Target Set):**
   - Exactly six events: `[279, 437, 520, 559, 1111003, 1111008]`.
   - Total sample count: **2,826 patches** (manifest: `KuroV2_grid_dict_test_0_100.gz`).
   - Expected sample distribution:
     - Event 279: 2,381 patches
     - Event 437: 169 patches
     - Event 520: 92 patches
     - Event 559: 42 patches
     - Event 1111003: 19 patches
     - Event 1111008: 123 patches
3. **Data Access Clarification & Sealed Test Quarantine Policy:**
   - **Access Record Clarification:** In prior setup exploration, the shared indexing pickle `KuroV2_grid_dict_test_0_100.gz` was unpickled in Python to verify dictionary key structure and query sample counts per event. The transcript shows that dictionary keys were queried and a sample metadata record (`00f35c21...`, belonging to event 118) was printed. No sealed-test image rasters, ground-truth label rasters, or test-event predictions were ever opened, read, or generated.
   - **Distinction Between Shared Metadata and Image/GT Access:** We distinguish between shared-manifest indexing metadata (event IDs, patch keys, relative folder paths) and underlying raster/label data or performance evaluation. We do not claim "zero metadata access" to the manifest file, nor do we claim proven pixel/label leakage since no sealed rasters were opened.
   - **Clarified Boundary Policy (Not Retroactive Approval):** Moving forward, shared indexing metadata is used *strictly and exclusively* to extract the allowlisted development records for events `[279, 437, 520, 559, 1111003, 1111008]`. All ten sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain quarantined: their images, ground-truth rasters, predictions, and evaluation metrics remain completely excluded.
   - **Defense-in-Depth Filter:** The evaluation dataset loader (`KuroEvaluationDataset`) filters all records by event allowlist upon initialization *and* re-verifies the allowlist and quarantine in `__getitem__` before resolving any directory or opening any raster file on disk.
4. **Independent Three-Seed Evaluation:**
   - Models from Seeds 0, 1, and 2 are evaluated independently on the exact same 2,826 patches.
   - Predictions are **not** pooled across seeds into an ensemble.
   - Metric outputs produce 18 event-level rows (3 seeds × 6 events), followed by per-metric arithmetic mean $\pm$ sample standard deviation ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$, $ddof=1$) across the three seeds.

---

## 2. Historical Gate S Status & Proposed Gate S-Target Review

### 2.1 Historical Definitions of Gate S (Source-Inner Validation)

In historical project protocols, "Gate S" was defined for model selection and pipeline validation on the 5 *source-inner validation events* (`[267, 275, 497, 518, 1111005]` defined in `configs/splits/kuro_source_inner_split_v1.json`):

1. **`docs/source_training_protocol_v1.md` (lines 195–202):**
   > "Gate S Acceptance Criteria:
   > - Overall validation Macro F1 $\ge 0.50$ across the 5 validation events.
   > - Validation flood recall $\ge 0.45$.
   > - Dry false alarm rate $\le 2.0\%$ on dry validation tiles.
   > - Training loss converged; no gradient explosion or numerical instability.
   > - Model must not produce all-background or all-flood trivial predictions."

2. **`docs/MASTER_EXECUTION_PLAN.md` (lines 382–395):**
   > "Gate S: Source Model Readiness
   > - Verified checkpoint exists and is reproducible across $\ge 2$ seeds.
   > - 3-class output functional (background, permanent water, flood).
   > - Model does not collapse to trivial all-background prediction.
   > - Macro F1 on validation split reaches reasonable baseline level.
   > - Code verification: dataset loading handles invalid masks correctly, no silent data omission."

### 2.2 Historical Evidence Audit for Source-Inner Gate S
- **Macro F1 $\ge 0.50$:** Supported by Week 3 replication evidence (Model A 3-seed inner-validation event-macro mean: $0.793205 \pm 0.011438$; per-seed F1: Seed 0: $0.781741$, Seed 1: $0.793257$, Seed 2: $0.804616$, verified from `docs/week3_seed012_source_comparison.md`). The figures $0.5454 \pm 0.0025$ and event list `[349, 368, 415, 452, 497]` reported in an earlier draft were unsupported reporting errors. Unassessed in Week 6 because all 27 events were dedicated to training without an inner validation split.
- **Flood Recall $\ge 0.45$:** Supported by Week 3 replication evidence. The figure $0.5092 \pm 0.0166$ reported in an earlier draft was an unsupported reporting error. Unassessed in Week 6.
- **Dry False Alarm Rate $\le 2.0\%$ on dry validation tiles:** **Unassessed.** No standalone dry-tile false-alarm benchmark was implemented or logged in historical runs.
- **Training Convergence & Numerical Stability:** Supported across all 3 seeds in both Week 3 and Week 6 (completed all epochs with finite losses, zero NaNs, zero gradient explosions).
- **Non-Trivial Multi-Class Output:** Supported across all 3 seeds (produced balanced, functional segmentations across background, permanent water, and flood classes).
- **Former Inner-Validation Evaluation:** Final 16-epoch all27 models were not evaluated on their former inner-validation events or on target data during this setup phase.

### 2.3 Finalized Evaluation Rules (Decisions Finalized Prior to First Week 7 Real-Data Evaluation)

These rules are recorded as explicit project decisions finalized before launching any real-data evaluation on target events, rather than as historical pre-registrations. We strictly separate **Technical Evaluation Validity** from **Target-Domain Performance Assessment**:

#### A. Technical Evaluation Validity (Binary Pass / Fail)
1. **Completeness & Coverage:** Exactly 2,826 of 2,826 target samples evaluated. Zero missing patches, zero duplicate sample IDs. Per-event sample counts match ground truth exactly (`279`: 2381, `437`: 169, `520`: 92, `559`: 42, `1111003`: 19, `1111008`: 123). Every sample ID and its valid-pixel count is recorded in `sample_valid_pixel_counts`.
2. **Double-Masking Rule:** Valid pixels defined strictly by $\text{valid} = (\text{valid\_mask} == 1) \land (\text{label} \ne 3)$.
3. **Numerical Validity:** Logits and probability maps contain zero `NaN` or `Inf` values across all 2,826 samples.
4. **Model State Immutability:** Pre-inference and post-inference parameter and buffer `SHA-256` hashes match bit-exactly across all three seeds.
5. **Protected-Event Exclusion:** Zero access, loading, or inference on the 10 sealed test events.
6. **Complete Six-Event Evaluability Requirement:**
   - All six events must be technically evaluable (`valid_pixels > 0`) for each seed before declaring a complete six-event baseline.
   - If any event has zero valid pixels across all its patches:
     - Sample-level and completeness records for that event are fully retained;
     - Overall technical validity is marked as **failed** (`technical_validity_passed = False`);
     - The report must state the exact count of evaluable events (e.g. `evaluable_events_count = 5`);
     - Any macro average computed over the remaining evaluable events is strictly labeled as **diagnostic only** (`macro_type = "reduced_event_diagnostic_only"`), not as a full six-event benchmark.

#### B. Target-Domain Performance Assessment (Diagnostic Reporting)
1. **Trivial Prediction Detection & Reporting:**
   - Detect and report at whole-run and per-event levels:
     - All-background predictions (events where `pred_flood_pixels == 0` and `pred_permanent_water_pixels == 0`).
     - All-permanent-water predictions (events where `pred_background_pixels == 0` and `pred_flood_pixels == 0`).
     - All-flood predictions (events where `pred_background_pixels == 0` and `pred_permanent_water_pixels == 0`).
   - Explicitly detect and report any event where zero flood is predicted despite ground-truth flood being present (`gt_flood_pixels > 0` but `pred_flood_pixels == 0`).
   - *Note on Granularity:* We do **not** require individual $224 \times 224$ patches to contain every predicted class, as many physical patches are naturally purely dry land or purely open water.
2. **No Retroactive Numerical Target Threshold:**
   - Target development events represent heterogeneous, out-of-distribution geographical domains (e.g., Event 559 has severe domain shift and historically low baseline $F_1 \approx 0.08$).
   - We do not impose an artificial numerical threshold (e.g. $F_1 \ge 0.50$) on target domains.
   - **Crucial Methodological Distinction:** Technical evaluation validity alone proves that the measurement was executed soundly, deterministically, and without data leakage; it does **not** prove that model performance on target domains is adequate. The unadapted baseline metrics establish the reference level against which subsequent test-time adaptation methods will be judged.
   - **Historical Gate S Status:** We do not claim historical source-inner Gate S is fully satisfied; the dry-tile false-alarm rate criterion remains unassessed.

---

## 3. Data Processing & Double-Masking Specification

### 3.1 Input Channels and Preprocessing
- **Sensory Modality:** 2-channel Sentinel-1 SAR amplitude: `[VV, VH]`.
- **Temporal Pairing:** Pre-event 2 (`SL2`) and Post-event (`MS1`).
- **Input Tensors:**
  - $x_A \in \mathbb{R}^{B \times 2 \times H \times W}$: Pre-event SAR (`SL2_IVV`, `SL2_IVH`) with native patch size $H \times W = 224 \times 224$.
  - $x_B \in \mathbb{R}^{B \times 2 \times H \times W}$: Post-event SAR (`MS1_IVV`, `MS1_IVH`) with native patch size $H \times W = 224 \times 224$.
- **Normalization (Frozen constants):**
  - $\mu = [0.0953, 0.0264]$
  - $\sigma = [0.0427, 0.0215]$
  - Input is clamped to $[0.0, 0.15]$ prior to standardization:
    $$\tilde{x} = \frac{\text{clamp}(x, 0.0, 0.15) - \mu}{\sigma}$$

### 3.2 Label Definition & Double-Masking Rule
- Class 0: Background / Non-water
- Class 1: Permanent Water
- Class 2: Flood / Newly flooded water (Target class)
- Class 3: Ignored / Unlabeled / Invalid
- Auxiliary Mask: `valid_mask` (0 = invalid sensor coverage, 1 = valid sensor coverage).

**Double-Masking Rule:**
A pixel $(u, v)$ is included in confusion matrix accumulation if and only if:
$$\text{valid}(u, v) = (\text{valid\_mask}(u, v) == 1) \land (\text{label}(u, v) \ne 3)$$
All pixels where $\text{valid}(u, v) == \text{False}$ are excluded from TP, FP, FN, TN, and all derived metrics.

---

## 4. Mathematical Definitions of Metrics

For each event $e \in \{279, 437, 520, 559, 1111003, 1111008\}$, let $\mathcal{V}_e$ denote the set of all valid pixels across all samples in event $e$.

### 4.1 Confusion Counts (Target Class 2)
$$\text{TP}_e = \sum_{p \in \mathcal{V}_e} \mathbb{I}(\hat{y}_p = 2 \land y_p = 2)$$
$$\text{FP}_e = \sum_{p \in \mathcal{V}_e} \mathbb{I}(\hat{y}_p = 2 \land y_p \ne 2)$$
$$\text{FN}_e = \sum_{p \in \mathcal{V}_e} \mathbb{I}(\hat{y}_p \ne 2 \land y_p = 2)$$
$$\text{TN}_e = \sum_{p \in \mathcal{V}_e} \mathbb{I}(\hat{y}_p \ne 2 \land y_p \ne 2)$$

### 4.2 Derived Performance Metrics & Zero-Valid-Pixel Rules
$$\text{Precision}_e = \frac{\text{TP}_e}{\text{TP}_e + \text{FP}_e + \epsilon}$$
$$\text{Recall}_e = \frac{\text{TP}_e}{\text{TP}_e + \text{FN}_e + \epsilon}$$
$$\text{F1}_e = \frac{2 \cdot \text{Precision}_e \cdot \text{Recall}_e}{\text{Precision}_e + \text{Recall}_e + \epsilon} = \frac{2 \cdot \text{TP}_e}{2 \cdot \text{TP}_e + \text{FP}_e + \text{FN}_e + \epsilon}$$
$$\text{IoU}_e = \frac{\text{TP}_e}{\text{TP}_e + \text{FP}_e + \text{FN}_e + \epsilon}$$
where $\epsilon = 10^{-7}$ prevents division by zero when positive pixels exist.

**Edge Cases & Zero-Valid-Pixel Handling:**
1. **Zero-Valid-Pixel Events:** If an event has zero valid pixels ($\mathcal{V}_e = \emptyset$, $\text{valid\_pixels} == 0$):
   - Flagged as `is_evaluable = False` and `evaluation_status = "technically_unevaluable_zero_valid_pixels"`.
   - Metrics are recorded as `None` (empty in CSV).
   - The event does **not** receive an artificial perfect score ($F_1 = 1.0$), and is excluded from equal-event macro averages to prevent skewing results.
2. **Genuine Flood-Free Events:** If an event has valid pixels ($\mathcal{V}_e \ne \emptyset$, $\text{valid\_pixels} > 0$), and both ground-truth flood and predicted flood are zero ($\text{TP}_e = 0, \text{FP}_e = 0, \text{FN}_e = 0, \text{TN}_e > 0$):
   - Flagged as `is_evaluable = True` and `evaluation_status = "evaluable_flood_free_perfect_negative"`.
   - Metrics are $1.0$ (perfect true-negative agreement).
3. **Completeness & Sample Tracking:**
   - Every evaluated sample ID and its valid-pixel count is recorded in `sample_valid_pixel_counts` in the summary output.
   - Invalid-only samples contribute zero confusion counts but are never silently dropped from completeness audits.

### 4.3 Aggregation Schemes
1. **Equal-Event Macro Average:**
   $$\overline{\text{Metric}}_{\text{macro}} = \frac{1}{|E|} \sum_{e \in E} \text{Metric}_e \quad (|E| = 6)$$
2. **Pooled (Micro) Average:**
   $$\text{TP}_{\text{pool}} = \sum_{e \in E} \text{TP}_e, \quad \text{FP}_{\text{pool}} = \sum_{e \in E} \text{FP}_e, \quad \text{FN}_{\text{pool}} = \sum_{e \in E} \text{FN}_e$$
   $$\text{F1}_{\text{pool}} = \frac{2 \cdot \text{TP}_{\text{pool}}}{2 \cdot \text{TP}_{\text{pool}} + \text{FP}_{\text{pool}} + \text{FN}_{\text{pool}} + \epsilon}$$
3. **Three-Seed Mean and Sample Standard Deviation:**
   For any metric $M$ computed on seed $s \in \{0, 1, 2\}$:
   $$\bar{M} = \frac{1}{3} \sum_{s=0}^2 M_s$$
   $$s_M = \sqrt{\frac{1}{3-1} \sum_{s=0}^2 (M_s - \bar{M})^2} \quad (ddof = 1)$$
   Reported in summary tables as: $\bar{M} \pm s_M$.

---

## 5. Deterministic Error Visualization Protocol

To enable rigorous, consistent visual inspection across model seeds (and subsequent TTA methods):
1. **Sample Selection:** Exactly 2 patches per event are selected deterministically by sorting all `sample_id` values alphanumerically and picking:
   - Index $0$ (the first patch).
   - Index $\lfloor N_e / 2 \rfloor$ (the median patch).
2. **Total Visualized Patches:** 6 events × 2 patches = 12 patches.
3. **Identical Sample Set Across Seeds:** The identical 12 patches are visualized for Seed 0, Seed 1, and Seed 2.
4. **Color Coding Convention:**
   - Pre-event SAR VV / Post-event SAR VV: Grayscale normalized display $[0, 0.15]$.
   - Ground Truth: Background (Black/Dark Gray), Permanent Water (Blue), Flood (Green), Invalid/Ignored (White).
   - Prediction Error Map:
     - True Positive (TP): Green (`#00FF00`)
     - False Positive (FP): Red (`#FF0000`)
     - False Negative (FN): Blue / Yellow (`#0088FF`)
     - True Negative (TN): Dark Gray (`#222222`)
     - Invalid / Ignored: Light Gray / White (`#DDDDDD`)
5. **Storage Path:** Saved under `outputs/week7_source_baseline/visualizations/seed{0,1,2}/`.

---

## 6. Artifact Integrity & Output Schema

The evaluation run writes the following standardized artifacts into `outputs/week7_source_baseline/`:
1. `source_baseline_seed{0,1,2}_event_metrics.csv`: 6 event rows per seed containing all confusion counts and metrics.
2. `source_baseline_seed{0,1,2}_summary.json`: Detailed machine-readable summary including execution time, macro/pooled metrics, worst-event metrics, and pre/post parameter hashes.
3. `source_baseline_three_seed_summary.json`: Aggregated 3-seed mean and sample standard deviation across all events and summary metrics.
4. `source_baseline_evaluation_manifest.json`: Cryptographic manifest containing config hashes, code Git commit, environment details, and output file hashes.
