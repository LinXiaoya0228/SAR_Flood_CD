# Week 7A Prelaunch Integrity Review: Access Records, Gate S Status, and Code Verification

**Document Version:** 1.0.0
**Date:** 2026-10-16
**Status:** Prelaunch Correction Completed — Awaiting Review Before Evaluation
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Checkpoints:** `outputs/week5_source_all27/source_snunet_vvvh_all27_seed{0,1,2}_v1_fp32/final_checkpoint.pt`
**Training Commit:** `6c63a5f8aa7ae2c608b145f0badc84b422b62ef1`

---

## 1. Access Record Clarification & Quarantine Boundary Policy

### 1.1 Factual Audit of Prior Exploration in Transcript
A review of the interaction transcript reveals the exact nature and scope of data access during prior configuration exploration:

1. **Manifest File Loaded:** The shared dictionary `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` was unpickled in Python to verify dictionary keys and inspect record structures.
2. **Metadata Inspected:**
   - The total number of keys in the test dictionary was checked (`67,490` keys).
   - In step 8687, the code attempted to look up integer event IDs directly via `d.get(e, [])`, returning 0 because the dictionary is keyed by patch UUID strings rather than event IDs.
   - In step 8689, the first dictionary key (`00f35c214e7b5cb493e0a9038ac74c0c`) and its associated metadata structure were printed. This printed record belongs to Event 118 (`actid: 118`, an arid event not in the development set and not in the sealed test set), displaying metadata such as file paths, dates, coverage, and bounding polygon.
   - In step 8691, the dictionary was scanned to aggregate patch counts for the six target development events, verifying 2,826 total patches (`279`: 2381, `437`: 169, `520`: 92, `559`: 42, `1111003`: 19, `1111008`: 123).
3. **Absence of Image and Ground-Truth Raster Access:**
   - At no point was any image raster file (`SL1_*.tif`, `SL2_*.tif`, `MS1_*.tif`) or ground-truth label file (`MK0_MLU_*.tif`, `MK0_MNA_*.tif`) opened or read for any sealed-test event.
   - No inference was performed on sealed test events, and no predictions or evaluation metrics were generated.
4. **Distinction Between Shared Indexing Metadata and Raster/Label Access:**
   - We maintain a strict boundary between shared-manifest metadata (patch IDs, event IDs, relative folder strings) and actual raster images, ground-truth masks, and performance metrics.
   - We do not claim "zero metadata access" to the manifest file, nor do we assert proven pixel or label leakage, as no sealed rasters were read or processed.

### 1.2 Clarified Future Boundary Policy (Not Retroactive Approval)
To prevent accidental data leakage or ambiguity going forward, the following policy is explicitly adopted:
- The shared manifest `KuroV2_grid_dict_test_0_100.gz` is used solely to identify and index the 2,826 records belonging to the six allowlisted development events (`[279, 437, 520, 559, 1111003, 1111008]`).
- The ten sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain permanently quarantined: their images, ground-truth labels, model predictions, and performance metrics are strictly excluded from all code execution.
- **Defense-in-Depth Loader Protection:** `src/data/evaluation_dataset.py` enforces filtering upon initial loading and includes a secondary assertion inside `__getitem__` immediately preceding directory resolution and file access. Any attempt to access a sealed test event halts execution with a `RuntimeError`.

---

## 2. Historical Gate S Status & Proposed Gate S-Target Framework

### 2.1 Unchanged Historical Source-Inner Gate S
The historical Gate S criteria defined in `docs/source_training_protocol_v1.md` and `docs/MASTER_EXECUTION_PLAN.md` govern model selection on the five *source-inner validation events* (`[267, 275, 497, 518, 1111005]` defined in `configs/splits/kuro_source_inner_split_v1.json`). Their empirical status is summarized as follows:

| Historical Gate S Criterion | Target Threshold | Empirical Status | Supporting Evidence |
| :--- | :--- | :--- | :--- |
| **Validation Macro F1** | $\ge 0.50$ | **Supported** (Week 3) / Unassessed (Week 6) | Model A Week 3 3-seed event-macro mean: $0.793205 \pm 0.011438$ (Seed 0: $0.781741$, Seed 1: $0.793257$, Seed 2: $0.804616$; verified from `docs/week3_seed012_source_comparison.md`). The previously reported figures ($0.5454 \pm 0.0025$, recall $0.5092 \pm 0.0166$, event list `[349, 368, 415, 452, 497]`) were unsupported reporting errors. (In Week 6, all 27 events were dedicated to training; no inner validation split was evaluated.) |
| **Validation Flood Recall** | $\ge 0.45$ | **Supported** (Week 3) / Unassessed (Week 6) | Supported in Week 3 replication. Unassessed in Week 6. |
| **Dry False Alarm Rate** | $\le 2.0\%$ on dry tiles | **Unassessed** | No standalone dry-tile false-alarm benchmark was implemented or logged in historical runs. |
| **Training Convergence** | Stable loss, zero explosion | **Supported** | All 3 seeds completed 16 epochs with finite losses, zero NaNs, and healthy gradient histories. |
| **Non-Degenerate Predictions** | Functional 3-class output | **Supported** | All 3 seeds produced valid non-trivial segmentation distributions across background, permanent water, and flood. |

*Note on Historical Models:* Final 16-epoch all27 models were not evaluated on their former inner-validation events or on target data during this setup phase.

### 2.2 Finalized Evaluation Rules (Decisions Finalized Prior to First Week 7 Real-Data Evaluation)
These rules are recorded as explicit project decisions finalized before launching any real-data evaluation on target events, rather than as historical pre-registrations. We strictly separate **Technical Evaluation Validity** from **Target-Domain Performance Assessment**:

#### A. Technical Evaluation Validity (Binary Pass / Fail)
1. **Sample Coverage & Integrity:** Exactly 2,826 / 2,826 development samples evaluated with zero omissions, zero duplicate sample IDs, and exact per-event counts matching ground truth. Every sample ID and its valid-pixel count is recorded in `sample_valid_pixel_counts`.
2. **Double-Masking Adherence:** Validity strictly enforced via $\text{valid} = (\text{valid\_mask} == 1) \land (\text{label} \ne 3)$.
3. **Numerical Integrity:** Zero `NaN` or `Inf` values across all batch logits and outputs.
4. **State Immutability:** Pre-inference and post-inference parameter and buffer SHA-256 hashes match bit-exactly across all three seeds.
5. **Quarantine Isolation:** Zero access or exposure to the ten sealed test events.
6. **Complete Six-Event Evaluability Requirement:**
   - All six events must be technically evaluable (`valid_pixels > 0`) for each seed before declaring a complete six-event baseline.
   - If any event has zero valid pixels across all its patches:
     - Sample-level and completeness records for that event are fully retained;
     - Overall technical validity is marked as **failed** (`technical_validity_passed = False`);
     - The report must state the exact count of evaluable events (e.g. `evaluable_events_count = 5`);
     - Any macro average computed over the remaining evaluable events is strictly labeled as **diagnostic only** (`macro_type = "reduced_event_diagnostic_only"`), not as a full six-event benchmark.

#### B. Target Performance Assessment & Collapse Reporting (Diagnostic)
1. **Trivial Prediction Detection:**
   - Whole-run and per-event tracking of:
     - All-background predictions (`pred_flood_pixels == 0` and `pred_permanent_water_pixels == 0`).
     - All-permanent-water predictions (`pred_background_pixels == 0` and `pred_flood_pixels == 0`).
     - All-flood predictions (`pred_background_pixels == 0` and `pred_permanent_water_pixels == 0`).
     - Zero flood predictions on flood-positive events (`gt_flood_pixels > 0` but `pred_flood_pixels == 0`).
   - *Spatial Granularity Note:* Individual $224 \times 224$ patches are not required to contain every class, as natural scenes are frequently purely dry land or purely water.
2. **No Retroactive Numerical Target Threshold:**
   - Target development events exhibit significant geographical and environmental heterogeneity (e.g. Event 559 has severe domain shift and historically low baseline $F_1 \approx 0.08$).
   - We do not retroactively impose an arbitrary numerical F1 threshold (such as $F_1 \ge 0.50$).
   - **Methodological Distinction:** Technical evaluation validity confirms that the measurement was executed soundly, deterministically, and without data leakage; it does **not** prove that model performance is adequate. Unadapted baseline scores serve as reference points for measuring adaptation delta.
   - **Historical Gate S Status:** We do not claim historical source-inner Gate S is fully satisfied; the dry-tile false-alarm rate criterion remains unassessed.

---

## 3. Inspection of Shared-Code Modifications

### 3.1 `src/evaluation/metrics.py`

#### Concise Git Diff
```diff
--- a/src/evaluation/metrics.py
+++ b/src/evaluation/metrics.py
@@ -100,6 +100,48 @@ def compute_flood_metrics(counts: Dict[str, int]) -> Dict[str, float]:
     }


+def compute_macro_metrics(event_rows: List[Dict[str, Any]]) -> Dict[str, float]:
+    """Computes unweighted equal-event macro metrics across event rows."""
+    if not event_rows:
+        return {}
+    precisions = [r["precision"] for r in event_rows]
+    recalls = [r["recall"] for r in event_rows]
+    f1s = [r["f1"] for r in event_rows]
+    ious = [r["iou"] for r in event_rows]
+    return {
+        "macro_precision": float(np.mean(precisions)),
+        "macro_recall": float(np.mean(recalls)),
+        "macro_f1": float(np.mean(f1s)),
+        "macro_iou": float(np.mean(ious)),
+        "median_precision": float(np.median(precisions)),
+        "median_recall": float(np.median(recalls)),
+        "median_f1": float(np.median(f1s)),
+        "median_iou": float(np.median(ious)),
+    }
+
+
+def compute_pooled_metrics(event_rows: List[Dict[str, Any]]) -> Dict[str, float]:
+    """Computes pooled confusion metrics across all event rows."""
+    if not event_rows:
+        return {}
+    pooled_tp = sum(r["tp"] for r in event_rows)
+    pooled_fp = sum(r["fp"] for r in event_rows)
+    pooled_fn = sum(r["fn"] for r in event_rows)
+    pooled_tn = sum(r["tn"] for r in event_rows)
+    pooled_counts = {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn, "tn": pooled_tn}
+    m = compute_flood_metrics(pooled_counts)
+    return {
+        "pooled_tp": pooled_tp,
+        "pooled_fp": pooled_fp,
+        "pooled_fn": pooled_fn,
+        "pooled_tn": pooled_tn,
+        "pooled_precision": m["precision"],
+        "pooled_recall": m["recall"],
+        "pooled_f1": m["f1"],
+        "pooled_iou": m["iou"],
+    }
```

#### Rationale & Verification of Conventions
- **Rationale:** Exposes pure, standalone functions for calculating equal-event macro averages and pooled confusion metrics directly from a list of event dictionaries, eliminating code duplication between `EventMetricAccumulator` and standalone evaluation pipelines.
- **Convention Verification:**
  - Zero-denominator handling preserves existing KuroSiwo conventions documented in `metrics.py`:
    - If `tp + fp == 0` and `tp + fn == 0` (empty flood in prediction and ground truth): Precision = 1.0, Recall = 1.0, F1 = 1.0, IoU = 1.0.
    - If `tp + fp == 0` and `tp + fn > 0` (complete missed detection): Precision = 0.0, Recall = 0.0, F1 = 0.0, IoU = 0.0.
    - If `tp + fp > 0` and `tp + fn == 0` (pure false alarm): Precision = 0.0, Recall = 0.0, F1 = 0.0, IoU = 0.0.
    - If valid pixels = 0 (invalid-only input where `valid_mask == 0` or all `label == 3`): `tp=0, fp=0, fn=0, tn=0` produces Precision = None, Recall = None, F1 = None, IoU = None. The event is flagged as technically unevaluable (`technically_unevaluable_zero_valid_pixels`) and excluded from equal-event macro averages to prevent artificial inflation.
    - If valid pixels > 0 and `tp=0, fp=0, fn=0, tn > 0` (genuine flood-free input): produces Precision = 1.0, Recall = 1.0, F1 = 1.0, IoU = 1.0 (perfect negative agreement).
  - Completeness audits record every sample ID and valid-pixel count in `sample_valid_pixel_counts`.

---

### 3.2 `src/models/snunet_wrapper.py`

#### Concise Git Diff
```diff
--- a/src/models/snunet_wrapper.py
+++ b/src/models/snunet_wrapper.py
@@ -66,7 +66,10 @@ class SNUNetWrapper(nn.Module):
         """Strictly loads the official checkpoint into SNUNetWrapper."""
         wrapper = cls(in_channels=in_channels, out_ch=out_ch, base_channel=base_channel)

-        ckpt = torch.load(checkpoint_path, map_location=device)
+        try:
+            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
+        except TypeError:
+            ckpt = torch.load(checkpoint_path, map_location=device)
         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
             state_dict = ckpt["model_state_dict"]
             metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
@@ -74,7 +77,12 @@ class SNUNetWrapper(nn.Module):
             state_dict = ckpt
             metadata = {}

-        wrapper.model.load_state_dict(state_dict, strict=True)
+        # Handle whether state_dict was saved from SNUNetWrapper (prefixed with 'model.')
+        # or directly from SNUNet_ECAM (unprefixed)
+        if any(k.startswith("model.") for k in state_dict.keys()):
+            wrapper.load_state_dict(state_dict, strict=True)
+        else:
+            wrapper.model.load_state_dict(state_dict, strict=True)
         wrapper.to(device)
         wrapper.freeze()
         return wrapper, metadata
```

#### Rationale & Compatibility Verification
- **Rationale:**
  1. PyTorch 2.6 changed the default of `torch.load` to `weights_only=True`. Checkpoints generated during training contain Python and NumPy metadata, which triggers `_pickle.UnpicklingError` unless loaded with `weights_only=False`.
  2. Training checkpoints saved with `train_source_model.py` saved `SNUNetWrapper` directly, prefixing parameter keys with `model.` (e.g. `model.conv0_0.conv1.weight`), whereas official checkpoints saved from raw `SNUNet_ECAM` lacked this prefix. The wrapper now transparently and strictly handles both key formats.
- **Strict Compatibility & Equivalence Verification:**
  - Evaluated on all three Week 6 Epoch 16 checkpoints (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed{0,1,2}_v1_fp32/final_checkpoint.pt`).
  - Compared direct manual loading:
    `wrapper.load_state_dict(torch.load(..., weights_only=False)['model_state_dict'], strict=True)`
    against `SNUNetWrapper.load_from_checkpoint(...)`.
  - Result: All 236 parameter tensors and buffer tensors matched bit-exactly across all three seeds.
  - Forward output verification on synthetic inputs confirmed bit-exact tensor equality between direct loading and `load_from_checkpoint`.

---

## 4. Test Suite Execution & Verification Evidence

In this turn, the full regression test suite was executed against the current codebase:
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest discover tests
```
Output:
```
Ran 140 tests in 54.742s

OK
```

*(Note on Test Counting & Regression Evidence: The entire test suite comprises 140 tests: 126 pre-existing tests across the repository plus 14 newly added evaluation unit tests in `tests/test_source_evaluation.py`. All 140 tests passed cleanly with exit code 0. We explicitly correct the prior draft statement that an earlier 133-test run occurred during Week 5B or proved the current code unregressed; this current 140-test execution is the empirical proof of complete regression-free health following the shared metrics modifications.)*

### Coverage of Tested Behaviors (14 Evaluation Tests)
1. `test_double_masking_rule`: Verifies $(\text{valid\_mask} == 1) \land (\text{label} \ne 3)$.
2. `test_known_confusion_and_metrics_math`: Verifies Precision, Recall, F1, IoU against hand-computed values.
3. `test_macro_vs_pooled_metrics_math`: Verifies distinct calculations for equal-event macro and pooled metrics.
4. `test_zero_denominators_and_invalid_only`: Tests edge cases for empty GT, empty prediction, empty union, and zero valid pixels (verifies metrics return `None`).
5. `test_collapse_audit_reporting`: Verifies detection of all-background, all-permanent-water, all-flood, and zero-flood-on-flood-positive events.
6. `test_sealed_test_events_quarantine_enforcement`: Tests immediate `RuntimeError` when sealed events are passed to dataset initialization.
7. `test_dataset_defense_in_depth_quarantine_rejection`: Tests that `__getitem__` halts before opening rasters if an unallowed or sealed event is encountered.
8. `test_mock_dataset_and_evaluator_execution`: Verifies end-to-end evaluation flow with synthetic tensors.
9. `test_model_immutability_verification`: Verifies detection and immediate halting if model parameters are mutated during evaluation.
10. `test_snunet_wrapper_loading_equivalence_synthetic`: Confirms bit-exact state and output equality between direct manual loading and `load_from_checkpoint`.
11. `test_three_seed_aggregation_math`: Verifies arithmetic mean and sample standard deviation ($ddof=1$) without inter-seed pooling.
12. `test_zero_valid_pixel_event_flagged_as_unevaluable`: Verifies zero-valid-pixel event is flagged as technically unevaluable, marks technical validity as failed, sets macro type to `reduced_event_diagnostic_only`, and excludes the event from macro F1.
13. `test_genuine_flood_free_event_perfect_negative`: Verifies genuine flood-free event with valid pixels receives F1=1.0 and is marked evaluable.
14. `test_invalid_samples_not_silently_dropped`: Verifies invalid-only samples are tracked in sample_valid_pixel_counts and not silently dropped.

---

## 5. Final Evaluation Settings & Proposed Files for Commit

### 5.1 Final Evaluation Settings
- **Protocol:** `week7_source_baseline_v1`
- **Config:** `configs/evaluation/source_baseline_tta_dev_protocol.json`
- **Model:** `SNUNet_ECAM` (`in_channels=2`, `out_ch=3`, `base_channel=32`, `canonical_preevent="pre_event_2"`)
- **Seeds:** `0, 1, 2` (evaluated independently; no ensembling)
- **Checkpoints:**
  - Seed 0: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` (`247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`)
  - Seed 1: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` (`72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a`)
  - Seed 2: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` (`0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd`)
- **Dataset Root:** `/usr1/home/s126mdg22_06/FYP_SARCD/datasets/KuroSiwo`
- **Manifest Path:** `/usr1/home/s126mdg22_06/FYP_SARCD/third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz`
- **Target Events (6):** `[279, 437, 520, 559, 1111003, 1111008]` (2,826 samples)
- **Quarantined Events (10):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (zero access)
- **Batch Size:** `32`
- **Output Directory:** `outputs/week7_source_baseline/`

### 5.2 Proposed Files for Commit
The following new and modified files constitute the clean Week 7A evaluation package:
1. `configs/evaluation/source_baseline_tta_dev_protocol.json` (new)
2. `docs/source_baseline_tta_dev_protocol.md` (new)
3. `docs/week7_prelaunch_integrity_review.md` (new)
4. `src/data/evaluation_dataset.py` (new)
5. `src/evaluation/evaluator.py` (new)
6. `tools/evaluate_source_baseline.py` (new)
7. `tests/test_source_evaluation.py` (new)
8. `src/evaluation/metrics.py` (modified: added standalone macro and pooled metric functions)
9. `src/models/snunet_wrapper.py` (modified: PyTorch 2.6 weights_only support and model prefix handling)

---

## 6. Git Status & Prelaunch Commands

### 6.1 Current Working Tree Status
```
On branch main
Changes not staged for commit:
  modified:   docs/interim_report_outline.md
  modified:   src/evaluation/metrics.py
  modified:   src/models/snunet_wrapper.py

Untracked files:
  configs/evaluation/
  docs/figures/
  docs/interim_report_draft_v1.md
  docs/interim_report_final_v2.md
  docs/interim_report_final_v3.md
  docs/interim_report_v2_fact_audit.md
  docs/interim_report_v3_correction_audit.md
  docs/source_baseline_tta_dev_protocol.md
  docs/week4_interim_evidence_map.md
  docs/week4_interim_video_slides.md
  docs/week4_progress.md
  docs/week4_supervisor_review_questions.md
  docs/week4_video_script_v1.md
  docs/week4_visual_review.md
  docs/week6_all27_training_closeout.md
  docs/week7_prelaunch_integrity_review.md
  exports/
  src/data/evaluation_dataset.py
  src/evaluation/evaluator.py
  tests/test_source_evaluation.py
  tools/evaluate_source_baseline.py
  tools/export_week4_package.py
  tools/make_week4_interim_figures.py
```
*(No files have been staged or committed automatically. Week 4 report files remain untracked and intact.)*

### 6.2 Preflight Validation Verification
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/evaluate_source_baseline.py --validate_only
```
Output:
`PREFLIGHT AUDIT PASSED: Evaluation setup is valid and frozen.`

### 6.3 Prepared Command for Baseline Run (Pending Review)
```bash
cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/evaluate_source_baseline.py \
  --config configs/evaluation/source_baseline_tta_dev_protocol.json \
  --device cuda:0
```
*(No inference or TTA was launched during this turn. All sealed-test partitions remain untouched.)*
