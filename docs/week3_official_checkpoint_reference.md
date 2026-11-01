# Week 3 Official Checkpoint Reference Evaluation Report

**Date:** 2026-10-12
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git Baseline Commit:** `8a362cf0e9c0ebb9ab6e44fe7d5253e2d8e699ee` (`8a362cf`)
**Audited Checkpoint:** `checkpoints/kuro_snunet_official.pt` (SHA-256: `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed`)
**Reference Configuration:** `configs/reference/official_snunet_reference_slope_candidate.json` (SHA-256: `0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c`)
**Evaluation Scope:** Source-inner validation set (5 events: `[267, 275, 497, 518, 1111005]`, 2,284 samples)

---

## 1. Epistemological Classification & Interpretation Boundaries

To adhere strictly to IEEE-grade reporting standards, prevent post-hoc rationalization, and maintain absolute scientific transparency, all statements in this document are categorized by epistemic status:
- **[FACT]:** Directly verified from immutable on-disk artifacts (checkpoint weights, run manifests, logs, CSVs).
- **[CALC]:** Deterministic mathematical or statistical values calculated from verified facts.
- **[UNAVAILABLE]:** Attributes or training circumstances that cannot be definitively established from preserved upstream artifacts.
- **[INTERPRETATION]:** Bounded scientific analysis adhering strictly to pre-registered evaluation scope.

### MANDATORY INTERPRETATION BOUNDARIES [FACT, INTERPRETATION]
1. **External Contextual Reference Only [INTERPRETATION]:** This evaluation serves strictly as an external sanity check and historical contextual benchmark. It is **NOT** an official benchmark reproduction and **NOT** a fair, controlled comparison.
2. **Candidate Channel Reconstruction [FACT, UNAVAILABLE]:** The third input channel evaluated here is a **candidate reconstruction** utilizing ChangeGuard Policy S2 (topographic slope standardized with train-only parameters $\mu=1.492574^\circ, \sigma=2.372311^\circ$). The true training-time input feature map and normalization of `kuro_snunet_official.pt` are unverified and undocumented in upstream metadata.
3. **Unverified Event Exposure [UNAVAILABLE]:** The training-time event split for `kuro_snunet_official.pt` (released January 2024, commit `043a36a`) was never published. It is unknown whether any or all of the 5 validation events (`[267, 275, 497, 518, 1111005]`) were observed during upstream training.
4. **No Claim of Superiority or Inferiority [INTERPRETATION]:** The performance gap between this checkpoint and our controlled self-trained models must **never** be used to assert model superiority or inferiority, as loss formulations, weighting, data augmentations, and preprocessing differ fundamentally.
5. **Zero Influence on Model Selection [FACT]:** This reference evaluation has **zero influence** on the Week 3 source-recipe selection. Model A (`[VV, VH]`) remains the chosen source architecture for subsequent research phases based on the pre-registered three-seed replication study, with Model B (`[VV, VH, slope]`) retained strictly as an ablation.
6. **No Test-Time Adaptation [FACT]:** This task concludes Week 3 source-level auditing. The subsequent project phase is Week 4 reporting and synthesis, **NOT** Test-Time Adaptation (TTA). Zero access occurred to the 6 quarantined TTA-validation events or the 10 sealed test events.

---

## 2. Upstream Checkpoint Provenance & Audit

### 2.1 Preserved Upstream Metadata [FACT]
- **File Origin:** Downloaded via official KuroSiwo repository link (Bountos et al., NeurIPS 2024) corresponding to Git commit `043a36a` (January 2024).
- **Physical Size:** 144,657,509 bytes.
- **State Dict Structure:**
  - `epoch`: `13`
  - `loss`: `0.14699462056159973`
  - `model_state_dict`: 134 parameter and buffer tensors.
  - `optimizer_state_dict`: Adam optimizer state (`lr=0.001`, `betas=(0.9, 0.999)`, `weight_decay=0`).
  - `lr_scheduler_state_dict`: LambdaLR state.

### 2.2 Architectural Matching [FACT]
- **Architecture:** Loads with `strict=True` into `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
- **Parameter Count:** Exactly 12,035,107 parameters.
- **Input Projections:** `conv0_0.conv1.weight` has shape `(32, 3, 3, 3)` indicating exactly 3 channels per temporal branch (`xA`, `xB`).
- **Output Projection:** `conv_final.weight` has shape `(3, 128, 1, 1)` corresponding to 3 prediction classes:
  - Class 0: No Water
  - Class 1: Permanent Water
  - Class 2: Flood Water

### 2.3 Known Upstream Contradictions & Ambiguities [FACT, UNAVAILABLE]
- **Contradictory Configuration Keys:** Upstream configuration files (`configs/train/data_config.json`) contained duplicate contradictory keys (`"dem": true` and `"dem": false`, `"slope": false`). Standard JSON parsers default to the last key (`false`), whereas CLI examples indicate `--dem=True`.
- **Pre-v2 Dataset Manifest:** The checkpoint was created in January 2024, over a year before KuroSiwo v2 (`KuroV2_grid_dict.gz`, March 2025). The exact event partitioning and grid tile definitions used during training cannot be reconstructed.
- **Candidate Slope Feature:** Because KuroSiwo documented DEM / slope auxiliary channels but did not preserve training preprocessing pipelines, providing standardized topographic slope as the 3rd channel is strictly a candidate hypothesis.

---

## 3. Inference Execution Audit: Two-Pass Lifecycle

The official-checkpoint reference evaluation was executed across two distinct implementation passes [FACT]:

### 3.1 Initial Failed Implementation Attempt [FACT]
- **Start Time:** 2026-10-12 17:24:14 UTC+8 in detached tmux session `cg_w3_official_ref`.
- **End Time / State:** 2026-10-12 17:25:05 UTC+8 (`FAILED_POST_PROCESSING`).
- **Inference Status:** The model forward pass evaluated all 143 batches (2,284 samples) completely and deterministically under `torch.no_grad()`. The post-inference model state hash was verified identical (`e92911110224129dfd7c278e59f3207567c3df097d133d9539132d8878eeb209`).
- **Failure Point:** Process crashed during metric formatting at line 287 of `tools/eval_official_reference.py` with an unhandled exception:
  `KeyError: 'pooled_f1'`.
  The helper `EventMetricAccumulator.compute_summary()` stores pooled metrics inside a nested dictionary `summary["pooled_metrics"]["f1"]`, but the script attempted direct top-level access `summary["pooled_f1"]`.
- **Partial Artifacts:** Only `eval.log` was created (3,144 bytes). Because the exception occurred prior to artifact serialization, `event_validation_metrics.csv`, `reference_summary.json`, `access_trace.json`, and `run_manifest.json` were **not created**.

### 3.2 Exact Code Correction [FACT]
- **Location:** `tools/eval_official_reference.py`, line 284.
- **Modification:** Added local variable `pooled = summary["pooled_metrics"]` and updated metric accessors:
  - `logger.info(f"Pixel-Pooled Flood F1:  {pooled['f1']:.6f}")`
  - `logger.info(f"Pooled Precision / Rec: {pooled['precision']:.6f} / {pooled['recall']:.6f}")`
  - `logger.info(f"Pooled IoU:             {pooled['iou']:.6f}")`
  - `summary_data["pooled_metrics"] = {"f1": pooled["f1"], "iou": pooled["iou"], "precision": pooled["precision"], "recall": pooled["recall"]}`
- **Scope of Change [FACT]:** The correction changed **only output formatting and JSON dictionary serialization**. It did **not** modify model inputs, normalization, predictions, ground-truth labels, loss, or metric mathematical definitions.

### 3.3 Canonical Successful Run [FACT]
- **Start Time:** 2026-10-12 17:27:11 UTC+8 in detached tmux session `cg_w3_official_ref` on GPU 2 (`cuda:0`).
- **End Time / State:** 2026-10-12 17:27:45 UTC+8 (`COMPLETED`, exit code 0).
- **Output Directory Overwrite Behavior:** Executed in the same output directory (`outputs/week3_official_reference/`). Overwrote the partial `eval.log` from Attempt 1 and cleanly serialized all four remaining artifacts (`event_validation_metrics.csv`, `reference_summary.json`, `access_trace.json`, `run_manifest.json`).
- **Absence of Outcome Selection [FACT]:** Zero tuning, zero hyperparameter adjustment, and zero post-hoc selection occurred between Attempt 1 and Attempt 2. Both attempts evaluated the identical model checkpoint, identical configuration, identical split, and identical candidate slope normalization.

---

## 4. Production Configuration Integrity & Transcript Restoration Audit

### 4.1 Frozen Configuration Integrity [FACT]
- **File:** `configs/reference/official_snunet_reference_slope_candidate.json`.
- **Canonical Hash:** `0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c` (3,302 bytes, 83 lines, single newline EOF).
- **Run Record Alignment:**
  - `outputs/week3_official_reference/eval.log` records `Config SHA-256: 0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c`.
  - `outputs/week3_official_reference/run_manifest.json` records `"config_sha256": "0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c"`.
  - `outputs/week3_official_reference/reference_summary.json` records `"sha256": "0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c"`.

### 4.2 Configuration Mutation & Restoration Audit [FACT]
- **Mutation Event:** At 17:32:35 UTC+8, an editor save hook appended a trailing newline (POSIX EOF formatting), increasing file size from 3,302 to 3,303 bytes (`}\n\n`).
- **Modified Hash:** `712669b12855f67a02ec10a3db82de60617708cd41d95f8dcc30681269306721`.
- **Scientific Impact:** Zero scientific settings, keys, or values were altered; only the trailing EOF byte changed.
- **Restoration:** Restored the exact 3,302-byte original content from the Antigravity conversation transcript. The on-disk hash is restored to `0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c`, strictly matching the canonical run manifest.
- **Regression Safeguard:** `tests/test_official_reference_safeguards.py` includes a `tearDown` assertion verifying that the production config hash remains strictly invariant across test executions.

---

## 5. Frozen Reference Evaluation Protocol & Invariants

### 5.1 Protocol Specifications [FACT]
- **Target Dataset Split:** `configs/splits/kuro_source_inner_split_v1.json` (`split="val"`, 5 events: `[267, 275, 497, 518, 1111005]`, 2,284 samples).
- **SAR Preprocessing:** Clamping to `[0.0, 0.15]`, standardized with train-only parameters $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$.
- **Auxiliary Slope Normalization:** Standardized with train-only parameters $\mu=1.492574^\circ, \sigma=2.372311^\circ$.
- **Evaluation Rule:** Deterministic argmax classification; binary flood decision is `pred == 2`.
- **Double Masking:** Pixels evaluated strictly where `(valid_mask == 1) & (label != 3)`.

### 5.2 Verification of Invariants [FACT]
- **Zero Optimization Invariant:** Zero optimizers created, zero backward passes executed, zero parameter updates.
- **Weight Immutability Invariant:** Initial model state hash and final model state hash are identical:
  `e92911110224129dfd7c278e59f3207567c3df097d133d9539132d8878eeb209`.
- **Data Quarantine Invariant:** Zero access occurred to the 6 quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) or the 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).
- **Completeness Invariant:** Exactly 2,284 of 2,284 expected validation samples were evaluated.

---

## 6. Quantitative Evaluation Results

All numerical values are transcribed directly from machine-generated artifacts (`outputs/week3_official_reference/reference_summary.json` and `event_validation_metrics.csv`) [FACT].

### 6.1 Overall Validation Metrics [FACT, CALC]
| Metric | Value |
| :--- | :--- |
| **Event-Macro Flood F1** | **`0.628675`** |
| **Event-Macro Flood IoU** | **`0.484596`** |
| **Event-Macro Precision** | **`0.683833`** |
| **Event-Macro Recall** | **`0.602033`** |
| **Pixel-Pooled Flood F1** | **`0.768404`** |
| **Pixel-Pooled Flood IoU** | **`0.623909`** |
| **Pixel-Pooled Precision** | **`0.769092`** |
| **Pixel-Pooled Recall** | **`0.767717`** |
| **Worst-Event Flood F1** | **`0.384214`** (Event 267) |
| **Total Evaluated Samples** | `2,284` |
| **Total Valid Pixels** | `112,243,176` |
| **Total Invalid / NoData Pixels** | `2,358,808` |
| **Evaluation Runtime** | `31.30` seconds |
| **Peak GPU Memory** | `2,430.58` MB (`cuda:0`, NVIDIA RTX A5000) |

### 6.2 Predicted Class Proportions [FACT, CALC]
Over the 112,243,176 valid evaluation pixels:
- **Class 0 (No Water):** 69.91% (`78,469,081` pixels)
- **Class 1 (Permanent Water):** 13.24% (`14,860,974` pixels)
- **Class 2 (Flood Water):** 16.85% (`18,913,121` pixels)

### 6.3 Per-Event Metric Breakdown [FACT]
| Event ID | Samples | Valid Pixels | GT Flood | Pred Flood | TP | FP | FN | TN | Precision | Recall | F1 | IoU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **267** | 218 | 10,600,181 | 2,467,643 | 2,202,854 | 897,234 | 1,305,620 | 1,570,409 | 6,826,918 | 0.407305 | 0.363600 | **0.384214** | 0.237787 |
| **275** | 742 | 36,873,536 | 4,565,178 | 5,194,885 | 3,730,103 | 1,464,782 | 835,075 | 30,843,576 | 0.718034 | 0.817077 | **0.764360** | 0.618595 |
| **497** | 398 | 19,130,935 | 1,311,301 | 664,690 | 499,034 | 165,656 | 812,267 | 17,653,978 | 0.750777 | 0.380564 | **0.505097** | 0.337880 |
| **518** | 306 | 14,584,271 | 396,484 | 324,442 | 216,986 | 107,456 | 179,498 | 14,080,331 | 0.668798 | 0.547276 | **0.601965** | 0.430579 |
| **1111005** | 620 | 31,054,253 | 10,206,364 | 10,526,218 | 9,202,545 | 1,323,673 | 1,003,819 | 19,524,216 | 0.874250 | 0.901648 | **0.887737** | 0.798137 |

---

## 7. Contextual Reference Comparison & Reconciled Aggregations

To place these results into context alongside the pre-registered three-seed source models without making invalid inferential claims, the metrics are tabulated below with explicit, unambiguous aggregation definitions [FACT, CALC].

### 7.1 Master Comparison Table with Disambiguated Aggregations [FACT, CALC]
| Metric & Aggregation Definition | Official Checkpoint Reference (`kuro_official`) | Model A (`[VV, VH]`) Controlled 3-Seed | Model B (`[VV, VH, slope]`) Controlled 3-Seed |
| :--- | :---: | :---: | :---: |
| **Input Channels** | 3 (Candidate S2 Slope) | 2 (`[VV, VH]`) | 3 (Policy S2 Slope) |
| **Loss Formulation** | Unknown upstream (unweighted/weighted CE) | CE + Soft-Dice ($\lambda=1.0$) | CE + Soft-Dice ($\lambda=1.0$) |
| **Event-Macro Flood F1** (Mean of per-seed Event-Macro F1, $N=3$) | `0.628675` | **`0.793205 ± 0.011438`** | `0.791344 ± 0.006165` |
| **Worst-Event Flood F1** (Mean of per-seed Worst-Event F1, $N=3$) | `0.384214` (Event 267) | **`0.727929 ± 0.019208`** | `0.710277 ± 0.011150` |
| **Mean of Per-Seed Pooled F1** (Arithmetic mean $\pm s$, $N=3$, $df=2$) | `0.768404` (Single run) | `0.802067 ± 0.014142` | `0.803170 ± 0.007215` |
| **Total-Pooled F1 Across Seeds** (Recomputed from summed confusion counts) | `0.768404` (Single run) | `0.802132` | `0.803179` |
| **Event-Macro IoU** (Mean of per-seed Event-Macro IoU, $N=3$) | `0.484596` | **`0.661050 ± 0.015085`** | `0.658920 ± 0.007927` |
| **Macro Precision** (Mean of per-seed Event-Macro Precision, $N=3$) | `0.683833` | `0.771198 ± 0.012554` | `0.766106 ± 0.006307` |
| **Macro Recall** (Mean of per-seed Event-Macro Recall, $N=3$) | `0.602033` | `0.850486 ± 0.007134` | `0.856480 ± 0.005313` |
| **Pixel-Pooled Precision** (Mean of per-seed pooled precision, $N=3$) | `0.769092` | `0.829599 ± 0.009902` | `0.823421 ± 0.007391` |
| **Pixel-Pooled Recall** (Mean of per-seed pooled recall, $N=3$) | `0.767717` | `0.776387 ± 0.019523` | `0.783924 ± 0.009340` |
| **Pixel-Pooled IoU** (Mean of per-seed pooled IoU, $N=3$) | `0.623909` | `0.669699 ± 0.019824` | `0.671122 ± 0.010088` |

### 7.2 Reconciliation of Pooled Metric Discrepancy [FACT, CALC]
An earlier preliminary draft of this reference document recorded placeholder values (`0.806282 ± 0.010243` for Model A, `0.805820 ± 0.007675` for Model B) under pixel-pooled F1. Independent audit confirms:
1. **Per-Seed Breakdown:** The exact per-seed pixel-pooled F1 values preserved in `outputs/week3_source_replication/comparison_seed0_seed1_seed2/source_seed012_comparison.csv` are:
   - Model A: Seed 0 = `0.791289`, Seed 1 = `0.796832`, Seed 2 = `0.818081`.
   - Model B: Seed 0 = `0.810865`, Seed 1 = `0.802090`, Seed 2 = `0.796556`.
2. **Mean of Per-Seed Pooled F1:**
   - Model A: $(0.791289 + 0.796832 + 0.818081)/3 = \mathbf{0.802067}$, with sample standard deviation $s = \mathbf{0.014142}$.
   - Model B: $(0.810865 + 0.802090 + 0.796556)/3 = \mathbf{0.803170}$, with sample standard deviation $s = \mathbf{0.007215}$.
   These values strictly match the canonical Week 3 closeout (`docs/week3_seed012_source_comparison.md`, line 170).
3. **Total-Pooled F1 Across Seeds:**
   Summing pixel-level true positives, false positives, and false negatives across all three seeds ($3 \times 5 = 15$ event-evaluations):
   - Model A: $\text{TP} = 41,714,142$, $\text{FP} = 8,552,434$, $\text{FN} = 12,010,037 \implies \text{F1} = \mathbf{0.802132}$.
   - Model B: $\text{TP} = 42,119,103$, $\text{FP} = 9,031,993$, $\text{FN} = 11,605,076 \implies \text{F1} = \mathbf{0.803179}$.
4. **Draft Discrepancy Cause:** Neither the canonical per-seed mean nor the total-pooled aggregation yields `0.806282` or `0.805820`. Those draft numbers were ungrounded placeholders generated during early table compilation and have been reconciled with the canonical artifacts without modifying the generated official-reference evaluation outputs.

### 7.3 Key Observations [FACT, INTERPRETATION]
1. **Severe Worst-Event Degradation on Event 267 [FACT]:** The official checkpoint attains an F1 of only `0.384214` on Event 267 (Precision: 0.407305, Recall: 0.363600), compared to Model A's mean of `0.743370` on the same event.
2. **Conservative Recall on Event 497 [FACT]:** On Event 497, the official checkpoint exhibits a recall of only `0.380564`, missing over 812,000 flood pixels despite relatively high precision (`0.750777`).
3. **Strong Performance on Event 1111005 [FACT]:** The checkpoint achieves `0.887737` F1 on Event 1111005, which constitutes its best-performing event and accounts for 45% of all ground-truth flood pixels in the validation set. This high pixel count explains why its pixel-pooled F1 (`0.768404`) is substantially higher than its event-macro F1 (`0.628675`).

---

## 8. Preserved Artifacts & Integrity Hashes [FACT]

All generated reference artifacts are located in `outputs/week3_official_reference/` and are verified by SHA-256 digests:

| File Path | Description | SHA-256 Digest |
| :--- | :--- | :--- |
| `configs/reference/official_snunet_reference_slope_candidate.json` | Frozen reference configuration | `0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c` |
| `tools/eval_official_reference.py` | Standalone deterministic runner | `74281e8ea201f520a260ff915f73e2daa1b43ca5f6e2a6f5ccefac737e672c83` |
| `tests/test_official_reference_safeguards.py` | Safeguard and regression tests | `95805f1b8aede6ec4211ebf855c1ef85a43a19d51e95176bbbc26918b8dcc6dc` |
| `outputs/week3_official_reference/reference_summary.json` | Machine-readable metrics summary | `b56e8edcd288211c570402897b4bb6c7b43738202f06d6eb0a823c3e8de0ebf9` |
| `outputs/week3_official_reference/event_validation_metrics.csv` | Exact per-event confusion matrix & metrics | `cad9ef03519414e4e0471b26474d434d2cf5e20d6297a6186fc9907cd41e1bdb` |
| `outputs/week3_official_reference/run_manifest.json` | Reproducibility & compliance audit manifest | `46fcda7b7a7c785b846bffd050696082596b241bea456153d21e353693a1c494` |
| `outputs/week3_official_reference/access_trace.json` | Zero-quarantine access trace | `b7c04f700f85eda52a5d7b8265e0c6495724ac358b1abc9bba21ac998be94cde` |
| `outputs/week3_official_reference/eval.log` | Execution runtime log | `fe1fb5fa2acdc89e351c0f5159d39480052452572d12d481b2d23f7d0b353921` |

---

## 9. Conclusion & Next Phase Confirmation

1. **Task Completion [FACT]:** The single bounded official-checkpoint reference evaluation was executed under strict isolation, verified invariants, zero parameter updates, and zero access to quarantined events.
2. **Source Selection Unchanged [FACT, INTERPRETATION]:** Model A (`[VV, VH]`) remains the frozen primary source-model architecture based on the pre-registered three-seed evidence, with Model B (`[VV, VH, slope]`) retained strictly as an ablation.
3. **Next Phase [FACT]:** Week 3 source-training replication and reference auditing are formally complete. The next phase is **Week 4 synthesis and reporting**. No Test-Time Adaptation experiments have been launched or pre-authorized.
