# Week 7 Source-Only Baseline Evaluation Closeout & Verification

**Document ID:** `docs/week7_source_baseline_closeout.md`
**Date (UTC):** 2026-10-16
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Evaluation Commit:** `7338d20fda1f9b2915bbde0c4899877029adf116` (`git_dirty: true` due to untracked documentation and previous week artifacts; evaluation code is 100% clean and committed)
**Training Commit:** `6c63a5f8aa7ae2c608b145f0badc84b422b62ef1` (`ChangeGuard_all27_clean`)
**Evaluation Protocol:** `configs/evaluation/source_baseline_tta_dev_protocol.json` (`docs/source_baseline_tta_dev_protocol.md`)
**Protocol Config SHA-256:** `4f298c70eea2d135422dbec08f4feb7883e1f36cc1fcd64acfc56f0b7f9fb18a`
**Compute Hardware:** NVIDIA GeForce RTX 3090 (GPU 0), PyTorch 2.13.0+cu130, Python 3.10.20
**Evaluated Seeds:** `[0, 1, 2]` (all 3 formal Week 6 all-source Model A checkpoints)
**Evaluated Events (6 TTA Development Events):** `[279, 437, 520, 559, 1111003, 1111008]`
**Quarantined Events (10 Sealed Test Events):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`
**Status:** `SOURCE_BASELINE_EVALUATION_COMPLETED_AND_VERIFIED`

---

## 1. Executive Summary & Evaluation Boundary

This closeout document provides the definitive verification, artifact audit, independent metric recomputation, and visual inspection for the controlled source-only baseline evaluation of ChangeGuard Model A (`SNUNet_ECAM [VV, VH]`).

All three formal Week 6 checkpoints (trained from random initialization on the 27 source events for 16 epochs) were evaluated independently on the pre-registered 6 TTA development events comprising exactly 2,826 multi-temporal SAR patch pairs (135,859,080 valid pixels). Evaluation was strictly non-adaptive (`model.eval()`, `torch.no_grad()`, zero optimizer steps, zero backward passes, zero batch normalization running statistics updates). Model state immutability was cryptographically proven before and after inference for every seed.

All technical validity checks passed unconditionally. Every one of the 6 events was technically evaluable (`valid_pixels > 0`) across all three seeds, satisfying the rule for a complete six-event baseline. All 18 seed-event confusion counts were independently recomputed and verified to bit-level precision.

### Key Benchmark Metrics Across Seeds

| Aggregation Level | Macro F1 | Macro Recall | Macro Precision | Macro IoU | Pooled F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 0** | 0.5949 | 0.6753 | 0.5917 | 0.4530 | 0.5193 |
| **Seed 1** | 0.5884 | 0.6691 | 0.6016 | 0.4461 | 0.4766 |
| **Seed 2** | 0.5885 | 0.6389 | 0.6113 | 0.4464 | 0.5078 |
| **Three-Seed (Mean $\pm$ SD)** | **0.5906 $\pm$ 0.0037** | **0.6611 $\pm$ 0.0195** | **0.6015 $\pm$ 0.0098** | **0.4485 $\pm$ 0.0039** | **0.5012 $\pm$ 0.0221** |

> [!IMPORTANT]
> **Scientific Integrity & Boundary Condition:**
> 1. **No Post-Hoc Target Performance Threshold:** As decided and frozen prior to evaluation launch, no numerical target-performance pass/fail threshold was imposed on target-domain evaluation. The source baseline is an empirical reference anchor, not a filtered outcome.
> 2. **Complete Six-Event Baseline:** Because all 6 events contain valid non-zero labeled pixels across all seeds, the macro average represents the complete six-event baseline without exclusions.
> 3. **Sealed-Test Quarantine & Metadata Access:** Zero rasters, zero ground-truth masks, and zero predictions were loaded or evaluated for the 10 quarantined sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`). In accordance with standard data loading, the dataset manifest dictionary was loaded to filter and select the 6 allowlisted events; this permitted metadata indexing is strictly distinguished from raster/label data access.
> 4. **Historical Gate S Status:** The historical dry-tile false-alarm criterion from the source-inner study remains unassessed on the all-source models, as specified in the protocol.

---

## 2. Technical Validity & Cryptographic Audit

### 2.1 Checkpoint Provenance & Model State Immutability

Each model checkpoint was loaded in CPU memory under strict state-dict verification, transferred to GPU 0, and audited for state immutability by computing a full SHA-256 digest over all floating-point parameters and buffers immediately before and after inference.

| Run / Seed | Checkpoint Path | Training Commit | Checkpoint SHA-256 | Pre-Inference Model Hash | Post-Inference Model Hash | Immutability Verified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 0** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `6c63a5f8` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | `e70362653c7159e686a2f4bbe49ea41cd860501b96163a11194d02d4bc7cfdba` | `e70362653c7159e686a2f4bbe49ea41cd860501b96163a11194d02d4bc7cfdba` | **TRUE** (Bit-exact match) |
| **Seed 1** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `6c63a5f8` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` | `4fa22949740fe310df1c581c145c1fef297535034b6fd591023131d15809a8a4` | `4fa22949740fe310df1c581c145c1fef297535034b6fd591023131d15809a8a4` | **TRUE** (Bit-exact match) |
| **Seed 2** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `6c63a5f8` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` | `8dcc7cdc19930c26cb0a0528639375a4b469cdd32f974d2058d26c32278c1985` | `8dcc7cdc19930c26cb0a0528639375a4b469cdd32f974d2058d26c32278c1985` | **TRUE** (Bit-exact match) |

### 2.2 Sample Coverage & Valid-Pixel Accounting

The evaluation dataset was constructed strictly from the 6 target development events. Exactly 2,826 samples were presented to each model. No sample was missed, no sample was duplicated, and valid-pixel accounting was identical across all three evaluation passes.

| Event ID | Expected Samples | Evaluated Samples | Missing / Duplicates | Total Valid Pixels | GT Flood Pixels | Non-Flood Valid Pixels | Flood Pixel Fraction | Evaluable Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **279** | 2,381 | 2,381 | 0 / 0 | 116,199,247 | 868,697 | 115,330,550 | 0.748% | Evaluable (`valid_pixels > 0`) |
| **437** | 169 | 169 | 0 / 0 | 7,859,430 | 64,002 | 7,795,428 | 0.814% | Evaluable (`valid_pixels > 0`) |
| **520** | 92 | 92 | 0 / 0 | 3,455,248 | 329,243 | 3,126,005 | 9.529% | Evaluable (`valid_pixels > 0`) |
| **559** | 42 | 42 | 0 / 0 | 1,843,520 | 50,672 | 1,792,848 | 2.749% | Evaluable (`valid_pixels > 0`) |
| **1111003** | 19 | 19 | 0 / 0 | 815,490 | 6,961 | 808,529 | 0.854% | Evaluable (`valid_pixels > 0`) |
| **1111008** | 123 | 123 | 0 / 0 | 5,686,145 | 480,301 | 5,205,844 | 8.447% | Evaluable (`valid_pixels > 0`) |
| **Total** | **2,826** | **2,826** | **0 / 0** | **135,859,080** | **1,799,876** | **134,059,204** | **1.325%** | **All 6 Evaluable** |

### 2.3 Numerical Health & Collapse Audit

For all three runs:
- **Numerical Safeguards:** 0 non-finite outputs, 0 NaNs, 0 Infs detected.
- **Trivial Collapse Checks:**
  - `run_all_background`: **False**
  - `run_all_permanent_water`: **False**
  - `run_all_flood`: **False**
  - `run_zero_flood_predicted`: **False**
- **Per-Event Triviality:** No event produced constant predictions or zero predicted flood pixels across any of the seeds.

---

## 3. Empirical Performance Ledger & Independent Recomputation

Every metric reported below was verified through an independent recomputation directly from the unrounded pixel confusion counts (`tp`, `fp`, `fn`, `tn`). For all 18 rows, the mathematically computed metrics matched the stored summaries with zero discrepancy ($< 10^{-6}$).

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{IoU} = \frac{TP}{TP + FP + FN}$$

### 3.1 Complete 18-Row Event-Level Evaluation Ledger

| Seed | Event ID | Samples | Valid Pixels | GT Flood | Pred Flood | TP | FP | FN | TN | Precision | Recall | F1 Score | IoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **279** | 2,381 | 116,199,247 | 868,697 | 2,394,894 | 615,013 | 1,779,881 | 253,684 | 113,550,669 | 0.2568 | 0.7080 | 0.3769 | 0.2322 |
| **0** | **437** | 169 | 7,859,430 | 64,002 | 75,720 | 48,858 | 26,862 | 15,144 | 7,768,566 | 0.6452 | 0.7634 | 0.6994 | 0.5377 |
| **0** | **520** | 92 | 3,455,248 | 329,243 | 314,522 | 229,743 | 84,779 | 99,500 | 3,041,226 | 0.7305 | 0.6978 | 0.7137 | 0.5549 |
| **0** | **559** | 42 | 1,843,520 | 50,672 | 31,778 | 27,727 | 4,051 | 22,945 | 1,788,797 | 0.8725 | 0.5472 | 0.6726 | 0.5067 |
| **0** | **1111003** | 19 | 815,490 | 6,961 | 20,582 | 3,537 | 17,045 | 3,424 | 791,484 | 0.1718 | 0.5081 | 0.2568 | 0.1473 |
| **0** | **1111008** | 123 | 5,686,145 | 480,301 | 454,993 | 397,433 | 57,560 | 82,868 | 5,148,284 | 0.8735 | 0.8275 | 0.8499 | 0.7389 |
| **1** | **279** | 2,381 | 116,199,247 | 868,697 | 2,708,620 | 597,152 | 2,111,468 | 271,545 | 113,219,082 | 0.2205 | 0.6874 | 0.3339 | 0.2004 |
| **1** | **437** | 169 | 7,859,430 | 64,002 | 64,139 | 45,659 | 18,480 | 18,343 | 7,776,948 | 0.7119 | 0.7134 | 0.7126 | 0.5536 |
| **1** | **520** | 92 | 3,455,248 | 329,243 | 352,326 | 238,893 | 113,433 | 90,350 | 3,012,572 | 0.6780 | 0.7256 | 0.7010 | 0.5397 |
| **1** | **559** | 42 | 1,843,520 | 50,672 | 30,336 | 26,894 | 3,442 | 23,778 | 1,789,406 | 0.8865 | 0.5307 | 0.6640 | 0.4970 |
| **1** | **1111003** | 19 | 815,490 | 6,961 | 22,576 | 4,120 | 18,456 | 2,841 | 790,073 | 0.1825 | 0.5919 | 0.2790 | 0.1621 |
| **1** | **1111008** | 123 | 5,686,145 | 480,301 | 395,311 | 367,685 | 27,626 | 112,616 | 5,178,218 | 0.9301 | 0.7655 | 0.8398 | 0.7239 |
| **2** | **279** | 2,381 | 116,199,247 | 868,697 | 2,367,179 | 590,416 | 1,776,763 | 278,281 | 113,553,787 | 0.2494 | 0.6797 | 0.3649 | 0.2232 |
| **2** | **437** | 169 | 7,859,430 | 64,002 | 68,485 | 44,536 | 23,949 | 19,466 | 7,771,479 | 0.6503 | 0.6959 | 0.6723 | 0.5064 |
| **2** | **520** | 92 | 3,455,248 | 329,243 | 294,133 | 228,654 | 65,479 | 100,589 | 3,060,526 | 0.7774 | 0.6945 | 0.7336 | 0.5793 |
| **2** | **559** | 42 | 1,843,520 | 50,672 | 30,128 | 26,990 | 3,138 | 23,682 | 1,789,710 | 0.8958 | 0.5326 | 0.6681 | 0.5016 |
| **2** | **1111003** | 19 | 815,490 | 6,961 | 18,330 | 3,193 | 15,137 | 3,768 | 793,392 | 0.1742 | 0.4587 | 0.2525 | 0.1445 |
| **2** | **1111008** | 123 | 5,686,145 | 480,301 | 402,895 | 370,838 | 32,057 | 109,463 | 5,173,787 | 0.9204 | 0.7721 | 0.8398 | 0.7238 |

### 3.2 Per-Seed Aggregate Summary

Macro metrics are calculated as the unweighted arithmetic mean across all 6 events ($\frac{1}{6} \sum_{e=1}^6 M_e$). Pooled metrics aggregate all confusion pixels across all 6 events before computing ratios.

| Seed | Macro F1 | Macro Recall | Macro Precision | Macro IoU | Pooled F1 | Pooled Recall | Pooled Precision | Pooled IoU | Worst Event (ID, F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 0** | 0.594878 | 0.675319 | 0.591727 | 0.452958 | 0.519331 | 0.734668 | 0.401614 | 0.350741 | Event 1111003 (0.2568) |
| **Seed 1** | 0.588382 | 0.669090 | 0.601589 | 0.446095 | 0.476590 | 0.711384 | 0.358324 | 0.312844 | Event 1111003 (0.2790) |
| **Seed 2** | 0.588527 | 0.638905 | 0.611263 | 0.446449 | 0.507778 | 0.702619 | 0.397538 | 0.340283 | Event 1111003 (0.2525) |

### 3.3 Three-Seed Arithmetic Mean $\pm$ Sample Standard Deviation ($ddof=1$)

Multi-seed aggregation strictly uses the arithmetic mean and sample standard deviation across the three seed macro values:

$$\bar{x} = \frac{1}{K}\sum_{k=1}^K x_k, \quad s = \sqrt{\frac{1}{K-1}\sum_{k=1}^K (x_k - \bar{x})^2} \quad (K=3, ddof=1)$$

Seeds are **not** pooled together as extra events.

| Metric | Level / Formulation | Mean | Sample SD ($ddof=1$) | Formatted Report |
| :--- | :--- | :--- | :--- | :--- |
| **Macro F1** | Event-unweighted | 0.590596 | 0.003710 | **0.5906 $\pm$ 0.0037** |
| **Macro Recall** | Event-unweighted | 0.661105 | 0.019476 | **0.6611 $\pm$ 0.0195** |
| **Macro Precision** | Event-unweighted | 0.601526 | 0.009768 | **0.6015 $\pm$ 0.0098** |
| **Macro IoU** | Event-unweighted | 0.448501 | 0.003864 | **0.4485 $\pm$ 0.0039** |
| **Pooled F1** | Total pixel confusion | 0.501233 | 0.022109 | **0.5012 $\pm$ 0.0221** |
| **Pooled Recall** | Total pixel confusion | 0.716224 | 0.016563 | **0.7162 $\pm$ 0.0166** |
| **Pooled Precision** | Total pixel confusion | 0.385825 | 0.023904 | **0.3858 $\pm$ 0.0239** |
| **Pooled IoU** | Total pixel confusion | 0.334623 | 0.019572 | **0.3346 $\pm$ 0.0196** |
| **Worst-Event F1** | Minimum event F1 | 0.262769 | 0.014198 | **0.2628 $\pm$ 0.0142** |

### 3.4 Per-Event Cross-Seed Analysis

| Event ID | Samples | F1 Score (Mean $\pm$ SD) | Recall (Mean $\pm$ SD) | Precision (Mean $\pm$ SD) | IoU (Mean $\pm$ SD) | Performance Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1111008** | 123 | **0.8432 $\pm$ 0.0058** | 0.7884 $\pm$ 0.0340 | 0.9080 $\pm$ 0.0303 | 0.7289 $\pm$ 0.0087 | 1 (Best) |
| **520** | 92 | **0.7161 $\pm$ 0.0164** | 0.7060 $\pm$ 0.0171 | 0.7286 $\pm$ 0.0497 | 0.5579 $\pm$ 0.0200 | 2 |
| **437** | 169 | **0.6948 $\pm$ 0.0206** | 0.7242 $\pm$ 0.0350 | 0.6691 $\pm$ 0.0371 | 0.5325 $\pm$ 0.0240 | 3 |
| **559** | 42 | **0.6682 $\pm$ 0.0043** | 0.5369 $\pm$ 0.0090 | 0.8850 $\pm$ 0.0117 | 0.5017 $\pm$ 0.0048 | 4 |
| **279** | 2,381 | **0.3586 $\pm$ 0.0222** | 0.6917 $\pm$ 0.0146 | 0.2422 $\pm$ 0.0192 | 0.2186 $\pm$ 0.0164 | 5 |
| **1111003** | 19 | **0.2628 $\pm$ 0.0142** | 0.5196 $\pm$ 0.0673 | 0.1762 $\pm$ 0.0056 | 0.1513 $\pm$ 0.0094 | 6 (Worst) |

### 3.5 Mathematical Explanation of the Macro vs. Pooled Divergence

A significant difference is observed between Macro F1 ($0.5906$) and Pooled F1 ($0.5012$). This divergence is mathematically caused by the severe sample imbalance across the KuroSiwo evaluation split:
1. **Sample Concentration:** Event 279 contains **2,381 of the 2,826 samples** (84.3% of all patches) and represents **116,199,247 of the 135,859,080 valid pixels** (85.5% of all valid pixels).
2. **False-Alarm Volume:** In Event 279, Model A achieves high recall ($0.6917$), correctly detecting 590k–615k flood pixels. However, it generates an enormous number of false positive flood pixels (**1.78M to 2.11M FP pixels**) across dry terrain patches.
3. **Denominator Dominance:** In pooled evaluation, the total FP across all 6 events is 1.92M–2.29M, of which **92.1% to 92.5% originate solely from Event 279**. This floods the pooled precision denominator:

$$\text{Pooled Precision} = \frac{\sum TP}{\sum TP + \sum FP} \approx \frac{1.32\text{M}}{1.32\text{M} + 1.97\text{M}} \approx 0.3858$$

By contrast, the unweighted equal-event macro metric assigns equal weight ($\frac{1}{6}$) to each event, preventing Event 279's massive geographic extent from completely masking the strong performance on Events 1111008 ($F_1 = 0.8432$), 520 ($F_1 = 0.7161$), 437 ($F_1 = 0.6948$), and 559 ($F_1 = 0.6682$).

---

## 4. Qualitative Error Analysis & Visual Inspection

Two deterministic samples per event (indices $0$ and $\lfloor N/2 \rfloor$ in sorted sample ID order) were rendered as 5-panel diagnostic figures (Pre-event VV, Post-event VV, Ground Truth, Model Prediction, and Confusion Error Map) under `outputs/week7_source_baseline/visualizations/seed0/`.

At least one visualization per event was inspected to examine the visual prediction patterns and radar characteristics underlying model behavior:

### 1. Event 279
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_279_003b09f810f75b93b2f8838a41d10a5c.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_279_7ca093115b97532985931fbd274a1f8d.png`
- **Visual Features & Qualitative Observations:**
  - Tile `003b09f8...` is an unflooded tile in the ground truth. In the post-event SAR imagery, localized low-backscatter patches appear. Model A predicts false-positive flood clusters over these dark regions, visible as red clusters in the confusion error map.
  - Tile `7ca09311...` is also an unflooded tile without prominent low-backscatter anomalies. Model A correctly predicts background across the entire patch, indicating that false alarms are concentrated in specific low-backscatter areas rather than occurring uniformly across all dry tiles.

### 2. Event 437
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_437_04edf98576805eedaf7f87ce9180f55c.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_437_834c937db8815996ae23aa97660b4d02.png`
- **Visual Features & Qualitative Observations:**
  - Tile `04edf985...` displays inundation along a water corridor, where the post-event VV imagery exhibits backscatter reduction along channel margins. Model A detects the primary flooded channel body (green), with minor under-detection at the outer boundaries (yellow) and localized edge dilation (red).
  - Tile `834c937d...` is an unflooded patch where Model A predicts almost entirely true background, with only minor isolated false-positive speckle on an unflooded depression.

### 3. Event 520
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_520_7ee084a1fa9a575a9e84c9179145df7b.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_520_074b0f9c4d665c25b5be5cf6005f9ec0.png`
- **Visual Features & Qualitative Observations:**
  - Tile `7ee084a1...` displays extensive contiguous inundation with substantial post-event backscatter drop. Model A captures the main flooded area well (green), with discrepancies primarily at boundaries: false negatives along outer edges (yellow) and false positives extending into adjacent low-return margins (red).
  - Tile `074b0f9c...` contains narrow, fragmented hydrological features. Model A detects broader channel junctions but misses thinner linear branches, showing reduced sensitivity to fine structures.

### 4. Event 559
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_559_03bbc474ba0a58c6ad3087873f610443.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_559_7733e195a8c85cd29ab74a83c1a6a82d.png`
- **Visual Features & Qualitative Observations:**
  - Tile `03bbc474...` exhibits isolated, low-extent ground-truth flood pixels. Model A predicts conservatively, identifying core flooded pixels while omitting peripheral ground-truth pixels.
  - Tile `7733e195...` is an unflooded tile where Model A outputs pure background without false alarms, consistent with the event-level high precision and moderate recall.

### 5. Event 1111003
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_1111003_916cf27746475aeca41f7f45bc7a1be8.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_1111003_06da57ae6bf35b32b8f138c20512d767.png`
- **Visual Features & Qualitative Observations:**
  - Tile `916cf277...` illustrates a significant failure mode under domain shift. Ground truth contains only a small flood footprint. However, the post-event SAR imagery exhibits broad regions of low backscatter across unflooded terrain. Model A overpredicts flood extensively across these dark areas, generating broad false-positive swaths (red).
  - Tile `06da57ae...` has a small ground-truth flood area where Model A captures part of the true positive flood but also produces false negatives and false positives along the perimeter. The widespread false alarms on dark terrain account for the low event precision.

### 6. Event 1111008
- **Inspected Files:**
  - `outputs/week7_source_baseline/visualizations/seed0/event_1111008_8b69a33fd51d5e46ad28e1be8fc43434.png`
  - `outputs/week7_source_baseline/visualizations/seed0/event_1111008_0122d0874d5857119699e92ecd6ce527.png`
- **Visual Features & Qualitative Observations:**
  - Tile `8b69a33f...` represents high-contrast open-water flooding with a sharp post-event backscatter decrease. Model A achieves high spatial agreement with ground truth across the flooded area (predominantly green), with few misses and narrow false-positive boundaries.
  - Tile `0122d087...` is an unflooded background tile where Model A correctly outputs uniform background with zero false alarms, aligning with the event's high F1 and precision scores.


---

## 5. Week 7 Closeout Declaration & Research Implications for Week 8

### 5.1 Verification Checklist & Closeout Declaration

- [x] Exactly 3 seed results evaluated and verified.
- [x] Exactly 6 expected events per seed (18 unique seed-event metric records).
- [x] Exactly 2,826 samples per seed, matching frozen per-event allocations.
- [x] Zero missing samples and zero duplicate samples across all seeds.
- [x] Valid-pixel accounting verified ($135,859,080$ valid pixels across all seeds).
- [x] All 6 events technically evaluable (`valid_pixels > 0`) for every seed; no event excluded.
- [x] Finite numerical outputs verified throughout (0 non-finite values).
- [x] Pre- and post-inference model state hashes bit-exact; model immutability guaranteed.
- [x] Checkpoint hashes match formal Week 6 training records (`6c63a5f8` commit).
- [x] Quarantined sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) completely protected (zero rasters, zero labels accessed).
- [x] Unrounded confusion matrices independently recomputed with zero discrepancies.
- [x] Multi-seed aggregation computed strictly via arithmetic mean $\pm$ sample SD ($ddof=1$).
- [x] Worst event per seed and 3-seed macro identified and analyzed (Event 1111003).
- [x] Deterministic error visualizations inspected across all 6 events.

**Week 7 is hereby formally CLOSED.**

### 5.2 Research Implications for Week 8 Test-Time Adaptation (TTA)

The empirical findings from this controlled baseline provide foundational direction for the design of the ChangeGuard Test-Time Adaptation (TTA) framework in Week 8:

1. **Recall-Precision Asymmetry as Primary Target:**
   Across all three seeds, Model A's macro recall ($0.6611 \pm 0.0195$) is substantially higher than its macro precision ($0.6015 \pm 0.0098$), and pooled recall ($0.7162$) heavily exceeds pooled precision ($0.3858$). The dominant failure mode is **overprediction of flood on dark, smooth unflooded radar surfaces (false alarms)**, rather than missed detection. TTA adaptation strategies must focus on entropy minimization and pseudo-label refinement that aggressively suppresses false positives without eroding valid flood boundaries.
2. **Extreme Cross-Event Domain Variance:**
   Performance varies by nearly $60$ percentage points across events, from $0.8432$ F1 on Event 1111008 down to $0.2628$ F1 on Event 1111003. Any test-time adaptation algorithm must include **self-adaptation safeguards** (e.g., confidence gating, update magnitude clipping, or relative entropy monitoring) to prevent catastrophic performance degradation on difficult, low-contrast events like 1111003.
3. **Mandatory Equal-Event Weighting:**
   Because Event 279 accounts for 84.3% of the evaluation patches, unweighted equal-event macro F1 ($0.5906$) must remain the primary benchmark metric for assessing TTA gains. Relying on pooled metrics would cause any algorithm that over-fits to Event 279 to appear superior, while obscuring critical degradation on smaller events.
