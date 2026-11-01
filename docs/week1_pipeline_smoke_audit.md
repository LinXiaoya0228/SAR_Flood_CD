# Week 1 Pipeline and Smoke Testing Audit Report

**Document ID:** `docs/week1_pipeline_smoke_audit.md`  
**Version:** 1.0.0-FINAL  
**Status:** `WEEK1_PIPELINE_VALIDATED — READY FOR CONTROLLED TRAINING`  
**Date:** 2026-10-06  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA GeForce RTX 3090 24GB)  
**Execution GPU:** Single GPU (GPU 0, `CUDA_VISIBLE_DEVICES=0`)  
**Base Commit (Week 0 Freeze):** `af164ec69aff0105fc462d94417582be79ffd8a4`  
**Python Runtime:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`  

---

## 1. Executive Summary

Week 1 of the controlled source-model phase for ChangeGuard has successfully completed all development, benchmarking, empirical validation, and smoke-testing milestones.

### Key Milestones Achieved
1. **Week 0 Entry Verification:** Confirmed HEAD commit `af164ec69aff0105fc462d94417582be79ffd8a4` on branch `main` with frozen 22/5 split, frozen model configs, and unviolated data quarantine.
2. **Policy S2 Slope Pipeline & Statistics Artifact:** Generated the train-only slope normalization statistics strictly across the 22 inner-train events (19,178 samples) without leaking validation or test data. Output artifact: [`outputs/week1/source_train_slope_statistics.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1/source_train_slope_statistics.json) (`bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`).
3. **Nodata Cliff Boundary Mitigation:** Solved the bilinear interpolation fringe problem on DEM nodata (-32768m) using 3-iteration binary dilation and nearest-neighbor elevation inpainting before gradient computation, reducing artificial boundary slopes from $89.4^\circ$ down to physically consistent terrain values ($0.7^\circ$--$3.8^\circ$).
4. **Frozen Loss Module:** Implemented and validated absent-class stable unweighted Cross-Entropy + Soft Dice loss ($0.5 \cdot \text{CE} + 0.5 \cdot \text{Dice}$, $\epsilon=1.0$, `ignore_index=3`) in [`src/evaluation/losses.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/evaluation/losses.py).
5. **Physical Microbatch & Memory Benchmarking:** Evaluated candidate physical microbatch sizes `[2, 4, 8, 16]` on GPU 0 under AMP FP16. Confirmed `physical_microbatch_size = 8` with `gradient_accumulation_steps = 4` (effective batch size 32) as optimal, preserving **87.3%** VRAM safety headroom. Documented in [`docs/week1_batch_memory_amendment.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week1_batch_memory_amendment.md).
6. **Controlled Dataset & Loader Pipeline:** Implemented [`src/data/source_dataset.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/data/source_dataset.py) supporting Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`) with double-mask validity filtering (`valid = (MNA == 1) & (MLU != 3)`).
7. **Checkpoint Resume Reproducibility Validation:** Proved bit-level parameter alignment and loss replication between Run A (10 uninterrupted steps) and Run B (5 steps $\to$ save $\to$ reload $\to$ 5 steps), achieving maximum parameter discrepancy of **$0.00$** and maximum step loss difference of **$4.77 \times 10^{-7}$**.
8. **Short Smoke Training Runs:** Executed 25 optimizer steps on GPU 0 for Model A (`outputs/week1_smoke/model_a/`) and Model B (`outputs/week1_smoke/model_b/`). Both runs showed monotonic loss convergence (Model A: 12.47 $\to$ 0.50; Model B: 14.05 $\to$ 0.64), clean validation macro F1 (0.7202 and 0.6862), and successful checkpoint reload.
9. **Full Test Suite:** 60 / 60 unit and integration tests passing repository-wide in 12.3s.
10. **Research Integrity Certification:** Zero access to the 6 TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) and 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`). Manifest `KuroV2_grid_dict_test_0_100.gz` remained permanently unopened.

---

## 2. Policy S2 Slope Pipeline & Empirical Census

### 2.1 Technical Mechanism
Under Policy S2, raw elevation GeoTIFFs (`MK0_DEM.tif`) are processed as follows:
- **Pixel Spacing:** Dynamically extracted from raster metadata transform: observed $\Delta x = 10.0\text{ m}, \Delta y = 10.0\text{ m}$ under EPSG:3857 (projected metric). Geographic degree coordinate systems fail loudly.
- **Nodata Sanitization:** Raw nodata sentinels ($\le -430.0\text{ m}$ or metadata `nodata`) are identified. Because upstream bilinear resampling produced an interpolation fringe with values between $-1000\text{ m}$ and $0\text{ m}$ immediately adjacent to valid terrain, the nodata mask is dilated by 3 iterations (footprint $7 \times 7$). The masked invalid region is inpainted with nearest valid elevation via Euclidean distance transform before gradient evaluation.
- **Metric Slope Angle:**
  $$\frac{\partial z}{\partial x} = \frac{z_{i, j+1} - z_{i, j-1}}{2 \Delta x}, \quad \frac{\partial z}{\partial y} = \frac{z_{i+1, j} - z_{i-1, j}}{2 \Delta y}$$
  $$\text{slope}_{\text{deg}} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \times \frac{180.0}{\pi}$$
- **Neutral Prior Imputation:** Pixels marked invalid in elevation are filled with the training-set slope mean ($\mu_{\text{slope}}$) prior to z-score standardization, yielding exactly $0.0$ in standardized feature space.

### 2.2 Census Across 22 Inner-Train Events (19,178 Samples)
- **Artifact:** [`outputs/week1/source_train_slope_statistics.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1/source_train_slope_statistics.json)
- **SHA-256 Digest:** `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`
- **Total Valid Pixels:** 955,664,099
- **Nodata Pixels:** 6,611,229 (0.687%)
- **Moments:**
  - Mean Slope: **$1.4926^\circ$**
  - Standard Deviation: **$2.3723^\circ$**
  - Minimum / Maximum: **$0.0000^\circ$ / $88.4581^\circ$**
- **Percentiles:**
  - $p_1 = 0.005^\circ$, $p_5 = 0.005^\circ$, $p_{10} = 0.225^\circ$, $p_{25} = 0.535^\circ$
  - Median ($p_{50}$) = **$0.8650^\circ$**
  - $p_{75} = 1.525^\circ$, $p_{90} = 2.925^\circ$, $p_{95} = 4.795^\circ$, $p_{99} = 12.695^\circ$

*Physical Sanity Comparison:* Upstream unnormalized slope code generated $\sigma=67.50$ due to treating degree coordinate spacing as metres and differentiating across unmasked -32768 cliffs. Policy S2 delivers physically authentic terrain slopes (coastal floodplain median $< 1^\circ$) with clean standardization.

---

## 3. Physical Microbatch & Memory Benchmarking (GPU 0)

Conducted on GPU 0 (NVIDIA GeForce RTX 3090, 24,124 MiB available VRAM) with effective batch size 32 under AMP FP16:

| Model | Physical Microbatch | Accumulation Steps | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Headroom | Throughput | Status |
|:-----:|:-------------------:|:------------------:|:-------------------:|:------------------:|:-------------:|:----------:|:------:|
| Model A | 2 | 16 | 827.24 MiB | 928.00 MiB | 96.1% | 41.46 s/s | PASS |
| Model A | 4 | 8 | 1,435.67 MiB | 1,576.00 MiB | 93.5% | 48.86 s/s | PASS |
| **Model A** | **8** | **4** | **2,559.77 MiB** | **3,024.00 MiB** | **87.5%** | **49.86 s/s** | **OPTIMAL** |
| Model A | 16 | 2 | 4,991.43 MiB | 5,990.00 MiB | 75.2% | 49.54 s/s | PASS |
| Model B | 2 | 16 | 831.91 MiB | 986.00 MiB | 95.9% | 42.59 s/s | PASS |
| Model B | 4 | 8 | 1,437.66 MiB | 1,588.00 MiB | 93.4% | 47.93 s/s | PASS |
| **Model B** | **8** | **4** | **2,654.01 MiB** | **3,068.00 MiB** | **87.3%** | **49.38 s/s** | **OPTIMAL** |
| Model B | 16 | 2 | 4,998.87 MiB | 6,020.00 MiB | 75.0% | 49.25 s/s | PASS |

**Locked Decision:** `physical_microbatch_size = 8` with `gradient_accumulation_steps = 4` is officially locked across both configs.

---

## 4. Checkpoint Resume Reproducibility Validation

To certify that training can be paused, interrupted by language server disconnects, and resumed without numerical drift, an empirical Run A vs Run B test was executed on GPU 0:
- **Run A:** 10 continuous training steps without interruption.
- **Run B:** 5 steps $\to$ save `step5_checkpoint.pt` $\to$ create fresh trainer $\to$ resume $\to$ 5 steps.
- **DataLoader Offset Handling:** Resuming accounts for consumed microbatches `(global_step * accumulation_steps) % len(loader)` to ensure exact batch alignment.

### Empirical Alignment Results
- **Maximum Step Loss Discrepancy:** **$4.77 \times 10^{-7}$**
- **Final Step Loss:**
  - Run A (Step 10): `0.798498`
  - Run B (Step 10): `0.798498`
  - Difference: `0.00e+00`
- **Maximum Parameter Difference:** **$0.00\text{e}+00$** (bit-exact identical model weights)
- **Validation Report:** [`outputs/week1/checkpoint_resume_validation.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1/checkpoint_resume_validation.json) (`f6acbaa2c6756b8fa0056bd468fb2cb866dd585bece4e096343605439f399460`)

---

## 5. Short Smoke Training Runs

Short controlled smoke training runs (25 optimizer steps $\times$ 4 microbatches = 800 training samples) were executed on GPU 0 using real inner-train events and validated on the 5 inner-validation events.

### 5.1 Model A (`[VV, VH]`)
- **Output Directory:** [`outputs/week1_smoke/model_a/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1_smoke/model_a)
- **Initial Step Loss:** `12.4734` (CE: 24.0535, Dice: 0.8932)
- **Final Step Loss (Step 25):** `0.5037` (CE: 0.6998, Dice: 0.3076)
- **Duration (25 steps):** 125.8 seconds
- **Inner-Validation Macro Flood F1:** **`0.7202`** (across 5 inner-val events)
- **Inner-Validation Macro Recall:** **`0.7499`**
- **Worst-Event Flood F1:** `0.5314` (Event 518)
- **Checkpoints:** `checkpoint_latest.pt` (144.7 MB), `checkpoint_best.pt` (144.7 MB)
- **Run Manifest:** [`outputs/week1_smoke/model_a/run_manifest.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1_smoke/model_a/run_manifest.json) (`a8473f57f9eb88c937d6ec55353df66e1706c03d63739506d5c899da7611d604`)

### 5.2 Model B (`[VV, VH, slope]`)
- **Output Directory:** [`outputs/week1_smoke/model_b/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1_smoke/model_b)
- **Slope Normalization Input:** Mean $1.4926^\circ$, Std $2.3723^\circ$
- **Initial Step Loss:** `14.0554` (CE: 27.1807, Dice: 0.9301)
- **Final Step Loss (Step 25):** `0.6404` (CE: 0.9865, Dice: 0.2942)
- **Duration (25 steps):** 121.5 seconds
- **Inner-Validation Macro Flood F1:** **`0.6862`** (across 5 inner-val events)
- **Inner-Validation Macro Recall:** **`0.7585`**
- **Worst-Event Flood F1:** `0.5372` (Event 518)
- **Checkpoints:** `checkpoint_latest.pt` (144.7 MB), `checkpoint_best.pt` (144.7 MB)
- **Run Manifest:** [`outputs/week1_smoke/model_b/run_manifest.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1_smoke/model_b/run_manifest.json) (`7dbb08f4506a378a63d4d4e6d2371e2b8cb4b8a514a2cbbc31e872e01708f8fd`)

---

## 6. Complete Artifact Inventory & SHA-256 Manifest

| Relative Path | Size | Description | SHA-256 Digest |
|:--------------|:----:|:------------|:---------------|
| `outputs/week1/source_train_slope_statistics.json` | 1,574 B | Policy S2 22-event train slope statistics | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` |
| `outputs/week1/memory_smoke_results.json` | 2,371 B | GPU 0 VRAM and throughput smoke benchmarks | `c2da571fb39b5daea2fae88d9dd4edc235d2644a19e9ed3448ea48037dc91475` |
| `outputs/week1/checkpoint_resume_validation.json` | 272 B | Run A vs Run B bit-level equivalence certificate | `f6acbaa2c6756b8fa0056bd468fb2cb866dd585bece4e096343605439f399460` |
| `outputs/week1_smoke/model_a/smoke_run_metrics.json` | 4,877 B | Model A 25-step loss and validation metrics | `c1e2ce1c06ecc444e62bf6953b1f1791b5cac82d00ce6262e0bec9bd96a956b7` |
| `outputs/week1_smoke/model_a/run_manifest.json` | 2,186 B | Model A provenance and integrity manifest | `a8473f57f9eb88c937d6ec55353df66e1706c03d63739506d5c899da7611d604` |
| `outputs/week1_smoke/model_b/smoke_run_metrics.json` | 4,866 B | Model B 25-step loss and validation metrics | `0e167471664fda9bcb0164820b3b9a1bbae7a6e4e5478e7fb40b36100f180456` |
| `outputs/week1_smoke/model_b/run_manifest.json` | 2,192 B | Model B provenance and integrity manifest | `7dbb08f4506a378a63d4d4e6d2371e2b8cb4b8a514a2cbbc31e872e01708f8fd` |
| `docs/week1_batch_memory_amendment.md` | 3,115 B | Physical microbatch and memory amendment | `57e3f8a0ffbb8c0919f1704db4f2e5ebf878f44ff53e8783451bb31c9e421061` |
| `src/data/source_dataset.py` | 9,634 B | Controlled source dataset pipeline | `b348d49a7ce1c9e6ce22aebdfad63fca418e27c1511c19bbf0281b3feea2775a` |
| `src/evaluation/losses.py` | 4,557 B | Unweighted CE + Soft Dice loss module | `eb6601ea187c3a07851d0879c3d973cb8c8e1467406a6c17fbb3078a6d45e59e` |
| `src/training/source_trainer.py` | 8,826 B | SourceTrainer training engine | `23a2f8c520a7b458d0c26aeecf63566f1e16fdf9cf9aa5c4c23f11d13783c180` |

---

## 7. Research Integrity & Isolation Certification

We certify that:
1. **Zero TTA-Validation Ingestion:** The 6 usable TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) were **never** accessed, read, loaded, or evaluated during any phase of Week 1.
2. **Zero Sealed-Test Ingestion:** All 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remained untouched. The benchmark test manifest `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` was never opened or parsed.
3. **Strict Train-Only Slope Normalization:** Elevation and slope moments were computed strictly across the 19,178 inner-train patches of the 22 approved inner-train events. Zero validation or test samples contributed to normalization moments.
4. **No Multi-Epoch Training Executed:** Only short 25-step smoke tests and 10-step resume tests were executed. Multi-epoch formal source training remains properly held for subsequent phases.
5. **No Automatic Commits Made:** Working tree modifications are staged for user inspection and manual commit approval.

---

## 8. Conclusion & Gate S Readiness

Week 1 has successfully established a rock-solid, fully verified controlled source-model training foundation. The codebase is fully unit-tested (60/60 tests passing), memory-safe, reproducible across checkpoint restarts, and certified against all data-leakage hazards.
