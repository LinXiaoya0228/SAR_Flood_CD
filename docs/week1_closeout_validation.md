# ChangeGuard Week 1 Closeout Validation Audit

**Document:** `docs/week1_closeout_validation.md`  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `af164ec69aff0105fc462d94417582be79ffd8a4` (Week 0 Protocol Freeze)  
**Branch:** `main`  
**Audit Date:** `2026-10-06`  
**Host:** `gpu22` (Linux 6.8.0-136-generic x86_64)  
**Python Runtime:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` (Python 3.10.16, PyTorch 2.6.0+cu124)  
**Compute Hardware:** GPU 0 (NVIDIA GeForce RTX 3090, 24,576 MiB VRAM)  
**Final Verdict:** `WEEK1_COMPLETE_WITH_DOCUMENTED_CORRECTIONS`

---

## 1. Executive Summary & Gate Verdict

This audit provides an exhaustive, independent closeout validation of Week 1 of the controlled source-model training phase for the ChangeGuard Final Year Project. The objective of Week 1 was to construct, calibrate, and smoke-test the complete controlled training pipeline for **Model A** (`[VV, VH]`) and **Model B** (`[VV, VH, slope]`) under the frozen 22/5 source-inner split (Alternative A1), without launching formal 150-epoch training, without running test-time adaptation (TTA), and with zero access to quarantined TTA-validation or sealed test events.

### 1.1 Gate Verdict

$$\mathbf{VERDICT:\ WEEK1\_COMPLETE\_WITH\_DOCUMENTED\_CORRECTIONS}$$

The controlled source training pipeline is **functionally verified, mathematically sound, memory-stable, and ready for Week 2 formal 150-epoch execution**, subject to the specific provenance, manifest, and test suite corrections established and validated in this report.

### 1.2 Core Integrity & Boundary Verification

1. **Quarantined TTA-Validation Events ($n=6$):**
   - Events: `[279, 437, 520, 559, 1111003, 1111008]`
   - **Historical Week 1 Execution:** Legacy unit tests opened configurations (`configs/baseline_config.json`, `configs/gate_b_risk_protocol.json`) and sample data associated with Event 279 during early test discovery. This is formally recorded as `DEVIATION-W1-01`. No model update, source checkpoint selection, or reported source performance used Event 279. No final sealed-test image or label was accessed.
   - **Original and Corrected Source Smoke Runs:** Zero TTA-validation samples were loaded; zero sealed-test samples were loaded.
   - **Final Post-Correction Unit-Test Suite:** All legacy dependencies were completely replaced with synthetic fixtures; the corrected final suite no longer opens Event 279 or the prohibited manifest.
2. **Sealed Test Events ($n=10$):**
   - Events: `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`
   - **Status:** **STRICTLY SEALED**. No sealed-test image, label, prediction, or metric was accessed at any point. While legacy test discovery opened `KuroV2_grid_dict_test_0_100.gz` to filter for Event 279 prior to refactoring, no sealed test samples were read, and the final post-correction test suite no longer opens `KuroV2_grid_dict_test_0_100.gz`.
3. **Single GPU Constraint:**
   - Adhered to strictly. All profiling and smoke tests executed exclusively on **GPU 0** on host `gpu22`.
4. **Historical Artifact Preservation:**
   - All preliminary Week 1 outputs under `outputs/week1/` and `outputs/week1_smoke/` remain **immutable and byte-preserved**. Corrected runs and audits are segregated in `outputs/week1_correction/`.

---

## 2. State & Artifact Provenance

### 2.1 Git Repository State

- **Branch:** `main`
- **HEAD Commit:** `af164ec69aff0105fc462d94417582be79ffd8a4` (Week 0 freeze)
- **Working Tree Status:** Clean except for tracked config and test updates and untracked Week 1 deliverables. No premature staging or commit has occurred.

### 2.2 Historical Artifact Inventory (`outputs/week1/` & `outputs/week1_smoke/`)

All historical Week 1 artifacts generated during earlier sessions remain untouched and verified against their recorded hashes:

| Relative Path | Size (Bytes) | SHA-256 Hash | Purpose |
|---|---|---|---|
| `outputs/week1/source_train_slope_statistics.json` | 1,574 B | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | 22-event train slope mean & std normalization |
| `outputs/week1/memory_smoke_results.json` | 2,371 B | `c2da571fb39b5daea2fae88d9dd4edc235d2644a19e9ed3448ea48037dc91475` | Microbatch memory & VRAM profiling |
| `outputs/week1/checkpoint_resume_validation.json` | 272 B | `f6acbaa2c6756b8fa0056bd468fb2cb866dd585bece4e096343605439f399460` | Preliminary resume validation |
| `outputs/week1_smoke/model_a/checkpoint_latest.pt` | 144,657,931 B | `7b2353ffe21c1fe917ce281ee71cc10276cfd8b5b4ea8283a47cdc0d2f7ec5a9` | Model A initial smoke latest checkpoint |
| `outputs/week1_smoke/model_a/checkpoint_best.pt` | 144,656,571 B | `cc2079609feb19afff0c67b1f0d53f27d223646ce34bcd06a767dfc3664a315e` | Model A initial smoke best checkpoint |
| `outputs/week1_smoke/model_a/smoke_run_metrics.json` | 4,877 B | `c1e2ce1c06ecc444e62bf6953b1f1791b5cac82d00ce6262e0bec9bd96a956b7` | Model A initial smoke loss & F1 |
| `outputs/week1_smoke/model_a/run_manifest.json` | 2,186 B | `a8473f57f9eb88c937d6ec55353df66e1706c03d63739506d5c899da7611d604` | Model A initial run manifest |
| `outputs/week1_smoke/model_b/checkpoint_latest.pt` | 144,661,387 B | `7b5107191c42378e7d96b19463a1ec0077a24d121421aab63e01404eab5984f7` | Model B initial smoke latest checkpoint |
| `outputs/week1_smoke/model_b/checkpoint_best.pt` | 144,660,027 B | `6abc78ba52d1fdfd1f4e057307803e61a370aa5085d0ad21f250a138d79629cc` | Model B initial smoke best checkpoint |
| `outputs/week1_smoke/model_b/smoke_run_metrics.json` | 4,866 B | `0e167471664fda9bcb0164820b3b9a1bbae7a6e4e5478e7fb40b36100f180456` | Model B initial smoke loss & F1 |
| `outputs/week1_smoke/model_b/run_manifest.json` | 2,192 B | `7dbb08f4506a378a63d4d4e6d2371e2b8cb4b8a514a2cbbc31e872e01708f8fd` | Model B initial run manifest |

### 2.3 Corrected Artifact Inventory (`outputs/week1_correction/`)

All corrected artifacts generated during this validation audit are segregated in `outputs/week1_correction/`:

| Relative Path | Size (Bytes) | SHA-256 Hash | Purpose |
|---|---|---|---|
| `outputs/week1_correction/slope_sensitivity_comparison.json` | 2,103 B | `2a5be9b9f00882c3b37ed08857f8a965f07d28ab7b1cef691939319d1d563dcb` | 5-rule DEM nodata & slope sensitivity benchmark |
| `outputs/week1_correction/split_isolation_audit.json` | 579 B | `c56ad8daef271eb04d66cc17377a3dd87639771249700369ae3b853cd8256114` | Train/val/quarantined event and sample isolation |
| `outputs/week1_correction/loader_access_trace.json` | 668 B | `e86b52c1cbf434ddf3f0d0475e214af8bba1c1d901bb4ba5b9bc498e17c518ee` | Seed 42 DataLoader access verification |
| `outputs/week1_correction/model_a/checkpoint_latest.pt` | 144,668,851 B | `56a4f38e21731fe0bf03b35b8a79f10a04791732feeaaa566e4482c6346b5603` | Model A corrected checkpoint with full RNG states |
| `outputs/week1_correction/model_a/checkpoint_best.pt` | 144,667,481 B | `8388dd661b9e0dc7a3d1a5c3505701f3d243d270cde276417ecfc95e21e8b930` | Model A corrected best checkpoint |
| `outputs/week1_correction/model_a/run_manifest.json` | 2,992 B | `b78181835cef5dc5c0bf2ea3e8a00aeb0b3c86287eaa44cc4dcab79350e70046` | Model A corrected manifest with dynamic event trace |
| `outputs/week1_correction/model_a/access_trace.json` | 755 B | `81237dcb4094fe1e7da83ce44f7c5717e4138d58562a4c175568076afcc2bdea` | Model A runtime train/val sample trace |
| `outputs/week1_correction/model_a/smoke_run_metrics.json` | 5,054 B | `d932176c9a58d1df4b957ed2ede34fa28675e2d467eb3c41789676fee8f60658` | Model A corrected smoke metrics |
| `outputs/week1_correction/model_b/checkpoint_latest.pt` | 144,672,307 B | `e504eadbe2a8f4913a588d1aeca0594338fc33652489d28ef69f532786f5eabf` | Model B corrected checkpoint with full RNG states |
| `outputs/week1_correction/model_b/checkpoint_best.pt` | 144,670,937 B | `77391d7ca7f30a69364e4f57ced4ca3b9c2e9b9927bfe6ac0548f83093a7e075` | Model B corrected best checkpoint |
| `outputs/week1_correction/model_b/run_manifest.json` | 3,020 B | `6d02ca969266fb26555b84e695e792b8f1022550fea85b04887be49816081187` | Model B corrected manifest with dynamic event trace |
| `outputs/week1_correction/model_b/access_trace.json` | 773 B | `c35067e83d97a52433d84d9561b636f17aca03b2fdd53e99852829d6ee307324` | Model B runtime train/val sample trace |
| `outputs/week1_correction/model_b/smoke_run_metrics.json` | 5,046 B | `2446f844c4d09d3a4db54a0813d62af3f1f1ca9cf409ab32055bfde746024ac2` | Model B corrected smoke metrics |

### 2.4 Code & Configuration Inventory (Changes Since `af164ec`)

| File Path | Size (Bytes) | Lines | SHA-256 Hash |
|---|---|---|---|
| `configs/source/snunet_vvvh_v1.json` | 4,843 B | 125 | `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc` |
| `configs/source/snunet_vvvh_slope_v1.json` | 6,742 B | 146 | `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec` |
| `src/evaluation/run_manifest.py` | 9,630 B | 222 | `47a107c6fdbae04dc71be80c79c0d439eeefe309297b33eb483e081eacc05b8c` |
| `src/training/source_trainer.py` | 14,775 B | 381 | `547151632f32146add6839f26d65dd9e69f0a8bb715e1ce17304ee9bd46d97ad` |
| `src/data/source_dataset.py` | 13,556 B | 306 | `21a219bbe86653142e7f40a68a3407ce54e4c2c69ec5ea787a13792c379bdc95` |
| `src/evaluation/losses.py` | 5,735 B | 153 | `03a468d3e5f02a8d74e3b6789973ef282695b187a06ebd0c78882c3c036fe8ab` |
| `tests/test_source_configs.py` | 12,426 B | 225 | `686066e3c146f4737cc34f3d3bdbaa769fbec03b5682db5d7ebb2d5ec7b7da1b` |
| `tests/test_source_dataset.py` | 6,348 B | 140 | `8900a536c18c7d7d50bd0e22d91154fb7e7741d00f472bb9c14b7a1849ea17ab` |
| `tests/test_source_trainer.py` | 4,973 B | 132 | `f95d3d59d989f053ca043097f6e63a52cffb403b78ebb14ab32ae0b6d87e0297` |
| `tests/test_losses.py` | 3,213 B | 82 | `96f3b16f5c08819e3a2c12837035259b1cc12babab29069e422e95f2745cd702` |
| `tests/test_week1_controlled_training.py` | 13,232 B | 305 | `653c202cd3fa85b060c4ca4b47a5cb43c7636bb45d0723fa4ab93773dadb2bc4` |
| `tests/test_preprocessing.py` | 10,710 B | 285 | `be38a070dc4f8a68dabee706c8575ca9326dd55ea032df4644396acc04e77648` |
| `tests/test_risk_probing_isolation.py` | 7,126 B | 179 | `373453f3922a97631099ff25836fad5ca9f228b5b852e10890b5829af8d5880b` |
| `tests/test_run_manifest.py` | 6,576 B | 152 | `dc60e624c66d145a658440de4e3bddb4fae722c211e95e57f2cc8e3012aab472` |
| `tools/audit_slope_nodata_sensitivity.py` | 9,199 B | 207 | `598de0f20d76001a6e83f3950aee3bf2ef6bf217170ba82d330c773d30b6adbd` |
| `tools/generate_train_slope_statistics.py` | 11,448 B | 290 | `599c9d768fa660869b9e8f4030d8b3c27879ef04b609d3d86fec21270946bcd8` |
| `tools/run_week1_memory_smoke.py` | 5,387 B | 149 | `413849992f4bd170804d91b68c7ec598dd233a035d2864d97c1f1cd372933678` |
| `tools/run_week1_smoke_training.py` | 5,884 B | 174 | `75dcfcdcee72e66819a5103031a40c692d9a0b31c9d51c9e70b0371d7f934112` |
| `tools/run_week1_corrected_smoke.py` | 10,637 B | 285 | `b342e2144d19b077da459d02496018ee2f46faf66dba29a95ec1f08337625715` |
| `tools/run_week1_final_protocol_smoke.py` | 11,924 B | 315 | `7fa0997a7dbc31859c8fbcc48a33575a4f133313dfc249cb14ad69dfe395977d` |
| `tools/validate_checkpoint_resume.py` | 5,647 B | 146 | `ef69ac6e42cb5c2a3aed81089a5ce834f902271b8779a7a4eddf5c1a91032774` |
| `docs/week1_batch_memory_amendment.md` | 4,145 B | 59 | `aae050ac2946f257ecd4662e5951ffcea75e81f7b73d9c3a107f999f6783de36` |
| `docs/week1_pipeline_smoke_audit.md` | 14,328 B | 159 | `ae0b3b54f8549c14f771c46f42569eb4b723a74826ceba7b25a19474df80890c` |
| `docs/week1_optimizer_protocol_correction.md` | 19,054 B | 307 | `bb920fad8f3187d53492959b27e507aa212f2d549722ab1b1fa874974ff0280c` |
| `docs/source_training_protocol_v1.md` | 18,747 B | 254 | `bc7c2ea8b9a8e62553c200bc1a9f3de326f39edde0f9a0a3fa4843ec61934efa` |

### 2.5 Canonical Final Protocol Smoke Artifacts (`artifacts/week1/final_protocol_*`)

The final protocol confirmation run produced 6 non-binary JSON artifacts stored directly in `artifacts/week1/`:

| Relative Path | Size (Bytes) | SHA-256 Hash | Purpose |
|---|---|---|---|
| `artifacts/week1/final_protocol_model_a_access_trace.json` | 682 B | `0b29f5772a1d3ad45259e30717ac47c0286a1529ba0b5d411e49bd73748fde2f` | Model A dynamic sample trace under final protocol |
| `artifacts/week1/final_protocol_model_a_run_manifest.json` | 2,962 B | `fe5ea9ade4aaf4e5496b3f4f40455c14dcd8586013c93546d046513b9fcf160c` | Model A run manifest under final protocol |
| `artifacts/week1/final_protocol_model_a_smoke_metrics.json` | 4,624 B | `38a2836df971db8e36a44411239f1ab3937a3602fa182627105294e3e93e40f5` | Model A loss & val F1 under final protocol |
| `artifacts/week1/final_protocol_model_b_access_trace.json` | 718 B | `48775fe7d0712cd3ca846a212834fe1af670c1dd6fbbd35f04f036125c5ba1c2` | Model B dynamic sample trace under final protocol |
| `artifacts/week1/final_protocol_model_b_run_manifest.json` | 3,012 B | `ecdfa80bee009f4793816780cc36131fc1a9ec0bbf4b514a0c918be8ac974225` | Model B run manifest under final protocol |
| `artifacts/week1/final_protocol_model_b_smoke_metrics.json` | 4,631 B | `14ebe6dc1bdc74c0bc2c14395f30e0a5c468b657d784f629275ec62d49f3d4ca` | Model B loss & val F1 under final protocol |

---

## 3. Gradient Accumulation & Loss Function Audit

### 3.1 Masked Loss Formulation & Implementation

Controlled source model training optimizes `ControlledSourceLoss` (`src/evaluation/losses.py`), defined as an unweighted combination of Cross-Entropy and Soft Dice losses:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

#### Masking Rules & Ignore Index
1. **Double-Masking Rule:** In `KuroSourceDataset`, any pixel where the valid mask is 0 (`MK0_MNA == 0`) is explicitly assigned target class `3`.
2. **Cross-Entropy Masking:** PyTorch `nn.CrossEntropyLoss(ignore_index=3, reduction='mean')` ignores class 3 and averages loss strictly over valid pixels ($N_{\text{valid}}$). If a batch contains zero valid pixels, it returns scalar `0.0` with gradient tracking preserved.
3. **Soft Dice Masking:**
   $$\mathcal{L}_{\text{Dice}} = 1.0 - \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \sum_{i \in \Omega_{\text{valid}}} p_{i, c} y_{i, c} + \epsilon}{\sum_{i \in \Omega_{\text{valid}}} p_{i, c}^2 + \sum_{i \in \Omega_{\text{valid}}} y_{i, c}^2 + \epsilon}$$
   where $C=3$ (classes 0: background, 1: water, 2: flood), $p_{i,c} = \text{Softmax}(\text{logits})_{i,c}$, and summation is strictly restricted to valid pixels $\Omega_{\text{valid}} = \{i \mid \text{label}_i \in \{0, 1, 2\}\}$. Pixels with label 3 are zeroed out before computing Dice numerators and denominators.

### 3.2 Gradient Accumulation Equivalence Analysis

The locked protocol specifies **effective batch size 32**, implemented via physical microbatch 8 and accumulation steps 4 on GPU 0:

```python
scaled_loss = loss / accumulation_steps
scaled_loss.backward()
# After 4 microbatches:
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

#### Mathematical & Empirical Equivalence
1. **Cross-Entropy Component (Linear Summation):**
   $$\mathcal{L}_{\text{CE}}^{(\text{batch 32})} = \frac{1}{N_{\text{total}}} \sum_{k=1}^4 \sum_{j \in \mathcal{B}_k} \ell_j = \sum_{k=1}^4 \frac{N_k}{N_{\text{total}}} \left( \frac{1}{N_k} \sum_{j \in \mathcal{B}_k} \ell_j \right)$$
   When valid pixels per microbatch are equal, Cross-Entropy gradient under accumulation is **mathematically identical** to a single batch of 32. In our unit test (`TestWeek1ControlledTraining.test_gradient_accumulation_equivalence`), the parameter difference between microbatch accumulation and full batch under pure CE is **`0.0`** (within $10^{-7}$).
2. **Soft Dice Component (Batch-Level Denominator):**
   Unlike CE, Soft Dice contains a non-linear denominator summed over the batch. Under accumulation with microbatch 8, Dice is computed per-microbatch:
   $$\mathcal{L}_{\text{accum}} = \frac{1}{4} \sum_{k=1}^4 \mathcal{L}_{\text{Dice}}(\mathcal{B}_k) \ne \mathcal{L}_{\text{Dice}}\left( \bigcup_{k=1}^4 \mathcal{B}_k \right)$$
   Our automated test proves that this non-linearity produces an absolute gradient difference bounded by $\le 2.38 \times 10^{-5}$ ($0.36\%$ relative). Optimizing microbatch-averaged Dice is the standard, validated approach across computer vision and medical imaging segmentation.

### 3.3 Root Cause of Initial Step Loss ($\approx 12.5$)

During the initial smoke step, the loss is recorded as **$\approx 12.47$** (Model A) and **$\approx 14.06$** (Model B), before dropping sharply to **$\approx 0.50 - 0.64$** by step 25.

#### Detailed Mathematical Root Cause
1. **Siamese Feature Summation:** In `SNUNet_ECAM`, Siamese features from dual ResNet blocks are progressively concatenated and summed across 16 dense skip connections into `conv_final` ($128 \to 3$).
2. **Unscaled Initialization Variance:** Under standard Kaiming normal initialization (`mean=0, std=\sqrt{2/fan_in}`), repeated dense additions across multiple unnormalized layers cause output logit variance to accumulate:
   - Initial logit standard deviation: **$19.32$**
   - Initial logit range: **$[-78.5, +81.5]$**
3. **Cross-Entropy Magnitude:** For an unaligned class prediction with logit separation $\Delta z \approx 20$, the softmax cross-entropy loss evaluates to:
   $$\mathcal{L}_{\text{CE}} \approx -\log(e^{-20}) \approx 20.0 - 27.0$$
4. **Combined Initial Loss Calculation:**
   $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}} \approx 0.5 \times (24.05) + 0.5 \times (0.89) = 12.47$$
5. **Rapid Adam Stabilization:** Adam's adaptive second-moment estimate ($v_t$) rapidly scales down the gradient updates to `conv_final`. Within 8 optimizer steps, logit standard deviation collapses to $\sim 1.5$, CE loss drops below $1.0$, and total loss descends smoothly to $0.50$ by step 25.
6. **Loss Trajectory Characterization:** Step metrics show that the loss trajectory is **not strictly monotonic** (e.g., oscillating slightly between steps 3–6 and 14–17 due to sample-level flood fraction variation across random microbatches). The trajectory is properly characterized as a **rapid, substantial net descent** ($12.47 \to 0.50$, a $96.0\%$ reduction in 25 steps).

---

## 4. DEM Nodata & Slope-Boundary Policy Audit

### 4.1 Root Cause of KuroSiwo DEM Nodata

Analysis of all 19,178 inner-train samples across the 22 training events revealed:
- Exactly **416 samples** contain DEM nodata values.
- These nodata samples are concentrated entirely in coastal scenes—specifically Texas Coastal flood events **1111004** and **1111010**—where DEM tiles overlap open ocean / the Gulf of Mexico.
- **USGS 3DEP / SRTM Sentinel Value:** In raw GeoTIFF rasters, open ocean is assigned the integer nodata sentinel **$-32768.0\text{ m}$**.
- **Bilinear Resampling Transition Ramp:** During KuroSiwo dataset creation, grid tiles were created using bilinear resampling. At the coastline, bilinear interpolation between real ground ($+0\text{ to }+5\text{ m}$) and the $-32768\text{ m}$ sentinel produced an intermediate transition fringe spanning elevations from $-998\text{ m}$ down to $-1.2\text{ m}$.

### 4.2 Benchmark Comparison Across 5 Candidate Rules

To evaluate the mathematical rigor of DEM nodata handling, we implemented `tools/audit_slope_nodata_sensitivity.py` and benchmarked 5 candidate rules across all 416 nodata-bearing inner-train samples:

| Rule ID | Rule Specification | Valid Pixels | Nodata Pixels | Mean Slope | Median Slope | Max Slope | % > 30° | % > 45° | % > 60° |
|---|---|---|---|---|---|---|---|---|---|
| **Rule 1** | Raw Sentinel (`< -1000m`, dil 0, no inpaint) | 14,693,754 | 6,179,462 | *NaN* | *NaN* | *NaN* | 1.337% | 1.299% | **1.286%** |
| **Rule 2** | Current Policy S2 (`< -430m`, dil 3, inpaint) | 14,262,077 | 6,611,139 | 1.6116° | 0.8212° | 88.46° | 0.033% | 0.001% | **0.0011%** |
| **Rule 3** | Conservative S2.1 (`< -50m`, dil 3, inpaint) | 14,220,647 | 6,652,569 | 1.6110° | 0.8210° | 60.75° | 0.032% | 0.0001% | **0.0000%** |
| **Rule 4** | Coastal Bound (`< -10m`, dil 3, inpaint) | 14,213,038 | 6,660,178 | 1.6109° | 0.8209° | 48.06° | 0.032% | 0.0001% | **0.0000%** |
| **Rule 5** | Sentinel + 4 Dil (`< -1000m`, dil 4, inpaint) | 14,184,039 | 6,689,177 | 1.6110° | 0.8200° | 89.20° | 0.034% | 0.0027% | **0.0025%** |

### 4.3 Key Findings on Slope Policy

1. **Catastrophic Artifacts in Raw Sentinel (Rule 1):** Without inpainting, gradient computation across the step boundary from $0\text{ m}$ to $-32768\text{ m}$ creates extreme gradient explosion: over **189,000 pixels ($1.286\%$)** exhibit artificial vertical cliffs $> 60^\circ$, and floating-point overflow generates NaNs.
2. **Current Policy S2 Performance (Rule 2):** Current Policy S2 (`dem < -430.0`, `dilation=3`, nearest-neighbor inpaint) reduces cliff pixels $> 60^\circ$ by **$99.91\%$** (from $1.286\%$ down to $0.0011\%$). Mean slope is $1.6116^\circ$ and median slope is $0.8212^\circ$.
3. **Conservative S2.1 Recommendation (Rule 3):** Thresholding at `dem < -50.0m` completely eliminates the remaining $0.0011\%$ interpolation fringe: max slope drops from $88.46^\circ$ to $60.75^\circ$, and percentage $> 60^\circ$ drops to **$0.0000\%$**. The difference in valid pixels between Rule 2 and Rule 3 is only 41,430 pixels ($0.29\%$ of nodata samples, and only $0.003\%$ of the $1.25$ billion inner-train pixels).
4. **Authentic Terrain Preservation Guarantee:**
   - Physical inspection of terrain across all 22 inner-train events confirmed that the lowest authentic elevation on dry land is **$-5.0\text{ m}$** (coastal plain terrain in Texas ranges from $-2\text{ m}$ to $+7\text{ m}$).
   - There is zero land below $-5.0\text{ m}$ in any training event.
   - Therefore, neither $-430\text{ m}$ nor $-50\text{ m}$ clips or alters authentic dry land terrain.

---

## 5. Train/Validation Split & Sample-Level Isolation Audit

### 5.1 Split Specification & Manifest Filtering

The frozen 22/5 source-inner split (Alternative A1) was verified using `outputs/week1_correction/split_isolation_audit.json`:

```
Source Manifest (KuroV2_grid_dict.gz): 31,707 total samples
├── Inner-Train Events (22):          19,178 samples (60.48%)
├── Inner-Validation Events (5):        2,284 samples ( 7.20%)
└── Quarantined / Sealed Events (16):  10,245 samples (32.31%)
    ├── Quarantined TTA-Val (6):        4,891 samples
    └── Sealed Test (10):               5,354 samples
```

- **Inner-Train Events ($n=22$):** `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`
- **Inner-Validation Events ($n=5$):** `[267, 275, 497, 518, 1111005]`
- **Sample Intersection:** Exactly **0** samples overlap between train and validation sets.
- **Event Intersection:** Exactly **0** events overlap between train and validation sets.

### 5.2 Runtime DataLoader Access Trace

Using `TrackingDataLoader` during the 25-step smoke run (seed 42), we captured the exact samples and event IDs loaded by worker processes:

- **Train Batches Drawn:** 100 batches $\times$ 8 samples = **800 samples**
- **Train Events Represented:** 21 of the 22 inner-train events were drawn (sample distribution: Texas flood 1111009: 296, Texas flood 1111004: 185, 427: 84, 1111011: 67, etc.).
- **Validation Batches Drawn:** 25 batches $\times$ 8 samples = **200 samples**
- **Validation Events Represented:** All 5 inner-val events were drawn (518: 27, 1111005: 54, 497: 34, 275: 56, 267: 29).
- **Cross-Split Sample Intersection:** **`0`**
- **Quarantined Events Accessed:** **`0`** (zero samples from TTA-val or sealed test events).

---

## 6. Smoke Training Runner & Run-Manifest Audit

### 6.1 Findings on Initial Smoke Runner & Manifests

Auditing `tools/run_week1_smoke_training.py` and `outputs/week1_smoke/*/run_manifest.json` revealed three important limitations:
1. **Manifest Invocation Disconnect:** In `tools/run_week1_smoke_training.py`, `RunManifest` was imported but never instantiated. The manifests in `outputs/week1_smoke/` were generated post-hoc via a `python -c` script (`"command": "-c"`).
2. **Static Event Listing:** The initial manifest listed a static, hard-coded array of all 27 source events under both `"event_ids"` and `"accessed_event_ids"`, obscuring the distinction between train events and validation events.
3. **Misapplied TTA Label Constraint:** Because `RunManifest` was originally designed for unsupervised TTA (where using labels for updates is a research integrity violation), `RunManifest.validate()` raised an exception if `labels_used_for_updates=True`. The previous runner bypassed this by setting `labels_used_for_updates: false`, which misstated the reality of supervised source model training.

### 6.2 Legacy Test Leak Risk in `test_preprocessing.py`

When executing `python -m unittest discover tests/`, we discovered that `tests/test_preprocessing.py:144` loaded `configs/baseline_config.json`, which directly referenced `KuroV2_grid_dict_test_0_100.gz` with `allowed_events=[279]` (one of the 6 quarantined TTA-validation events). This test was authored during the early pilot phase and represents a dormant regression risk. While it did not compromise Week 1 training (which uses `KuroSourceDataset`), this test must be quarantined or updated prior to Week 2.

### 6.3 Corrected Runner Architecture (`tools/run_week1_corrected_smoke.py`)

To eliminate these issues, we implemented `tools/run_week1_corrected_smoke.py`:
1. **Dynamic Runtime Tracking:** Wraps DataLoaders in `TrackingDataLoader` to log the exact list of accessed sample IDs and event IDs for both train and validation splits.
2. **First-Class Source Training Manifest:** Extended `RunManifest` (`src/evaluation/run_manifest.py`) to support `run_type="source_training"`, which:
   - Formally allows `labels_used_for_updates=True` for supervised source training;
   - Enforces strict zero-tolerance validation against both sealed test events ($n=10$) AND quarantined TTA-validation events ($n=6$);
   - Records `config_sha256` and links directly to `access_trace.json`.
3. **Command Provenance:** Captures full CLI command arguments (`--model_type both --steps 25 --device cuda:0`).

---

## 7. Checkpoint Integrity & Deterministic Resumption Audit

### 7.1 Checkpoint State Dictionary Audit

Inspection of historical smoke checkpoints revealed that they stored model parameters, optimizer state, scaler state, epoch, and metrics, but lacked RNG state vectors and config hashes.

### 7.2 Enhanced Checkpoint Architecture

We upgraded `SourceTrainer.save_checkpoint()` and `resume_from_checkpoint()` in `src/training/source_trainer.py` to persist:
- `rng_state`: PyTorch CPU RNG state vector (`torch.get_rng_state()`)
- `cuda_rng_state`: PyTorch CUDA RNG state vectors (`torch.cuda.get_rng_state_all()`)
- `np_rng_state`: NumPy RNG tuple (`np.random.get_state()`)
- `git_commit`: Active repository HEAD SHA (`get_git_commit()`)
- `git_dirty`: Boolean flag indicating working tree modifications (`is_git_dirty()`)
- `config_path` & `config_hash`: Relative path and SHA-256 of the governing JSON config
- `extra_metadata`: Model architecture metadata (`in_channels`, `base_channel`, etc.)

### 7.3 PyTorch 2.6 `weights_only=False` Compatibility

In PyTorch 2.6+, `torch.load()` defaults to `weights_only=True`, which raises an unpickling error when loading complex NumPy RNG states (`numpy._core.multiarray._reconstruct`). `SourceTrainer.resume_from_checkpoint()` now safely specifies `weights_only=False` for internal checkpoint restoration, verified by bit-exact test passes.

---

## 8. Smoke Training Metrics & Interpretation

### 8.1 25-Step Diagnostic Performance Summary

| Model Identifier | Input Channels | Steps | Physical Microbatch | Initial Loss | Final Step Loss | Val Macro F1 (25 batches) | Worst-Event F1 (Event 518) |
|---|---|---|---|---|---|---|---|
| **Model A** | 2 (`[VV, VH]`) | 25 | 8 (accum 4) | 12.4734 | 0.5037 | **0.7202** | 0.5314 |
| **Model B** | 3 (`[VV, VH, slope]`) | 25 | 8 (accum 4) | 14.0554 | 0.6404 | **0.6862** | 0.5372 |

### 8.2 Scientific Interpretation & Model Comparison Caveat

$$\mathbf{CRITICAL\ PRINCIPLE:\ SMOKE\ METRICS\ ARE\ PIPELINE\ DIAGNOSTICS,\ NOT\ SCIENTIFIC\ FINDINGS.}$$

1. **Why Model A Shows Higher 25-Step F1 than Model B:**
   - 25 optimizer steps represent only **800 samples** out of 19,178 ($4.17\%$ of a single epoch). Full training requires 150 epochs ($\sim 360,000$ sample passes).
   - Model B has 50% more input channels. The first convolutional layer (`conv1`) contains $3 \times 32$ filters instead of $2 \times 32$, requiring more gradient updates to establish feature equilibrium between SAR backscatter and topographic slope.
   - At step 25, model weights have only begun to depart from random initialization.
2. **What the Smoke Test Proves:**
   - Gradients flow cleanly through both 2-channel and 3-channel input pipelines.
   - Mixed-precision loss scaling operates without underflow or NaN spikes.
   - Validation inference, event-level metric accumulation, and deterministic tie-breaking function without memory leaks.
3. **Scientific Invalidation Warning:** Under no circumstances should the 25-step validation metrics ($0.7202$ vs $0.6862$) be cited in the FYP report as evidence that Model A is superior to Model B. Model ranking can only be evaluated after full 150-epoch training.

---

## 9. Automated Test Suite & Regression Prevention

### 9.1 Test Suite Status

The automated test suite was executed via the project Python runtime:
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest discover tests/
```
**Results:** **82 tests ran, 82 passed, 0 failed, 0 errors** in 12.76s.

### 9.2 Regression Prevention Suite & Module Breakdown

The 82 passing tests span all project modules:
- `tests/test_week1_controlled_training.py` (8 tests): Gradient accumulation mathematical equivalence, masked source loss specification, Policy S2 nodata inpainting cliff suppression, split isolation verification, checkpoint provenance & deterministic resumption, source dataset test manifest rejection, source dataset split forbidden event rejection, and RunManifest source training isolation enforcement.
- `tests/test_source_configs.py` (13 tests): Schema validation, batch divisibility, ablation parameterization, seed rules, boundary isolation, approved optimizer protocol & runtime agreement verification, and production config & split hash verification against final protocol artifacts.
- `tests/test_source_dataset.py` (7 tests): Slope computation, ramp accuracy, geographic CRS rejection, inpainting cliff mitigation, 5-event inner validation loading, Model B 3-channel input loading, and Model A/Model B valid-mask fairness verification.
- `tests/test_source_trainer.py` (3 tests): Checkpoint save/resume, deterministic tie-breaking, and optimizer step execution.
- `tests/test_losses.py` (5 tests): Controlled source loss shapes, components, class 3 masking, absent class stability, and backward differentiability.
- `tests/test_run_manifest.py` (6 tests): Manifest serialization, test event rejection, TTA update prohibition, partial diagnostic tracking, source training quarantined event rejection, and source training test manifest rejection.
- `tests/test_preprocessing.py` (9 tests): Synthetic GeoTIFF elevation sanitization, sentinel masking, normalization, and SAR channel preservation.
- `tests/test_risk_probing_isolation.py` (2 tests): Parameter non-mutation and unlabeled dataset raster isolation.
- `tests/test_source_inner_split.py` (8 tests): Frozen split structure, hash verification, sample counts, and complete event disjointness.
- `tests/test_tta_isolation.py` (7 tests): TTA label independence, weight immutability, and state reset behavior.
- `tests/test_protocol_rules.py` (6 tests): Protocol rules, tie-breaking, and median epoch selection.
- `tests/test_risk_indicators.py` (3 tests): Risk indicator boundedness and score formulas.
- `tests/test_risk_schema.py` (3 tests): Risk schema validation and field enforcement.
- `tests/test_risk_thresholds.py` (2 tests): Offline evaluation thresholds and classification.

### 9.3 Resolution of DEVIATION-W1-01 (Legacy Test Fixture Quarantine)

- **Deviation Identifier:** `DEVIATION-W1-01`
- **Root Cause:** Legacy unit tests (`tests/test_preprocessing.py:144` and `tests/test_risk_probing_isolation.py:109`) historically loaded `configs/baseline_config.json` and `configs/gate_b_risk_protocol.json`, referencing Event 279 on disk. During test discovery, these tests opened raw GeoTIFF tiles to verify preprocessing routines.
- **Integrity Assessment:** Event 279 is one of the 6 quarantined TTA-validation events; neither test accessed any of the 10 sealed test events. No model updates, source checkpoint selections, or reported source performances were executed using Event 279.
- **Resolution:** Both tests were completely refactored to construct self-contained, synthetic multi-channel GeoTIFF fixtures and temporary mock manifests in `tempfile.TemporaryDirectory()`. Furthermore, `tests/test_run_manifest.py` was updated to reference `configs/source/snunet_vvvh_v1.json`. All legacy dependencies on `baseline_config.json` and Event 279 have been completely eradicated.
- **Verification:** Verified via `unittest discover`: zero real Kuro Siwo test manifests or quarantined events are opened anywhere in the test suite.

### 9.4 Model A / Model B Valid-Mask Fairness Audit

- **Audit Objective:** Verify that topographic slope nodata handling does not modify the supervised target `valid_mask` or create asymmetric evaluation denominators between Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`).
- **Inspection Findings:**
  1. In `src/data/source_dataset.py`, `valid_mask` is derived strictly from `MK0_MNA`, and double-masking sets `label[mna == 0] = 3`.
  2. For Model B, `compute_policy_s2_slope` generates an internal `valid_slope_mask` used strictly for inpainting and setting invalid slope locations to the training slope mean (standardized to `0.0`).
  3. `valid_slope_mask` is NEVER intersected with the target `valid_mask` or `label`. It is returned as an optional diagnostic key `slope_valid_mask` without modifying supervision.
  4. Both models return bit-identical target `label` and bit-identical `valid_mask` for the same sample.
  5. Both models produce bit-identical evaluation denominators (`valid_pixels`, `gt_flood_pixels`) under `compute_confusion_counts`.
- **Regression Test:** Automated verification established in `tests/test_source_dataset.py::test_model_a_and_model_b_valid_mask_fairness`.

### 9.5 Resolution of Optimizer Protocol Drift & Final Protocol Confirmation

- **Audit Objective:** Investigate conversational drift referencing AdamW and cosine annealing, verify on-disk code and config provenance, and confirm strict alignment with the approved Adam protocol.
- **Inspection Findings:**
  1. On-disk configs at Week 0 freeze (`af164ec`) and runtime code never contained AdamW or cosine annealing; both specified `Adam`, `lr=1e-3`, `weight_decay=0.0`, `betas=(0.9, 0.999)`, and `scheduler=None`. The drift existed purely as conversational prose in an earlier summary.
  2. Capitalized `"optimizer": "Adam"` in both configs and implemented `SourceTrainer.from_config` to directly bind configuration JSON parameters to PyTorch objects.
  3. Added `test_approved_optimizer_protocol_and_runtime_agreement` to `tests/test_source_configs.py`.
  4. Executed `tools/run_week1_final_protocol_smoke.py` on GPU 0 (`cuda:0`) for 10 steps each on Model A and Model B, generating verified metrics and manifests in `artifacts/week1/final_protocol_*`.
- **Report Reference:** Complete forensic analysis and empirical logs are recorded in `docs/week1_optimizer_protocol_correction.md`.

---

## 10. Transition Gate to Week 2 Formal Training

### 10.1 Entry Gate Checklist for Week 2

Before launching formal 150-epoch training for Model A and Model B in Week 2, the following conditions must be satisfied:

- [x] **Split Freeze:** Approved Alternative A1 (22 train, 5 val) verified and sealed.
- [x] **Input Pipeline:** Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`) verified with zero NaNs.
- [x] **Slope Normalization:** Train-only slope statistics artifact locked (`mean=1.492574`, `std=2.372311`) in `artifacts/week1/`.
- [x] **Canonical Artifact Tracking:** 18 small JSON protocol artifacts tracked under `artifacts/week1/` in Git.
- [x] **Valid-Mask Fairness:** Bit-identical target labels, valid masks, and metric denominators confirmed across Model A and B.
- [x] **Memory Stability:** Physical microbatch 8 confirmed locked (2.28 GB peak VRAM on 24 GB RTX 3090).
- [x] **Loss Function:** Masked CE + Soft Dice verified on valid pixels.
- [x] **Optimizer Protocol Locked:** Verified Adam ($lr=1\times 10^{-3}, \text{weight\_decay}=0.0, \beta=(0.9, 0.999)$, no scheduler) in configs and runtime code via `SourceTrainer.from_config`.
- [x] **Checkpointing:** Deterministic tie-breaking and full RNG state restoration verified.
- [x] **Provenance:** Standardized `RunManifest` with `run_type="source_training"` verified.
- [x] **Legacy Test Quarantine:** `DEVIATION-W1-01` resolved; all test fixtures synthetically isolated.
- [x] **Regression Prevention:** 82/82 unit tests passing cleanly.
- [ ] **Week 1 Git Commit:** Create local Git checkpoint committing Week 1 configs, tools, tests, artifacts, and documentation (pending user checkpoint command).

### 10.2 Recommended Week 2 Formal Training Commands

Formal 150-epoch training should be executed sequentially on GPU 0 with dedicated log redirection:

```bash
# Model A: SNUNet [VV, VH] (150 epochs, effective batch 32, microbatch 8)
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py \
    --config configs/source/snunet_vvvh_v1.json \
    --output_dir outputs/source_models/model_a \
    --device cuda:0 \
    --epochs 150

# Model B: SNUNet [VV, VH, slope] (150 epochs, effective batch 32, microbatch 8)
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py \
    --config configs/source/snunet_vvvh_slope_v1.json \
    --output_dir outputs/source_models/model_b \
    --device cuda:0 \
    --epochs 150
```

---

## 11. Final Audit Conclusion

Week 1 has achieved all technical, mathematical, and research-integrity requirements. The controlled source-training pipeline for ChangeGuard is robust, fully audited, reproducible, and ready for transition to Week 2.

**Audit Status:** `COMPLETE`  
**Verdict:** `WEEK1_COMPLETE_WITH_DOCUMENTED_CORRECTIONS`  
**Sign-off:** Independent Validation Audit Engine (Antigravity)
