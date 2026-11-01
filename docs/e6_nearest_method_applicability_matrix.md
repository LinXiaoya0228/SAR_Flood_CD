# ChangeGuard Work Package E6.3: Nearest-Method Applicability Matrix

**Status:** COMPLETE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**HEAD at Decision:** `1311bad`
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.3 (commit `24ffcc9`)
**Branch:** C — Empirical Study
**Inspection Protocol:** Primary sources only; all URLs, DOIs, repositories, versions, and licenses verified as of 2026-10-21. Unverified fields are marked `UNKNOWN` with access dates.

> [!IMPORTANT]
> **Summary of Disposition:**
> - **CoTTA:** `SCHEDULE AS FORMAL BASELINE PROTOCOL` (Protocol created in `docs/protocols/e6_cotta_compatibility_protocol_v1.md`; implementation deferred to E7).
> - **EATA & SAR:** `IMPLEMENTED AND EVALUATED` (Historically completed in E1.4 across 12 streams each; retained as audited historical baselines).
> - **Source-Only, AdaBN, Tent:** `IMPLEMENTED AND EVALUATED` (Historically completed in E1.4; retained as primary controls).
> - **NOTE, RoTTA, TTA-Monitor, CIM-TTDA, PA-TTA (Huang et al., 2026), ASR:** `DISCUSS ONLY` (Detailed technical justifications provided).

---

## 1. ChangeGuard Target Task Specification

For applicability assessment, the ChangeGuard deployment task is defined as:

| Specification Dimension | ChangeGuard Plan V3 Target Value |
| :--- | :--- |
| **Prediction Task** | Dense 3-class semantic change segmentation: Class 0 (Land), Class 1 (Permanent Water), Class 2 (Flood Water) |
| **Architecture** | SNUNet-ECAM (Siamese UNet backbone with Ensemble Channel Attention Modules) |
| **Input Modality** | Bi-temporal paired Synthetic Aperture Radar (SAR) rasters: Sentinel-1 GRD ($T_1$ pre-flood, $T_2$ post-flood; dual-pol VV+VH) |
| **Class Distribution** | Extreme spatial class imbalance: Land ($\approx 95\%$) $\gg$ Permanent Water ($\approx 3\%$) $\gg$ Flood ($\approx 2\%$) |
| **Spatial Resolution** | $512 \times 512$ pixel patch pairs; valid sensory pixels masked by binary sensor validity raster $M_{NA}$ |
| **Source Data Access** | Strictly source-free during adaptation (zero source images/labels accessible at deployment) |
| **Target Label Access** | Strictly unsupervised streaming ($0.0\%$ target labels accessible; label-guarded batches) |
| **Prediction Timing** | `PRE_UPDATE` primary prediction: Predictions frozen on disk before computing gradients or applying updates |
| **Adaptation Scope** | Online streaming test-time adaptation across sequential batches (batch size = 16) |
| **Reset Semantics** | Episodic per-event reset (default operational setting) and continual multi-event adaptation (comparative setting) |
| **Evaluation Unit** | 6 development events ($N = 2,826$ samples, 179 physical batches); 10 sealed test events (strictly quarantined) |

---

## 2. Comprehensive Nearest-Method Applicability Matrix

### 2.1 CoTTA (Continual Test-Time Adaptation)

- **Exact Paper Title:** Continual Test-Time Domain Adaptation
- **Authors:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai
- **Year & Venue / Status:** 2022, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), pp. 7201–7211
- **DOI / Official Publisher URL:** [10.1109/CVPR52729.2022.00707](https://doi.org/10.1109/CVPR52729.2022.00707) | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
- **Official Repository URL:** [https://github.com/qinenergy/cotta](https://github.com/qinenergy/cotta)
- **Inspected Repository Version / Commit:** GitHub default branch commit `c4cfbd3` (inspected 2026-10-21)
- **License:** MIT License
- **Original Task & Modality:** Continual image classification on corrupted streams (CIFAR-10-C, CIFAR-100-C, ImageNet-C) and driving scene semantic segmentation under shifting weather (Cityscapes $\to$ ACDC); optical 2D RGB imagery
- **Source-Data Assumptions:** Source-free during adaptation. Pretrained source weights $\theta_0$ are stored in memory and accessed for stochastic source restoration.
- **Retraining Assumptions:** Zero source retraining. Operates on off-the-shelf pretrained models.
- **Adaptation State:** Continual adaptation state. Student network $\theta_s$ is updated via gradient descent; Teacher network $\theta_t$ is updated via Exponential Moving Average (EMA, momentum $\alpha = 0.999$).
- **Prediction Timing:** `PRE_UPDATE` primary prediction: Augmentation-averaged teacher output serves as both the target prediction and pseudo-label for the current batch before student backpropagation.
- **Deployment-Time Labels / Metadata:** Zero target labels. No domain boundary indicators.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Error Accumulation Reduction:* Mean Teacher EMA smoothing combined with multi-view augmentation-averaged predictions.
  2. *Catastrophic Forgetting Prevention:* Stochastic Source Restoration (SSR): At each step, every trainable parameter has a probability $p = 0.01$ of being reset bit-exactly to its initial source value $\theta_0$.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** High methodological compatibility. Augmentation averaging and teacher EMA can be applied directly to SNUNet-ECAM.
- **Deviations Required:**
  1. Dense pixel-level cross-entropy over valid sensory mask $M_{NA} = 1$ with threshold $\tau = 0.90$.
  2. Joint spatial augmentation across paired SAR channels ($x_A, x_B$) to preserve temporal pixel registration.
  3. Evaluation under both native continual mode and deployment-standard per-event reset mode.
- **Whether Deviations Preserve Defining Method:** Yes. Teacher EMA, augmentation averaging, and stochastic source restoration are fully preserved.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5` (Deferred in E1/E2 in favor of standalone consistency self-training).
- **Future Role under Branch C:** `SCHEDULE AS FORMAL BASELINE PROTOCOL`. Protocol specified in `docs/protocols/e6_cotta_compatibility_protocol_v1.md`.
- **Limitations of Project Formulation:** Official CoTTA uses random color/contrast jittering and Gaussian blur on optical RGB images; SAR inputs cannot tolerate color jittering without destroying radar cross-section calibration, restricting augmentations strictly to geometric transforms (flips).

---

### 2.2 EATA (Efficient Test-Time Adaptation without Forgetting)

- **Exact Paper Title:** Efficient Test-Time Model Adaptation without Forgetting
- **Authors:** Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan
- **Year & Venue / Status:** 2022, International Conference on Machine Learning (ICML 2022), PMLR 162:16888–16905
- **DOI / Official Publisher URL:** [https://proceedings.mlr.press/v162/niu22a.html](https://proceedings.mlr.press/v162/niu22a.html) | [arXiv:2204.02610](https://arxiv.org/abs/2204.02610)
- **Official Repository URL:** [https://github.com/mr-eggplant/EATA](https://github.com/mr-eggplant/EATA)
- **Inspected Repository Version / Commit:** GitHub default branch commit `8c3b7a5` (inspected 2026-10-21)
- **License:** MIT License
- **Original Task & Modality:** Out-of-distribution image classification on corrupted benchmarks (CIFAR-10-C, CIFAR-100-C, ImageNet-C); optical RGB imagery
- **Source-Data Assumptions:** Requires access to source training data once prior to test deployment to compute diagonal Fisher Information Matrix (FIM). Source-free during adaptation.
- **Retraining Assumptions:** Zero retraining. Requires offline forward-backward pass on source data to generate Fisher diagonal.
- **Adaptation State:** Updates Batch Normalization affine parameters ($\gamma, \beta$) online via entropy loss regularized by parameter displacement weighted by Fisher information: $\frac{\alpha}{2} \sum_i F_i (\theta_i - \theta_{0,i})^2$.
- **Prediction Timing:** `PRE_UPDATE` primary prediction.
- **Deployment-Time Labels / Metadata:** Zero target labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Reliability Filtering:* Discards uninformative or noisy samples where predictive entropy exceeds threshold $E_0$.
  2. *Redundancy Filtering:* Discards samples whose predicted distribution is within distance $\epsilon$ of moving-average predictions.
  3. *Fisher Regularization:* Restrains adaptation drift on parameters deemed critical for source task performance.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Fully compatible. Implemented in `src/changeguard/baseline_adapters.py` (`E1EATAAdapter`).
- **Deviations Required:** Patch-level valid-pixel entropy calculation; Fisher diagonal computed on source SAR training patches.
- **Whether Deviations Preserve Defining Method:** Yes. Active sample selection, redundancy filtering, and Fisher regularization are fully preserved.
- **Historical Project Status:** `IMPLEMENTED AND EVALUATED` (Completed across 12 streams in Work Package E1.4; audited in `docs/e1_4_matched_baselines_closeout.md`).
- **Future Role under Branch C:** `RETAINED AS AUDITED HISTORICAL BASELINE BENCHMARK`. Provides reference evidence across 12 streams (Seeds 0, 1, 2 × Orders A, B × Reset/Continual).
- **Limitations of Project Formulation:**
  Under the upstream entropy threshold $E_0 = 0.40 \ln(3) \approx 0.4394$, background land dominance causes patch-level entropy to remain uniformly below threshold on the 6 development events, resulting in zero gradient updates applied across all 12 streams. EATA numerically collapsed to AdaBN (Macro F1 = 0.5793 vs AdaBN 0.5793).

---

### 2.3 SAR (Towards Stable Test-Time Adaptation in Dynamic Wild World)

- **Exact Paper Title:** Towards Stable Test-Time Adaptation in Dynamic Wild World
- **Authors:** Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Mingkui Tan
- **Year & Venue / Status:** 2023, International Conference on Learning Representations (ICLR 2023, Notable-Top-5% / Oral)
- **DOI / Official Publisher URL:** [https://openreview.net/forum?id=g2YraF75Tj](https://openreview.net/forum?id=g2YraF75Tj) | [arXiv:2302.12400](https://arxiv.org/abs/2302.12400)
- **Official Repository URL:** [https://github.com/mr-eggplant/SAR](https://github.com/mr-eggplant/SAR)
- **Inspected Repository Version / Commit:** GitHub default branch commit `b67c1e3` (inspected 2026-10-21)
- **License:** BSD 3-Clause License
- **Original Task & Modality:** Robust classification under wild test conditions (mixed corruptions, small batch size, label shift) on ImageNet-C, ImageNet-R; optical RGB imagery
- **Source-Data Assumptions:** Source-free during adaptation. Pretrained source weights $\theta_0$ stored for model recovery reset.
- **Retraining Assumptions:** Zero retraining required for standard BN affine adaptation. (Authors recommend GroupNorm/LayerNorm models to eliminate BN estimation error, which would require retraining).
- **Adaptation State:** Updates BN affine parameters using Sharpness-Aware Minimization (SAM); tracks loss moving average; resets model to $\theta_0$ if loss drops below recovery threshold $e_m$.
- **Prediction Timing:** `PRE_UPDATE` primary prediction.
- **Deployment-Time Labels / Metadata:** Zero target labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Reliability Filtering:* Filters out large-gradient / high-entropy samples ($H(x) \ge E_0$).
  2. *Sharpness-Aware Minimization:* Encourages weights toward flat loss basins to prevent over-adaptation.
  3. *Model Recovery:* Monitors loss Exponential Moving Average; triggers rollback to $\theta_0$ upon detecting model collapse.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Fully compatible. Implemented in `src/changeguard/baseline_adapters.py` (`E1SARAdapter`).
- **Deviations Required:** Valid-pixel masking; SAM optimization on BN affine parameters of Siamese UNet.
- **Whether Deviations Preserve Defining Method:** Yes. Two-stage SAM optimization, reliability filtering, and model recovery are preserved.
- **Historical Project Status:** `IMPLEMENTED AND EVALUATED` (Completed across 12 streams in Work Package E1.4; audited in `docs/e1_4_matched_baselines_closeout.md`).
- **Future Role under Branch C:** `RETAINED AS AUDITED HISTORICAL BASELINE BENCHMARK`. Provides reference evidence across 12 streams (Seeds 0, 1, 2 × Orders A, B × Reset/Continual; Macro F1 = 0.5801 ± 0.0137; 8 harmful, 4 beneficial streams).
- **Limitations of Project Formulation:**
  Evaluated on standard BatchNorm SNUNet-ECAM rather than GroupNorm. SAM two-stage forward pass doubles computation per active batch. Model recovery was never triggered during dev6 runs because reliability filtering suppressed the extreme gradients that trigger collapse.

---

### 2.4 NOTE (Robust Continual Test-time Adaptation Against Temporal Correlation)

- **Exact Paper Title:** NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation
- **Authors:** Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, Sung-Ju Lee
- **Year & Venue / Status:** 2022, Advances in Neural Information Processing Systems (NeurIPS 2022), vol. 35, pp. 27253–27266
- **DOI / Official Publisher URL:** [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae5746b4bb92b774b77f10e4453086eb-Abstract.html) | [arXiv:2208.05117](https://arxiv.org/abs/2208.05117)
- **Official Repository URL:** [https://github.com/TaesikGong/NOTE](https://github.com/TaesikGong/NOTE)
- **Inspected Repository Version / Commit:** GitHub default branch commit `41de2a8` (inspected 2026-10-21)
- **License:** `UNKNOWN` (no explicit license file in repository root as of 2026-10-21 access date; default all rights reserved)
- **Original Task & Modality:** Continual test-time adaptation under non-i.i.d., temporally correlated streams (CIFAR-10-C, ImageNet-C, Cityscapes-C); 2D optical RGB imagery
- **Source-Data Assumptions:** Source-free during adaptation.
- **Retraining Assumptions:** Replaces standard BatchNorm with Instance-Aware Batch Normalization (IABN), which requires architectural substitution.
- **Adaptation State:** Continual memory state. Maintains Prediction-Balanced Reservoir Sampling (PBRS) replay buffer across the stream; updates IABN affine parameters using entropy minimization on replay batches.
- **Prediction Timing:** `PRE_UPDATE` primary prediction on incoming batch before buffer insertion.
- **Deployment-Time Labels / Metadata:** Zero target labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Instance-Aware Batch Normalization (IABN):* Corrects normalization statistics for out-of-distribution instances under non-i.i.d. conditions.
  2. *Prediction-Balanced Reservoir Sampling (PBRS):* Re-balances class frequencies in memory buffer to counteract temporal correlation.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Low-to-moderate. PBRS assumes discrete single-label image classification instances; dense segmentation patches have multi-class mixtures (Land + Water + Flood within single patch). IABN requires modifying SNUNet-ECAM BN layers.
- **Deviations Required:** Redesigning PBRS reservoir sampling for continuous spatial class mixture vectors; buffer storage of high-dimensional multi-channel SAR rasters; replacing BN layers in frozen architecture.
- **Whether Deviations Preserve Defining Method:** Questionable. PBRS relies on discrete class assignments to balance the memory reservoir; in dense change detection with 95% land pixels per patch, reservoir balancing breaks down.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5`.
- **Future Role under Branch C:** `DISCUSS ONLY — NOT SCHEDULED`.
- **Justification:** Protocol conflict with episodic per-event reset; reservoir buffer formulation is incompatible with multi-class dense mixture patches without substantial ad-hoc heuristic redesign; unlicensed upstream repository.

---

### 2.5 RoTTA (Robust Test-Time Adaptation in Dynamic Scenarios) — Source-Knowledge Restoration

- **Exact Paper Title:** Robust Test-Time Adaptation in Dynamic Scenarios
- **Authors:** Longhui Yuan, Binhui Xie, Shuang Li
- **Year & Venue / Status:** 2023, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023), pp. 15922–15932
- **DOI / Official Publisher URL:** [10.1109/CVPR52729.2023.01528](https://doi.org/10.1109/CVPR52729.2023.01528) | [arXiv:2303.13899](https://arxiv.org/abs/2303.13899)
- **Official Repository URL:** [https://github.com/BIT-DA/RoTTA](https://github.com/BIT-DA/RoTTA)
- **Inspected Repository Version / Commit:** GitHub default branch commit `8f7b3d2` (inspected 2026-10-21)
- **License:** MIT License
- **Original Task & Modality:** Robust test-time adaptation under dynamic scenarios (continually changing target distributions with non-i.i.d. sample streams) on CIFAR-10-C, CIFAR-100-C, ImageNet-C; optical RGB imagery
- **Source-Data Assumptions:** Source-free during adaptation. Pretrained source weights $\theta_0$ stored in memory for Robust Batch Normalization (R-BN) anchor and Category-balanced Memory Bank.
- **Retraining Assumptions:** Zero source retraining. Operates on pretrained checkpoint.
- **Adaptation State:** Continual adaptation state; Category-balanced Memory Bank dynamically stores recent features and pseudo-labels; updates model using memory replay and Robust Batch Normalization (R-BN).
- **Prediction Timing:** `PRE_UPDATE` primary prediction from student/teacher before memory replay update.
- **Deployment-Time Labels / Metadata:** Zero target ground-truth labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Robust Batch Normalization (R-BN):* Blends source running statistics with current target batch statistics using an information-theoretic distance gate to prevent normalization collapse.
  2. *Category-Balanced Memory Bank:* Mitigates label distribution shift by maintaining class-balanced replay samples for test-time regularization.
  3. *Timeliness-decayed memory update:* Evicts stale samples to track changing target domains.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Moderate. R-BN source-statistic blending is conceptually adjacent to ChangeGuard's E3 anchor screening and AdaBN. However, Category-Balanced Memory Bank assumes discrete image-level classification categories; dense SAR patches contain spatial mixtures with overwhelming Land dominance.
- **Deviations Required:** Replacing patch memory bank with pixel-level memory or spatial region sampling; re-engineering class-balanced eviction for dense masks.
- **Whether Deviations Preserve Defining Method:** Partial. R-BN statistic blending can be preserved, but the Category-Balanced Memory Bank cannot be cleanly transferred to dense change segmentation without altering its defining algorithm.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5` (E3 evaluated source-anchor regularization via probability/logit/feature distances; E3.3 closed out with no anchor candidate promoted).
- **Future Role under Branch C:** `DISCUSS ONLY — NOT SCHEDULED`.
- **Justification:** E3 already evaluated the source-knowledge preservation hypothesis in depth across 4 anchor families ($\lambda \in [0.01, 1.00]$); best anchor achieved only $+0.05$ pp Macro F1 vs AdaBN. Re-implementing RoTTA's memory bank would duplicate settled E3 anchor findings while adding memory bank overhead on high-resolution SAR rasters.

---

### 2.6 TTA-Monitor (Monitoring Risks in Test-Time Adaptation) — Test-Time Risk Monitoring

- **Exact Paper Title:** Monitoring Risks in Test-Time Adaptation
- **Authors:** Mona Schirmer, Metod Jazbec, Christian Andersson Naesseth, Eric Nalisnick
- **Year & Venue / Status:** 2025, Advances in Neural Information Processing Systems (NeurIPS 2025); arXiv preprint [arXiv:2507.08721](https://arxiv.org/abs/2507.08721) [cs.LG]
- **DOI / Official Publisher URL:** [10.48550/arXiv.2507.08721](https://doi.org/10.48550/arXiv.2507.08721) | [NeurIPS Proceedings 2025](https://neurips.cc/virtual/2025/poster/116904)
- **Official Repository URL:** [https://github.com/monasch/tta-monitor](https://github.com/monasch/tta-monitor)
- **Inspected Repository Version / Commit:** GitHub default branch commit `9d2a1b4` (inspected 2026-10-21)
- **License:** `UNKNOWN` (no explicit license file in repository root as of 2026-10-21 access date; default all rights reserved)
- **Original Task & Modality:** Online risk monitoring and failure detection during test-time adaptation of vision models (ImageNet-C, CIFAR-C); optical RGB imagery
- **Source-Data Assumptions:** Access to a calibration dataset from the source domain or validation domain to calibrate sequential testing thresholds prior to test deployment.
- **Retraining Assumptions:** Zero model retraining. Operates alongside any off-the-shelf TTA method (Tent, EATA, SAR).
- **Adaptation State:** External sequential monitor running in parallel with the adaptation process; accumulates non-conformity or uncertainty martingale scores.
- **Prediction Timing:** `PRE_UPDATE` / `POST_UPDATE` monitoring: Computes martingale or sequential risk statistic at each step; raises an alarm and stops adaptation or triggers fallback when cumulative risk crosses calibrated bound.
- **Deployment-Time Labels / Metadata:** Zero test labels. Requires offline calibration data to compute valid test thresholds.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Sequential Hypothesis Testing / Martingales:* Tests whether online adaptation is reducing or inflating target classification risk over time.
  2. *False-Alarm-Controlled Stopping Rule:* Statistically bounds the probability of false alarms on safe adaptation streams while detecting catastrophic drift.
  3. *Fallback Action:* Once alarm fires, model halts adaptation or rolls back to the source checkpoint.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** High conceptual relevance to ChangeGuard's temporal-risk objective. However, sequential martingales assume exchangeable calibration samples or stationary risk baselines. In SAR flood events, streams are short (e.g. Event 1111003 has 2 batches; Event 559 has 3 batches), which is too short for asymptotic sequential martingale power.
- **Deviations Required:** Formulating dense pixel-level non-conformity scores; adapting martingale stopping thresholds to highly non-stationary 2-to-149 batch flood event streams.
- **Whether Deviations Preserve Defining Method:** Partial. The statistical guarantees of sequential testing depend heavily on exchangeability and calibration length; applying it to 6 development flood events with extreme sequence length disparity ($N \in [2, 149]$ batches) invalidates asymptotic false-alarm guarantees.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5` (E5 evaluated 10 pre-update temporal risk scores directly).
- **Future Role under Branch C:** `DISCUSS ONLY — NOT SCHEDULED`.
- **Justification:** Gate E demonstrated that pre-update risk scores invert across model seeds (KL AUROC collapses from 1.000 to 0.000). Sequential statistical monitoring cannot overcome the fundamental absence of a directional signal without labeled target calibration data; upstream repository lacks an open-source license.

---

### 2.7 CIM-TTDA (Coastal Inundation Mapping via Test-Time Domain Adaptation)

- **Exact Paper Title:** Improving Cross-Event Generalization for SAR-Based Coastal Inundation Mapping via Test-Time Domain Adaptation *(Conference: Event-wise Test-Time Domain Adaptation for Robust SAR Coastal Inundation Mapping Across Flood Events)*
- **Authors:** Wantai Chen, Chong Wang, Zimeng Zhao, Xiaofeng Li
- **Year & Venue / Status:** 2026, Journal of Geophysical Research: Machine Learning and Computation; presented at PIERS 2026 (Progress In Electromagnetics Research Symposium)
- **DOI / Official Publisher URL:** [10.1029/2025JH001145](https://doi.org/10.1029/2025JH001145) | [PIERS 2026 Session 1A2a](https://piers.org)
- **Official Repository URL:** `UNKNOWN` (no public repository released or linked in searchable literature as of 2026-10-21 access date)
- **Inspected Repository Version / Commit:** `UNKNOWN` (unreleased upstream code)
- **License:** `UNKNOWN`
- **Original Task & Modality:** Single-temporal and multi-temporal Sentinel-1 dual-polarization (VV + VH) Synthetic Aperture Radar (SAR) coastal inundation and water body mapping; dense 2D SAR rasters
- **Source-Data Assumptions:** Pretrained on large-scale source SAR datasets (e.g. S1S2-Water); source-free during target adaptation.
- **Retraining Assumptions:** Requires pretraining segmentation backbone on dual-polarization SAR water datasets.
- **Adaptation State:** Event-wise unsupervised adaptation; maintains EMA teacher-student network; applies class-balanced cross-entropy loss with hard sample mining on dual-pol backscatter.
- **Prediction Timing:** `PRE_UPDATE` primary inference via EMA teacher.
- **Deployment-Time Labels / Metadata:** Zero target ground-truth labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Class-Balanced Loss:* Counters extreme land-water imbalance in flood scenes.
  2. *Hard Sample Mining:* Focuses gradient updates on high-ambiguity boundary pixels.
  3. *Stochastic Restoration:* Prevents parameter drift across adaptation iterations.
  4. *Geo-Aware Sampling:* Reduces spatial redundancy in large raster tiles.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Highest domain relevance among all literature. Directly targets SAR flood and coastal inundation mapping using Sentinel-1 imagery. However, CIM-TTDA focuses primarily on water segmentation rather than bi-temporal change detection (Land vs Water vs Flood change classes), and uses single-event or concatenated inputs rather than Siamese bi-temporal difference reasoning (SNUNet-ECAM).
- **Deviations Required:** Adapting binary/water inundation objective to 3-class change detection; adapting input from dual-pol single-time to 4-channel bi-temporal paired rasters; reimplementing without reference code.
- **Whether Deviations Preserve Defining Method:** Questionable without upstream source code. Any reimplementation would be a ChangeGuard-authored composite heuristic rather than a faithful baseline reproduction.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5` (E4 evaluated class-aware loss weighting K1-K4, finding neutral-to-negative net outcomes).
- **Future Role under Branch C:** `DISCUSS ONLY — NOT SCHEDULED`.
- **Justification:** Primary source code is unreleased/unknown; reimplementation from paper abstract alone risks confounding baseline failure with reimplementation artifacts; E4 already established that class-balanced and flood-weighted loss formulations do not prevent adaptation degradation on this benchmark.

---

### 2.8 PA-TTA (Test-Time Adaptation for Photovoltaic Installation Growth Assessment)

> [!NOTE] Acronym Disambiguation:
> **PA-TTA (Huang et al., TGRS 2026)** is the relevant parameter-updating test-time adaptation method for remote sensing change assessment investigated here. It must not be confused with the unrelated 2023 J-STARS landslide study (Chen et al., IEEE J-STARS 2023, DOI: 10.1109/JSTARS.2022.3233043) that used "probability-average test-time augmentation" under the identical acronym. The 2023 landslide method performs zero parameter updates ($\Delta \theta = 0$) and is a static test-time augmentation technique, whereas PA-TTA (Huang et al., 2026) is an adaptive parameter-updating TTA method.

- **Exact Paper Title:** Test-Time Adaptation for Photovoltaic Installation Growth Assessment
- **Authors:** Rui Huang, Huanjing Yue, Jianhua Guo, Xiaoqiang Lu, Jingyu Yang
- **Year & Venue / Status:** 2026, IEEE Transactions on Geoscience and Remote Sensing (TGRS), Accepted June 11, 2026; IEEE Xplore Document ID 11563891
- **DOI / Official Publisher URL:** [10.1109/TGRS.2026.3703807](https://doi.org/10.1109/TGRS.2026.3703807) | [IEEE Xplore Document 11563891](https://ieeexplore.ieee.org/document/11563891)
- **Official Repository URL:** [https://github.com/DonMuv/PA-TTA](https://github.com/DonMuv/PA-TTA)
- **Inspected Repository Version / Commit:** Commit `30ed674cf89547fbae806128afa15de8596d578a` (inspected 2026-10-21)
- **License:** `UNKNOWN` (no explicit license file in repository root as of 2026-10-21 access date; default all rights reserved)
- **Original Task & Modality:** High-resolution optical satellite imagery (bitemporal RGB image pairs) for photovoltaic (solar panel) installation growth assessment / change detection
- **Paired-Image Formulation & Architecture:** Bitemporal paired optical image inputs $(x_1, x_2)$ processed through a Bitemporal Image Transformer (`BASE_Transformer` / BIT_CD, Chen et al., 2021) coupled with a Cross-Attention Context Aware (`CCA`) module for context alignment
- **Source-Data / Retraining Requirements:** Uses a source model pretrained on bitemporal photovoltaic datasets (`best_ckpt.pt`); source training data are not required during test-time adaptation. However, generating pseudo-labels requires an external vision foundation model (Segment Anything Model, SAM) to supply boundary guidance masks.
- **Adaptation State:** Online/transductive parameter adaptation updating Batch Normalization affine parameters (`nn.BatchNorm2d` scale $\gamma$ and shift $\beta$, collected via `collect_params`).
- **Pseudo-Label Mechanism:** Cross-image context fusion with SAM foundation model guidance. Merges candidate predictions (`logits.argmax(dim=1)`) with external SAM masks (`sam_mask`) using `merge_batch_masks_by_overlap` ($IoU \ge 0.70$) and edge-to-area ratio tensor filtering (`edge_to_area_ratio_tensor`) to generate refined pseudo-label masks for cross-entropy supervision.
- **Catastrophic-Forgetting Protection:** Retains deep copies of the initial source model as an anchor (`model_anchor`) and maintains an exponential moving average model (`model_ema`), with reset capability (`load_model_and_optimizer`).
- **Prediction Timing:** Multi-epoch / transductive evaluation in upstream code (`trainer.py` iterates `MAX_EPOCH` epochs over target dataset before calling `inference()`). In streaming mode, SAM guidance masks must be precomputed or computed concurrently, preceding BN affine updates.
- **Deployment-Time Labels / Metadata:** Zero target ground-truth change labels. Requires external SAM masks as auxiliary prompts.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Low-to-incompatible.
  1. *Optical vs Radar Modality:* PA-TTA relies fundamentally on SAM (Segment Anything Model) prompts for spatial boundary refinement. SAM was trained on natural RGB images and exhibits severe hallucination and boundary collapse when applied to dual-polarization SAR speckle rasters without specialized radar adapter layers.
  2. *Binary vs 3-Class Segmentation:* Upstream PA-TTA is formulated strictly for binary change (PV Growth vs Background; `output_nc=2`). ChangeGuard requires 3-class dense segmentation (Land vs Permanent Water vs Flood).
  3. *Siamese UNet vs Transformer:* Upstream model architecture is `BASE_Transformer` (BIT_CD) with specific CCA token dimensions; ChangeGuard's frozen source backbone is SNUNet-ECAM.
  4. *Computational Budget & VRAM:* Incorporating SAM or SAM-derived mask generation on $512 \times 512$ patches would require an additional large foundation model in VRAM (SAM ViT-B/H requires $\ge 4.5$ GB alone), violating ChangeGuard's single-batch bounded memory constraint ($\le 2.92$ GB VRAM).
- **Modifications Required for ChangeGuard:**
  1. Substituting Sentinel-1 dual-pol paired rasters for optical RGB inputs.
  2. Adapting the SAM guidance module to operate without SAM (or with an unvalidated radar foundation model).
  3. Extending CCA and cross-entropy from binary change to 3-class mutual exclusivity.
  4. Constraining multi-epoch training (`trainer.py`) to single-pass online streaming `PRE_UPDATE` prediction.
- **Whether Deviations Preserve Defining Method:** No. Removing the SAM guidance module and edge-to-area ratio mask fusion strips away the defining contribution of PA-TTA, reducing it to standard Tent-style BN adaptation with feature anchoring.
- **Historical Project Status:** `NOT PREVIOUSLY IMPLEMENTED IN E1-E5` (Evaluated conceptually in literature reviews during Week 10; no code integration).
- **Final Disposition:** `DISCUSS ONLY — NOT SCHEDULED`.
- **Justification:** PA-TTA's defining mechanism relies on Segment Anything Model (SAM) foundation features that cannot operate reliably on SAR backscatter, introduces excessive memory overhead, requires multi-epoch adaptation loops that violate ChangeGuard's streaming PRE_UPDATE contract, and lacks an open-source license. Removing SAM guidance destroys the defining method.

---

### 2.9 SAR Model Recovery & ASR (Adaptive and Selective Reset) — Prediction-Verification / Rollback

- **Exact Paper Title:**
  - *Primary 9a:* Towards Stable Test-Time Adaptation in Dynamic Wild World [Section 3.3: Model Recovery] (Niu et al., ICLR 2023)
  - *Primary 9b:* When and Where to Reset Matters for Long-Term Test-Time Adaptation [Adaptive and Selective Reset (ASR)] (Lim, Hwang, Lee, ICLR 2026 / arXiv:2603.03796)
- **Authors:**
  - *9a:* Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Mingkui Tan
  - *9b:* Taejun Lim, Joong-Won Hwang, Kibok Lee
- **Year & Venue / Status:**
  - *9a:* 2023, International Conference on Learning Representations (ICLR 2023, Oral)
  - *9b:* 2026, International Conference on Learning Representations (ICLR 2026); arXiv preprint [arXiv:2603.03796](https://arxiv.org/abs/2603.03796)
- **DOI / Official Publisher URL:**
  - *9a:* [https://openreview.net/forum?id=g2YraF75Tj](https://openreview.net/forum?id=g2YraF75Tj) | [arXiv:2302.12400](https://arxiv.org/abs/2302.12400)
  - *9b:* [10.48550/arXiv.2603.03796](https://doi.org/10.48550/arXiv.2603.03796) | [OpenReview Forum](https://openreview.net/forum?id=uN6Jqg0b27)
- **Official Repository URL:**
  - *9a:* [https://github.com/mr-eggplant/SAR](https://github.com/mr-eggplant/SAR)
  - *9b:* [https://github.com/taejunlim/ASR](https://github.com/taejunlim/ASR)
- **Inspected Repository Version / Commit:**
  - *9a:* Commit `b67c1e3` (inspected 2026-10-21)
  - *9b:* Commit `f10c3b8` (inspected 2026-10-21)
- **License:**
  - *9a:* BSD 3-Clause License
  - *9b:* MIT License
- **Original Task & Modality:** Continual test-time adaptation of vision classifiers under corruption and domain shifts; optical RGB imagery
- **Source-Data Assumptions:** Source-free during adaptation. Pretrained source weights $\theta_0$ stored for rollback/reset.
- **Retraining Assumptions:** Zero source retraining.
- **Adaptation State:** Updates parameters online; monitors post-step metric (entropy moving average in SAR; layer-wise Fisher drift in ASR); triggers parameter rollback or layer-selective reset to $\theta_0$ when metric crosses safety boundary.
- **Prediction Timing:** Two-tier timing: `PRE_UPDATE` primary prediction; `POST_UPDATE` evaluation triggers rollback of parameter state for *subsequent* batches.
- **Deployment-Time Labels / Metadata:** Zero target labels.
- **Protection / Restoration / Monitoring Mechanism:**
  1. *Post-Update Verification:* Checks whether candidate update produced non-finite values, severe entropy collapse, or large parameter displacement.
  2. *Parameter Rollback / Selective Reset:* Restores weights to pre-step checkpoint $\theta_{t-1}$ or resets selected layers to source weights $\theta_0$.
- **Compatibility with Paired Dense 3-Class SAR Change Segmentation:** Highly compatible. SAR model recovery is already implemented in `src/changeguard/baseline_adapters.py:800-820, 1070-1080`.
- **Deviations Required:** None for SAR recovery (already implemented); for ASR, layer-wise Fisher drift tracking on Siamese encoder.
- **Whether Deviations Preserve Defining Method:** Yes. The rollback/reset mechanism is directly preserved.
- **Historical Project Status:** `IMPLEMENTED AND EVALUATED` (SAR model recovery evaluated as integral component of SAR across 12 streams in E1.4; finite-gradient post-step rollback implemented in `E1EATAAdapter`).
- **Future Role under Branch C:** `RETAINED AS AUDITED HISTORICAL BASELINE BENCHMARK`.
- **Limitations of Project Formulation:** In ChangeGuard E1.4 evaluations, the SAR loss EMA rarely breached the empirical recovery threshold $e_m \approx 0.0318$ because conservative reliability filtering already prevented high-entropy updates. Consequently, model recovery was triggered in zero streaming batches during dev6 runs, demonstrating that static threshold rollback is inactive when combined with conservative sample filtering.

---

## 3. Comprehensive Historical Baseline Status Ledger

The following methods were already implemented and audited in ChangeGuard prior to Work Package E6:

| Method Identifier | Historical Project Status | Work Package | Streams Executed | Future Role under Branch C | Key Implementation Limitation vs Official Method |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Source-Only** (`source_only_frozen_running_stats`) | `IMPLEMENTED AND EVALUATED` | E1.4, E5 | 3 streams (S0, S1, S2) | Primary fixed reference across all evaluations | None (exact standard inference) |
| **AdaBN** (`adabn_matched_pre_update` / BN-no-update) | `IMPLEMENTED AND EVALUATED` | E1.4, E1.5, E5 | 12 streams (3 seeds × 2 orders × 2 reset modes) | Negative control isolating normalization shift from parameter updates | Uses test batch statistics only without momentum smoothing across events |
| **Tent** (`tent_matched_pre_update`) | `IMPLEMENTED AND EVALUATED` | E1.4 | 12 streams (3 seeds × 2 orders × 2 reset modes) | Audited historical entropy-minimization baseline | Pixel-wise Shannon entropy over valid mask; no spatial confidence gating |
| **EATA** (`eata_v2_fisher_matched_pre_update`) | `IMPLEMENTED AND EVALUATED` | E1.4 | 12 streams (3 seeds × 2 orders × 2 reset modes) | Audited historical sample-selection & Fisher baseline | Default entropy threshold ($E_0 \approx 0.439$) yielded zero updates on dev6, collapsing to AdaBN |
| **SAR** (`sar_matched_pre_update` with recovery) | `IMPLEMENTED AND EVALUATED` | E1.4 | 12 streams (3 seeds × 2 orders × 2 reset modes) | Audited historical sharpness-aware & rollback baseline | Evaluated on BatchNorm SNUNet-ECAM; recovery threshold inactive due to conservative filtering |
| **Candidate Consistency** (τ=0.90, Horiz Flip) | `IMPLEMENTED AND EVALUATED` | E2, E3, E4, E5 | 9 streams (E2), 4 streams (E3), 4 streams (E4), 3 cells (E5) | Primary adaptive candidate under study | Single-view consistency; lacks EMA teacher smoothing |

---

## 4. Applicability Summary and Baseline Scheduling

| Method | Primary Venue | License | Task Relevance | Upstream Code | Project Status | Future Role under Branch C |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CoTTA** | CVPR 2022 | MIT | High | Available | New Baseline | **SCHEDULE AS FORMAL BASELINE PROTOCOL** |
| **EATA** | ICML 2022 | MIT | Moderate | Available | Evaluated (E1.4) | **RETAINED AS AUDITED HISTORICAL BASELINE** |
| **SAR** | ICLR 2023 | BSD 3-Clause | Moderate | Available | Evaluated (E1.4) | **RETAINED AS AUDITED HISTORICAL BASELINE** |
| **NOTE** | NeurIPS 2022 | UNKNOWN | Moderate | Available | New | **DISCUSS ONLY** (Protocol conflict; unlicensed) |
| **RoTTA** | CVPR 2023 | MIT | Moderate | Available | New | **DISCUSS ONLY** (Duplicates settled E3 anchor findings) |
| **TTA-Monitor** | NeurIPS 2025 | UNKNOWN | High | Available | New | **DISCUSS ONLY** (Direction instability; unlicensed) |
| **CIM-TTDA** | JGR/PIERS 2026 | UNKNOWN | High | UNKNOWN | New | **DISCUSS ONLY** (Unreleased upstream source code) |
| **PA-TTA** | IEEE TGRS 2026 | UNKNOWN | High | Available | New | **DISCUSS ONLY** (SAM dependency invalid on SAR; multi-epoch; unlicensed) |
| **ASR / SAR Recovery** | ICLR 2023/2026 | BSD/MIT | High | Available | Evaluated (E1.4) | **RETAINED AS AUDITED HISTORICAL BASELINE** |

> [!NOTE]
> Per `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.3: *"At most one new closest-method implementation may be scheduled."* CoTTA is the single newly scheduled baseline, with full implementation details governed by `docs/protocols/e6_cotta_compatibility_protocol_v1.md`.

---

## 5. Cross-Reference

| Document | Role |
| :--- | :--- |
| `configs/changeguard/e6_decision_contract_v1.json` | Formalized decision rules |
| `docs/e6_gate_e_decision.md` | Gate E evidence ledger |
| `docs/e6_primary_study_definition.md` | Branch C empirical study definition |
| `docs/e6_claim_to_experiment_map.md` | Claim-to-experiment map |
| `docs/protocols/e6_cotta_compatibility_protocol_v1.md` | CoTTA compatibility protocol |
| `docs/e6_final_route_decision.md` | Final integrated route decision |
