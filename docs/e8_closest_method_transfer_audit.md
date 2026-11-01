# ChangeGuard Work Package E8: Closest-Method Transfer Audit

**Status:** APPROVED & FROZEN (FACTUALLY CORRECTED)
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E8 (Sections 13.1–13.4)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/e6_nearest_method_applicability_matrix.md`
- `docs/protocols/e6_cotta_compatibility_protocol_v1.md`
- `docs/protocols/e7_branch_c_empirical_protocol_v1.md`
- `docs/protocols/e8_cross_event_confirmation_protocol_v1.md`
- `docs/e7_final_closeout.md`

---

## 1. Executive Summary & Audit Framing

Before freezing the Work Package E8 confirmation matrix, this audit conducts a primary paper-and-code transfer inspection of the closest published task-specific test-time adaptation (TTA) and risk-monitoring methods:
1. **PA-TTA** (Huang et al., IEEE TGRS 2026): Remote-sensing bitemporal photovoltaic change assessment.
2. **CIM-TTDA** (Chen et al., JGR-MLC / PIERS 2026): SAR-based coastal inundation mapping.
3. **Source-Restoration, Risk-Monitoring & Parameter-Updating Baselines:**
   - SAR: Sharpness-Aware and Reliable Test-Time Adaptation (Niu et al., ICLR 2023)
   - ASR: Adaptive and Selective Reset (Lim et al., ICLR 2026)
   - TTA-Monitor: Sequential Risk Monitoring (Schirmer et al., NeurIPS 2025)
4. **CoTTA** (Wang et al., CVPR 2022): Continual test-time domain adaptation via Mean Teacher EMA, multi-view augmentation averaging, and Stochastic Source Restoration (SSR).

### Key Audit Framing Principles:
- **Baseline Role:** Work Package E8 evaluates **CoTTA** as the strongest currently executable, fully compatible generic test-time adaptation baseline in the causal online streaming track.
- **Protocol Boundary, Not Scientific Defeat:** **PA-TTA and CIM-TTDA are not claimed to have been defeated.** Their exclusion from the primary online confirmation matrix is strictly a **protocol-comparability decision**, not empirical evidence that they fail on SAR change detection.
- **Separate Transductive Track:** A separate event-transductive evaluation track may be scheduled after E8 if required for final publication claims, operating under an explicit declared protocol that matches their multi-epoch transductive formulation.

---

## 2. In-Depth Methodological Transfer Audits

### 2.1 PA-TTA: Test-Time Adaptation for Photovoltaic Installation Growth Assessment

- **Primary Citation:** Rui Huang, Huanjing Yue, Jianhua Guo, Xiaoqiang Lu, Jingyu Yang, "Test-Time Adaptation for Photovoltaic Installation Growth Assessment," *IEEE Transactions on Geoscience and Remote Sensing* (TGRS), vol. 64, Art no. 11563891, accepted June 11, 2026. DOI: [10.1109/TGRS.2026.3703807](https://doi.org/10.1109/TGRS.2026.3703807).
- **Official Upstream Repository:** `https://github.com/DonMuv/PA-TTA`
- **Inspected Git Revision:** Commit `30ed674cf89547fbae806128afa15de8596d578a` (inspected 2026-10-21).
- **Upstream License Status:** No explicit open-source license file is present in the repository root as of the inspection date. Under standard international copyright law and GitHub Terms of Service, all rights remain reserved to the authors; downstream researchers possess no explicit grant for redistribution or code modification, presenting a formal licensing and external reproducibility constraint.
- **Target Task & Modality in Upstream:** High-resolution optical satellite imagery (bitemporal RGB image pairs, $512 \times 512$) for detecting photovoltaic (PV) solar panel installation growth.
- **Required Architecture & Modules:**
  - Backbone: `BASE_Transformer` (BIT_CD, Bitemporal Image Transformer; Chen et al., 2021).
  - Context Alignment: Cross-Attention Context Aware (`CCA`) module computing cross-attention across bitemporal token embeddings.
- **Priors & Pseudo-Label Generation:**
  - Upstream data loader explicitly consumes pre-computed auxiliary masks named `sam_mask`.
  - In `trainer.py`, candidate predictions (`logits.argmax(dim=1)`) are merged with `sam_mask` using `merge_batch_masks_by_overlap` ($IoU \ge 0.70$) and edge-to-area ratio tensor filtering (`edge_to_area_ratio_tensor`).
  - Output space and loss functions are structured around binary change detection (`output_nc = 2`: background vs change).
- **Adaptation Loop & Timing:**
  - Upstream runner (`trainer.py`) executes a **multi-epoch transductive adaptation loop**: it repeatedly cycles over the target event dataset for `MAX_EPOCH` (e.g., 5–10 epochs) to update BN affine parameters before final inference.
  - This multi-epoch transductive setup is fundamentally different from ChangeGuard's causal online single-pass `PRE_UPDATE` primary streaming protocol (where predictions are generated and durably persisted before any parameter update on the current batch).
- **Memory & Model Objects:**
  - The codebase initializes auxiliary anchor/EMA model objects (`model_anchor`, `model_ema`), but in the active adaptation loop of `trainer.py`, gradient descent specifically updates the active model's `nn.BatchNorm2d` scale ($\gamma$) and shift ($\beta$) parameters via cross-entropy against SAM-guided pseudo-labels.
- **Comparability Assessment for Primary Online Matrix:**
  - PA-TTA is **not directly comparable in the primary online streaming matrix**. Integrating PA-TTA directly into the online table would require substantial structural modifications that violate fair baseline comparison:
    1. *Guidance Modality:* SAM was trained on natural RGB optical imagery; its behavior on uncalibrated, speckle-affected dual-polarization SAR imagery has not been evaluated in this benchmark, and using it without specialized SAR adaptation is unvalidated.
    2. *Class Cardinality:* The upstream implementation is hard-coded for binary change detection (`output_nc = 2`), so transferring it to ChangeGuard's 3-class segmentation task (Land, Permanent Water, Flood Water) requires redesigning the pseudo-label assignment logic and loss formulation.
    3. *Temporal & Information Timing:* Its multi-epoch transductive loop requires having the entire event data available upfront, conflicting with causal single-pass streaming.
- **Applicability Decision:**
  **`NOT DIRECTLY COMPARABLE IN THE PRIMARY ONLINE MATRIX — RETAIN AS A SEPARATE TRANSDUCTIVE TASK-TRANSFER CANDIDATE`**
  - A faithful empirical evaluation of PA-TTA would require an explicitly declared separate protocol defining:
    - SAR-compatible spatial guidance;
    - Three-class pseudo-label semantics;
    - Full-event data availability;
    - Multi-epoch adaptation;
    - Formal compute and memory measurement under the transductive loop.
  - This separate transductive track is retained as a candidate for future investigation but is **not implemented or scheduled during the primary online E8 confirmation matrix**.

---

### 2.2 CIM-TTDA: SAR Coastal Inundation Mapping via Test-Time Domain Adaptation

- **Primary Citation:** Wantai Chen, Chong Wang, Zimeng Zhao, Xiaofeng Li, "Improving Cross-Event Generalization for SAR-Based Coastal Inundation Mapping via Test-Time Domain Adaptation," *Journal of Geophysical Research: Machine Learning and Computation* (JGR-MLC), 2026; presented at PIERS 2026. DOI: [10.1029/2025JH001145](https://doi.org/10.1029/2025JH001145).
- **Official Upstream Repository:** **No official implementation was located during this audit.**
- **Target Task & Modality in Upstream:** Single-temporal and post-event Sentinel-1 dual-polarization (VV + VH) Synthetic Aperture Radar (SAR) coastal inundation and water body mapping.
- **Adaptation Protocol:**
  - **Full-event, multi-epoch transductive adaptation:** Adapts parameters by repeatedly cycling over all tiles of a flooded coastal region across multiple epochs before final inference.
- **Mechanisms Described in Publication:**
  - Teacher–student pseudo-labeling with Exponential Moving Average (EMA).
  - Hard-sample mining focusing gradient updates on high-ambiguity water-land boundary pixels.
  - Class-balanced cross-entropy loss countering extreme spatial water-land imbalance.
  - Stochastic parameter restoration preventing unbounded drift.
  - Geo-aware spatial patch sampling.
- **Differences from ChangeGuard Protocol:**
  - *Task:* CIM-TTDA performs water mapping (binary: Water vs Non-water). ChangeGuard performs bi-temporal 3-class change detection (Land vs Permanent Water vs Flood).
  - *Input:* CIM-TTDA processes single post-event rasters. ChangeGuard processes Siamese paired bi-temporal rasters ($T_1$ pre-flood, $T_2$ post-flood; 4 channels total).
  - *Timing:* CIM-TTDA is a multi-epoch transductive offline process. ChangeGuard is an online, single-pass streaming protocol where predictions are durably persisted `PRE_UPDATE` before the next batch arrives.
- **Relationship to CoTTA:**
  - CoTTA shares foundational teacher–student EMA and stochastic parameter restoration principles.
  - However, CoTTA operates strictly in causal online streaming mode, uses multi-view spatial augmentation averaging, and does not perform hard-sample boundary mining or class-balanced loss weighting.
  - **CoTTA is NOT equivalent to CIM-TTDA.** They share foundational teacher–student and parameter restoration concepts, but differ in sampling, loss formulation, and transductive vs streaming timing.
- **Applicability Decision:**
  **`NOT DIRECTLY COMPARABLE IN THE PRIMARY ONLINE MATRIX (SEPARATE TRANSDUCTIVE CANDIDATE) — NO OFFICIAL CODE LOCATED`**
  - Retained for literature discussion. Without an official reference implementation, re-implementing CIM-TTDA from paper description alone risks introducing confounding implementation artifacts.

---

### 2.3 Source-Restoration, Risk-Monitoring & Parameter-Updating Baselines

#### A. SAR: Sharpness-Aware and Reliable Test-Time Adaptation (Niu et al., ICLR 2023)
- **Primary Citation:** Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Mingkui Tan, "Towards Stable Test-Time Adaptation in Dynamic Wild World," *ICLR 2023* (Oral).
- **Core Mechanism:** Updates BN affine parameters using Sharpness-Aware Minimization (SAM) to steer parameters toward flat loss minima, filtered by batch entropy reliability ($H(x) < E_0$). As a safeguard against model collapse, it includes a *model recovery* component that tracks loss exponential moving averages and triggers a parameter reset if the moving average drops below a threshold $e_m$.
- **Terminology Clarification:** The method is **SAR** (Sharpness-Aware and Reliable Test-Time Adaptation). Model recovery is one internal safeguard within SAR, not the method title.
- **ChangeGuard Status:** Evaluated across 12 streams in Work Package E1.4 (`E1SARAdapter`). Model recovery was never triggered during development streams because reliability filtering suppressed the extreme gradients that trigger collapse.
- **Disposition:** `REUSED HISTORICAL BASELINE` (12 streams, 72 event rows already completed and verified).

#### B. ASR: Adaptive and Selective Reset (Lim et al., ICLR 2026 / arXiv:2603.03796)
- **Primary Citation:** Taejun Lim, Joong-Won Hwang, Kibok Lee, "When and Where to Reset Matters for Long-Term Test-Time Adaptation," *ICLR 2026*.
- **Mechanism:** Tracks parameter divergence or feature drift relative to source representations; selectively resets specific layers or triggers episodic resets upon detecting distribution shift.
- **ChangeGuard Applicability:** Selective layer-wise reset requires stationary drift tracking across long streams. In short, non-stationary SAR flood streams ($N \in [2, 149]$ batches), empirical drift estimates exhibit high small-sample variance.
- **Disposition:** `LITERATURE DISCUSSION ONLY`.

#### C. TTA-Monitor: Monitoring Risks in Test-Time Adaptation (Schirmer et al., NeurIPS 2025)
- **Primary Citation:** Mona Schirmer, Metod Jazbec, Christian Andersson Naesseth, Eric Nalisnick, "Monitoring Risks in Test-Time Adaptation," *NeurIPS 2025*.
- **Mechanism:** Constructs sequential non-conformity martingales with false-alarm-controlled stopping boundaries to halt adaptation or revert to source upon detecting risk inflation.
- **ChangeGuard Applicability:** Sequential testing requires exchangeable calibration samples and directional risk scores. Gate E proved that pre-update unsupervised risk scores invert across source seeds (Seed 0 KL AUROC = 1.000; Seed 1 KL AUROC = 0.000). A statistical monitor cannot provide false-alarm control when the underlying proxy signal has zero directional stability.
- **Disposition:** `LITERATURE DISCUSSION ONLY`.

---

### 2.4 CoTTA: Continual Test-Time Domain Adaptation (Wang et al., CVPR 2022)

- **Primary Citation:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai, "Continual Test-Time Domain Adaptation," *CVPR 2022*, pp. 7201–7211. DOI: [10.1109/CVPR52729.2022.00707](https://doi.org/10.1109/CVPR52729.2022.00707).
- **Official Repository:** `https://github.com/qinenergy/cotta` (commit `c4cfbd3`, MIT License).
- **Core Mechanisms:**
  1. *Mean Teacher EMA:* Teacher weights updated smoothly ($\alpha = 0.999$) from student: $\theta_t \leftarrow \alpha \theta_t + (1 - \alpha) \theta_s$.
  2. *Multi-View Augmentation Averaging:* Averages predictions over 4 geometric transformations (Identity, H-flip, V-flip, Rot180) to produce stable teacher pseudo-labels.
  3. *Stochastic Source Restoration (SSR):* At each step, every trainable parameter has probability $p = 0.01$ of being restored bit-exactly to initial source weights $\theta_0$.
  4. *Confidence Thresholding:* High-confidence pseudo-label cross-entropy ($\tau = 0.90$).
- **Implementation in ChangeGuard:**
  - Implemented in `src/changeguard/cotta_adapter.py` (`E7CoTTAAdapter`).
  - Passed 16 unit tests (`tests/test_e7_cotta_adapter.py`) with zero GPU calls.
  - Passed 11 runner integration tests (`tests/test_e7_empirical_runner.py`).
  - Successfully verified in bounded real-data GPU smoke test v2 on Event 520 (Exit code 0, Macro F1 = 0.7257, 2 updates, 0 skips, zero NaNs/Infs).
- **Evaluation Modes:**
  - `cotta_continual`: Carries student/teacher/optimizer state continuously across events.
  - `cotta_reset`: Resets student, teacher, optimizer, and SSR RNG to initial source checkpoint at the onset of each event.
- **Disposition:** `FAITHFUL COMPATIBLE PORT — APPROVED FOR FORMAL E8 CONFIRMATION` (12 physical streams, 72 event rows to be executed).

---

## 3. Applicability Decision Matrix

| Method | Primary Citation | Venue / Year | Target Modality | Adaptation Scope | ChangeGuard Applicability Decision | Detailed Scientific Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Source-Only** | Plan V3 Reference | — | Bi-temporal SAR | Static | **EXACT REPRODUCTION** | 3 streams (Seeds 0, 1, 2); completed in E1.4; verified immutable reference. |
| **BN-No-Update** | Diagnostic Control | — | Bi-temporal SAR | Streaming (no-up) | **EXACT REPRODUCTION** | 6 streams (Seeds 0, 1, 2 × Orders A, B; BS=16); completed in E1.5; verified. |
| **AdaBN** | Li et al. | 2016 | Bi-temporal SAR | Streaming (EMA BN) | **EXACT REPRODUCTION** | 12 streams; completed in E1.4; verified across all factor combinations. |
| **Tent** | Wang et al. | ICLR 2021 | Bi-temporal SAR | Streaming (Entropy) | **FAITHFUL COMPATIBLE PORT** | 12 streams; completed in E1.4; matched learning rate $1\times 10^{-4}$ and Adam optimizer. |
| **EATA** | Niu et al. | ICML 2022 | Bi-temporal SAR | Streaming (Fisher) | **FAITHFUL COMPATIBLE PORT** | 12 streams; completed in E1.4; Fisher computed on source SAR training events. |
| **SAR** | Niu et al. | ICLR 2023 | Bi-temporal SAR | Streaming (SAM) | **FAITHFUL COMPATIBLE PORT** | Sharpness-Aware and Reliable TTA; 12 streams completed in E1.4. |
| **CoTTA** | Wang et al. | CVPR 2022 | Bi-temporal SAR | Continual & Reset | **FAITHFUL COMPATIBLE PORT** | 12 streams scheduled for E8; verified by 16 CPU tests and v2 GPU smoke test. |
| **PA-TTA** | Huang et al. | TGRS 2026 | Optical RGB | Multi-epoch Transductive | **NOT DIRECTLY COMPARABLE (SEPARATE TRANSDUCTIVE CANDIDATE)** | Structure consumes SAM optical guidance and runs multi-epoch transductive loop; retain as separate transductive candidate. |
| **CIM-TTDA** | Chen et al. | JGR-MLC 2026 | Single-pol SAR | Multi-epoch Transductive | **NOT DIRECTLY COMPARABLE (SEPARATE TRANSDUCTIVE CANDIDATE) — NO OFFICIAL CODE LOCATED** | No official implementation located during audit; water mapping rather than bi-temporal change; transductive protocol. |
| **NOTE** | Gong et al. | NeurIPS 2022 | Optical RGB | Continual Replay | **LITERATURE DISCUSSION ONLY** | Reservoir sampling assumes discrete classification instances; breaks down on dense 95% land SAR mixtures; unlicensed code. |
| **RoTTA** | Yuan et al. | CVPR 2023 | Optical RGB | Continual Memory | **LITERATURE DISCUSSION ONLY** | Class-balanced memory bank assumes discrete classification; E3 already evaluated source anchors ($\Delta \text{F1} \le +0.05$ pp). |
| **TTA-Monitor** | Schirmer et al. | NeurIPS 2025 | Optical RGB | Sequential Martingale | **LITERATURE DISCUSSION ONLY** | Martingale false-alarm control requires stationary exchangeability; Gate E showed unsupervised risk signals invert across seeds. |
| **ASR** | Lim et al. | ICLR 2026 | Optical RGB | Selective Reset | **LITERATURE DISCUSSION ONLY** | Selective layer reset requires stationary drift tracking; excessive small-sample variance on short flood streams. |

---

## 4. Methodological Integrity & Protocol Standards

1. **Protocol Boundary Distinction:** The exclusion of PA-TTA and CIM-TTDA from the primary online matrix is a protocol-comparability decision. It must not be cited as evidence that these methods are inferior or "defeated" on SAR change detection.
2. **CoTTA as Generic Benchmark:** CoTTA is evaluated as the strongest currently executable and fully compatible generic TTA baseline under causal streaming.
3. **Future Transductive Track:** If necessary for final publication claims, a separate transductive evaluation track comparing PA-TTA and transductive SAR adaptation may be conducted under an explicitly declared transductive protocol after the E8 confirmation matrix is sealed.
