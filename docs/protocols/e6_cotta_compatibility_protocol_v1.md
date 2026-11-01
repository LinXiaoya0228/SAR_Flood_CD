# ChangeGuard Protocol: CoTTA Compatibility and Execution Protocol (v1)

**Status:** FROZEN PROTOCOL SPECIFICATION (Implementation Deferred to E7)
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Document:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Section 11.3 (commit `24ffcc9`)
**Applicability Reference:** `docs/e6_nearest_method_applicability_matrix.md`, Section 2.1
**Execution Boundary:** This document freezes the protocol specification only. No code implementation or GPU execution is performed during Work Package E6.

---

## 1. Upstream Method Provenance & Verification

- **Exact Upstream Paper Title:** Continual Test-Time Domain Adaptation
- **Authors:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai
- **Publication Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), pp. 7201–7211
- **DOI:** [10.1109/CVPR52729.2022.00707](https://doi.org/10.1109/CVPR52729.2022.00707) | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
- **Official Repository URL:** [https://github.com/qinenergy/cotta](https://github.com/qinenergy/cotta)
- **Inspected Upstream Commit:** Commit `c4cfbd3` (inspected as of 2026-10-21)
- **License:** MIT License (verified in repository root)

---

## 2. Defining CoTTA Components to Preserve

To qualify as a valid, faithful evaluation of CoTTA on the ChangeGuard benchmark, the implementation must preserve all three defining algorithmic pillars:

1. **Mean Teacher Network with Exponential Moving Average (EMA):**
   A dedicated teacher network $\theta_t$ tracks the student parameters $\theta_s$ via:
   $$\theta_t \leftarrow \alpha \theta_t + (1 - \alpha) \theta_s$$
   where $\alpha = 0.999$ is the upstream smoothing coefficient. The teacher parameters are never updated directly via gradient descent.

2. **Augmentation-Averaged Teacher Predictions:**
   For each incoming test batch, the teacher model evaluates multiple stochastically augmented views $\mathcal{A}_k(x)$ ($k = 1, \dots, K$, with $K = 4$ views). The predicted probability maps are transformed back via inverse augmentation $\mathcal{A}_k^{-1}$ and averaged:
   $$\bar{p}_t(x) = \frac{1}{K} \sum_{k=1}^K \mathcal{A}_k^{-1}\left( \text{Softmax}\left( f( \mathcal{A}_k(x); \theta_t ) \right) \right)$$
   The resulting probability map $\bar{p}_t(x)$ generates the pseudo-label $\hat{y}_t = \arg\max \bar{p}_t(x)$ and guides the student update.

3. **Stochastic Source Restoration (SSR):**
   To counteract catastrophic forgetting across continual streaming steps, at every adaptation step, each trainable parameter element has an independent probability $p = 0.01$ of being reset bit-exactly to its initial source value $\theta_0$:
   $$\theta_s^{(j)} \leftarrow \begin{cases} \theta_0^{(j)} & \text{with probability } p = 0.01 \\ \theta_s^{(j)} & \text{with probability } 1 - p \end{cases}$$

---

## 3. Mathematical & Algorithmic Formulation for ChangeGuard

### 3.1 Paired Bi-Temporal SAR Inputs & Geometric Augmentations
The ChangeGuard input consists of bi-temporal paired Sentinel-1 SAR rasters:
$$x = (x_A, x_B) \in \mathbb{R}^{B \times 4 \times H \times W}$$
where $x_A$ is the pre-flood acquisition (VV, VH) and $x_B$ is the post-flood acquisition (VV, VH).

**Augmentation Constraint:**
Augmentations must preserve spatial correspondence between $x_A$ and $x_B$ and cannot alter radar backscatter calibration. Photometric jittering, contrast shifting, and color jittering are forbidden. Permitted augmentation group $\mathcal{A}$ comprises exact spatial transformations:
- $\mathcal{A}_1$: Identity ($\text{rot0}$)
- $\mathcal{A}_2$: Horizontal Flip ($\text{hflip}$)
- $\mathcal{A}_3$: Vertical Flip ($\text{vflip}$)
- $\mathcal{A}_4$: Horizontal + Vertical Flip ($\text{rot180}$)

Each transformation $\mathcal{A}_k$ is applied identically to both $x_A$ and $x_B$:
$$\mathcal{A}_k(x_A, x_B) = (\mathcal{A}_k(x_A), \mathcal{A}_k(x_B))$$
Inverse transformations $\mathcal{A}_k^{-1}$ invert the spatial coordinates of the output logits before probability averaging.

### 3.2 Sensory-Valid Mask Enforcement
Predictions and loss computation are strictly masked by the binary sensor validity mask $M_{NA} \in \{0, 1\}^{B \times H \times W}$. Invalid pixels ($M_{NA} = 0$) contribute zero loss and zero gradient:
$$V = \mathbb{I}(M_{NA} == 1)$$
$$\mathcal{L}_{\text{cotta}}(\theta_s) = \frac{1}{\sum V} \sum_{b,h,w} V_{b,h,w} \cdot \ell_{\text{CE}}\left( f(x)_{b,h,w}; \hat{y}_{t, b, h, w} \right) \cdot \mathbb{I}\left( \max \bar{p}_{t, b, h, w} \ge \tau \right)$$
where $\tau = 0.90$ is the standard confidence threshold matching the ChangeGuard baseline contract.

### 3.3 Zero Target-Label Constraint
The implementation must operate strictly through `LabelGuardedBatch`. The ground-truth tensor `batch.get_labels()` is inaccessible during both inference and adaptation. Any reference to ground-truth labels during adaptation will raise an immediate fatal runtime error.

### 3.4 Primary Prediction Timing
Primary predictions are strictly `PRE_UPDATE`. The primary output recorded for the current batch is the augmentation-averaged teacher prediction $\bar{p}_t(x)$ evaluated *before* the student gradient update is computed. Post-update predictions are diagnostic only and cannot overwrite primary evaluation records.

---

## 4. Operational Modes: Continual vs. Episodic Reset

CoTTA was conceived for continual adaptation, whereas disaster response deployments often operate on discrete flood events. To provide a rigorous scientific evaluation, the implementation must evaluate two distinct modes:

1. **Continual Mode (`cotta_continual`):**
   The student $\theta_s$, teacher $\theta_t$, and optimizer states carry over across all event boundaries without reset. This directly tests the upstream hypothesis: does stochastic source restoration prevent representation collapse over a long, multi-event stream?

2. **Episodic Reset Mode (`cotta_reset_per_event`):**
   At each event boundary, student parameters $\theta_s$ and teacher parameters $\theta_t$ reset bit-exactly to the frozen source checkpoint $\theta_0$, and optimizer state is reinitialized. Within each event, teacher EMA, augmentation averaging, and stochastic source restoration operate normally. This isolates intra-event error accumulation mitigation.

Both modes are executed under identical seeds and presentation orders.

---

## 5. Permitted & Prohibited Deviations

| Component | Upstream CoTTA Formulation | ChangeGuard Formulation | Permitted? | Justification |
| :--- | :--- | :--- | :---: | :--- |
| **Augmentation Set** | Color jitter, Gaussian blur, random crop | Horizontal/vertical flips only | **Yes (Required)** | SAR backscatter is radiometrically calibrated; photometric alterations destroy physical scattering signatures. |
| **Spatial Aggregation** | Image-level classification | Dense pixel-level cross-entropy | **Yes (Required)** | ChangeGuard is a dense 3-class segmentation task. |
| **Confidence Mask** | Unfiltered cross-entropy | $\tau = 0.90$ pixel-level mask | **Yes (Permitted)** | Matches ChangeGuard candidate baseline protocol; prevents noisy pseudo-labels on ambiguous background pixels. |
| **Reset Semantics** | Continual only | Both Continual and Reset-per-Event | **Yes (Permitted)** | Tests upstream continual claims while assessing operational event-reset protocol. |
| **Target Labels** | Zero target labels | Zero target labels | **Strict Requirement** | Mandatory unsupervised setting. |
| **Gradient Clipping** | None | None | **Strict Requirement** | No heuristic gradient clipping without experimental evidence. |

---

## 6. Verification and Acceptance Criteria (E7 Preflight)

Before launching any formal GPU evaluation in Work Package E7, the CoTTA implementation must satisfy the following verification sequence:

### 6.1 CPU Synthetic Contract Tests (`tests/test_e7_cotta_adapter.py`)
1. **Teacher EMA Invariance:** Verify that a student forward-backward-step updates teacher weights by exactly $\theta_t \leftarrow \alpha \theta_t + (1-\alpha) \theta_s$.
2. **Augmentation Inversion Identity:** Verify on synthetic spatial patterns that $\mathcal{A}_k^{-1}(\mathcal{A}_k(T)) \equiv T$ for all $k \in \{1, 2, 3, 4\}$.
3. **Stochastic Restoration Distribution:** Run 100,000 parameter updates on dummy weights; verify that the empirical restoration rate is within $0.010 \pm 0.002$.
4. **Valid Mask Sealing:** Verify that zero gradients flow to parameters from pixels where $M_{NA} = 0$.
5. **Bit-Exact Source Reset:** In reset mode, verify that `adapter.reset_to_source()` restores `state_dict` bit-for-bit to $\theta_0$ SHA-256.

### 6.2 Bounded GPU Smoke Test
- **Target Event:** Event 520 (USA/Florence, small development event).
- **Execution Budget:** Exactly 2 batches ($N = 32$ samples).
- **Verification:** Peak VRAM $\le 8.0$ GB; zero CUDA NaNs; zero non-finite losses; execution time $\le 60$ seconds.

### 6.3 Incompatibility & Rejection Criteria
CoTTA will be formally rejected as `INCOMPATIBLE` if:
1. Teacher EMA updates cause gradient explosion or non-finite logits on any development event.
2. Augmentation averaging introduces spatial misalignment artifacts that degrade Macro F1 below AdaBN on $>50\%$ of development events.
3. Stochastic source restoration introduces random parameter noise that destabilizes Batch Normalization running statistics under batch size 16.

---

## 7. Artifacts & Logging Requirements

When executed in E7, the CoTTA evaluation will produce:
- Execution config: `configs/changeguard/e7_cotta_dev6_v1.json`
- Frozen output directory: `outputs/e7_cotta_evaluation/`
- Prediction caches: `frozen_predictions.npz` (per-sample spatial predictions)
- Stream summary: `stream_summary.json` containing exact pixel-level confusion matrices, Precision, Recall, Macro F1, and Pooled F1.
- All records written in English; zero sealed-test access.
