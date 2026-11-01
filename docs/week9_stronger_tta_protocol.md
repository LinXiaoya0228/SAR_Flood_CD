# Week 9 Stronger TTA Protocol & Baseline Benchmarking Specification

> **STATUS: DRAFT / IMPLEMENTED FOR CPU VERIFICATION**  
> *This document specifies the research protocol, mathematical formulations, transfer rules, and experimental design for evaluating stronger published Test-Time Adaptation (TTA) baselines in Week 9. Verification is conducted under strict CPU tests only; no GPU execution or real-data adaptation is launched without explicit user authorization.*

**Document ID:** `docs/week9_stronger_tta_protocol.md`  
**Configuration File:** `configs/evaluation/week9_stronger_tta_protocol.json`  
**Date (UTC):** 2026-10-17  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Target Model:** Model A (`SNUNet_ECAM [VV, VH]`, 2 channels, base 32, FP32)  
**Source Checkpoint Commit:** `6c63a5f8aa7ae2c608b145f0badc84b422b62ef1` (`outputs/week5_source_all27/`)  
**Target Events (6 TTA Development Events):** `[279, 437, 520, 559, 1111003, 1111008]` (2,826 samples, 135,859,080 valid pixels)  
**Approved Source Training Events (27 Events):** `[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]` (21,462 patches, defined in `configs/splits/kuro_source_all27_split_v1.json`)  
**Quarantined Events (10 Sealed Test Events):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (Strictly zero access)  

---

## 1. Scientific Motivation & Context

In Week 8, the formal evaluation matrix for standard unsupervised TTA—Adaptive Batch Normalization (AdaBN) and reference-aligned Test-Time Entropy Minimization (Tent)—decisively satisfied **Gate A2** (`harm_observed: True`). Across 30 physical streams, 36 logical cells, and 216 logical event evaluations, adaptation induced severe, reproducible performance degradation ($\Delta F_1 \le -0.0100$, dropping up to $-0.0959$) in at least two of three identical checkpoint seeds across both stream presentation orders (Order A and Order B) for 15 method/reset/event combinations on the 6 development events.

### The Question for Week 9
Standard Tent adapts unconditionally on all incoming batches by minimizing vanilla Shannon entropy, making it vulnerable to:
1. **Unreliable gradients** from high-entropy, noisy, or ambiguous pixels/patches.
2. **Redundant gradient updates** on trivial, repeated background scenes.
3. **Catastrophic forgetting** of source knowledge, as normalization parameters drift unconstrained.
4. **Sharp/unstable minima**, where parameter updates lead to loss spikes under severe domain shifts.

Week 9 investigates whether **stronger published TTA algorithms with active sample filtering and regularization** can overcome these vulnerabilities in dual-input SAR flood change detection, or whether unsupervised adaptation continues to exhibit systematic failure modes that necessitate an autonomous risk controller.

The primary algorithms selected for evaluation are:
- **EATA (Efficient Anti-Forget Test-Time Adaptation, ICML 2022)**: Introduces entropy-based reliability filtering, moving-average cosine redundancy filtering, and Fisher-information anti-forgetting regularization.
- **SAR-TTA (Sharpness-Aware and Reliable TTA, ICLR 2023 Oral)**: Introduces reliable sample filtering, Sharpness-Aware Minimization (SAM) to steer parameters toward flat loss basins, and an automated model recovery scheme.
- **CoTTA (Continual Test-Time Adaptation, CVPR 2022)**: Preserved as a secondary fallback baseline.

---

## 2. Upstream Codebase Provenance & Licensing

To maintain strict scientific reproducibility and licensing integrity, the evaluated methods are anchored to the official upstream reference repositories:

| Algorithm | Venue | Official Repository | Exact Commit SHA | Date | License | Upstream Source Files |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **EATA** | ICML 2022 | `https://github.com/mr-eggplant/EATA` | `f739b3668cc7617e9b9f1979c1a358497a3472c3` | 2023-05-19 | MIT License | `eata.py`, `main.py` |
| **SAR-TTA** | ICLR 2023 (Oral) | `https://github.com/mr-eggplant/SAR` | `20f6e24b17525f34503510afccedc0629b67b7c4` | 2023-09-08 | BSD 3-Clause | `sar.py`, `sam.py`, `main.py` |
| **CoTTA** *(Fallback)* | CVPR 2022 | `https://github.com/qinenergy/cotta` | `c212a204b32be4005092e4323105a24a29ad2952` | 2024-06-11 | MIT License | `cota.py` |

Both EATA and SAR were authored by the same research group (Niu et al.) under permissive open-source licenses (MIT and BSD 3-Clause), facilitating a unified and mathematically consistent implementation.

---

## 3. Systematic Distinctions: Upstream Defaults vs. Transfer Rules vs. Safety Fixes

To maintain absolute scientific transparency, we clearly separate:
1. **Upstream Defaults**: Original parameters designed for ImageNet classification ($C=1000$).
2. **Project Transfer Rules**: Approved adjustments required to transfer methods to dense 3-class SAR change detection.
3. **Implementation Safety Fixes**: Algorithmic safeguards preventing silent failures, division-by-zero, parameter corruptions, or data leakage.

| Dimension | (1) Upstream Default | (2) Approved Project Transfer Rule | (3) Implementation Safety Fix |
| :--- | :--- | :--- | :--- |
| **Task & Architecture** | Single RGB image classification ($C=1000$) | Dense bi-temporal change detection ($C=3$, dual 2-channel `[VV, VH]` SAR) | Strict preservation of frozen `SNUNet_ECAM` architecture and weights |
| **Entropy Granularity** | Scalar image entropy $H = -\sum p_c \ln p_c$ | Patch-level sensory-valid mean entropy $H_b = \frac{1}{V_b} \sum_{\mathcal{V}_b} H_{b,i,j}$ | Squeezes masks safely; returns 0 if valid pixel count $V_b = 0$ |
| **Sensory Masking** | None (natural RGB images) | Sensory radar validity `MK0_MNA == 1` only | Zero ground-truth access; `MK0_MLU` strictly excluded from adaptation |
| **Reliability Threshold ($E_0$)** | EATA: $\frac{\ln(1000)}{2}-1 \approx 2.45$; SAR: $0.40 \ln(1000) \approx 2.76$ | Scaled to $E_0 = 0.40 \times \ln(3) \approx \mathbf{0.4394449}$ | Avoids EATA's classification formula which yields negative ($-0.4507$) for $C=3$ |
| **Redundancy Threshold ($d_{\text{margin}}$)** | $d_{\text{margin}} = 0.05$ (EATA) | $d_{\text{margin}} = 0.05$ maintained | Compares patch-mean probability vector against EMA $\bar{p} \in \Delta^2$ |
| **Fisher Regularization ($\alpha$)** | $\alpha = 2000.0$ (EATA) | $\alpha = 2000.0$ maintained | Precomputed on source training split; no gradients to non-affine weights |
| **SAM Perturbation Radius ($\rho$)** | $\rho = 0.05$ (SAR) | $\rho = 0.05$ maintained | If stage 2 fails/aborts, safely restores perturbation to original weights |
| **Recovery Threshold ($e_m$)** | Hardcoded `ema < 0.2` in `sar.py` | Configured: $0.2 \times \frac{\ln(3)}{\ln(1000)} \approx \mathbf{0.031808}$ | Replaces hardcoded constant with configurable class-scaled threshold |
| **Batch Size ($B$)** | $B = 64$ | $B = 16$ (retaining all tail batches) | Tail batch numerical checks; handles varying batch sizes safely |
| **Optimizer & Learning Rate** | SGD $\text{lr} = (0.00025 / 64) \times B \times 2 = 1.25 \times 10^{-4}$, momentum $0.9$ | SGD $\text{lr} = 1.25 \times 10^{-4}$, momentum $0.9$, weight_decay $0.0$ | Documented difference from Week 8 Adam ($\text{lr}=10^{-4}$) |
| **Empty Selection Handling** | Not explicitly guarded in upstream loop | If $|\mathcal{S}| = 0$, log skipped update cleanly | No backward pass, zero grad step, no threshold relaxation |

### 3.1 Optimizer & Learning Rate Provenance: SGD vs. Week 8 Adam
- **Week 8 Tent**: Followed Wang et al. (ICLR 2021) which specified Adam with $\text{lr} = 10^{-4}$, $\beta = (0.9, 0.999)$, and $\text{weight\_decay} = 0.0$.
- **Week 9 EATA and SAR**: The official codebases by Niu et al. explicitly adopt SGD with momentum $0.9$, weight decay $0.0$, and a linear scaling rule based on batch size:
  $$\text{lr} = \left( \frac{0.00025}{64} \right) \times B \times 2 = \frac{0.00025}{64} \times 16 \times 2 = \mathbf{1.25 \times 10^{-4}}$$
  Adopting SGD ($\text{lr}=1.25 \times 10^{-4}$, $\text{momentum}=0.9$) adheres directly to the upstream authors' validated optimizer configuration for both EATA and SAR.

---

## 4. Method Specifications

### 4.1 EATA (Efficient Anti-Forget Test-Time Adaptation)

EATA incorporates three complementary mechanisms into test-time adaptation:

#### 1. Reliable Sample Selection (Entropy Filtering)
Filter out ambiguous patches:
$$\mathcal{S}_{\text{reliable}} = \{b \in \mathcal{B} \mid H_b < E_0\}$$
where $E_0 = 0.40 \ln(3) \approx 0.4394449$.

#### 2. Non-Redundant Sample Selection (Cosine Similarity Filtering)
Maintain a running exponential moving average of predicted class probabilities across newly selected active samples (matching upstream `eata.py`):
$$\bar{p} \leftarrow 0.9 \, \bar{p} + 0.1 \left( \frac{1}{|\mathcal{S}_{\text{active}}|} \sum_{b \in \mathcal{S}_{\text{active}}} p_b \right)$$
For each reliable sample $b \in \mathcal{S}_{\text{reliable}}$, compute cosine similarity with the historical mean:
$$\text{Sim}(p_b, \bar{p}) = \frac{p_b \cdot \bar{p}}{\|p_b\|_2 \|\bar{p}\|_2 + 10^{-12}}$$
Filter out redundant samples that do not provide informative diversity:
$$\mathcal{S}_{\text{active}} = \{b \in \mathcal{S}_{\text{reliable}} \mid |\text{Sim}(p_b, \bar{p})| < d_{\text{margin}}\}$$
where $d_{\text{margin}} = 0.05$ (upstream default).

#### 3. Reweighted Entropy Loss
To penalize uncertain samples even within the active set:
$$\mathcal{L}_{\text{entropy}} = \frac{1}{|\mathcal{S}_{\text{active}}|} \sum_{b \in \mathcal{S}_{\text{active}}} w_b H_b, \quad \text{where } w_b = \frac{1}{\exp(H_b - E_0)}$$

#### 4. Anti-Forgetting Fisher Regularization
To preserve source knowledge and prevent parameter drift, EATA adds an elastic weight consolidation penalty against the initial source checkpoint parameters $\theta^0$ (matching upstream `eata.py:126` with coefficient $\alpha$):
$$\mathcal{L}_{\text{fisher}} = \sum_{l \in \Theta_{\text{affine}}} \alpha F_l (\theta_l - \theta_l^0)^2$$
where $F_l$ is the diagonal Fisher information value for parameter $\theta_l$, and $\alpha = 2000.0$ (upstream trade-off hyperparameter).

The total objective optimized per batch is:
$$\mathcal{L}_{\text{EATA}} = \mathcal{L}_{\text{entropy}} + \mathcal{L}_{\text{fisher}}$$

#### EATA Safe Execution & State Management:
- **Empty Batch Handling:** If $|\mathcal{S}_{\text{active}}| = 0$, the update is skipped entirely: no backward pass, no optimizer step, and `num_skipped_updates` is incremented.
- **Prediction Timing:** Predictions returned for evaluation are computed during the initial forward pass prior to parameter update (or without second post-update forward).
- **Episodic Reset (`reset_per_event`):** Restores model weights to initial source state, reinitializes SGD optimizer, resets running probability vector $\bar{p} = \text{None}$, and resets step and sample counters.

---

### 4.2 EATA Fisher Information Matrix Preparation Protocol

The Fisher information matrix $F$ measures parameter sensitivity on the source distribution. To ensure scientific integrity and eliminate data leakage:

1. **Source Data Isolation:** Fisher computation MUST use **only approved source training events** (`outputs/week5_source_all27/`, Events `[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]`). Target development events (`[279, 437, 520, 559, 1111003, 1111008]`) and sealed test events are strictly forbidden.
2. **Raw Data Paths from Frozen Source Configuration:**
   - Dataset Root: `/usr1/home/s126mdg22_06/FYP_SARCD/datasets/KuroSiwo`
   - Manifest Path: `/usr1/home/s126mdg22_06/FYP_SARCD/third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz`
   - Split Configuration: `configs/splits/kuro_source_all27_split_v1.json` (SHA-256: `601b21c81f7f1e26984fe6507a1532dbfa373d72b29346103fb5f0a1ac38d5e1`)
3. **Deterministic Sample Selection:** Exactly $N = 2,000$ source patches selected without replacement from the 21,462 available source patches using selection seed `2026`. The exact list of selected sample IDs is deterministic and identical across all three source checkpoint seeds.
4. **Empirical Fisher Estimator: Upstream Batch-Mean vs. Implemented Per-Patch Accumulation:**  
   There is a critical mathematical and implementation distinction between upstream EATA and the ChangeGuard implementation:
   
   - **Upstream EATA Estimator (`main.py` lines 152–170):**  
     Upstream evaluates source data in mini-batches of size $B=64$ with the model in `train()` mode (`track_running_stats=False`), computing batch statistics across the mini-batch. Loss is `train_loss_fn(outputs, targets)` (`nn.CrossEntropyLoss(reduction='mean')`), yielding batch-mean loss $\bar{\ell} = \frac{1}{B} \sum_{b=1}^B \ell_b$. Upstream squares the resulting batch gradient:
     $$F_k^{\text{upstream}} = \frac{1}{M} \sum_{m=1}^{M} \left( \frac{1}{B} \sum_{b=1}^B \frac{\partial \ell_{m, b}}{\partial \theta_k} \right)^2 = \frac{1}{M B^2} \sum_{m=1}^M \sum_{b=1}^B \sum_{b'=1}^B g_{m, b, k} g_{m, b', k}$$
     This averages **squared batch-mean gradients**, introducing cross-sample covariance products ($b \ne b'$) and an implicit $1/B$ suppression factor relative to individual sample sensitivity.
   
   - **ChangeGuard Implementation (`tools/prepare_eata_fishers.py`):**  
     ChangeGuard evaluates the 2,000 selected patches individually ($B=1$ microbatching) with the model in `eval()` mode. In this BN context, frozen source running statistics (`running_mean`, `running_var`) normalize activations, avoiding singleton batch variance collapse. For each patch $i \in \{1, \dots, N\}$, cross-entropy against predicted pseudo-labels $\hat{c}^{(i)}$ is averaged over valid sensory pixels ($\mathcal{V}_i$), backward is executed, and squared per-patch gradients are accumulated:
     $$F_k^{\text{per-patch}} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\partial \ell_i}{\partial \theta_k} \right)^2$$
     This computes the classical diagonal empirical Fisher Information Matrix without cross-sample covariance terms.
   
   - **Methodological Deviation Notice:**  
     These two estimators are **not mathematically equivalent**. We explicitly document this as a deliberate methodology deviation from upstream classification code, designed to eliminate mini-batch cross-terms and singleton BN distortion. Per independent review and user directive, the unused and unsupported `batch_mean` branch is explicitly rejected, establishing the exact per-patch estimator (`exact_per_sample` / `per_patch`) as the sole canonical method.

5. **Constructor Defaults vs. Upstream CLI Defaults:**
   - In `eata.py` (`EATA.__init__`): `e_margin = math.log(1000)/2 - 1` ($\approx 2.4539$), `d_margin = 0.05`, `fisher_alpha = 2000.0`. (Note: in `forward_and_adapt_eata`, the function default is `fisher_alpha=50.0`, but the class constructor explicitly passes `2000.0`).
   - In upstream EATA `main.py`: parser sets `--fisher_size 2000`, `--fisher_alpha 2000.0`, `--d_margin 0.05`, with fixed learning rate $0.00025$.
   - **Learning Rate Rule Provenance:** The batch-scaled learning rate formula:
     $$\text{lr} = \left( \frac{0.00025}{64} \right) \times B \times 2 \quad (\text{for } B < 32)$$
     originates from upstream SAR (`SAR/main.py` lines 169–180). For canonical adaptation batch size $B=16$, this evaluates to $\text{lr} = 1.25 \times 10^{-4}$, which is our approved frozen learning rate for both EATA and SAR.

6. **Batch Size Separation:**
   - **Canonical Adaptation Batch Size:** Fixed at $B = 16$ in `stream_execution_policy.batch_size`. Conflicting formal CLI overrides are strictly rejected by the runner.
   - **Fisher Batch Size:** Evaluated with microbatch size $1$ in `fisher_information_protocol.fisher_batch_size`, strictly separate from the adaptation stream batch size.

7. **Storage & Provenance:**
   The Fisher diagonal and initial parameter weights are saved as frozen artifact files in a dedicated cache directory separate from evaluation runs:
   `artifacts/eata_fishers/source_all27_seed{0,1,2}_fisher.pt`  
   A companion metadata JSON records the selected sample IDs, split SHA-256, data paths, loss reduction, and accumulation formula.

---

### 4.3 SAR-TTA (Sharpness-Aware and Reliable TTA)

SAR addresses adaptation instability and gradient noise under shifting test streams using three core components:

#### 1. Reliable Entropy Minimization
Identical to EATA, samples with $H_b \ge E_0 = 0.40 \ln(3) \approx 0.4394449$ are excluded from optimization.
$$\mathcal{L}_{\text{SAR}}(\theta) = \frac{1}{|\mathcal{S}_{\text{reliable}}|} \sum_{b \in \mathcal{S}_{\text{reliable}}} H_b(\theta)$$

#### 2. Sharpness-Aware Minimization (SAM)
Rather than seeking a local minimum that might be sharp and fragile under domain shifts, SAR optimizes for flat minima by seeking parameter regions where loss remains low under worst-case weight perturbations within radius $\rho = 0.05$:

**Step 1 (Ascent / Perturbation):**
Compute gradient on reliable samples:
$$g_1 = \nabla_\theta \mathcal{L}_{\text{SAR}}(\theta)$$
Compute optimal weight perturbation:
$$\hat{\epsilon}(\theta) = \rho \frac{g_1}{\|g_1\|_2 + 10^{-12}}$$
Climb to local maximum:
$$\theta_{\text{perturbed}} = \theta + \hat{\epsilon}(\theta)$$

**Step 2 (Descent / Flatness Update):**
Re-evaluate reliable samples and compute gradients at perturbed weights $\theta_{\text{perturbed}}$:
$$g_2 = \nabla_{\theta_{\text{perturbed}}} \mathcal{L}_{\text{SAR}}(\theta_{\text{perturbed}})$$
Restore original weights and apply base optimizer step using $g_2$:
$$\theta \leftarrow \theta - \eta \cdot \text{BaseOptimizer}(g_2)$$

#### 3. Model Recovery Scheme (Anti-Collapse Reset)
SAR tracks a running exponential moving average of the adapted batch loss:
$$\text{EMA} \leftarrow 0.9 \, \text{EMA} + 0.1 \, \mathcal{L}_{\text{SAR}}(\theta_{\text{perturbed}})$$
If the EMA drops below the configured recovery threshold $e_m$, or if loss/gradient anomalies occur, the adapter automatically triggers a **model recovery reset**, restoring parameters $\theta$ and optimizer state to the initial source checkpoint.

**Threshold Calibration for 3-Class Segmentation:**  
Upstream SAR sets $e_m = 0.20$ for ImageNet ($C=1000, \ln(1000) \approx 6.907755 \implies 0.2 / \ln(1000) \approx 0.02895$).  
For 3-class segmentation:
$$e_m = 0.2 \times \frac{\ln(3)}{\ln(1000)} \approx 0.2 \times \frac{1.098612}{6.907755} = \mathbf{0.031808}$$
If $\text{EMA} < 0.031808$, the model has collapsed to a single-class deterministic state, triggering recovery.

#### SAR Safe Execution & State Management:
- **Empty Stage 1 Handling:** If $|\mathcal{S}_{\text{reliable}}| = 0$, update is skipped cleanly; no SAM steps are taken.
- **Empty Stage 2 / Abort Handling:** If stage 1 succeeded in perturbing weights to $\theta_{\text{perturbed}}$, but stage 2 cannot proceed (e.g. no samples pass stage 2 or numerical anomaly occurs), the adapter **safely restores weights to original unperturbed values $\theta$** before returning.
- **Prediction Timing:** Predictions returned for evaluation are computed during the initial forward pass prior to SAM perturbation.
- **Episodic Reset (`reset_per_event`):** Restores model weights to initial source state, reinitializes SGD/SAM optimizer, resets loss EMA to `None`, and resets step and recovery counters.

---

## 5. Experimental Matrix & Matched Comparison Design

### 5.1 Matched Matrix Structure
To directly isolate the algorithmic effect of EATA and SAR against Week 8 findings, all experimental conditions are strictly matched:
- **Source Checkpoints:** Same 3 frozen Model A checkpoints (Seeds 0, 1, 2).
- **Stream Orders:** Order A (chronological, ascending sample ID) and Order B (reversed, descending sample ID).
- **Reset Modes:** `reset_per_event` (episodic) and `continual` (cross-event persistence).
- **Batch Size:** 16 (retaining all partial tail batches).
- **Target Data:** Same 6 development events (2,826 samples total).

### 5.2 Stream Accounting: 24 New Physical Streams
- **EATA Physical Streams:** $3 \text{ seeds} \times 2 \text{ orders} \times 2 \text{ reset modes} = \mathbf{12 \text{ streams}}$.
- **SAR Physical Streams:** $3 \text{ seeds} \times 2 \text{ orders} \times 2 \text{ reset modes} = \mathbf{12 \text{ streams}}$.
- **Total New Physical Streams:** $\mathbf{24 \text{ streams}}$.

### 5.3 Comparative Matrix Ledger (60 Logical Cells)
Rather than re-running Source-Only, AdaBN, and Tent, the Week 9 evaluation ledger will **explicitly import the verified 36 logical cells from Week 8**, yielding a comprehensive 60-cell comparative ledger:

| Method ID | Method Category | Physical Execution in W9 | Reused from W8 | Logical Cells | Logical Event Rows |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `source_only` | Unadapted Source Control | 0 | 12 (from 6 physical) | 12 | 72 |
| `adabn` | Statistics Adaptation | 0 | 12 | 12 | 72 |
| `tent` | Unconstrained Entropy Min | 0 | 12 | 12 | 72 |
| `eata` | Filtered Entropy + Fisher | **12** | 0 | 12 | 72 |
| `sar` | Flat-Minima Entropy + Recovery | **12** | 0 | 12 | 72 |
| **Total** | | **24** | **36** | **60** | **360** |

All imported rows will be cryptographically audited against the Week 8 frozen SHA-256 digests.
> [!NOTE]
> The generation of this combined 360-row comparative ledger is an explicit postprocessing analysis task to be performed after the 24 new physical Week 9 streams have completed and verified. It is not generated during initial runner execution, and Week 9 is not claimed to be complete until both physical execution and postprocessing synthesis are concluded.

---

## 6. Bounded Diagnostic Smoke Test Protocol

Prior to launching any formal 24-stream matrix execution, a bounded diagnostic smoke test must be executed and audited:

### Smoke Test Parameters
- **Target Model:** Seed 0 (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`).
- **Stream Order:** Order A only.
- **Diagnostic Workload:** Exactly the first 19 samples from Event 437 and first 19 samples from Event 1111003 (38 samples per stream, identical to Week 8 smoke test).
- **Methods Evaluated:** EATA and SAR.
- **Reset Modes:** `reset_per_event` and `continual`.
- **Total Smoke Streams:** Exactly 4 physical streams.

### Diagnostic Verification Criteria
1. **Fisher Loading & Alignment:** Confirm EATA loads `source_all27_seed0_fisher.pt` and correctly maps Fisher diagonal tensors to model parameters.
2. **SAM Dual-Step Mechanics:** Confirm SAR executes both forward-backward steps, scales weight perturbations by $\rho$, and restores weights without memory leaks.
3. **Validity Mask Handling:** Confirm zero NaNs or Infs when batches contain varying numbers of valid sensory pixels.
4. **Filtering Ledger:** Verify that `num_reliable_samples`, `num_redundant_samples`, and `ema_loss` are correctly logged at each batch.
5. **Zero GT Access:** Verify that no dataloader or adapter code accesses `MK0_MLU` during forward/backward steps.

---

## 7. Untested Future Research Hypothesis (Entropy-Distribution Risk Prediction)

> **Untested Future Hypothesis (Entropy-Distribution Risk Prediction):**  
> While EATA and SAR apply static scalar thresholds ($E_0$) derived from global class counts, real-world SAR disaster streams experience dynamic, non-uniform distribution shifts. We hypothesize that a lightweight unsupervised metric measuring statistical divergence (e.g. Wasserstein distance) between the target stream's predictive entropy distribution and a calibrated source reference could serve as a predictive indicator of adaptation degradation (targeting an aspirational benchmark of AUROC $\ge 0.70$ in future controller evaluations).  
> *Status:* **Untested Future Research Hypothesis**. No controller or risk prediction score has yet been implemented, evaluated, or claimed as a completed contribution.

---

## 8. Document Status & Implementation Scope

- **Implementation Status:** Adapters, Fisher extraction logic, matrix runner, and focused CPU verification tests implemented.
- **Execution Boundary:** Verified under CPU synthetic tests. Real-data Fisher calculation, GPU execution, and formal matrix runs remain strictly paused pending user review.
