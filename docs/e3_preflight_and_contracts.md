# ChangeGuard Plan V3: Work Package E3 Preflight Specification & Anchor Contract

**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 8: Block E3)  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Execution Environment:** Miniconda `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`)  
**Base Terminal Commit:** `2d37db8f1884c946f95f63870a2aba7f7d502184`  
**Status:** IMPLEMENTER PREFLIGHT SPECIFICATION & MEASUREMENT CONTRACT (LOCKED)

---

## 1. Executive Context & Methodological Role

Under Plan V3 Section 8, Work Package **`E3`** systematically investigates whether regularizing adaptation toward an **immutable source-reference model** prevents adaptation collapse or improves test-time adaptation performance for bi-temporal SAR flood change detection.

### Critical Epistemic Guardrails
1. **Source Reference, Not Ground Truth:** The frozen source model is an **immutable source-reference model**, not ground truth. Source agreement is not correctness. A large anchor penalty can force the candidate model to mimic the unadapted source model without detecting or preventing true adaptation risk.
2. **Internal Evaluation Platform:** The base consistency recipe ($\tau = 0.90$ with identity-weak vs. horizontal-flip-strong augmentation) is retained from Block E2 strictly as an **internal evaluation platform**, *not* as a promoted standalone method. Neither confidence filtering nor geometric consistency beat matched controls in Block E2 (`NONE RETAINED`).
3. **No Dynamic Hyperparameter Searching:** Exactly nine screening cells are defined across three families and three fixed weights $\lambda \in \{0.10, 0.50, 1.00\}$. No intermediate or post-hoc lambda values, projection layers, extra teachers, or event-specific tuning will be introduced after observing development metrics.

---

## 2. Workload & Resource Realities

- **Sample and Batch Accounting:**  
  The 6 development events contain exactly $N = 2,826$ patch pairs ($256 \times 256$ pixels, 2 temporal channels $[VV, VH]$ each).  
  At batch size 16 with natural tail, this produces exactly **179 batches total across all six events**:
  - Event 279: 728 samples $\to$ 46 batches (45 full of 16, 1 tail of 8)
  - Event 437: 412 samples $\to$ 26 batches (25 full of 16, 1 tail of 12)
  - Event 520: 490 samples $\to$ 31 batches (30 full of 16, 1 tail of 10)
  - Event 559: 496 samples $\to$ 31 batches (31 full of 16, 0 tail)
  - Event 1111003: 244 samples $\to$ 16 batches (15 full of 16, 1 tail of 4)
  - Event 1111008: 456 samples $\to$ 29 batches (28 full of 16, 1 tail of 8)
  - Average event size: $\approx 471$ samples per event (not 471 batches).
- **VRAM Usage Bound:**  
  In Block E2, peak reserved VRAM reached approximately $18.4 \text{ GB}$. Work Package E3 introduces a second (frozen source) model branch, retained computation graphs for decomposed gradient telemetry, and intermediate feature tensor caching. Memory consumption cannot be predicted theoretically and is treated as **unknown until empirically measured via real smoke tests**.

---

## 3. Anchor-View Contract & Geometric Semantics

### 3.1 Historical Definition and Implementation Decision
In historical Week 13 (`C3_anchored_consistency_v1`) and the Plan V3 weak-strong consistency framework:
- The candidate model evaluates the **weak view** $x_{weak} = (x_A, x_B)$ to obtain pseudo-labels and confidence masks $c_{weak} \ge \tau$.
- The candidate model evaluates the **strong view** $x_{strong} = (\text{flip}(x_A, \text{dims}=[-1]), \text{flip}(x_B, \text{dims}=[-1]))$, which is realigned into the weak/original coordinate frame via horizontal flip. Parameter gradients are backpropagated into this realigned strong pass.
- The immutable source-reference model evaluates the **weak/original view** $x_{weak} = (x_A, x_B)$ under frozen training running statistics.

### 3.2 Formal Anchor-View Decision
**We formally declare and lock the Weak-Source versus Aligned-Strong-Candidate anchor-view convention across all three anchor families:**
- Source target: evaluated on original weak view $x_{weak}$.
- Candidate prediction/representation: evaluated on strong view $x_{strong}$ and realigned back into weak coordinates before computing anchor discrepancies.

### 3.3 Physical and Scientific Interpretation
Because source targets are evaluated on the weak view and candidate representations are evaluated on the realigned strong view:
> **Explicit Guardrail:** The raw anchor loss $\mathcal{L}_{anc}$ may reflect **both** normalization-context differences and view/augmentation differences, in addition to parameter drift. It must **not** be attributed solely to model parameter drift.

This convention is applied consistently across Probability, Logit, and Feature anchor families.

---

## 4. Formal Mathematical Contract for All Three Anchor Families

For each batch step, given inputs $x_A, x_B \in \mathbb{R}^{B \times 2 \times H \times W}$ and binary sensory valid mask $M_{valid} \in \{0, 1\}^{B \times H \times W}$:

### 4.1 Base Consistency Component
- Weak candidate logits: $z_{cand, weak} = f_\theta(x_A, x_B)$
- Weak candidate probabilities: $p_{cand, weak} = \text{Softmax}(z_{cand, weak}, \text{dim}=1)$
- Selection mask: $\Omega_{sel} = \{ (b, h, w) \mid M_{valid}(b, h, w) = 1 \;\land\; \max_c p_{cand, weak, c}(b, h, w) \ge 0.90 \}$
- Strong candidate logits: $z_{cand, strong} = f_\theta(\text{flip}(x_A, [-1]), \text{flip}(x_B, [-1]))$
- Aligned candidate logits: $z_{cand, aligned} = \text{flip}(z_{cand, strong}, [-1])$
- Aligned candidate log-probabilities: $\log p_{cand, aligned} = \text{LogSoftmax}(z_{cand, aligned}, \text{dim}=1)$
- Base consistency loss:
  $$\mathcal{L}_{cons} = \begin{cases} 
  \frac{1}{|\Omega_{sel}|} \sum_{i \in \Omega_{sel}} \sum_{c=0}^2 \text{detach}(p_{cand, weak, c}(i)) \left[ \log(\text{detach}(p_{cand, weak, c}(i)) + 10^{-12}) - \log p_{cand, aligned, c}(i) \right], & |\Omega_{sel}| > 0 \\
  0.0, & |\Omega_{sel}| = 0
  \end{cases}$$

### 4.2 Family 1: Frozen-Source Probability Anchor (KL Divergence)
- Immutable source-reference probabilities: $p_{src} = \text{detach}(\text{Softmax}(f_{\theta_{src}}(x_A, x_B), \text{dim}=1))$
- Valid mask: $\Omega_{valid} = \{ (b, h, w) \mid M_{valid}(b, h, w) = 1 \}$
- KL Direction: **Forward KL** ($D_{KL}(p_{src} \parallel p_{cand, aligned})$):
  $$\mathcal{L}_{anc, prob} = \begin{cases}
  \frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \sum_{c=0}^2 p_{src, c}(i) \left[ \log(p_{src, c}(i) + 10^{-12}) - \log p_{cand, aligned, c}(i) \right], & |\Omega_{valid}| > 0 \\
  0.0, & |\Omega_{valid}| = 0
  \end{cases}$$
- Total objective: $\mathcal{L}_{total} = \mathcal{L}_{cons} + \lambda_{prob} \cdot \mathcal{L}_{anc, prob}, \quad \lambda_{prob} \in \{0.10, 0.50, 1.00\}$

### 4.3 Family 2: Frozen-Source Logit Anchor (Shift-Invariant Centered MSE)
- Immutable source-reference logits: $z_{src} = \text{detach}(f_{\theta_{src}}(x_A, x_B)) \in \mathbb{R}^{B \times 3 \times H \times W}$
- Aligned candidate logits: $z_{cand, aligned} \in \mathbb{R}^{B \times 3 \times H \times W}$
- Per-pixel class-mean centering (invariance to common-mode additive shifts):
  $$\tilde{z}_{src, c}(i) = z_{src, c}(i) - \frac{1}{3} \sum_{k=0}^2 z_{src, k}(i)$$
  $$\tilde{z}_{cand, c}(i) = z_{cand, aligned, c}(i) - \frac{1}{3} \sum_{k=0}^2 z_{cand, aligned, k}(i)$$
- Fixed scale: Temperature $T = 1.0$ (no label-tuned scaling)
- Raw logit anchor loss:
  $$\mathcal{L}_{anc, logit} = \begin{cases}
  \frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \frac{1}{3} \sum_{c=0}^2 \left( \tilde{z}_{cand, c}(i) - \tilde{z}_{src, c}(i) \right)^2, & |\Omega_{valid}| > 0 \\
  0.0, & |\Omega_{valid}| = 0
  \end{cases}$$
- Total objective: $\mathcal{L}_{total} = \mathcal{L}_{cons} + \lambda_{logit} \cdot \mathcal{L}_{anc, logit}, \quad \lambda_{logit} \in \{0.10, 0.50, 1.00\}$

### 4.4 Family 3: Frozen-Source Feature Anchor (Penultimate ECAM Bottleneck)
- Selected Layer: Dense input to `model.conv_final` ($C = 128$ channels, $H = 256, W = 256$)
- Immutable source-reference features: $F_{src} = \text{detach}(\text{Hook}(f_{\theta_{src}}(x_A, x_B))) \in \mathbb{R}^{B \times 128 \times H \times W}$
- Aligned candidate features: $F_{cand, aligned} = \text{flip}(\text{Hook}(f_\theta(x_{strong})), [-1]) \in \mathbb{R}^{B \times 128 \times H \times W}$
- Normalization: Per-pixel unit $L_2$ channel normalization $\hat{F}(i) = \frac{F(i)}{\|F(i)\|_2 + 10^{-8}}$
- Bounded cosine distance metric ($d(i) \in [0, 2]$):
  $$d(i) = 1 - \langle \hat{F}_{cand, aligned}(i), \hat{F}_{src}(i) \rangle = \frac{1}{2} \|\hat{F}_{cand, aligned}(i) - \hat{F}_{src}(i)\|_2^2$$
- Raw feature anchor loss:
  $$\mathcal{L}_{anc, feat} = \begin{cases}
  \frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \left( 1 - \langle \hat{F}_{cand, aligned}(i), \hat{F}_{src}(i) \rangle \right), & |\Omega_{valid}| > 0 \\
  0.0, & |\Omega_{valid}| = 0
  \end{cases}$$
- Total objective: $\mathcal{L}_{total} = \mathcal{L}_{cons} + \lambda_{feat} \cdot \mathcal{L}_{anc, feat}, \quad \lambda_{feat} \in \{0.10, 0.50, 1.00\}$

---

## 5. Frozen Source-Branch Immutability Contract

1. **Physical Storage Independence:** Source model is an entirely separate object in Python. `data_ptr()` intersection between candidate and source parameters/buffers must be empty (`set()`).
2. **Evaluation Mode:** Source model is permanently set to `eval()`.
3. **No Gradients:** All source parameters have `requires_grad = False` and are strictly excluded from the optimizer.
4. **No Running-Stat Updates:** Source BatchNorm layers keep `track_running_stats = True` with frozen running statistics from the training checkpoint.
5. **Cryptographic Checksum Invariance:** Full SHA-256 state-dict checksum of source model is verified before and after each stream:
   $$\text{SHA256}(\theta_{src})_{\text{start}} == \text{SHA256}(\theta_{src})_{\text{end}}$$
6. **Zero Gradient Accumulation:** Source parameters are checked to ensure `p.grad is None` after every step.

---

## 6. Expanded Gradient Telemetry Specification

Equal numeric $\lambda$ values across different loss formulations do not imply equal gradient regularization strength. To ensure full visibility without mutating optimizer updates:
1. **Gradient Extraction:**
   $$g_{cons} = \nabla_\theta \mathcal{L}_{cons}$$
   $$g_{anc} = \nabla_\theta \mathcal{L}_{anc}$$
   $$g_{total} = g_{cons} + \lambda \cdot g_{anc}$$
2. **Recorded Step Metrics:**
   - `loss_cons_raw`: $\mathcal{L}_{cons}$
   - `loss_anc_raw`: $\mathcal{L}_{anc}$
   - `loss_anc_weighted`: $\lambda \cdot \mathcal{L}_{anc}$
   - `loss_total`: $\mathcal{L}_{total}$
   - `grad_norm_consistency`: $\|g_{cons}\|_2$ (raw consistency gradient norm)
   - `grad_norm_anchor_raw`: $\|g_{anc}\|_2$ (raw anchor gradient norm)
   - `grad_norm_anchor_weighted`: $\lambda \cdot \|g_{anc}\|_2$ (weighted anchor gradient norm)
   - `grad_norm_total`: $\|g_{total}\|_2$ (total gradient norm)
   - `grad_norm_ratio_weighted`: $\frac{\lambda \|g_{anc}\|_2}{\|g_{cons}\|_2 + 10^{-12}}$ (weighted anchor to consistency gradient ratio)
   - `grad_cosine_conflict`: $\cos(g_{cons}, g_{anc}) = \frac{\langle g_{cons}, g_{anc} \rangle}{\|g_{cons}\|_2 \|g_{anc}\|_2 + 10^{-12}}$
   - `update_norm_l2`: $\|\theta_{t+1} - \theta_t\|_2$ (Euclidean parameter displacement after optimizer step)
   - `bn_weight_displacement`: $\|w_{cand} - w_{src}\|_2$
   - `bn_bias_displacement`: $\|b_{cand} - b_{src}\|_2$
   - `prob_disagreement_pre`: $\frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \frac{1}{2} \|p_{cand, weak}(i) - p_{src}(i)\|_1$
3. **Telemetry Non-Interference:** Component gradients are extracted into temporary tensors without altering `p.grad`, parameter buffers, or optimizer momentum state.

---

## 7. Exact Global Method-Promotion Boolean Logic (Plan V3 Section 1.3)

Plan V3 Section 1.3 defines the method-promotion target verbatim:
> "The original aspiration of approximately 1-2 percentage points of macro-F1 improvement, or meaningful worst-event improvement without an average loss, remains a method-promotion target."

Reproduced in exact Boolean logic:
$$\text{IsPromoted} \iff (\Delta \text{MacroF1} \ge 0.01) \lor (\Delta \text{WorstEvent} > 0 \land \Delta \text{MacroF1} \ge 0.0)$$
relative to matched controls (AdaBN: $0.573297$; Base Consistency: $0.573318$; Worst-Event Event 279: $0.2650$).

This exact rule will be evaluated mechanically in the deterministic verifier without post-hoc modifications.

---

## 8. Historical Reuse Ledger Summary

| Cell ID | Condition | Status | Action |
| :--- | :--- | :---: | :--- |
| **RUN-W13-C3** | $\lambda = 0.50$ Probability KL | `HISTORICAL_CONTEXT_ONLY` | Lacks decomposed gradient norms, cosine conflict, and BN displacements. **Executed fresh in E3.** |
| **All Other Cells**| Logit & Feature ($\lambda \in \{0.10, 0.50, 1.00\}$) | `MISSING` | **Executed fresh in E3.** |

---

## 9. Locked Nine-Cell Screening Matrix

| Cell ID | Family | Weight ($\lambda$) | Stream Order | Reset Mode | Checkpoint | Trainable Scope |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`E3_PROB_01`** | Probability (KL) | $0.10$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_PROB_05`** | Probability (KL) | $0.50$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_PROB_10`** | Probability (KL) | $1.00$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_LOGIT_01`**| Logit (Centered MSE)| $0.10$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_LOGIT_05`**| Logit (Centered MSE)| $0.50$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_LOGIT_10`**| Logit (Centered MSE)| $1.00$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_FEAT_01`** | Feature (ECAM) | $0.10$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_FEAT_05`** | Feature (ECAM) | $0.50$ | Order A | Event | Seed 0 | Candidate BN Affine |
| **`E3_FEAT_10`** | Feature (ECAM) | $1.00$ | Order A | Event | Seed 0 | Candidate BN Affine |

---

## 10. Offline Interpretation Metrics Contract (Computed Post-Persistence Only)

To prevent misinterpreting simple adaptation suppression as successful regularized adaptation, all anchor cells undergo rigorous offline decomposition against both the matched Base Consistency baseline (`E2_CONF_90`) and the immutable source-reference model after predictions are durably persisted and ground-truth labels are legally unlocked.

### 10.1 Four Mandatory Epistemic Guardrails
1. **Source agreement is NOT correctness:** The frozen source model is an unadapted baseline, not ground truth. A model that achieves high source agreement may simply be failing to adapt to valid domain shifts.
2. **Approaching source predictions is NOT sufficient evidence of successful anchoring:** Mere reduction in parameter or prediction drift does not establish that the anchor prevented failure modes; it may simply have frozen the model.
3. **Aggregate F1 improvement may result from suppressing adaptation rather than detecting risk:** If an anchor improves average F1 by broadly muting updates across all events, it represents conservative inertia rather than selective risk-aware adaptation.
4. **Beneficial-adaptation losses must be reported alongside recovered harmful adaptations:** Anchoring inevitably incurs a trade-off. Any recovery of harmful base errors must be weighed against the forfeiture of beneficial adaptations where the base method successfully adapted.

### 10.2 Mathematical Metric Definitions (Over Double-Masked Valid Pixels $\Omega_{valid}$)
Let $Y \in \{0, 1\}$ be binary ground truth flood ($Y = \mathbb{I}(\text{label} == 2)$), $A$ be candidate anchor prediction, $B$ be matched base consistency prediction (`E2_CONF_90`), and $S$ be immutable source prediction.
- **Source-Candidate Prediction Disagreement Rate:** $\frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \mathbb{I}(A(i) \ne S(i))$
- **Source Agreement Rate:** $\frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \mathbb{I}(A(i) == S(i))$
- **Base Consistency Agreement Rate:** $\frac{1}{|\Omega_{valid}|} \sum_{i \in \Omega_{valid}} \mathbb{I}(A(i) == B(i))$
- **Harmful Base Error Recovery Count & Rate:**
  - Count: $\sum_{i \in \Omega_{valid}} \mathbb{I}(S(i) == Y(i) \;\land\; B(i) \ne Y(i) \;\land\; A(i) == Y(i))$
  - Rate relative to harmful base errors: $\frac{\text{HarmfulRecoveryCount}}{\sum \mathbb{I}(S(i) == Y(i) \land B(i) \ne Y(i)) + 10^{-12}}$
- **Beneficial Adaptation Retention Count & Rate:**
  - Count: $\sum_{i \in \Omega_{valid}} \mathbb{I}(S(i) \ne Y(i) \;\land\; B(i) == Y(i) \;\land\; A(i) == Y(i))$
  - Rate relative to beneficial adaptations: $\frac{\text{BeneficialRetentionCount}}{\sum \mathbb{I}(S(i) \ne Y(i) \land B(i) == Y(i)) + 10^{-12}}$
- **Beneficial Adaptation Loss Count & Rate:**
  - Count: $\sum_{i \in \Omega_{valid}} \mathbb{I}(S(i) \ne Y(i) \;\land\; B(i) == Y(i) \;\land\; A(i) \ne Y(i))$
  - Rate relative to beneficial adaptations: $\frac{\text{BeneficialLossCount}}{\sum \mathbb{I}(S(i) \ne Y(i) \land B(i) == Y(i)) + 10^{-12}}$
  - Conservation Invariant: $\text{RetentionCount} + \text{LossCount} \equiv \text{TotalBeneficialAdaptations}$
- **Regression from Base Rate:** $\frac{\sum \mathbb{I}(B(i) == Y(i) \land A(i) \ne Y(i))}{\sum \mathbb{I}(B(i) == Y(i)) + 10^{-12}}$
- **Correction over Base Rate:** $\frac{\sum \mathbb{I}(B(i) \ne Y(i) \land A(i) == Y(i))}{\sum \mathbb{I}(B(i) \ne Y(i)) + 10^{-12}}$


