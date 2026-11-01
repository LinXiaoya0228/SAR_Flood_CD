# Week 13 Frozen-Source Probability-Anchor Protocol (DRAFT)

**Protocol ID:** `week13_source_anchor_dev6_v1_DRAFT`  
**Status:** `DRAFT_PROTOCOL_EXECUTION_UNAUTHORIZED`  
**Evidence Level:** `PROPOSED_DEVELOPMENT_PROTOCOL`  
**Governing Context:** Follows Week 12 exploratory results ([`docs/week12_exploratory_results_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week12_exploratory_results_v2.md))

---

## 1. Motivation & Scientific Rationale

In the Week 12 exploratory ablation, continuous unconstrained test-time adaptation (both Tent entropy minimization and Evidence Consistency C1/C2) degraded performance relative to the frozen source model on 4 out of 6 development events, suffering severe false-positive inflation on Event 279 ($F_1$ dropped from $0.3769$ down to $0.3004$, with predicted flood pixels increasing by $\sim 1.0 \times 10^6$).

Confidence filtering alone (C2 vs C1) filtered 11.21% of sensory pixels but had virtually no metric impact on aggregate degradation ($\Delta F_1 = +0.000305$).

To constrain continuous parameter drift, Week 13 specifies a **frozen-source probability-anchor** comparison:
- **Baseline Reference (C2):** Evidence consistency with confidence filtering ($\tau \ge 0.90$).
- **Anchored Candidate (C3):** The identical C2 objective plus a detached source-probability regularization term:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cons}} + \lambda_{\text{anc}} \mathcal{L}_{\text{anc}}$$

---

## 2. Mathematical Formulation & Architecture Specification

### 2.1 The Two Branches
1. **Candidate Branch ($f_{\text{cand}}$):**
   - Active adapter model with BatchNorm modules set to `train()` and `track_running_stats = False`.
   - Normalizes incoming batch using current batch statistics ($\mu_B, \sigma^2_B$).
   - Computes weak view logits $z_{\text{weak}} = f_{\text{cand}}(x_A, x_B)$ and strong view logits $z_{\text{strong}} = f_{\text{cand}}(\text{flip}(x_A), \text{flip}(x_B))$.
   - Softmax probabilities: $p_{\text{cand}} = \text{softmax}(z_{\text{weak}}, \text{dim}=1)$.
2. **Frozen Source Branch ($f_{\text{source}}$):**
   - Frozen source baseline checkpoint loaded on device in `eval()` mode.
   - Preserves frozen source training running statistics ($\mu_{\text{source}}, \sigma^2_{\text{source}}$).
   - Wrapped under `torch.no_grad()`; zero gradients, no backward computation, no optimizer attachment.
   - Computes source probabilities: $p_{\text{source}} = \text{softmax}(f_{\text{source}}(x_A, x_B), \text{dim}=1)$.

### 2.2 Objective Functions
1. **Consistency Term ($\mathcal{L}_{\text{cons}}$):**
   Identical to C2 ($\tau = 0.90$):
   $$\Omega_{\text{sel}} = \{ i \in \Omega_{\text{valid}} \mid \max_{c \in \{0, 1, 2\}} p_{\text{weak}, i}(c) \ge 0.90 \}$$
   $$\mathcal{L}_{\text{cons}} = \frac{1}{\max(1, |\Omega_{\text{sel}}|)} \sum_{i \in \Omega_{\text{sel}}} D_{\text{KL}}\left( \text{detach}(p_{\text{weak}, i}) \,\|\, p_{\text{strong}, i} \right)$$
2. **Source Probability Anchor Term ($\mathcal{L}_{\text{anc}}$):**
   Measures cross-entropy / Kullback-Leibler divergence between the detached frozen source distribution and the candidate prediction:
   $$\mathcal{L}_{\text{anc}} = \frac{1}{|\Omega_{\text{valid}}|} \sum_{i \in \Omega_{\text{valid}}} D_{\text{KL}}\left( \text{detach}(p_{\text{source}, i}) \,\|\, p_{\text{cand}, i} \right)$$
   where $\Omega_{\text{valid}}$ is the sensory-valid pixel set ($MK0\_MNA == 1$).

### 2.3 Gradient Flow & Trainable Scope
- Gradients flow **strictly** through $p_{\text{strong}, i}$ (in $\mathcal{L}_{\text{cons}}$) and $p_{\text{cand}, i}$ (in $\mathcal{L}_{\text{anc}}$).
- The detached source branch has `requires_grad = False` and runs inside `torch.no_grad()`.
- Trainable scope remains identical to Week 12: BatchNorm affine parameters only ($\gamma, \beta$).

### 2.4 Hyperparameter Budget
- **Anchor Weight ($\lambda_{\text{anc}}$):** Default fixed budget $\lambda_{\text{anc}} = 0.50$ (or a bounded exploration $\{0.1, 0.5, 1.0\}$).
- **Learning Rate:** $1.25 \times 10^{-4}$, SGD momentum $0.9$, weight decay $0$.
- **Batch Size:** 16 with natural tail.
- **Timing:** Strictly `PRE_UPDATE` predictions persisted before ground-truth access.

---

## 3. Matched Controls & Scoping Invariants

1. **Target Events:** Strictly the 6 KuroSiwo development events (`[279, 437, 520, 559, 1111003, 1111008]`). The 10 sealed test events remain forbidden.
2. **Checkpoints:** Frozen SNUNet_ECAM VV/VH seed 0 source checkpoint (`final_checkpoint.pt`).
3. **Stream Configuration:** `order_a` ascending deterministic order; `reset_per_event = true`.
4. **Evaluation Invariant:** Double-masked offline scoring ($\text{valid\_mask} == 1 \land \text{label} \ne 3$) with strict confusion conservation $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels}$.

---

## 4. Known Limitations & Failure Modes

1. **Source Error Ceiling (Anchor Limitation):**
   The source-probability anchor penalizes divergence from the frozen source model. While this protects against severe drift and false positive explosion on long streams, it inherently anchors the candidate to source errors. If the source model suffers severe misclassifications due to water-body backscatter ambiguities, the anchor bounds the candidate's ability to adapt away from those source errors.
2. **Batch Statistics Mismatch:**
   $p_{\text{source}}$ is generated using frozen population running statistics, whereas $p_{\text{cand}}$ is generated using current batch statistics. This distribution shift introduces a non-zero baseline anchor penalty even at initialization.
3. **Computational Overhead:**
   Evaluating the frozen source branch requires an additional forward pass per batch, increasing per-batch inference time.

---

## 5. Execution Authorization Notice

> [!WARNING]
> This protocol is a **DRAFT SPECIFICATION ONLY**. Execution, experiment launches, and gating implementations are **STRICTLY UNAUTHORIZED** until explicit future approval.
