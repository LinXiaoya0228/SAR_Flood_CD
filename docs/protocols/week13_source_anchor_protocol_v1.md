# Week 13 Frozen-Source Probability-Anchor Protocol

**Protocol Version:** `week13_source_anchor_dev6_v1`  
**Status:** `APPROVED_FOR_WEEK13_SOURCE_ANCHOR_EXECUTION`  
**Evidence Level:** `CONTROLLED_DEVELOPMENT_ABLATION`  
**Governing Context:** Follows Week 12 exploratory results ([`docs/week12_exploratory_results_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week12_exploratory_results_v2.md)) and Week 12 Matched BN Diagnostic ([`docs/week12_bn_diagnostic_comparison_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week12_bn_diagnostic_comparison_v1.md)).

---

## 1. Motivation & Scientific Objectives

In Week 12, continuous unanchored test-time adaptation (both Tent entropy minimization and Evidence Consistency C1/C2) degraded performance relative to the frozen source baseline. The Week 12 Matched BN Diagnostic demonstrated that:
1. Evaluating under batch-statistic BatchNorm with zero parameter updates accounted for virtually the entire aggregate drop ($F_1 = 0.5733$ vs $0.5949$ source).
2. Gradient updates under unconstrained consistency (C1 and C2) changed fewer than 1 in 10,000 pixels across the 6 development events, failing to correct for the batch-statistic domain shift.

Week 13 specifies a single, rigorously controlled condition: **C3 Anchored Evidence Consistency** (`C3_anchored_consistency_v1`).
C3 introduces an explicit frozen-source probability anchor to provide stabilizing guidance toward calibrated source predictions during test-time adaptation.

---

## 2. Mathematical Formulation & Architecture Specification

### 2.1 Dual-Model Architecture

1. **Candidate Branch ($f_{\text{candidate}}$):**
   - Active adapter initialized from frozen source checkpoint seed 0.
   - All BatchNorm modules set to `train()` mode with `track_running_stats = False`.
   - Normalizes weak and strong views using current-batch statistics ($\mu_B, \sigma_B^2$).
   - Computes weak view logits $z_{\text{weak}} = f_{\text{candidate}}(x_{\text{weak}})$ and strong view logits $z_{\text{strong}} = f_{\text{candidate}}(x_{\text{strong}})$, where $x_{\text{strong}} = \text{flip}_{\text{horizontal}}(x_{\text{weak}})$.
   - Weak probabilities: $p_{\text{weak}} = \text{softmax}(z_{\text{weak}}, \text{dim}=1)$.
   - Strong probabilities: $p_{\text{strong}} = \text{softmax}(z_{\text{strong}}, \text{dim}=1)$.
   - Aligned strong probabilities: $p_{\text{candidate\_aligned\_strong}} = \text{flip}_{\text{horizontal}}(p_{\text{strong}})$.

2. **Frozen Source Branch ($f_{\text{source}}$):**
   - Separate, frozen instance of SNUNet_ECAM initialized from the exact same source checkpoint.
   - Preserves training running statistics (`eval()` mode with `track_running_stats = True`).
   - Strictly wrapped under `torch.no_grad()`; parameters are immutable with `requires_grad = False`.
   - Computes source probabilities on the weak view:
     $$p_{\text{source\_detached}} = \text{softmax}(f_{\text{source}}(x_{\text{weak}}), \text{dim}=1)$$

### 2.2 Objective Function

The total adaptation loss per batch is defined as:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{consistency}} + \lambda_{\text{anc}} \cdot \mathcal{L}_{\text{anchor}}$$

where the hyperparameter budget specifies exactly:
$$\lambda_{\text{anc}} = 0.50$$

#### 1. Consistency Loss Term ($\mathcal{L}_{\text{consistency}}$)
Identical to Week 12 Condition C2 ($\tau = 0.90$):
$$\Omega_{\text{sel}} = \{ i \in \Omega_{\text{valid}} \mid \max_{c \in \{0, 1, 2\}} p_{\text{weak}, i}(c) \ge 0.90 \}$$
$$\mathcal{L}_{\text{consistency}} = \frac{1}{\max(1, |\Omega_{\text{sel}}|)} \sum_{i \in \Omega_{\text{sel}}} D_{\text{KL}}\left( \text{detach}(p_{\text{weak}, i}) \,\|\, p_{\text{candidate\_aligned\_strong}, i} \right)$$
If $|\Omega_{\text{sel}}| = 0$, $\mathcal{L}_{\text{consistency}} = 0$.

#### 2. Source Anchor Loss Term ($\mathcal{L}_{\text{anchor}}$)
Penalizes divergence of the candidate model's aligned strong prediction from the frozen source's calibrated prediction over all sensory-valid pixels:
$$\mathcal{L}_{\text{anchor}} = \frac{1}{|\Omega_{\text{valid}}|} \sum_{i \in \Omega_{\text{valid}}} D_{\text{KL}}\left( p_{\text{source\_detached}, i} \,\|\, p_{\text{candidate\_aligned\_strong}, i} \right)$$
where:
$$D_{\text{KL}}(p \,\|\, q) = \sum_{c \in \{0, 1, 2\}} p(c) \cdot (\log p(c) - \log q(c))$$
over all pixels where sensory mask $M_{K0} \cdot M_{NA} == 1$.

### 2.3 Trainable Parameters & Gradient Flow
- **Trainable Scope:** Selected BatchNorm affine parameters ($\gamma, \beta$) of $f_{\text{candidate}}$ only.
- **Excluded:** Backbone weights, attention parameters, classifier heads, and all running buffers.
- **Source Model:** Strictly detached; zero gradients, parameters remain byte-for-byte identical throughout the run.
- **Candidate Weak Branch:** Detached when serving as pseudo-label in $\mathcal{L}_{\text{consistency}}$.
- **Candidate Strong Branch:** Receives gradients from both $\mathcal{L}_{\text{consistency}}$ (for selected confident pixels) and $\mathcal{L}_{\text{anchor}}$ (for all valid pixels).

---

## 3. Matched Experimental Controls

| Setting | Value / Policy |
|---|---|
| **Target Condition** | Exactly 1 condition: `C3_anchored_consistency_v1` |
| **Events** | Strictly the 6 KuroSiwo development events: `[279, 437, 520, 559, 1111003, 1111008]` |
| **Test Events** | 10 sealed test events remain strictly forbidden |
| **Stream Ordering** | `order_a` (lexicographic ascending path order) |
| **Stream Reset** | `reset_per_event = true` (adapter resets to source checkpoint at the start of each event) |
| **Batch Size** | 16 with natural tail (matching Weeks 11 and 12) |
| **Optimizer** | SGD, learning rate $1.25 \times 10^{-4}$, momentum $0.9$, weight decay $0$ |
| **Step Budget** | Exactly 1 optimizer step per batch |
| **Prediction Timing** | Strictly `PRE_UPDATE` predictions persisted before ground-truth access |
| **Double-Masked Scoring** | Offline evaluation over $M_{K0} \cdot M_{NA} == 1$ and $\text{label} \ne 3$ |
| **Confusion Conservation** | Strict assertion: $\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels}$ |
| **Failure Recovery** | Check finite loss, finite gradients, and post-step candidate state; rollback on failure |

---

## 4. Execution Authorization & Preconditions

- **Authorized Output Path:** [`outputs/week13_source_anchor/approved_dev6_s0_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week13_source_anchor/approved_dev6_s0_v1/)
- **Configuration File:** [`configs/changeguard/week13_source_anchor_dev6_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/week13_source_anchor_dev6_v1.json)
- **Status Flag:** `APPROVED_FOR_WEEK13_SOURCE_ANCHOR_EXECUTION`

