# ChangeGuard Protocol: Branch C Empirical Evaluation Protocol (v1)

**Status:** ACTIVE PROTOCOL SPECIFICATION (Branch C)
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Sections 11.4, 12.4, 12.5, 13.1–13.4
- `docs/e6_final_route_decision.md` (Verdict: `E6 COMPLETE — BRANCH C`)
- `docs/e6_primary_study_definition.md`
- `docs/e6_claim_to_experiment_map.md`
- `docs/e6_nearest_method_applicability_matrix.md`
- `docs/protocols/e6_cotta_compatibility_protocol_v1.md`

---

## 1. Scientific Context & Branch C Rationale

Work Package E6 formally evaluated Gate E on the completed canonical exploratory cell (Seed 0 / Order A) and two orthogonal repeat cells (Order repeat: Seed 0 / Order B; Seed repeat: Seed 1 / Order A). Gate E criteria 2, 3, and 6 failed:
- Pre-update displacement signals measure adaptation magnitude rather than direction, and signal AUROC inverts across source checkpoint seeds (e.g., `source_to_candidate_kl` collapses from 1.0000 on Seed 0 to 0.0000 on Seed 1).
- No evaluated adaptation component provides stable, reproducible improvement over source-only inference across seeds.

In accordance with `docs/MASTER_EXECUTION_PLAN_V3.md` Section 12.4, ChangeGuard proceeds along **Branch C: Controlled Empirical Study**. Branch C explicitly rejects:
- Accept/reject decision gates or selective adaptation classifiers.
- Generic-plus-change score fusion policies.
- Learned risk predictors or heuristic threshold calibration.
- Claims of advance harm prediction.

Instead, Branch C provides a rigorous, controlled empirical characterization of test-time adaptation dynamics on paired multi-temporal SAR flood segmentation streams.

---

## 2. Core Research Questions (RQs)

Branch C investigates four core scientific questions:

- **RQ1 (Adaptation Fragility & Seed Contingency):** Under what conditions does online test-time adaptation degrade, maintain, or improve flood-class segmentation performance on paired multi-temporal SAR streams?
- **RQ2 (Factor Sensitivity):** How do adaptation outcomes vary across source checkpoint initializations (Seeds 0, 1, 2), stream presentation orders (Order A vs. Order B), and event-specific geographical/distributional characteristics?
- **RQ3 (Nearest SOTA Benchmark — CoTTA):** Does the nearest structurally compatible continual TTA method (CoTTA, Wang et al., CVPR 2022) with mean-teacher EMA, augmentation averaging, and stochastic source restoration prevent adaptation collapse or improve performance over source-only and standard TTA (Tent, EATA, SAR)?
- **RQ4 (Continual vs. Episodic Dynamics):** How does continual adaptation across an unbroken multi-event stream compare against episodic per-event reset adaptation under identical seeds and presentation orders?

---

## 3. Evaluated Methods & Matched Controls

The empirical study evaluates eight unified methods:

| Method ID | Method Name | Description | Key Hyperparameters | State Carrying |
| :--- | :--- | :--- | :--- | :---: |
| `source_only` | Source-Only Baseline | Frozen training BN running stats, zero updates | N/A | Immutable |
| `adabn` | AdaBN / BN-no-update | Test-time BN moving average, zero gradient updates | Momentum 0.1 | Episodic / Continual |
| `tent` | Tent Baseline | Test-time entropy minimization over BN affine parameters | lr = 1e-4, Adam | Episodic / Continual |
| `eata` | EATA Baseline | Redundancy/reliability filtered Fisher regularized TTA | lr = 1.25e-4, d_margin = 0.05, α = 2000 | Episodic / Continual |
| `sar` | SAR Baseline | Sharpness-aware minimization with model recovery | lr = 1.25e-4, ρ = 0.05, e_m = 0.031808 | Episodic / Continual |
| `candidate_e2` | Candidate Consistency | Weak-strong consistency (horizontal flip, τ = 0.90) | lr = 1.25e-4, SGD momentum 0.9 | Episodic / Continual |
| `cotta_continual` | CoTTA Continual | Mean teacher EMA + 4-view aug-avg + SSR (p=0.01) | lr = 1e-4, Adam, α = 0.999, p = 0.01 | **Continual** |
| `cotta_reset` | CoTTA Reset-per-Event | Mean teacher EMA + 4-view aug-avg + SSR (p=0.01) | lr = 1e-4, Adam, α = 0.999, p = 0.01 | **Episodic Reset** |

---

## 4. Benchmark Dataset & Stream Specifications

### 4.1 Development Events
All primary empirical evaluations and factor analyses are conducted on the six declared development flood events:

| Event ID | Name / Region | Sample Count ($N$) | Batch Count (BS=16) | Target Class Characteristics |
| :---: | :--- | :---: | :---: | :--- |
| `279` | Spain / Ebro | 1,176 | 74 | Large flood extent |
| `437` | UK / Severn | 536 | 34 | Medium flood extent |
| `520` | USA / Florence | 474 | 30 | Medium flood extent |
| `559` | USA / Midwest | 196 | 13 | Small flood extent |
| `1111003` | Bangladesh | 210 | 14 | Small flood extent |
| `1111008` | Myanmar | 234 | 15 | Medium flood extent |
| **Total** | **6 Events** | **2,826** | **179** | **Full Dev Partition** |

### 4.2 Sealed-Test Event Quarantine
The following ten events constitute the sealed-test partition and are **strictly quarantined**:
`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.
Zero access to sealed-test data or metadata is permitted during implementation, test execution, or preflight validation.

### 4.3 Stream Presentation Orders
1. **Order A (`order_a`):**
   Sequence: `[279, 437, 520, 559, 1111003, 1111008]`
   Within-event ordering: deterministic sorted `sample_id` ascending.
2. **Order B (`order_b`):**
   Sequence: `[1111008, 1111003, 559, 520, 437, 279]`
   Within-event ordering: deterministic sorted `sample_id` descending.

### 4.4 Source Checkpoints
All primary source models use SNUNet-ECAM trained on the all-27 dataset for 16 epochs in FP32:
- Seed 0: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`
  SHA-256: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
- Seed 1: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt`
  SHA-256: `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a`
- Seed 2: `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt`
  SHA-256: `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd`

---

## 5. Algorithmic Protocol: CoTTA for SAR Change Detection

The CoTTA adapter (`E7CoTTAAdapter`) implements the three defining upstream pillars adapted to paired bi-temporal SAR change detection without invalid photometric perturbations:

1. **Mean Teacher EMA:**
   Dedicated teacher network $\theta_t$ tracks student $\theta_s$ via $\theta_t \leftarrow \alpha \theta_t + (1 - \alpha) \theta_s$ ($\alpha = 0.999$). Teacher has `requires_grad_(False)` and never receives direct gradients.
2. **Augmentation-Averaged Teacher Predictions ($K=4$):**
   Teacher evaluates four exact spatial transforms: $\mathcal{A}_1$ (Identity), $\mathcal{A}_2$ (Horizontal Flip), $\mathcal{A}_3$ (Vertical Flip), $\mathcal{A}_4$ (Horizontal + Vertical Flip / Rot180).
   Each transform is applied identically to $(x_A, x_B)$. Softmax probability maps are inverted via $\mathcal{A}_k^{-1}$ and averaged:
   $$\bar{p}_t(x) = \frac{1}{4} \sum_{k=1}^4 \mathcal{A}_k^{-1}\left( \text{Softmax}(f(\mathcal{A}_k(x_A, x_B); \theta_t)) \right)$$
3. **Stochastic Source Restoration (SSR):**
   At each step after teacher EMA, each trainable parameter element has an independent probability $p = 0.01$ of resetting bit-exactly to initial source value $\theta_0$:
   $$\theta_s^{(j)} \leftarrow \begin{cases} \theta_0^{(j)} & \text{with probability } p = 0.01 \\ \theta_s^{(j)} & \text{with probability } 0.99 \end{cases}$$
4. **Sensory Masking & Confidence Thresholding:**
   Loss is evaluated only on sensory-valid pixels ($M_{NA} == 1$) where teacher confidence $\max \bar{p}_t \ge \tau = 0.90$.
5. **Primary Prediction Timing:**
   Primary predictions recorded for evaluation are strictly the `PRE_UPDATE` teacher output $\bar{p}_t(x)$.

---

## 6. Cross-Event Generalization & LOEO Semantics (Branch C)

Under Branch C, there is no policy threshold calibration. Leave-One-Event-Out (LOEO) evaluation assesses cross-event generalization of baseline performance without pseudo-replication:
- **Held-Out Folds:** Each development event $e \in \{279, 437, 520, 559, 1111003, 1111008\}$ is evaluated independently as the held-out target event.
- **Independence:** No held-out event data or labels influence any upstream configuration or model selection.
- **Reporting Metrics:**
  - **Event-Macro F1:** Unweighted mean of per-event F1 scores across the 6 events: $\text{Macro } F_1 = \frac{1}{6} \sum_{e} F_1(e)$.
  - **Pooled F1:** Global F1 computed by summing raw confusion matrix cells ($TP, FP, FN$) across all valid pixels in all events: $\text{Pooled } F_1 = \frac{2 \sum TP}{2 \sum TP + \sum FP + \sum FN}$.
  - **Per-Event Delta:** $\Delta F_1(e) = F_1^{\text{method}}(e) - F_1^{\text{source}}(e)$.
  - **Worst-Event F1:** $\min_{e} F_1(e)$.
  - **Harm / Benefit Counts:** Number of events with $\Delta F_1 \le -0.0100$ (harm) vs. $\Delta F_1 \ge +0.0100$ (benefit).

---

## 7. Execution Boundaries: Smoke vs. Formal

| Scope | Script / Tool | Events / Batches | Hardware Target | Execution Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Unit Verification** | `unittest discover` | Synthetic CPU tensors | CPU only | Zero CUDA calls; runs on any workstation |
| **GPU Smoke Test** | `tools/run_e7_empirical_evaluation.py --smoke` | Event 520, 2 batches ($N=32$) | GPU (gpu22) | Stopped at clean Git checkpoint before launch |
| **Formal Matrix** | `tools/run_e7_empirical_evaluation.py` | Full dev6 (179 batches) | GPU (gpu22) | Multi-stream formal evaluation in E7/E8 |

---

## 8. Fail-Closed Verification Requirements

1. **Storage Aliasing:** Candidate, source, and teacher models must share zero memory pointers (`data_ptr()`).
2. **Label Quarantine:** Ground-truth labels are completely inaccessible during adaptation. Passing any label key raises `ValueError`.
3. **Finite State:** Any non-finite loss, gradient, or parameter update triggers an immediate rollback to pre-step state without advancing step count.
4. **Reproducibility:** All runners verify SHA-256 digests of configs, protocols, and model checkpoints before execution.
