# ChangeGuard Plan V3: Work Package E4 Preflight Specification & Class-Aware Contract

**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 9: Block E4)  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Execution Environment:** Miniconda `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`)  
**Base Terminal Commit:** `ec5e7258c2ce589f734029a0b60f32b39cb53074`  
**Audit & Revision Commit:** E4.1b Research-Validity Audit & Contract Correction  
**Status:** IMPLEMENTER PREFLIGHT SPECIFICATION & MEASUREMENT CONTRACT (LOCKED & AUDITED)

---

## 1. Executive Context & Methodological Role

In ChangeGuard Work Package **`E4`**, we systematically investigate isolated class-aware adaptation interventions (K1–K4) designed to overcome the fundamental failure mode of self-training under extreme class imbalance.

### The Problem: Extreme Imbalance and Pseudo-Label Collapse
In Week 14 offline diagnostic audits, we proved that under standard confidence thresholding ($\tau = 0.90$), majority background classes overwhelm the pseudo-label pool:
- Over 27 source events ($1.06\times 10^9$ valid pixels), the empirical class distribution is:
  - Class 0 (Land / Background): $74.70\%$ ($\pi_0 = 0.7469878$)
  - Class 1 (Permanent Water): $7.04\%$ ($\pi_1 = 0.0704130$)
  - Class 2 (Flood Water / Change): $18.26\%$ ($\pi_2 = 0.1825992$)
- In high-confidence selection ($\tau = 0.90$), Land frequently comprises **up to $97.2\%$** of selected pseudo-labels.
- The candidate model reinforces majority background features, causing gradual class collapse, permanent water degradation, and sharp false-alarm surges ($c_{02}$ Land $\to$ Flood and $c_{12}$ Water $\to$ Flood).

### Epistemic Guardrails & Research Invariants
1. **Internal Evaluation Platform, Not Promoted Method:** The fixed base consistency recipe ($\tau = 0.90$, identity-weak vs. horizontal-flip-strong augmentation, Adam lr=$10^{-4}$, BN-affine adaptation) is retained strictly as an **internal evaluation platform** (`E4_BASE` $\equiv$ `E2_CONF_90`). In Block E2 and E3, consistency and source anchoring both yielded `NONE RETAINED` relative to matched unadapted controls. `E4_BASE` is NOT a promoted baseline.
2. **Immutable Source-Reference Model, Not Ground Truth:** The frozen source model is an immutable reference, never ground truth. Source agreement is explicitly distinct from correctness.
3. **Strict Component Isolation (No Combinations):** Interventions K1, K2, K3, and K4 are tested strictly in isolation against `E4_BASE`. Plan V3 forbids multi-intervention combinations during component screening.
4. **Predeclared Parameter Settings:** Each intervention evaluates exactly one predeclared setting derived from research validity principles or the source calibration distribution; no post-hoc grid search or dynamic parameter tuning on development events.
5. **Air-Tight Ground-Truth Sealing Lifecycle:** Adaptation operates strictly on sensory rasters ($x_A, x_B$) and binary sensory valid mask `MK0_MNA`. Ground truth labels `MK0_MLU` are locked behind `LabelGuardedBatch` with fail-closed permissions throughout the multi-batch streaming adaptation loop (Phase 1). Labels are accessed strictly post-persistence in Phase 2 for offline evaluation.

---

## 2. Workload & Resource Realities

- **Sample and Batch Accounting Across 6 Development Events:**
  The development sequence consists of $N = 2,826$ patch pairs ($256 \times 256$, 2 channels $[VV, VH]$):
  - **Event 279:** 728 samples $\to$ 46 batches (45 of 16, 1 of 8)
  - **Event 437:** 412 samples $\to$ 26 batches (25 of 16, 1 of 12)
  - **Event 520:** 490 samples $\to$ 31 batches (30 of 16, 1 of 10)
  - **Event 559:** 496 samples $\to$ 31 batches (31 of 16, 0 tail)
  - **Event 1111003:** 244 samples $\to$ 16 batches (15 of 16, 1 of 4)
  - **Event 1111008:** 456 samples $\to$ 29 batches (28 of 16, 1 of 8)
  - **Total Development Workload:** 179 batches per cell across the 6 events.
- **Empirical GPU Memory Consumption:**
  In the isolated GPU smoke run on Event 520 (RTX 3090, 24 GB VRAM), peak reserved memory was measured at **$20,024.0\text{ MB}$** ($\approx 19.56\text{ GB}$). VRAM headroom is $\approx 4.5\text{ GB}$, fully safe for batch size 16 execution.

---

## 3. Formal Intervention Contracts & Mathematical Formulations

All five screening cells share the identical model architecture (`SNUNet_ECAM`, seed 0, checkpoint SHA-256 `247d3584...`), optimizer (`Adam`, $\text{lr}=10^{-4}$, $\beta_1=0.9$, $\beta_2=0.999$, $\text{wd}=0.0$), reset mode (`reset_per_event`), order (`Order A`), and batch size ($16$).

### Cell 0: `E4_BASE` (Fixed Unmodified Consistency Base)
- **Mathematical Objective:**
  $$\mathcal{L}_{base} = -\frac{1}{|\Omega_{base}|} \sum_{i \in \Omega_{base}} \sum_{c=0}^2 p_{weak, c}(i) \log p_{aligned, c}(i)$$
  where $\Omega_{base} = \{ i \mid M_{valid}(i) = 1 \;\land\; \max_c p_{weak, c}(i) \ge 0.90 \}$.
- If $|\Omega_{base}| = 0$, update is cleanly skipped (`SKIPPED_EMPTY_MASK`, zero gradient).
- **Control Reuse Provenance Audit:** In Section 8 below, `E4_BASE` is audited against `E2_CONF_90` across all 13 protocol dimensions. Because `E4_BASE` and `E2_CONF_90` are 100% bit-exact identical, `run_e4_class_aware_screening.py` supports `--reuse-base` to reconstruct `E4_BASE` directly from existing cached predictions without redundant GPU computation.

---

### Cell 1: `E4_K1_CLASS_BALANCED` (Class-Balanced Confidence Selection)
- **Motivation & Scientific Revision (E4.1b Audit):**
  The original proposal assigned quotas based on source calibration priors $Q_c = \lfloor N_{valid} \cdot \pi_c \rfloor$. However, because source proportions are $(0.747, 0.070, 0.183)$, this preserved the extreme background dominance ($74.7\%$ Land quota) and did not balance classes.
  Under the audited and locked E4.1b contract, K1 enforces genuine class balancing via a uniform batch-level per-class cap.
- **Eligible Set:**
  Let $\Omega_{base} = \{ i \mid M_{valid}(i) = 1 \;\land\; \max_c p_{weak, c}(i) \ge 0.90 \}$.
  Eligible pixel count:
  $$N_{eligible} = |\Omega_{base}| = \sum_{c=0}^2 A_c, \quad \text{where } A_c = |\{ i \in \Omega_{base} \mid \hat{y}_{cand}(i) = c \}|$$
- **Uniform Per-Class Quota Cap:**
  $$C_{batch} = \begin{cases} \lfloor N_{eligible} / 3 \rfloor & \text{if } \lfloor N_{eligible} / 3 \rfloor > 0 \\ 1 & \text{if } N_{eligible} > 0 \text{ and } \lfloor N_{eligible} / 3 \rfloor == 0 \\ 0 & \text{if } N_{eligible} == 0 \end{cases}$$
- **Realised Selection Rule:**
  For each class $c \in \{0, 1, 2\}$:
  $$R_c = \min(A_c, C_{batch})$$
  The top-$R_c$ pixels for class $c$ are selected based on candidate confidence $p_{weak, c}(i)$.
- **Deterministic Lexicographical Tie-Breaking:**
  To guarantee exact cross-device determinism without floating-point precision truncation, ties in confidence are resolved strictly via a two-stage stable lexicographical sort:
  1. Primary key: confidence $p_{weak, c}(i)$ descending.
  2. Secondary key: spatial raster index $i$ ascending.
  In PyTorch:
  ```python
  spatial_order = torch.argsort(flat_spatial_idx, descending=False, stable=True)
  conf_order = torch.argsort(confs[spatial_order], descending=True, stable=True)
  selected_indices = spatial_order[conf_order[:R_c]]
  ```
  Zero floating-point perturbations ($10^{-12}$) are used.
- **Empty-Class Non-Substitution Invariant:**
  If a class has zero available confident pixels ($A_c = 0$), $R_c = 0$, and unused quota capacity is **strictly discarded**. Unused quota slots are NEVER redistributed or transferred to other classes.

---

### Cell 2: `E4_K2_FLOOD_WEIGHTED` (Flood-Aware Loss Weighting)
- **Motivation:** Boosts the adaptation signal for minority flood pixels without altering streaming order or discarding spatial context.
- **Source-Calibrated Weights:**
  From the empirical ratio $\pi_0 / \pi_2 \approx 74.70 / 18.26 \approx 4.09$:
  $$w_0 = 1.0, \quad w_1 = 1.0, \quad w_2 = 4.0$$
- **Weighted Loss Objective:**
  $$\mathcal{L}_{K2} = \frac{\sum_{i \in \Omega_{base}} w(\hat{y}_{cand}(i)) \cdot \ell_{cons}(i)}{\sum_{i \in \Omega_{base}} w(\hat{y}_{cand}(i))}$$
  where $\ell_{cons}(i) = -\sum_{c=0}^2 p_{weak, c}(i) \log p_{aligned, c}(i)$.
- **Denominator Invariant:**
  The denominator is strictly the sum of effective weights across selected pixels $W = \sum_{i \in \Omega_{base}} w_i$. This normalizes the loss scale, preventing gradient magnitude explosion when flood pixels are selected.

---

### Cell 3: `E4_K3_PERM_PROTECTED` (Source-Predicted Permanent-Water Exclusion)
- **Motivation:** Prevents permanent water bodies from being erroneously converted into flood change.
- **Source Exclusion Mask (Frozen Source Argmax):**
  $$\mathcal{M}_{perm} = \{ i \mid M_{valid}(i) = 1 \;\land\; \hat{y}_{src}(i) == 1 \}$$
  where $\hat{y}_{src} = \arg\max_c p_{src, c}$ is evaluated on the immutable frozen source model.
- **Excluded Adaptation Mask:**
  $$\Omega_{K3} = \Omega_{base} \setminus \mathcal{M}_{perm}$$
- **Definitional Boundary & Non-Invariance Caveat:**
  Excluding source-permanent-water pixels from the adaptation loss mask prevents candidate loss gradients on those spatial locations. However, this does **NOT** guarantee invariant predictions on those pixels, because the shared batch normalization affine parameters adapt across the rest of the image.
- **Zero-Leakage Invariant:**
  Exclusion relies strictly on the frozen source reference model predictions ($\hat{y}_{src} == 1$). Absolutely no ground truth labels (`MK0_MLU`) or ground truth water masks are accessed during adaptation.
- **Threshold Removal:**
  The unreferenced threshold `tau_perm = 0.50` is formally removed; K3 operates strictly on the discrete source predicted argmax $\hat{y}_{src} == 1$.
- **Telemetry:**
  Tracks total protected/excluded pixels, mean source confidence on excluded pixels, and candidate drift away from source permanent water ($\sum_{i \in \mathcal{M}_{perm}} \mathbb{I}[\hat{y}_{cand}(i) \ne 1]$).

---

### Cell 4: `E4_K4_FLOOD_PRESERVED` (Frozen-Source Flood/Change Preservation)
- **Motivation:** Protects against catastrophic flood erasure while actively monitoring flood over-expansion.
- **Source Flood Detection Mask:**
  $$\mathcal{M}_{src\_flood} = \{ i \mid M_{valid}(i) = 1 \;\land\; p_{src, 2}(i) \ge \tau_{src\_flood} \}, \quad \tau_{src\_flood} = 0.50$$
- **Auxiliary Preservation Loss:**
  $$\mathcal{L}_{pres} = -\frac{1}{|\mathcal{M}_{src\_flood}|} \sum_{i \in \mathcal{M}_{src\_flood}} \log(p_{aligned, 2}(i) + 10^{-12})$$
- **Total Adaptation Objective:**
  $$\mathcal{L}_{K4} = \mathcal{L}_{base} + \lambda_{pres} \cdot \mathcal{L}_{pres}, \quad \lambda_{pres} = 0.50$$
- **Erasure and Expansion Accounting:**
  Telemetry explicitly tracks and separates:
  - **Erasure Rate:** $\frac{\sum_{i: M_{valid}=1} \mathbb{I}[\hat{y}_{src}(i) == 2 \;\land\; \hat{y}_{cand}(i) \ne 2]}{\sum_{i: M_{valid}=1} \mathbb{I}[\hat{y}_{src}(i) == 2]}$ (denominator: source flood pixels).
  - **Expansion Rate:** $\frac{\sum_{i: M_{valid}=1} \mathbb{I}[\hat{y}_{src}(i) \ne 2 \;\land\; \hat{y}_{cand}(i) == 2]}{\sum_{i: M_{valid}=1} \mathbb{I}[\hat{y}_{src}(i) \ne 2]}$ (denominator: source non-flood pixels).

---

## 4. Offline Diagnostic Interpretation Schema (Post-Persistence)

To prevent ground-truth leakage into the adaptation loop, all diagnostic interpretation is conducted **strictly post-persistence** after predictions have been frozen on disk.

For each cell and each event, the following metrics are computed:
1. **$3 \times 3$ Confusion Matrix:** $C_{ij} = \sum \mathbb{I}[y = i \;\land\; \hat{y}_{cand} = j]$, with guaranteed conservation $\sum_{i,j} C_{ij} = N_{valid}$.
2. **Class-Wise Metrics:** Precision, recall, and F1 for all three classes (Land, Permanent Water, Flood).
3. **Separated False Positives & Negatives with Explicit Denominators:**
   - $c_{02}$ (Land mistaken for Flood): count, fraction of true Land ($C_{02} / \sum_j C_{0j}$), share of total FP.
   - $c_{12}$ (Permanent Water mistaken for Flood): count, fraction of true Water ($C_{12} / \sum_j C_{1j}$), share of total FP.
   - $c_{20}$ (Flood missed as Land): count, fraction of true Flood ($C_{20} / \sum_j C_{2j}$), share of total FN.
   - $c_{21}$ (Flood missed as Water): count, fraction of true Flood ($C_{21} / \sum_j C_{2j}$), share of total FN.
4. **Source-to-Candidate Transition Matrix:** $T_{ij} = \sum \mathbb{I}[\hat{y}_{src} = i \;\land\; \hat{y}_{cand} = j]$.
5. **Source-Relative Error Dynamics:**
   - Harmful transitions: Source 3-class correct, Candidate 3-class wrong ($\hat{y}_{src} = y \;\land\; \hat{y}_{cand} \ne y$).
   - Beneficial transitions: Source 3-class wrong, Candidate 3-class correct ($\hat{y}_{src} \ne y \;\land\; \hat{y}_{cand} = y$).
   - Preserved source errors: Source wrong and Candidate reproduces source error ($\hat{y}_{src} \ne y \;\land\; \hat{y}_{cand} = \hat{y}_{src}$).
6. **Comparison with Base Consistency:**
   - Harmful base error recovery: Source correct, Base wrong, Candidate correct.
   - Lost beneficial adaptations: Source wrong, Base correct, Candidate wrong.

---

## 5. Quarantined GPU Smoke Validation Evidence

Initial smoke validation was executed in E4.1 on Event 520 (32 samples, 2 batches of 16) on GPU (`cuda:0`), with all artifacts written to `outputs/e4_class_aware/smoke_v1/`.

> **Quarantine Notice:** These initial smoke artifacts are preserved in `outputs/e4_class_aware/smoke_v1/` for execution safety verification only. They were produced under the preliminary exploratory definitions and are NOT used for intervention selection or comparative ranking. The formal screening directory `outputs/e4_class_aware/v1/` remains strictly uncreated pending authorization of Work Package E4.2.

| Cell ID | Intervention | Confidence $\tau$ | Key Parameters | Updates Applied | Peak VRAM | Source Hash Invariant |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **`E4_BASE`** | `base` | $0.90$ | None | 2 / 2 | $20,020.0\text{ MB}$ | Bit-identical |
| **`E4_K1_CLASS_BALANCED`** | `k1_balanced_selection` | $0.90$ | Uniform cap $\lfloor N_{eligible}/3 \rfloor$ | 2 / 2 | $20,024.0\text{ MB}$ | Bit-identical |
| **`E4_K2_FLOOD_WEIGHTED`** | `k2_flood_weighting` | $0.90$ | $w = (1.0, 1.0, 4.0)$ | 2 / 2 | $20,024.0\text{ MB}$ | Bit-identical |
| **`E4_K3_PERM_PROTECTED`** | `k3_perm_protection` | $0.90$ | Source argmax $\hat{y}_{src} == 1$ | 2 / 2 | $20,024.0\text{ MB}$ | Bit-identical |
| **`E4_K4_FLOOD_PRESERVED`** | `k4_flood_preservation` | $0.90$ | $\tau_{flood} = 0.50, \lambda = 0.50$ | 2 / 2 | $20,024.0\text{ MB}$ | Bit-identical |

### Smoke Invariant Verifications:
1. **Bit-Exact Source Immutability:** Across all 5 cells, the frozen source model parameter SHA-256 hash was verified bit-identical before and after each event.
2. **Double Masking & Conservation:** For all cells, total valid pixels evaluated = $1,188,597$. Confusion conservation $\text{TP} + \text{FP} + \text{FN} + \text{TN} = N_{valid}$ verified down to the single pixel.
3. **Execution Safety:** Zero NaN, non-finite loss, or CUDA memory errors encountered.
4. **Segregation:** All smoke outputs strictly isolated under `outputs/e4_class_aware/smoke_v1/`. The formal screening directory `outputs/e4_class_aware/v1/` remains uncreated.

---

## 6. Unit Test Verification Suite

A dedicated unit test suite [`tests/test_e4_class_aware_contracts.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tests/test_e4_class_aware_contracts.py) verifies all mathematical invariants and isolation boundaries:

```bash
$ python -m unittest tests/test_e4_class_aware_contracts.py
...............
----------------------------------------------------------------------
Ran 15 tests in 7.735s

OK
```

The 15 unit tests rigorously cover:
1. Physical storage independence and zero shared memory pointers between candidate and frozen-source models.
2. Identical object rejection (`ValueError` / `RuntimeError`).
3. K1 uniform quota capping ($C = \lfloor N_{eligible}/3 \rfloor$) and deterministic lexicographical tie-breaking (primary: confidence descending, secondary: spatial index ascending).
4. K1 empty-class non-substitution (unused quotas strictly discarded; no capacity transfer).
5. K1 independence from ground truth and future samples.
6. K2 weighted loss denominator ($W = \sum w_i$) and full prediction coverage.
7. K3 source-permanent-water exclusion ($\hat{y}_{src} == 1$, zero GT access) and verification that predictions are not guaranteed invariant under parameter displacement.
8. K4 flood preservation, erasure, and expansion accounting.
9. Offline interpretation metrics confusion and transition conservation.
10. Empty valid mask boundary handling (cleanly skips update without NaN or non-finite loss).
11. Sealed-test event quarantine enforcement (sealed-test events strictly disjoint from development set).
12. `LabelGuardedBatch` fail-closed label access and zero-leakage parameter invariance.
13. Complete runner ground-truth sealing lifecycle: verifies that during multi-batch streaming adaptation, calling `batch.mark_persisted()` or `batch.get_labels()` raises a fail-closed `PermissionError`.
14. Multi-step source model immutability across multiple adaptation updates.
15. End-to-end instantiation and step execution across all 5 cells.

In addition, the combined cross-block regression suite passes cleanly:
```bash
$ python -m unittest tests/test_e2_1_confidence_grid.py tests/test_e2_2_augmentation_grid.py tests/test_e2_3_repeat_matrix.py tests/test_e3_anchor_contracts.py tests/test_e4_class_aware_contracts.py
........................................................
----------------------------------------------------------------------
Ran 56 tests in 2.186s

OK
```

---

## 7. Plan V3 Promotion Criteria for Work Package E4.2

Under Plan V3 Section 9, a class-aware intervention is promoted to the repeat validation stage (Seed 1 / Order B) if and only if:
1. **Substantial Aggregate Improvement:**
   $$\Delta \text{Macro F1} = \text{Macro F1}_{cell} - \text{Macro F1}_{E4\_BASE} \ge +0.010$$
   across the 6 development events.
2. **Tail Event Safety:**
   $$\Delta \text{WorstEvent F1} = \text{WorstEvent F1}_{cell} - \text{WorstEvent F1}_{E4\_BASE} \ge 0.000$$
   The intervention must not degrade performance on the worst-performing event (Event 1111003).
3. **Repeat Consistency:**
   Any cell meeting criteria 1 and 2 must reproduce positive gains under Seed 1 / Order B before any claim of robustness is accepted.

---

## 8. Matched Control Reuse Audit (`E4_BASE` vs `E2_CONF_90`)

Before launching E4.2 GPU execution, we perform a field-by-field matched control provenance audit between `E4_BASE` and `E2_CONF_90` (`outputs/e2_confidence_grid/v1/E2_CONF_90`):

| Protocol Dimension | `E2_CONF_90` Specification | `E4_BASE` Specification | Exact Match? |
| :--- | :--- | :--- | :---: |
| **Model Architecture** | `SNUNet_ECAM` (Model A, 2 channels VV/VH) | `SNUNet_ECAM` (Model A, 2 channels VV/VH) | **YES** |
| **Checkpoint SHA-256** | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`* | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`* | **YES** |
| **Model Seed** | `0` | `0` | **YES** |
| **Streaming Order** | `Order A` (natural temporal sequence) | `Order A` (natural temporal sequence) | **YES** |
| **Reset Mode** | `reset_per_event` | `reset_per_event` | **YES** |
| **Batch Size** | `16` (tail batch: remainder samples) | `16` (tail batch: remainder samples) | **YES** |
| **Optimizer & LR** | `Adam` ($\text{lr}=10^{-4}, \beta_1=0.9, \beta_2=0.999, \text{wd}=0.0$) | `Adam` ($\text{lr}=10^{-4}, \beta_1=0.9, \beta_2=0.999, \text{wd}=0.0$) | **YES** |
| **Adaptation Objective** | Consistency ($\tau=0.90$, cross-entropy) | Consistency ($\tau=0.90$, cross-entropy) | **YES** |
| **Augmentation Views** | Identity weak vs Horizontal Flip strong | Identity weak vs Horizontal Flip strong | **YES** |
| **Prediction Timing** | `PRE_UPDATE` on unaugmented weak views | `PRE_UPDATE` on unaugmented weak views | **YES** |
| **Target Events** | 6 Dev Events ($N=2,826$, 179 batches) | 6 Dev Events ($N=2,826$, 179 batches) | **YES** |
| **Masking Semantics** | Double-masking ($M_{valid} = \text{MK0\_MNA} \land M_{bounds}$) | Double-masking ($M_{valid} = \text{MK0\_MNA} \land M_{bounds}$) | **YES** |
| **Ground-Truth Sealing** | Labels locked during adaptation loop | Labels locked during adaptation loop | **YES** |

*\*Note on Checkpoint SHA-256 Provenance:* The authoritative SHA-256 of the physical checkpoint file `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` on disk is `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`. An earlier draft table transcribed a discordant string `247d35848529...` due to manual cut-and-paste from draft notes; audit confirmed that all runs (`E2_CONF_90`, `E4_BASE`, `E4_K1`, `E4_K2`, `E4_K3`, `E4_K4`) loaded this exact same physical checkpoint file.

### Provenance Audit Finding:
`E4_BASE` is **100% bit-exact mathematically and operationally identical** to `E2_CONF_90`. Re-running `E4_BASE` on GPU would produce identical weights and predictions, consuming redundant GPU hours.
`tools/run_e4_class_aware_screening.py` supports the `--reuse-base` flag, allowing the runner to load existing predictions from `outputs/e2_confidence_grid/v1/E2_CONF_90`, reconstruct all offline diagnostic interpretation metrics ($3\times 3$ confusion, error breakdowns, transitions, and baseline comparisons), and populate `outputs/e4_class_aware/v1/E4_BASE/` cleanly and deterministically.

3. **Repeat Consistency:**
   Any cell meeting criteria 1 and 2 must reproduce positive gains under Seed 1 / Order B before any claim of robustness is accepted.
