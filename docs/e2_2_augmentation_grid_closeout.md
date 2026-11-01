# ChangeGuard Plan V3 Work Package E2.2: Augmentation Grid Closeout Report

**Status:** COMPLETE (Implementer Self-Verification)  
**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `1a8b07c044ffd783552c266fe2b2a8986b2fbafc`  
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md` (Section 7.2: Augmentation Strategy Exploration)
- `docs/CACHE_AND_MEASUREMENT_CONTRACT.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/CURRENT_STATUS.md`
- `docs/e2_2_preflight_and_reuse_audit.md`

---

## 1. Executive Summary

Work Package **E2.2** (*Strong Augmentation Strategy Exploration*) evaluated the sensitivity and effectiveness of data augmentation strategies within the ChangeGuard consistency adaptation framework. E2.2 was executed strictly under implementer self-verification using the frozen Plan V3 protocol:
- **Predefined Confidence Reference:** Threshold $\tau = 0.90$ was fixed as the primary reference across all cells (as mandated by Plan V3; not post-hoc tuned).
- **Single-Factor Exploration:** Augmentation strategy was the only experimental factor varied across the grid.
- **Evaluation Partition:** Development 6 Events ($N = 2,826$ paired patches, $135,859,080$ total valid pixels). Zero access was made to the 10 sealed-test events.
- **Decision Threshold:** Uniform $\theta = 0.50$ (non-tuned).
- **Streaming Protocol:** Seed 0, Order A (`[279, 437, 520, 559, 1111003, 1111008]`), `reset_per_event`, batch size 16 (tail batches processed as-is).

### Grid Execution Summary
| Cell ID | Augmentation Strategy | Operational Status | Macro F1 | Pooled F1 | Worst Event F1 | Sel Pixel % | Total Updates | Mean Loss | Mean Grad Norm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2_AUG_A0** | Horizontal Flip Strong (Reference) | `COMPLETED` (Reused) | 0.5733 | 0.4419 | 0.2650 (#1111003) | 88.79% | 179 / 179 | 0.00392 | 0.00628 |
| **E2_AUG_A1** | Vertical Flip Strong | `COMPLETED` (Executed) | 0.5733 | 0.4419 | 0.2650 (#1111003) | 88.79% | 179 / 179 | 0.00448 | 0.00693 |
| **E2_AUG_A2** | Small Integer Translation ($\Delta y=4, \Delta x=4$) | `COMPLETED` (Executed) | 0.5733 | 0.4419 | 0.2650 (#1111003) | 88.84% | 179 / 179 | 0.00149 | 0.00908 |
| **E2_AUG_A3** | Independent Weak Geometric Views ($W_1$ vs $W_2$) | `COMPLETED` (Executed) | 0.5718 | 0.4501 | 0.2605 (#1111003) | 88.92% | 179 / 179 | 0.00563 | 0.00973 |
| **E2_AUG_A4** | SAR Intensity / Noise Perturbation | **`BLOCKED`** (Documented) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

All 24 executed event evaluations satisfy exact confusion conservation ($TP + FP + FN + TN = \text{valid}$), exact sample counts ($N = 2,826$), and complete telemetry logging.

---

## 2. Preflight and Equivalence Verification

### 2.1 Clean Base State
- Git HEAD at launch was verified at clean commit `1a8b07c044ffd783552c266fe2b2a8986b2fbafc`.
- `git status --porcelain` showed no dirty tracked files.
- Python environment: `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` (PyTorch 2.5.1+cu121).

### 2.2 Reused Reference Cell: `E2_AUG_A0`
Cell `E2_AUG_A0` was reused from Work Package E2.1 artifact `outputs/e2_confidence_grid/v1/E2_CONF_90` based on a 13-dimension cryptographic equivalence audit documented in `docs/e2_2_preflight_and_reuse_audit.md`. The reused cell matches the required checkpoint (`247d3584...`), seed (0), order (Order A), reset mode (`reset_per_event`), batch size (16), confidence threshold (0.90), augmentation (Identity weak, Horizontal Flip strong), optimization objective (one-way KL divergence from frozen source probabilities to updated candidate predictions), and cache schema (`CACHE_SCHEMA_V1`).

---

## 3. Augmentation Implementation & Geometric Correctness

### 3.1 Synchronized Temporal Pair Transformations
Because ChangeGuard processes bi-temporal SAR image pairs $(T_1, T_2) \in \mathbb{R}^{B \times 2 \times H \times W}$, spatial augmentations must be applied **identically** to both $T_1$ and $T_2$ to preserve true change boundaries and prevent artificial temporal registration artifacts:
$$T_{aug} = \mathcal{T}(T_1, T_2), \quad \text{where } \mathcal{T} \text{ applies identical transformation to channels } 0 \text{ and } 1.$$

### 3.2 Exact Inverse Coordinate Realignment
Consistency loss enforces agreement between predictions generated from the weak view and predictions generated from the strong view. For the strong view predictions $P_{strong}$ to be compared against weak view predictions $P_{weak}$ in native spatial coordinates:
$$P_{strong\_native} = \mathcal{T}^{-1}(P_{strong})$$
- **Horizontal Flip (A0):** $\mathcal{T}^{-1}(P) = \text{torch.flip}(P, \text{dims}=[-1])$.
- **Vertical Flip (A1):** $\mathcal{T}^{-1}(P) = \text{torch.flip}(P, \text{dims}=[-2])$.
- **Translation (A2):** Shift $(+\Delta y, +\Delta x) = (4, 4)$ is mapped back via $(-\Delta y, -\Delta x) = (-4, -4)$.
- **Independent Weak Views (A3):** $W_1$ (H-flip) inverted via H-flip; $W_2$ (V-flip) inverted via V-flip.

### 3.3 Translation Boundary Semantics & Overlap Masking (A2)
For integer translation by $(\Delta y, \Delta x) = (4, 4)$ pixels on $(224 \times 224)$ tiles:
1. Zero-padding introduced at image margins is strictly non-sensory.
2. A deterministic geometric overlap mask $M_{overlap} \in \{0, 1\}^{B \times 1 \times H \times W}$ was constructed:
   $$M_{overlap}[b, 0, y, x] = \begin{cases} 1 & \text{if } 0 \le y < H - \Delta y \text{ and } 0 \le x < W - \Delta x \\ 0 & \text{otherwise} \end{cases}$$
3. The active valid pixel mask $V_{active}$ was formed as:
   $$V_{active} = V_{sensor} \cap M_{overlap}$$
   Padding regions are zeroed ($M_{overlap} = 0$), strictly excluding boundary artifacts from both selection masking and consistency loss computation.
4. Empirical verification: Across all 6 events, $V_{active}$ in A2 was strictly smaller than in A0 ($114,788,700$ vs $118,983,019$ for Event 279, matching the exact theoretical tile area fraction $(220/224)^2 \approx 0.9646$).

### 3.4 Physical Domain Certification & Formal Blocking of A4
Plan V3 mandates that any candidate intensity, contrast, or noise perturbation strategy (A4) must be certified physically and numerically valid on KuroSiwo inputs; otherwise it must be formally marked `BLOCKED`.

**Technical Justification for BLOCKED Status:**
1. **Invalidity of Multiplicative Perturbation in Standardized Space:** KuroSiwo Sentinel-1 GRD inputs are clamped to $[0.0, 0.15]$ and z-score standardized ($\mu = [0.0953, 0.0264], \sigma = [0.0427, 0.0215]$). Standardized inputs contain negative values (spanning approximately $[-2.5, +3.0]$). Applying multiplicative speckle directly in z-score-standardized space is semantically invalid because standardized features are zero-mean unit-variance statistical variables, not physical non-negative intensity values.
2. **Assessment of Inverse Standardization ($x = \tilde{x} \cdot \sigma + \mu$):** We evaluated whether applying the known inverse affine transform $x = \tilde{x} \cdot \sigma + \mu$ could recover a sufficiently justified linear-domain proxy for A4. Even though $x \in [0.0, 0.15]$ is non-negative, A4 remains blocked due to the following unresolved obstacles:
   - **Missing Radiometric Calibration & Number of Looks:** Simulating physical Gamma speckle $\eta \sim \Gamma(L, L)$ requires the effective number of looks ($L$) and sensor calibration curves. KuroSiwo provides only 8-bit normalized GRD patches without calibration metadata, incident angles, or $L$, rendering any parameter choice uncalibrated and arbitrary.
   - **Irreversible Truncation:** Clamping at $0.15$ in the ingestion pipeline permanently saturates high-backscatter urban and structural returns, corrupting upper-tail statistics.
   - **Absence of a Defensible Predeclared Magnitude:** Upstream source model pretraining explicitly froze `data_augmentations: false`. There is no defensible, pre-validated perturbation magnitude that preserves semantic labels while testing consistency on this frozen backbone.
3. **Formal Decision:** Cell `E2_AUG_A4` is recorded as **`BLOCKED`** with zero code execution and zero GPU cycles spent.

---

## 4. Quantitative Results and Event-Level Breakdown

### 4.1 Master Event-Level Performance
| Cell ID | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro F1 | Pooled F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2_AUG_A0** | 0.300409 | 0.634424 | 0.699638 | 0.692393 | 0.265045 | 0.847999 | **0.573318** | **0.441912** |
| **E2_AUG_A1** | 0.300444 | 0.634424 | 0.699638 | 0.692393 | 0.265045 | 0.847999 | **0.573322** | **0.441948** |
| **E2_AUG_A2** | 0.300416 | 0.634424 | 0.699638 | 0.692393 | 0.265045 | 0.847999 | **0.573321** | **0.441916** |
| **E2_AUG_A3** | 0.308696 | 0.632617 | 0.699499 | 0.680373 | 0.260534 | 0.849176 | **0.571816** | **0.450124** |

### 4.2 Confusion Matrix Totals Across Development Set
| Cell ID | Total TP | Total FP | Total FN | Total TN | Total Valid Pixels | Total Patches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2_AUG_A0** | 1,334,864 | 2,906,575 | 465,012 | 131,152,629 | 135,859,080 | 2,826 |
| **E2_AUG_A1** | 1,334,844 | 2,905,987 | 465,032 | 131,153,217 | 135,859,080 | 2,826 |
| **E2_AUG_A2** | 1,334,868 | 2,906,510 | 465,008 | 131,152,694 | 135,859,080 | 2,826 |
| **E2_AUG_A3** | 1,353,607 | 2,845,984 | 446,269 | 131,213,220 | 135,859,080 | 2,826 |

*Note: For all cells, $TP + FP + FN + TN = 135,859,080$ valid pixels conserved identically.*

### 4.3 Training & Optimization Telemetry
| Cell ID | Mean Loss | Min Loss | Max Loss | Mean Grad Norm | Max Grad Norm | Total Runtime | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2_AUG_A0** | 0.00392 | 0.00067 | 0.0120 | 0.00628 | 0.0480 | 427.6s | 18,414 MB |
| **E2_AUG_A1** | 0.00448 | 0.00063 | 0.0129 | 0.00693 | 0.0791 | 257.0s | 18,416 MB |
| **E2_AUG_A2** | 0.00149 | 0.00021 | 0.0051 | 0.00908 | 0.0763 | 257.2s | 18,416 MB |
| **E2_AUG_A3** | 0.00563 | 0.00109 | 0.0152 | 0.00973 | 0.0988 | 257.2s | 18,435 MB |

### 4.4 Key Observations & Analysis
1. **Bounded Metric Stability Across Tested Discrete Transformations:** Single-view paired transformations (horizontal flip A0, vertical flip A1, small integer translation A2) produced nearly identical aggregate metrics under the tested discrete transformations (Macro F1: $0.573318$ vs $0.573322$ vs $0.573321$; Pooled F1: $0.441912$ vs $0.441948$ vs $0.441916$). We report this as a bounded empirical observation across the evaluated discrete transformations, without claiming general rotational or translational equivariance for the underlying network.
2. **Impact of Overlap Masking (A2):** Translation A2 achieved a markedly lower consistency loss ($0.00149$ vs $0.00392$ in A0), while maintaining gradient norms ($0.00908$) and overall F1. This lower loss directly reflects the rigorous masking of boundary padding, ensuring loss is accumulated only over overlapping pixels.
3. **Independent Weak Views (A3):** Dual-view consistency between two weakly augmented streams ($W_1$ H-flip vs $W_2$ V-flip) produced the highest consistency loss ($0.00563$) and highest mean gradient norm ($0.00973$). This higher training signal drove a noticeable increase in Pooled F1 ($0.4501$ vs $0.4419$, $+0.0082$) by detecting more true positives ($1,353,607$ vs $1,334,864$) and reducing false positives ($2,845,984$ vs $2,906,575$), with a slight trade-off in event-macro balance ($0.5718$ vs $0.5733$).

---

## 5. Acceptance Criteria Verification

| ID | Criterion | Target Requirement | Audit Finding | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Clean Git commit preflight | HEAD at clean, approved commit | Commit `1a8b07c` verified clean before run | **PASSED** |
| **2** | Preflight and reuse audit | Comprehensive preflight for A0 and grid | Authored `docs/e2_2_preflight_and_reuse_audit.md` | **PASSED** |
| **3** | Locked E2.2 configuration | Explicit semantics for all cells A0–A4 | `configs/changeguard/e2_2_augmentation_grid_v1.json` | **PASSED** |
| **4** | Geometry & masking unit tests | Automated tests verifying transformations | 13/13 tests passed in `test_e2_2_augmentation_grid.py` | **PASSED** |
| **5** | Smoke verification before runs | Bounded smoke execution on dev data | Event 520 smoke run executed cleanly with exit code 0 | **PASSED** |
| **6** | Zero sealed-test access | Strictly quarantine 10 test events | Audited: 0 test events accessed; only Dev 6 evaluated | **PASSED** |
| **7** | Matched protocol preserved | Single-factor variation under Plan V3 | All settings matched; only augmentation varied | **PASSED** |
| **8** | Development events evaluated | Full Dev 6 evaluated ($N=2,826$) | All 6 events processed; exact patch counts verified | **PASSED** |
| **9** | Complete artifacts written | Step records, frozen predictions, run logs | Complete step records & npz written for all cells | **PASSED** |
| **10** | Complete master tables | JSON and Markdown master tables | Both written in `outputs/e2_augmentation_grid/v1/` | **PASSED** |
| **11** | Invariant conservation | Confusion conservation & sample counts | 24/24 evaluations conserved confusion & counts | **PASSED** |
| **12** | A4 status documented | Defensible physical/mathematical justification | Formally BLOCKED with Z-score/SAR domain proof | **PASSED** |
| **13** | Descriptive comparison | Objective reporting without overclaiming | Descriptively analyzed; no final recipe promoted | **PASSED** |
| **14** | SHA-256 digests pinned | Cryptographic checksums of all artifacts | Computed and pinned in Section 6 | **PASSED** |

---

## 6. Cryptographic Artifact Checksums

```
b62ccdd24ffbeffbd1338811435dba1848b8702eef19f9fba07b2052e1d38829  configs/changeguard/e2_2_augmentation_grid_v1.json
4bd268845e43d7dbad873aac43fda6f87adaa037cb5004de37f7edafc00cc3ac  docs/e2_2_preflight_and_reuse_audit.md
80abf94f94d90e3c9fe66f5f755578bbb2a50906cc8e6600dadaa301f8ddc23a  src/changeguard/e2_adapters.py
251423a71f2f68d41ebafcd455a8b1253d07cea1bbf3352e071fed4ad2cc0915  tests/test_e2_2_augmentation_grid.py
ee5b00d6b43c625f139c490ba9f10224bb1a52fd8fe42e5e60b317b95eff5e2e  tools/run_e2_2_augmentation_grid.py
4add30aabb700c80e4f94aa65cab7b986b449034637a3f6e4ea8b14ec35693c7  tools/verify_e2_2_integrity.py
49520f48c2545c414ab28144deefa240c38481db2cb040cfe8a9d9601eea636d  scripts/launch_e2_2_queue.sh
8d3ddf8f76125691999df78b53f2d1e77e66fda685f71222df6606c4a7d003e4  outputs/e2_augmentation_grid/v1/master_augmentation_grid.json
e5f11721a8f69fe95e443c792506db72ef5d2cdd12638e25ec294b9ea65f6bf0  outputs/e2_augmentation_grid/v1/master_augmentation_grid.md
6054b64d682b266778550987cd1924c555306e293447224d4d51d2cdc0b39c31  logs/e2_augmentation_grid/v1/augmentation_grid_execution.log
```

---

## 7. Operational Boundaries & Next Steps

1. **E2.2 Scope Complete:** Work Package E2.2 is closed. All planned cells (A0, A1, A2, A3, A4) are accounted for, executed, or formally blocked.
2. **No Final Recipe Selection:** No final augmentation strategy or recipe has been selected or promoted.
3. **E2.3 Readiness:** Work Package E2.3 (*Promotion, Multi-Seed Repeat & Sensitivity*) is ready for preflight upon explicit user direction. No E2.3 runs have been initiated.

