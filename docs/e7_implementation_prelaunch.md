# ChangeGuard Work Package E7: Branch C Implementation & Prelaunch Report

**Status:** READY FOR COMMIT AND BOUNDED SMOKE
**Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git HEAD at Start:** `1b9bdc5df2d2b22f31ae9fbf54cc86e2201fcf29`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E7 (Sections 12.4, 12.5) & Block E8 (Sections 13.1–13.4)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/e6_primary_study_definition.md`
- `docs/e6_claim_to_experiment_map.md`
- `docs/e6_nearest_method_applicability_matrix.md`
- `docs/protocols/e6_cotta_compatibility_protocol_v1.md`
- `docs/protocols/e7_branch_c_empirical_protocol_v1.md`

---

## 1. Executive Summary

Work Package E7 implements the engineering infrastructure, baseline adapters, execution runners, and analysis engines required for **Branch C: Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams**.

In strict accordance with the governing mandate:
- **No policy or decision gates are implemented.** Branch C explicitly rejects accept/reject gates, generic-plus-change score fusion, learned risk classifiers, U/C threshold calibration, reduce actions, and claims of advance harm prediction.
- **The implementation stops at the clean Git checkpoint before real GPU smoke execution.** Zero GPU execution has been launched from uncommitted code.
- **Zero sealed-test data or metadata is accessed.** Quarantined partition `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` is verified untouched.
- **Pure CPU synthetic test suites pass with zero CUDA calls.** 25 focused unit tests covering adapter dynamics, numerical safeguards, spatial inversions, stochastic restoration, non-aliasing, and LOEO cross-event generalization passed cleanly.

---

## 2. What Is Implemented

The following nine files constitute the complete E7 Branch C implementation package:

### 2.1 Protocols & Machine-Readable Configurations
1. **`docs/protocols/e7_branch_c_empirical_protocol_v1.md`:**
   Governing empirical protocol detailing research questions (RQ1–RQ4), evaluated baseline methods, presentation orders (Order A, Order B), source checkpoint seeds (0, 1, 2), development events (dev6), sensory-valid masking, PRE_UPDATE primary prediction timing, LOEO generalization semantics under Branch C, and fail-closed integrity rules.
2. **`configs/changeguard/e7_branch_c_empirical_v1.json`:**
   Formal machine-readable configuration defining the complete empirical evaluation matrix across eight methods (`source_only`, `adabn`, `tent`, `eata`, `sar`, `candidate_e2`, `cotta_continual`, `cotta_reset`), three source seeds, two stream presentation orders, and 179 formal dev batches ($N=2,826$ samples).
3. **`configs/changeguard/e7_cotta_smoke_v1.json`:**
   Bounded GPU smoke configuration targeting Event 520 (USA/Florence), exactly 2 batches ($N=32$ samples, BS=16, Seed 0), with explicit VRAM ($\le 8.0$ GB) and timeout ($\le 60$ s) budgets.

### 2.2 SOTA Baseline Adapter: CoTTA
4. **`src/changeguard/cotta_adapter.py` (`E7CoTTAAdapter`):**
   Implements the official CVPR 2022 Continual Test-Time Domain Adaptation (CoTTA) method adapted to paired bi-temporal SAR flood segmentation:
   - **Mean Teacher Network with EMA:** Teacher $\theta_t$ tracks student $\theta_s$ via $\theta_t \leftarrow \alpha \theta_t + (1 - \alpha) \theta_s$ ($\alpha = 0.999$). Teacher has `requires_grad_(False)` and never receives direct gradients.
   - **Augmentation-Averaged Teacher Predictions ($K=4$):** Evaluates four exact SAR spatial transformations: $\mathcal{A}_1$ (Identity), $\mathcal{A}_2$ (Horizontal Flip), $\mathcal{A}_3$ (Vertical Flip), $\mathcal{A}_4$ (Rot180 / H+V Flip) applied jointly to $(x_A, x_B)$. Softmax maps are inverted via $\mathcal{A}_k^{-1}$ and averaged: $\bar{p}_t(x) = \frac{1}{4} \sum_{k=1}^4 \mathcal{A}_k^{-1}(\text{Softmax}(f(\mathcal{A}_k(x); \theta_t)))$.
   - **Stochastic Source Restoration (SSR):** At each step, each trainable parameter element has independent probability $p = 0.01$ of resetting bit-exactly to initial source checkpoint $\theta_0$, controlled via a dedicated deterministic `torch.Generator`.
   - **Trainable Parameters:** Selected BatchNorm affine parameters ($\gamma, \beta$) only; conv weights frozen; tracking stats disabled.
   - **Sensory-Valid Masking:** Loss is evaluated strictly on valid pixels ($M_{NA} == 1$) where teacher confidence $\max \bar{p}_t \ge \tau = 0.90$.
   - **Storage Non-Aliasing:** Enforces and verifies zero shared parameter or buffer pointers between student, teacher, and source models.
   - **Timing:** Primary recorded prediction is strictly the PRE_UPDATE teacher output $\bar{p}_t(x)$.
   - **Operational Modes:** Supports both continual adaptation (`cotta_continual`) and episodic per-event reset (`cotta_reset_per_event`).
   - **Fail-Closed Safeguards:** Rollback to pre-step state upon non-finite loss, gradients, or post-step weights.
5. **`src/changeguard/__init__.py`:**
   Exports `E7CoTTAAdapter` in the public ChangeGuard library namespace.

### 2.3 Evaluation & Comparative Analysis Runners
6. **`tools/run_e7_empirical_evaluation.py`:**
   Orchestration runner supporting `--config`, `--validate_only`, `--smoke`, `--stream_id`, and `--dry_run`:
   - Validates git cleanliness, dataset manifest SHA-256, split definition SHA-256, and all source checkpoint SHA-256 digests.
   - Quarantines sealed-test events with immediate fatal exceptions.
   - Provides safe resumability: skips already completed streams with verified manifests.
   - Double-masking offline evaluation: $(M_{NA} == 1) \& (M_{LU} \ne 3)$.
   - Exports `stream_summary.json`, `event_metrics.csv`, `completion_status.json`, and `manifest.json`.
7. **`tools/analyze_e7_empirical_comparison.py`:**
   Offline comparative engine and LOEO cross-event analyzer:
   - Aggregates multi-stream evaluation summaries across methods, seeds, orders, and events.
   - Computes event-macro and pooled metrics (Precision, Recall, F1, IoU) and per-event $\Delta F1$ relative to Source-Only.
   - Categorizes outcomes into Harmed ($\Delta F1 \le -0.0100$), Benefited ($\Delta F1 \ge +0.0100$), and Neutral ($|\Delta F1| < 0.0100$).
   - Implements Branch C LOEO cross-event generalization evaluation (6 held-out event folds without policy calibration or pseudo-replication).
   - Generates `e7_master_comparison_table.csv`, `e7_loeo_generalization_report.json`, and `e7_empirical_synthesis_summary.json`.

### 2.4 Unit Test Suites
8. **`tests/test_e7_cotta_adapter.py`:**
   16 focused unit tests covering non-aliasing, source immutability, exact EMA mathematics, spatial augmentation inversion, SSR binomial distribution ($0.010 \pm 0.002$), bit-exact restoration, sensory mask sealing, confidence thresholding, safe skipping on empty batches, bit-exact source reset, fail-closed rollback on NaNs, strict label isolation, PRE_UPDATE timing, deterministic RNG seeds, and pure CPU execution.
9. **`tests/test_e7_empirical_runner.py`:**
   9 focused unit tests verifying preflight verification, sealed-test quarantine, SHA-256 validation, overwrite prohibition, stream completion detection, LOEO 6-fold isolation without cross-event leakage, and delta F1 synthesis arithmetic.

---

## 3. What Is Tested & Verified

All unit tests and preflight checks were executed in the project environment (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`):

| Test Suite | Tests Run | Result | Duration | Execution Mode |
| :--- | :---: | :---: | :---: | :---: |
| `tests/test_e7_cotta_adapter.py` | 16 | **OK** | 1.66s | Pure CPU |
| `tests/test_e7_empirical_runner.py` | 9 | **OK** | 1.77s | Pure CPU |
| `Combined E7 Verification Suite` | 25 | **OK** | 10.06s | Pure CPU |
| `tools/run_e7_empirical_evaluation.py --validate_only` | Preflight | **PASSED** | 0.8s | Preflight Audit |
| `tools/run_e7_empirical_evaluation.py --smoke --validate_only` | Preflight | **PASSED** | 0.8s | Preflight Audit |
| `tests/test_e5_temporal_risk_analysis.py` (Regression) | 12 | **OK** | 60.21s | Regression Audit |

### Verified Checkpoint & Split Hashes
- **Seed 0 Checkpoint:** `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` (VERIFIED)
- **Seed 1 Checkpoint:** `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` (VERIFIED)
- **Seed 2 Checkpoint:** `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` (VERIFIED)
- **KuroSiwo Split:** `fbf30fe0bab7b90c468aaf0fbdab7960a5cf76dd49dfb3ca51f1e01668fba85f` (VERIFIED)
- **KuroSiwo Manifest:** `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad` (VERIFIED)

---

## 4. What Is Smoke-Ready

The bounded GPU smoke test is fully configured and ready for execution:
- **Configuration:** `configs/changeguard/e7_cotta_smoke_v1.json`
- **Target Event:** Event 520 (USA / Florence)
- **Batch Count:** Exactly 2 batches ($N=32$ samples, batch size 16)
- **Model / Seed:** SNUNet-ECAM, Seed 0
- **Runner Command:**
  ```bash
  /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/run_e7_empirical_evaluation.py --smoke
  ```
- **Execution Budget:** VRAM $\le 8.0$ GB, execution time $\le 60$ seconds.

---

## 5. What Is Not Yet Executed

- **GPU Smoke Test:** NOT yet executed. Held at the clean Git checkpoint per governing instructions.
- **Formal Empirical Evaluation Matrix:** NOT yet executed.

---

## 6. Unresolved Issues & Risks

- **Zero unresolved issues.** All components conform strictly to Plan V3 specifications and pass all synthetic verification checks.

---

## 7. Scope Boundaries (Branch C)

The following items are explicitly **out of scope** and **not implemented**:
- Accept/reject decision gates or selective adaptation classifiers.
- Generic-plus-change score fusion policies.
- Learned risk predictors or heuristic threshold calibration.
- Claims of advance harm prediction.
- Real GPU smoke or formal runs prior to clean Git commit.

---

## 8. Exact Git Add and Commit Commands

To commit this clean implementation stage to HEAD:

```bash
git add \
  configs/changeguard/e7_branch_c_empirical_v1.json \
  configs/changeguard/e7_cotta_smoke_v1.json \
  docs/protocols/e7_branch_c_empirical_protocol_v1.md \
  docs/e7_implementation_prelaunch.md \
  src/changeguard/__init__.py \
  src/changeguard/cotta_adapter.py \
  tests/test_e7_cotta_adapter.py \
  tests/test_e7_empirical_runner.py \
  tools/analyze_e7_empirical_comparison.py \
  tools/run_e7_empirical_evaluation.py

git diff --cached --check

git commit -m "feat(e7): implement branch c empirical evaluation suite, cotta adapter, and loeo runner"
```
