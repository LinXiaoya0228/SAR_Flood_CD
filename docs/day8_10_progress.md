# ChangeGuard Days 8–10 Execution Progress Ledger

**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host / Device:** `gpu22` / NVIDIA GeForce RTX 3090  
**Python Environment:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`  
**Initial Baseline Commit (Gate A):** `214150c2ad22e09814888748e49f71fc9678a62e`  
**Starting Date:** October 4, 2026  
**Security Boundary:** 10 Sealed Test Events strictly untouched: `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.  
**Usable Validation Events:** Exactly 6 events: `[279, 437, 520, 559, 1111003, 1111008]`.

---

## Phase 0: Finish and Checkpoint Days 1–7

### 0.1 Reproducibility Bugfix & Audit Refinements
- **Bug Fixed in `tools/run_gate_b_offline_evaluation.py`:** Added definition `y_harm_10 = df_joined["hidden_is_harmful_thresh10"].values` at line 280. Threshold remains strictly $\Delta\text{F1} \le -0.10$. No primary thresholds or formulas altered.
- **Focused Regression Test:** Added `tests/test_risk_thresholds.py` covering all four harm thresholds ($0.0, -0.01, -0.02, -0.10$) and offline evaluation variable integration.
- **Audit Refinement in `docs/day6_7_gate_b_audit.md`:** Replaced overstrong statement ("sole legitimate discriminatory measures" -> "more informative discriminatory summaries in this high-prevalence setting"). Confirmed statement that generic drift's strength is primarily cross-event separation.
- **Unit Test Results:** 40 / 40 tests passed (21 discovery suite tests including regression test, 4 completeness tests, 6 metric tests, 9 preprocessing tests).
- **Frozen Experimental Artifacts Integrity:** All 15 baseline artifacts re-hashed and verified 100% identical.

### 0.2 Days 1–7 Git Checkpoint
- **Status:** COMPLETED
- **Commit Message:** `feat: complete Days 1-7 ChangeGuard pilot gates`
- **Resulting Commit Hash:** `ca4d18e2cea9ebc9803476b5d462ff44c9b126c1`
- **Files Committed:** 18 files (3058 insertions, 10 deletions)

---

## Day 8: TTA Robustness Matrix

### 8.1 Inventory of Matrix Cells
| Method | Configuration / LR | Seed(s) | Status | Artifact Provenance / Manifest |
| :--- | :---: | :---: | :---: | :--- |
| `source_only` | LR = 0.0 | Seed 0 | **COMPLETE** | `outputs/day3_5_tta/manifest_source_only_seed0.json` |
| `adabn` | LR = 0.0 | Seeds 0, 1, 2 | **COMPLETE** | `outputs/day3_5_tta/manifest_adabn_seed{0,1,2}.json` |
| `tent` | LR = 1e-5 | Seeds 0, 1, 2 | **COMPLETE** | `outputs/day3_5_tta/manifest_tent_lr1e-5_seed{0,1,2}.json` |
| `tent` | LR = 3e-5 | Seeds 0, 1, 2 | **COMPLETE** | Executed under Day 8 protocol (`outputs/day8_robustness/`) |
| `tent` | LR = 1e-4 | Seeds 0, 1, 2 | **COMPLETE** | `outputs/day3_5_tta/manifest_tent_lr1e-4_seed{0,1,2}.json` |

### 8.2 Frozen Protocol Specification
- **Configuration File:** `configs/day8_tta_robustness_protocol.json`
- **SHA-256 Digest:** `6aeeb4446da108ba5312409fff943cf10540e3ce51cd8dc3d19a013bdab44df7`
- **Missing Cells Executed:** Tent at lr=$3\times 10^{-5}$ across Seeds 0, 1, 2 (3 runs total, 2,826 samples each).
- **Execution Log:** `outputs/day8_robustness/logs/day8_execution.log` (Status: SUCCESS).

### 8.3 Day 8 Artifacts and SHA-256 Hashes
- `outputs/day8_robustness/robustness_event_metrics.csv`: `ed525d5947dd2c1831ce557c4df77b3df756844e6c80bef92d71c11489a630a7`
- `outputs/day8_robustness/robustness_summary.csv`: `c9ac25e28de191462385f2296fd906e968d3ae68740fe27305ad6e3fbe72ed60`
- `outputs/day8_robustness/run_manifest.json`: `5de7949bc03472266b9234033c7ae7f9aae9aa14127d11fbe503a663744bbca3`
- `docs/day8_robustness_audit.md`: `b144dc9520b4669cec8a50c1f5d3cbfe683bf912b2c0618d60d537fa95162e2d`

### 8.4 Empirical Findings & Conclusions
1. **LR Tuning Does Not Cure Degradation:** Macro F1 drop across Tent learning rates ($10^{-5}$, $3\times 10^{-5}$, $10^{-4}$) is virtually flat ($-0.0212$, $-0.0208$, $-0.0194$) relative to source-only ($0.5646$).
2. **AdaBN Degrades Similarly:** Pure BN statistics adaptation without optimization also drops macro F1 to $0.5431$ ($\Delta = -0.0215$) and pooled F1 from $0.7020$ to $0.3469$.
3. **Severe Asymmetry:** Large events with extensive baseline predictions (Event 279, Event 437) suffer major drops in precision and F1 (Event 279: $0.6301 \to 0.2241$, $\Delta = -0.4060$; Event 437: $0.7481 \to 0.4921$, $\Delta = -0.2560$). In contrast, Event 559 improves dramatically ($0.0816 \to 0.6675$, $\Delta = +0.5859$) due to sensitivity gain from near-zero source recall ($0.0426 \to 0.5405$).
4. **Conclusion:** Adaptation failure is not an over-aggressive learning rate artifact, but an intrinsic consequence of test-time distribution estimation under class-imbalanced, sparse-change SAR distributions.

---

## Day 9: Event-Level Adaptation-Suitability Feasibility

### 9.1 Protocol & Specification
- **Configuration File:** `configs/day9_event_suitability_protocol.json`
- **SHA-256 Digest:** `2a5893d6a5901f4e8f8ffa780eca7316254fa1ee542acc98c9aa9d0f07c20a4c`
- **Primary Unit:** 1 event $\times$ 1 predeclared order (Seed 0 primary, Seeds 1 & 2 sensitivity).
- **Primary Window:** First batch only (`batch_idx == 1`, $N=16$ samples).
- **Secondary Window:** First two batches (`batch_idx in [1, 2]`, $N=32$ samples).
- **Cross-Validation Scheme:** Leave-One-Event-Out (LOEO) across 6 validation events.
- **Script Executed:** `tools/run_day9_event_suitability.py` (Status: SUCCESS).

### 9.2 Day 9 Artifacts and SHA-256 Hashes
- `outputs/day9_event_suitability/event_level_features.csv`: `9ffa0b0ec029b87c98e556312c4ced6d226723df08799dcb2442663f75320c14`
- `outputs/day9_event_suitability/loeo_decisions.csv`: `b3adddbfbd4aa3eee3c79e43543495af8035f577c106277d03b38c42da675522`
- `outputs/day9_event_suitability/policy_comparison.csv`: `7d05b67851aa4024ccf78511f624d37e48b88b4b248e6c4589e47ce032651eaf`
- `outputs/day9_event_suitability/run_manifest.json`: `59b4c9b4aa8d2be8509c6fa13022b0825a310ffe3909fd93e2d0b8061ebbdaa7`
- `docs/day9_event_suitability_audit.md`: `d6e50a7c52794b37e492d32d1abc5ccc8f11e27a98cd2766a273cee522187be9`

### 9.3 Empirical Findings & Conclusions
1. **Single-Class Fold Degeneration:** In the 6 validation events, only 1 event (Event 559) exhibits positive adaptation ($\Delta\text{F1} \ge +0.01$). In Fold 559, the training set contains 0 beneficial events, compelling the training optimizer to select `reject_all`. Consequently, held-out Event 559 is rejected out-of-fold.
2. **Degeneration to Reject-All:** The LOEO selector on generic BN drift (`bn_affine_drift_l2`) rejects adaptation across 100% of events, matching Source-Only macro F1 ($0.5646$) with 0% selection frequency.
3. **In-Sample Overfitting vs Out-of-Sample Failure:** While in-sample inspection shows Event 559 has distinct low drift and minimal source flood area, an out-of-sample classifier cannot discover this without seeing Event 559.
4. **Oracle Potential:** The theoretical Oracle selector achieves $0.6637$ macro F1 (+0.0991 over source-only), proving that event gating is conceptually sound but statistically unlearnable from a 6-event validation set with 1 positive sample.

---

## Day 10: Two-Week Pilot Final Synthesis

### 10.1 Deliverable & Specifications
- **Comprehensive Report:** `docs/two_week_pilot_final_report.md`
- **SHA-256 Digest:** `0bcf595de775505dcaff0ce5f88b33556b6e3768769e3b6f70b248558f46cc47`
- **Test Suite Status:** 40 / 40 passing tests verified.
- **Sealed Test Boundary:** 10 test events completely sealed (zero access confirmed).

### 10.2 Final Pilot Decision
- **Verdict:** `PIVOT_FAILURE_CHARACTERIZATION`
- **Proposed FYP Title:** *"When Does Test-Time Adaptation Fail in Earth Observation? Characterizing Failure Modes, Distribution Shifts, and Safety Boundaries in SAR Disaster Change Detection"*
- **Rationale:** Standard CV TTA fails on SAR flood mapping due to class imbalance and speckle-driven entropy collapse. A diagnostic, benchmark, and failure-characterization thesis provides deep scientific novelty, eliminates false controller risks, and is 100% achievable by March 2027.
- **Git State:** Local baseline checkpoint `ca4d18e2cea9ebc9803476b5d462ff44c9b126c1`:
  - Tracked files: zero modified and zero staged.
  - Days 8–10: eight untracked files.
  - Days 8–10 not yet committed.


