# RST5 Closeout: Recipe Freeze and Reproducibility Handoff

## 1. Stage Identification and Executive Status

- **Stage ID**: RST5 (Recipe Freeze and Reproducibility Handoff)
- **Status**: COMPLETED
- **Preceding Milestone**: RST4 Discovery Staircase (`ce76b80288954d52e66e4875169c30022887a068`)
- **Next Milestone**: RST6 Confirmatory Training Execution (NOT_STARTED)
- **Repository Root**: `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`

---

## 2. Key Accomplishments

### 2.1 RST4 Step Accounting & Boundary Accumulator Audit
- Traced the reported theoretical (7,488) vs realized (7,496) optimizer steps across the four RST4 discovery runs (R0-R3).
- Identified root cause: inter-epoch accumulation bleed caused by accumulating microbatches globally without resetting upon `optimizer.step()`, leading to an extra step on the first microbatch of 8 subsequent epochs.
- Confirmed all four discovery runs executed identical steps (7,496) and observed identical samples (239,472), ensuring absolute fairness across R0, R1, R2, and R3.
- Resolved for confirmatory runs in `src/robust_source_training/runner.py` by maintaining an epoch-local accumulator `accum_microbatches` that resets on every `optimizer.step()`, yielding strictly deterministic tail flushes: exactly 416 steps/epoch (6,656 total) for Fold A, and exactly 458 steps/epoch (7,328 total) for Fold B.
- Documented in `docs/robust_source_training/rst5_rst4_step_accounting_audit_v1.md`.

### 2.2 Label Semantics and Class Index Audit
- Confirmed semantic consistency from raw KuroSiwo labels through data loader, loss functions, and evaluation:
  - Class 0: Dry ground / no water (background)
  - Class 1: Permanent water body (rarer, upweighted to ~1.4 - 1.5)
  - Class 2: Newly inundated flood water (positive evaluation class, target of Macro-F1)
  - Class 3: Ignore index (`MK0_MNA == 0` or `MK0_MLU == 3`), masked out of both CE and soft-Dice losses.
- Documented in `docs/robust_source_training/rst5_label_semantics_audit_v1.md`.

### 2.3 Frozen Recipe R2 & Empirical Class Weights
- Pre-registered Recipe R2 (`M1=True, M2=True, M3=False`) and matched baseline R0 (`M1=False, M2=False, M3=False`).
- Computed training class weights for Fold A (Train B+C, 13,302 patches): `[0.486491, 1.409167, 1.104342]`.
- Computed training class weights for Fold B (Train A+C, 14,655 patches): `[0.543519, 1.499552, 0.956929]`.
- Enforced strict normalization ($\sum w_c = 3.0$) and bounding ($[0.5, 2.0]$ before normalization).
- Frozen in `configs/robust_source_training/rst5_frozen_recipe_r2_v1.json`.

### 2.4 Confirmatory Matrix & Execution Ledger
- Generated exactly 12 confirmatory configuration files in `configs/robust_source_training/rst5_confirmation/`:
  - Fold A: 3 seeds for R0, 3 seeds for R2
  - Fold B: 3 seeds for R0, 3 seeds for R2
- Generated `rst5_confirmation_matrix_v1.json` and `rst5_confirmation_ledger_v1.json` detailing:
  - 12 physical runs
  - 18 unique held-out evaluation events (9 per fold)
  - 108 event-level evaluation rows
- Enforced `launch_authorized: false` and `preparation_only: true` on all 12 configs.

### 2.5 Confirmatory Decision Rules (P1, P2, P3)
- Formalized pre-registered statistical and scientific thresholds in `configs/robust_source_training/rst5_confirmation_decision_rule_v1.json`:
  - **Criterion P1 (Average Superiority)**: $\Delta \text{Macro-F1} \ge +0.010$, not driven by single event, no precision/recall collapse $\ge 0.050$.
  - **Criterion P2 (Tail Reliability)**: Bottom-3 $\Delta \text{Macro-F1} \ge +0.020$, mean $\Delta \ge -0.005$.
  - **Criterion P3 (Seed Stability)**: Relative SD reduction $\ge 20\%$ in both folds, mean $\Delta \ge -0.005$.

### 2.6 Pre-Launch Static Validation & Dry Runs
- Executed `validate_only()` and CPU-only `dry_run()` across all 12 configs.
- Output report `outputs/robust_source_training/rst5_prelaunch_validation/readiness_report.json` confirmed `all_validations_passed: true` for all 12 runs.

### 2.7 Interruption, Resume, and Test Suite
- Tested synthetic interruption and state equality across model weights, optimizer, and RNG states.
- Tested strict rejection of mismatched config/fold hashes and train/eval event allowlists.
- All 69 tests across the repository robust source training test suite passed:
  - `tests/test_rst2_infrastructure.py`: 22 passed
  - `tests/test_rst3_components.py`: 21 passed
  - `tests/test_rst4_discovery.py`: 14 passed
  - `tests/test_rst5_confirmation.py`: 12 passed

---

## 3. Core Artifact Integrity Hashes (SHA-256)

| Artifact Path | SHA-256 Digest |
|---|---|
| `configs/robust_source_training/rst5_frozen_recipe_r2_v1.json` | `c707daa669d88260b8db517adbd9e238cf5f50a725c91d6e6640e5e11b6e6fab` |
| `docs/robust_source_training/rst5_frozen_recipe_r2_v1.md` | `a38782660e94918197d22be95c56bac6c4a1a553063216d1fd2f0301e76122e2` |
| `configs/robust_source_training/rst5_folda_class_weights_v1.json` | `c3bbb6bae14d98710798883132aab623c2cc554c226f796c1e15ff87591640bd` |
| `configs/robust_source_training/rst5_foldb_class_weights_v1.json` | `d84e797fd63a097124231c130a9728950d2bf3e676c937f99790fe4dcb239a83` |
| `configs/robust_source_training/rst5_confirmation_matrix_v1.json` | `ec5f5e424e5c9ec432f866620c3268c14fa9e5550510796768011251bd7bf2a1` |
| `configs/robust_source_training/rst5_confirmation_ledger_v1.json` | `1116c55f2aef1be71409f084f228b45720dde4c4a5bad5759a549d22f9f97db1` |
| `configs/robust_source_training/rst5_confirmation_decision_rule_v1.json` | `b347a1f903a4e57e9ca8e7835c03fb65cef89a732c0feb6eeadfbab68a48db35` |
| `docs/robust_source_training/rst5_confirmation_decision_rule_v1.md` | `d8073556805e4408f591076164e10f9c5ad69b07140d11d1f7cc8efe115f42e5` |
| `docs/robust_source_training/rst5_rst4_step_accounting_audit_v1.md` | `c9c7ba529597d691d0f1d49534fc96dfb4e25387befd43d44f31fb4cc1521ed3` |
| `docs/robust_source_training/rst5_label_semantics_audit_v1.md` | `dd8e19c61a5f524c963a3cc83cc90dccb5420693abafd6a0d01d80cf12f0d523` |
| `docs/robust_source_training/rst5_monitoring_and_recovery_v1.md` | `c78332d2634d51d87a4443d2db73c6609e766e9a7f1f7ca6989a347a83064ce5` |
| `docs/robust_source_training/rst5_reproducibility_handoff_v1.md` | `a96fb21338e77aa52df00b44854c9ed66ab04faba46a784f1899ff96daa0563a` |
| `docs/robust_source_training/rst6_run_closeout_template_v1.md` | `b420011efbfe76c3f636f719fa18bb3f01869d9dca9f21c5643d25a17296b987` |
| `docs/robust_source_training/rst6_final_analysis_template_v1.md` | `90a4dafe1b498b5f026f5bbcc8daf1aada67da3da44a04552d2eb6255b9504a3` |
| `outputs/robust_source_training/rst5_prelaunch_validation/readiness_report.json` | `9f76219cbb184e9c87a436c3abcb305af28474396291b431767c4042f36c53d8` |
| `tests/test_rst5_confirmation.py` | `4f05e00af819f1af86bffee12ab8b83c330948c2360d1af2629b71b2e4eccfe1` |
| `src/robust_source_training/runner.py` | `ff5538a655e8ed9019d91b3250566619487ba34d95aebcca13472c903b62803b` |

---

## 4. Research Integrity Guarantees

1. **Zero Confirmatory Group Evaluations**: Neither Fold A nor Fold B evaluation events were evaluated by any model during RST5.
2. **Zero Forbidden Target Event Access**: The 16 forbidden target events (`[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`) were strictly excluded and verified untouched.
3. **Historical Document Preservation**: Historical E11 documents remain byte-identically intact:
   - `docs/e11_final_scientific_conclusions.md` SHA-256: `26832a93be8efd85ada522c39e9334a678c6b3860f3379efe479567d879b3f36`
   - `docs/e11_final_test_closeout.md` SHA-256: `2e0d460ce088a8a6883ed5cdfadb61da01d9baaa3bf2b28249efcd6bcfd87ff8`
4. **Execution Hold Enforced**: All 12 confirmatory configuration files remain in `launch_authorized: false` and `preparation_only: true`. No GPU training runs were initiated.

---

## 5. Route Status and Handoff

- **RST0**: COMPLETED
- **RST1**: COMPLETED
- **RST2**: COMPLETED
- **RST3**: COMPLETED
- **RST4**: COMPLETED
- **RST5**: COMPLETED
- **RST6**: NOT_STARTED (Ready for launch authorization)
