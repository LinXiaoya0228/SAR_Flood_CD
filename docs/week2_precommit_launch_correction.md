# Week 2 Pre-Commit Launch Correction and Provenance Record

**Date:** 2026-10-07
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Host:** `gpu22`
**Base Commit (HEAD at Launch):** `0f6ffa547ce7f2fa9b6c5d2de089a8372b81d06e` (`feat: validate controlled source training pipeline`)
**Status:** `NONCANONICAL_PRECOMMIT_ATTEMPT_PRESERVED`

---

## 1. Executive Summary and Reason for Correction

Ahead of the calendar schedule, controlled source-model training for Week 2 seed 0 was initiated on 2026-10-07:
- Model A: `source_snunet_vvvh_seed0_v1` on physical GPU 1
- Model B: `source_snunet_vvvh_slope_seed0_v1` on physical GPU 2

All scientific parameters—including frozen JSON configurations, 22/5 source-inner event splits, hardware isolation, learning rate, optimizer settings, and zero protected data access—were verified and respected.

However, an audit of the launch provenance revealed that the newly implemented production launcher (`tools/train_source_model.py`), trainer epoch-methods (`src/training/source_trainer.py`), and corresponding unit tests (`tests/test_train_source_model.py`) were uncommitted in the working tree when the processes were launched. Consequently, the generated run manifests recorded base commit `0f6ffa547ce7f2fa9b6c5d2de089a8372b81d06e` with `git_dirty: true`. A reproducible scientific benchmark requires that every formal training artifact point directly to a clean, immutable Git commit containing the complete code necessary to replicate the run.

In accordance with the FYP ChangeGuard Research Integrity Protocol:
1. Both running jobs were terminated and their outputs preserved intact.
2. The initial runs have been classified as `NONCANONICAL_PRECOMMIT_ATTEMPT` and excluded from all formal Week 2 scientific analysis.
3. The production training code was audited, hardened with an in-loop stop check, verified with unit tests, and prepared for an isolated Git checkpoint commit before formal training may begin.

---

## 2. Initial Launch Details

### 2.1 Launch Commands Executed
```bash
# Model A [VV, VH] Seed 0
tmux new-session -d -s cg_w2_a_s0 "cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard && CUDA_VISIBLE_DEVICES=1 /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py --config configs/source/snunet_vvvh_v1.json --output_dir outputs/week2_source_seed0/source_snunet_vvvh_seed0_v1 --run_id source_snunet_vvvh_seed0_v1 --device cuda:0 --seed 0"

# Model B [VV, VH, slope] Seed 0
tmux new-session -d -s cg_w2_b_s0 "cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard && CUDA_VISIBLE_DEVICES=2 /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py --config configs/source/snunet_vvvh_slope_v1.json --output_dir outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v1 --run_id source_snunet_vvvh_slope_seed0_v1 --device cuda:0 --seed 0"
```

### 2.2 Execution Identifiers
- **Model A Run ID:** `source_snunet_vvvh_seed0_v1`
  - Output directory: `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v1/`
  - Process PID: `1934181`
  - Physical GPU: GPU 1 (NVIDIA GeForce RTX 3090, Bus `0000:1A:00.0`)
  - PyTorch device: `cuda:0` (isolated via `CUDA_VISIBLE_DEVICES=1`)
  - Config SHA-256: `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc`
- **Model B Run ID:** `source_snunet_vvvh_slope_seed0_v1`
  - Output directory: `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v1/`
  - Process PID: `1934756`
  - Physical GPU: GPU 2 (NVIDIA GeForce RTX 3090, Bus `0000:67:00.0`)
  - PyTorch device: `cuda:0` (isolated via `CUDA_VISIBLE_DEVICES=2`)
  - Config SHA-256: `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec`
  - Slope Statistics SHA-256: `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`
- **Source Split Config:** `configs/splits/kuro_source_inner_split_v1.json`
  - Split SHA-256: `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3`

---

## 3. Actual Termination Sequence and Characterization

The shutdown sequence unfolded as follows:

1. `SIGINT` (signal 2) was sent to Model A (PID `1934181`). Although the launcher's signal handler registered the signal and printed an acknowledgment to the log, the process remained active because the Python execution loop inside `train_epoch()` did not break mid-epoch.
2. A tmux `Ctrl+C` was also transmitted to session `cg_w2_a_s0`. The signal was received by the process group, but the process again remained active in its batch loop.
3. `SIGHUP` (signal 1) was subsequently sent to terminate Model A (PID `1934181`) and Model B (PID `1934756`). Both main processes and their respective DataLoader worker subprocesses exited immediately upon receiving `SIGHUP`.
4. `SIGKILL` / `kill -9` was **never used**.
5. Because neither job had completed Epoch 1, **no model checkpoint existed** (`best_checkpoint.pt` and `last_checkpoint.pt` were never created).
6. Because the processes terminated via `SIGHUP` while inside `train_epoch()`, the launcher did not itself finish writing an `INTERRUPTED` status to disk before termination (manifests retained their initial on-disk status `INITIALIZED`).
7. After both processes exited, `run_manifest.json` and `train.log` in each noncanonical v1 directory were manually annotated with the interruption reason and noncanonical classification:
   ```text
   Stopped intentionally because the production training code had not been committed before formal launch.
   ```
8. This post-run manual annotation modified provenance metadata and text logs only.
9. The empty/history-header CSV files (`train_history.csv`, `validation_history.csv`, `event_validation_metrics.csv`) and all numerical training results were **not manually edited**.
10. Both initial v1 attempts are strictly excluded from all formal Week 2 analysis.

### Characterization Statement
The jobs were terminated without `SIGKILL`, but the shutdown was not an application-confirmed graceful checkpoint shutdown.

---

## 4. Progress Observed Prior to Interruption

Neither model completed Epoch 1; no model checkpoint was generated.

- **Model A (`source_snunet_vvvh_seed0_v1`):**
  - Last logged batch: `Batch 0600/2398` (Epoch 1)
  - Last logged step: `Step 00150`
  - Loss at Step 150: Total: `0.2316` | CE: `0.2310` | Dice: `0.2321` | LR: `0.001000`
- **Model B (`source_snunet_vvvh_slope_seed0_v1`):**
  - Last logged batch: `Batch 0600/2398` (Epoch 1)
  - Last logged step: `Step 00150`
  - Loss at Step 150: Total: `0.2207` | CE: `0.2104` | Dice: `0.2311` | LR: `0.001000`

---

## 5. Artifact Ledger: Pre- and Post-Annotation Hashes

All artifacts generated by these preliminary attempts have been preserved under their respective output directories. Both pre-annotation (immediately after process exit) and post-annotation hashes from the execution transcript are recorded below.

### 5.1 Model A Artifacts (`outputs/week2_source_seed0/source_snunet_vvvh_seed0_v1/`)
| File | Pre-Annotation Size | Pre-Annotation SHA-256 | Post-Annotation Size | Post-Annotation SHA-256 | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `event_validation_metrics.csv` | 112 B | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | 112 B | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | Unmodified |
| `train_history.csv` | 138 B | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | 138 B | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | Unmodified |
| `validation_history.csv` | 239 B | `ba10013c51f356ad09975b742d8789048782b2ad6c76cc9736ee5942045ca5a5` | 239 B | `ba10013c51f356ad09975b742d8789048782b2ad6c76cc9736ee5942045ca5a5` | Unmodified |
| `run_manifest.json` | 2,845 B | `a5704ade06c5fbb0abc361158890796c2ca6d57af63e80c03c6ce353e50e56af` | 3,103 B | `60c2ad81d912a0aa7f8b28f3edb107d9b666e93dacf5b5a23ea22b28527f23bd` | Annotated with interruption metadata |
| `train.log` | 1,922 B | `e429958ddf9aea67bb769c220b1fd742642aaa99e4220d10492eba2f36db1167` | 2,348 B | `d47059dd021a60c0ba14cdb4f91ec29f21e64f3b2c687f74871cb9be64676178` | Annotated with termination notice |

### 5.2 Model B Artifacts (`outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v1/`)
| File | Pre-Annotation Size | Pre-Annotation SHA-256 | Post-Annotation Size | Post-Annotation SHA-256 | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `event_validation_metrics.csv` | 112 B | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | 112 B | `5faae337784f9c18aae9ed6040532a6a812292904222513b9eea44dc2bbd11d3` | Unmodified |
| `train_history.csv` | 138 B | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | 138 B | `351c19cc2e13b7a9c6b12750e275c45c3f71543004bcdf944d862bff38f207f7` | Unmodified |
| `validation_history.csv` | 239 B | `ba10013c51f356ad09975b742d8789048782b2ad6c76cc9736ee5942045ca5a5` | 239 B | `ba10013c51f356ad09975b742d8789048782b2ad6c76cc9736ee5942045ca5a5` | Unmodified |
| `run_manifest.json` | 2,917 B | `a768cd38d9dd9a0ba2bb15ce2e4ab9c5a6b3009b3269650bd873e8b7333d3449` | 3,175 B | `a3261082455c3e7e829ae3b7cb217538edb59b706521f7c55885b670c2d14e44` | Annotated with interruption metadata |
| `train.log` | 2,186 B | `7e175b32d62bad4b93ba4ae39713b630732c32bb74cfd35c53b6f4534aca9714` | 2,328 B | `84de96ad4a5b84b500fcd39fc425f090a941b8a7218e00199ff48b53d4821f2d` | Annotated with termination notice |

---

## 6. Code Hardening Following Interruption

To prevent future signal-interruption stalls:
1. `SourceTrainer` in `src/training/source_trainer.py` was hardened with a `stop_requested` flag initialized to `False` in `__init__()`.
2. Inside `train_epoch()`, an explicit in-loop check (`if getattr(self, "stop_requested", False): break`) was added at the batch level to guarantee clean, prompt loop termination upon receiving any interruption signal.
3. The launcher `tools/train_source_model.py` was updated so its signal handler immediately flags `trainer.stop_requested = True`.
4. This hardening change was validated: all syntax checks passed and the full test suite passed (**85/85 tests OK**).

---

## 7. Research Integrity Affirmations

1. **Zero Access to TTA-Validation Events:** Neither job accessed events `[279, 437, 520, 559, 1111003, 1111008]`.
2. **Zero Access to Sealed Final-Test Events:** Neither job accessed events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`.
3. **Sealed Test Archive:** `KuroV2_grid_dict_test_0_100.gz` was never opened.
4. **Data Isolation:** Only approved source-inner training events (`[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]`) were loaded.
5. **No Numerical Result Editing:** No generated metric or numerical training result was manually edited. Only post-run provenance metadata and text logs were annotated.
6. **Scientific Exclusion:** Both initial v1 attempts are classified as `NONCANONICAL_PRECOMMIT_ATTEMPT` and are excluded from all formal Week 2 analysis.

---

## 8. Proposed Production-Code Commit

The following three implementation files (and this provenance document) are proposed for the Week 2 checkpoint commit:

1. `src/training/source_trainer.py`:
   - Adds `train_epoch()` with microbatch size 8, gradient accumulation 4, effective batch size 32, AMP autocast, and in-loop `stop_requested` check.
   - Enhances `evaluate()` to compute and return validation loss components (`loss_total`, `loss_ce`, `loss_dice`).
2. `tools/train_source_model.py`:
   - Production multi-epoch training launcher with frozen hash guards, early stopping (patience 20), Section 7.2 deterministic tie-breaking checkpoint selection, triple-history CSV logging, runtime access auditing, and signal handling.
3. `tests/test_train_source_model.py`:
   - Focused unit tests verifying access guards, checkpoint ledger updates, and CSV logging schemas.
4. `docs/week2_precommit_launch_correction.md`:
   - Factual record of pre-commit launch correction, termination sequence, pre/post artifact hashes, and code hardening.

### Verification Status
- Syntax checks passed (`python -m py_compile`).
- Test suite passed: **85/85 tests OK** (`CUDA_VISIBLE_DEVICES=1 python -m unittest discover tests/`).
