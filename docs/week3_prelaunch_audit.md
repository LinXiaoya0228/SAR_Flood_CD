# Week 3 Source Replication Pre-Launch Audit Report

**Date:** 2026-10-10
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git Branch & HEAD:** `main` at commit `06b8d11b8d9aabbd5aaad1fae2537fd3d5ccc83d` (`06b8d11`)
**Audit Purpose:** Pre-flight provenance, configuration, lock manifest, and test suite freeze for Week 3 multi-seed replication.

---

## 1. Files Created and Modified

### Created Files (New Week 3 Artifacts):
1. `configs/source/snunet_vvvh_seed1_v2_fp32.json` — Model A seed 1 production configuration.
2. `configs/source/snunet_vvvh_slope_seed1_v2_fp32.json` — Model B seed 1 production configuration.
3. `configs/source/snunet_vvvh_seed2_v2_fp32.json` — Model A seed 2 preregistered configuration.
4. `configs/source/snunet_vvvh_slope_seed2_v2_fp32.json` — Model B seed 2 preregistered configuration.
5. `configs/source/source_training_week3_replication_lock.json` — One-way immutable replication lock manifest.
6. `docs/week3_source_replication_protocol.md` — Week 3 source replication protocol specification.
7. `tests/test_week3_replication_safeguards.py` — Dedicated unit tests for replication invariants and seed consistency.
8. `docs/week3_prelaunch_audit.md` — This pre-launch audit document.

### Modified Files:
1. `tools/train_source_model.py`:
   - Updated `validate_provenance_and_config` to support multi-model lock manifests by checking `lock["models"]` via relative config path and configuration ID (with backward-compatible fallback to legacy `model_a`/`model_b`).
   - Added protocol specification alias (`lock.get("protocol") or lock.get("governing_protocol")`).
   - Added strict seed consistency assertion: if `--seed` is passed via CLI, it must strictly match the config's declared seed; if omitted, it defaults safely to the config's declared seed.
   - Updated `main()` argument parser to default `--seed` to `None`, eliminating accidental silent divergence between CLI seed and immutable configuration.

---

## 2. Cryptographic Hashes (SHA-256)

### 2.1 Governance & Protocol Documents
| File | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Week 2 Closeout** | `docs/week2_seed0_closeout.md` | `8b474593616c2ec1a8c1b2254956a321a72a29ffc4428f89488bae128c15ee86` | Committed (`06b8d11`) |
| **Parent Protocol v2** | `docs/source_training_protocol_v2.md` | `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144` | Frozen |
| **Week 3 Replication Protocol** | `docs/week3_source_replication_protocol.md` | `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927` | Frozen |

### 2.2 Immutable Lock Manifests
| Lock Manifest | Path | SHA-256 Hash | Scope |
| :--- | :--- | :--- | :--- |
| **Week 2 Seed-0 Lock** | `configs/source/source_training_v2_lock.json` | `7139dbdd2031fd5654b4414299bf80e20295ebdba13bb507b64a59f10a50fbd3` | Frozen (seed 0) |
| **Week 3 Replication Lock** | `configs/source/source_training_week3_replication_lock.json` | `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677` | Frozen (seeds 0, 1, 2) |

### 2.3 Frozen Data Splits & Preprocessing Artifacts
| Artifact | Path | SHA-256 Hash | Scope |
| :--- | :--- | :--- | :--- |
| **Frozen Split Manifest** | `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | Frozen (22 train, 5 val) |
| **Train-Only Slope Stats** | `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | Frozen (mean=1.4926, std=2.3723) |

### 2.4 Versioned Source Model Configurations
| Model & Seed | Configuration Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Model A Seed 0** | `configs/source/snunet_vvvh_v2_fp32.json` | `528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3` | Completed |
| **Model B Seed 0** | `configs/source/snunet_vvvh_slope_v2_fp32.json` | `e99e980a9afc0f72b526ccff5484e4aadaabece12c5a542bd20f94be74ef6b56` | Completed |
| **Model A Seed 1** | `configs/source/snunet_vvvh_seed1_v2_fp32.json` | `fdcad256e5710588b193dadd902ab2442630c1c8f494c43bf424c4087ae8dccf` | **Required** |
| **Model B Seed 1** | `configs/source/snunet_vvvh_slope_seed1_v2_fp32.json` | `4845bcca17962aed26a4231b2a692d1d10834e8c548c5e9802759ec2e24b6f78` | **Required** |
| **Model A Seed 2** | `configs/source/snunet_vvvh_seed2_v2_fp32.json` | `9cd06aae01319079bb597125324cef212a2f0ffa02cab71853b57dea0c35c852` | Preregistered |
| **Model B Seed 2** | `configs/source/snunet_vvvh_slope_seed2_v2_fp32.json` | `663ff15f847582ade21be9a1aead13395bea37287aedf5c6e1bec0a8dffd9833` | Preregistered |

---

## 3. Exact Configuration Differences & Invariance Audit

### 3.1 Across Seeds of the Same Model (A0 vs A1 vs A2, and B0 vs B1 vs B2)
A programmatic verification confirmed that across seeds of the same model:
- **Scientific Invariance:** Architecture, input channels, clamping, normalization, label mapping, loss weights, optimizer (Adam), learning rate (0.001), weight decay (0.0), precision (`fp32`), microbatch size (8), gradient accumulation (4), effective batch size (32), split manifest, early stopping patience (20), max epochs (150), and tie-breaking metrics are **100% identical**.
- **Declared Differences:**
  - `optimization_protocol.seeds`: `[0]` vs `[1]` vs `[2]`.
  - `configuration_id`: `model_a_radar_only_v2_fp32` vs `model_a_radar_only_seed1_v2_fp32` vs `model_a_radar_only_seed2_v2_fp32` (and Model B equivalents).
  - `proposed_canonical_run_id`: `source_snunet_vvvh_seed0_v3_fp32` vs `source_snunet_vvvh_seed1_v3_fp32` vs `source_snunet_vvvh_seed2_v3_fp32` (and Model B equivalents).
  - `lock_manifest_path`: `configs/source/source_training_v2_lock.json` vs `configs/source/source_training_week3_replication_lock.json`.
  - `governing_protocol`: `docs/source_training_protocol_v2.md` vs `docs/week3_source_replication_protocol.md`.

### 3.2 Between Paired Model A and Model B (A1 vs B1, and A2 vs B2)
- **Scientific Differences:**
  - `architecture.in_channels_per_branch`: 2 (Model A) vs 3 (Model B).
  - `input_specification.channels_per_branch`: `["vv", "vh"]` vs `["vv", "vh", "slope"]`.
  - `input_specification.slope_policy`: Present only in Model B (`artifacts/week1/source_train_slope_statistics.json`, SHA-256 `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`).
- **All Other Fields:** 100% identical.

---

## 4. Test Suite Execution & Results

### 4.1 Focused Week 3 Replication Safeguards (`tests/test_week3_replication_safeguards.py`)
- `test_seed_fields_match_intended_integers`: **PASS** (seeds are exactly [1] and [2]).
- `test_seed0_configs_remain_unchanged`: **PASS** (seed-0 hashes strictly preserved).
- `test_paired_ab_configs_differ_only_in_approved_channel_and_slope_fields`: **PASS** (channels 2 vs 3).
- `test_seed_replications_differ_only_in_seed_and_metadata`: **PASS** (zero scientific drift across seeds).
- `test_protected_events_remain_excluded_across_all_configs_and_locks`: **PASS** (train and val disjoint from quarantined and sealed events).
- `test_all_config_and_lock_hashes_resolve`: **PASS** (all 4 replication configs and parent configs resolve in `validate_provenance_and_config`).
- `test_draft_or_mismatched_configs_cannot_launch`: **PASS** (seed mismatch, unapproved seed, mismatched run_id, and tampered config all fail fast).
- **Result:** **7/7 tests passed in 0.016s.**

### 4.2 Full Repository Test Suite Discovery
- Command: `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m unittest discover -s tests`
- Total Tests: **108 tests** (101 previous tests + 7 new Week 3 safeguards).
- Execution Time: 58.5s.
- **Result:** **108/108 tests passed (100% OK).**

---

## 5. Event Isolation & Compliance Confirmations

1. **Accessed Event IDs During Pre-Launch Preparation:**
   - Preflight and safeguard tests accessed **zero dataset imagery or pickles**.
   - Lightweight assertions read only JSON metadata from `configs/splits/kuro_source_inner_split_v1.json` to verify event ID lists.
2. **Zero Quarantined TTA-Validation Access:**
   - Quarantined events `[279, 437, 520, 559, 1111003, 1111008]` were accessed **0 times** across all audit actions.
3. **Zero Sealed-Test Access:**
   - Sealed events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` were accessed **0 times** across all audit actions.
4. **Zero Training, Inference, or TTA Execution:**
   - No training loop, evaluation loop, inference script, or TTA process was launched.

---

## 6. Git Status and Recommended Commit Workflow

### 6.1 `git status`
```
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   tools/train_source_model.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	configs/source/snunet_vvvh_seed1_v2_fp32.json
	configs/source/snunet_vvvh_seed2_v2_fp32.json
	configs/source/snunet_vvvh_slope_seed1_v2_fp32.json
	configs/source/snunet_vvvh_slope_seed2_v2_fp32.json
	configs/source/source_training_week3_replication_lock.json
	docs/week3_prelaunch_audit.md
	docs/week3_source_replication_protocol.md
	tests/test_week3_replication_safeguards.py

no changes added to commit (use "git add" and/or "git commit -a")
```

### 6.2 Recommended Staging Command & Commit Message
Per instructions, changes have **not** been staged or committed automatically. When ready, execute:

```bash
git add \
  configs/source/snunet_vvvh_seed1_v2_fp32.json \
  configs/source/snunet_vvvh_seed2_v2_fp32.json \
  configs/source/snunet_vvvh_slope_seed1_v2_fp32.json \
  configs/source/snunet_vvvh_slope_seed2_v2_fp32.json \
  configs/source/source_training_week3_replication_lock.json \
  docs/week3_source_replication_protocol.md \
  docs/week3_prelaunch_audit.md \
  tests/test_week3_replication_safeguards.py \
  tools/train_source_model.py

git commit -m "feat: freeze week3 source replication protocol and seed configs"
```
