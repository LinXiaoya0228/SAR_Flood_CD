# Week 0 Closeout and Controlled Source Protocol Freeze

**Document Version:** 1.0.0-FINAL  
**Status:** `WEEK0_FROZEN_COMPLETE — READY FOR WEEK 1 SMOKE TESTING`  
**Date:** 2026-10-06  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA RTX 3090 24GB)  
**Git Branch:** `main` | **HEAD:** `0d8dcec24c8c56609b4e318e317e66f5390c176b`  

---

## 1. Executive Summary

Week 0 of the controlled-source-model phase has successfully completed all research integrity audits, statistical census calculations, split sensitivity evaluations, and configuration freezes. 

The user's approved decisions have been formally implemented:
1. **Source-Inner Split:** Alternative A1 is approved and frozen in [`configs/splits/kuro_source_inner_split_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/splits/kuro_source_inner_split_v1.json).
2. **Model A:** 2-channel pure radar baseline `[VV, VH]` is frozen in [`configs/source/snunet_vvvh_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/snunet_vvvh_v1.json).
3. **Model B:** 3-channel topographic ablation `[VV, VH, slope]` under Policy S2 is frozen in [`configs/source/snunet_vvvh_slope_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/snunet_vvvh_slope_v1.json).
4. **Execution Protocol:** The comprehensive protocol is frozen in [`docs/source_training_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v1.md).
5. **Quality Assurance:** 46 / 46 unit tests pass cleanly in 15.4s.

No neural network models were trained, no forward or backward passes were executed, no TTA or validation inferences were run, and zero test rasters or labels were touched.

---

## 2. Approved Source-Inner Split (Alternative A1)

### 2.1 Event Allocations
- **Approved 5 Inner-Validation Events:**
  ```text
  [267, 275, 497, 518, 1111005]
  ```
  *(Total samples: 2,284 / 10.64%; valid pixels: 112,243,176)*
- **Approved 22 Inner-Training Events:**
  ```text
  [118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]
  ```
  *(Total samples: 19,178 / 89.36%; valid pixels: 950,602,654)*

### 2.2 Split-Selection Rationale
- **Empirical Superiority:** Alternative A1 replaces tiny Event `496` (21 samples) with medium Event `275` (742 samples, mid-flood tertile). This expands validation sample mass to 2,284 patches (10.64%) and reduces the maximum multi-dimensional discrepancy between train and val to just **3.30%** (vs 8.60% in the proposed split).
- **Class Balance:** Pooled permanent water discrepancy is just **0.22%** (val 6.85% vs train 7.06%); macro flood discrepancy is **1.26%** (val 15.62% vs train 14.36%).
- **Pure Label Statistics Basis:** Chosen strictly from source label statistics in `outputs/week0/source_event_statistics.csv` without any model performance or validation feedback.

### 2.3 Partition Verification
- `inner_train` event count: exactly 22.
- `inner_val` event count: exactly 5.
- Disjointness: `inner_train ∩ inner_val = ∅`.
- Union: exactly equals all 27 official source-training events.
- Exclusion: zero TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) appear.
- Exclusion: zero sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) appear.

---

## 3. Approved Model Definitions & Preprocessing

### 3.1 Model A (Controlled Pure Radar Baseline)
- **Architecture:** `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
- **Temporal Inputs:** $(x_A, x_B) = (\text{pre\_event\_2}, \text{post\_event})$.
- **Per-Time Channels:** `[VV, VH]`.
- **Output Classes:** 0 No Water, 1 Permanent Water, 2 Flood. (Class 3 Invalid excluded by valid mask).
- **Verified SAR Preprocessing Order:**
  1. Clamp raw linear amplitude to $[0.0, 0.15]$.
  2. Replace non-finite values (NaN, $+\infty$, $-\infty$) with $0.15$.
  3. Z-score normalize: $(x - \mu) / \sigma$ with $\mu=[0.0953, 0.0264], \sigma=[0.0427, 0.0215]$.

### 3.2 Model B (Controlled Topographic Ablation)
- **Architecture:** `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
- **Temporal Inputs:** $(x_A, x_B) = (\text{pre\_event\_2}, \text{post\_event})$.
- **Per-Time Channels:** `[VV, VH, slope]`.
- **Identical Static Channel:** Identical slope raster fed to both temporal branches.
- **Explicit Provenance Statement:** Model B is a **controlled scientific ablation study**, evaluating whether topographic slope aids flood discrimination. Model B does **not** claim that slope was used in the official checkpoint. The official checkpoint third channel remains **unknown**.

### 3.3 Approved Slope Policy (Policy S2 Requirements)
1. Read $x, y$ pixel spacing dynamically from DEM GeoTIFF transform matrix.
2. Verify projected CRS with metric units; fail loudly on geographic degrees.
3. Observe $10\text{ m}$ spacing as empirical data, do not blindly hard-code.
4. Identify DEM nodata from metadata and sentinel values ($< -1000\text{ m}$ via `kuro_dem_sanitize_v1`).
5. Compute slope only over sanitized valid elevation support to eliminate boundary cliffs.
6. Physically explicit slope angle: $\text{slope}_{\text{deg}} = \arctan(\sqrt{(dz/dx)^2 + (dz/dy)^2}) \times 180.0 / \pi$.
7. Guarantee spatial alignment with SAR ($224 \times 224$).
8. Replace invalid slope pixels with train-set mean before normalization (yielding $0.0$ after z-score standardization).
9. Calculate normalization statistics ($\mu_{\text{slope}}, \sigma_{\text{slope}}$) strictly over the 22 approved inner-training events. Zero TTA-val or test data.
10. Record SHA-256 hash of `outputs/week1/source_train_slope_statistics.json`.
11. Formal launch guard: Model B formal multi-epoch training is **blocked** until the slope statistics artifact exists and passes validation.

---

## 4. Approved Sample & Optimization Protocol

### 4.1 Sample Policy
- All 19,178 patches belonging to the 22 inner-training events are retained.
- Zero filtering of zero-flood patches; zero artificial oversampling of flood patches.
- Preserves the authentic geographical background prior (15.22% zero-flood patches in train).

### 4.2 Optimization & Loss Protocol
- **Optimizer:** Adam ($lr=1.0 \times 10^{-3}$, $wd=0.0$, PyTorch default betas `(0.9, 0.999)`, no scheduler).
- **Exact Loss Function:** Unweighted Cross-Entropy + Soft Dice:
  $$L_{\text{total}} = 0.5 \cdot \text{CE}(\text{logits}, y, \text{ignore\_index}=3) + 0.5 \cdot \text{Dice}_{\text{soft}}(\text{softmax}(\text{logits}), y, \text{classes}=[0,1,2], \epsilon=1.0)$$
- **Absent-Class Stability:** The smooth Dice formulation ($\epsilon=1.0$) guarantees strictly positive denominators, yields $0.0$ loss for correct class absence, and produces smooth bounded gradients for false alarms without division by zero.
- **Budget:** 150 max epochs; early-stopping patience 20 epochs.
- **Selection Metric:** Event-Macro Flood F1 across the 5 inner-validation events.

### 4.3 Deterministic Epoch Tie-Breaking
1. Higher event-macro Flood F1;
2. Higher worst-event Flood F1;
3. Higher event-macro Flood Recall;
4. Earlier epoch (smaller integer).

### 4.4 Final Retraining Rule & Seeds
- Random seeds: `[42, 43, 44]`.
- Locked retraining epoch: $E_{\text{final}} = \text{median}(E_{42}^*, E_{43}^*, E_{44}^*)$.
- Final source models retrained from scratch across all 27 events (21,462 patches) for exactly $E_{\text{final}}$ epochs with **zero validation tuning**.

### 4.5 Batch Size Policy & Hardware Safeguards
- Target effective batch size: 32.
- Precision: Automatic Mixed Precision (AMP FP16).
- Physical microbatch status: `TO_BE_CONFIRMED_BY_WEEK1_MEMORY_SMOKE_TEST`.
- Accumulation steps: $32 / \text{physical\_microbatch\_size}$ with exact divisibility.
- Candidate microbatches for Week 1 test: `[2, 4, 8, 16]`.
- Formal training remains **blocked** until microbatch is confirmed in a versioned Week 1 protocol amendment.

---

## 5. Artifact & Cryptographic Integrity Register

| Artifact / File Path | SHA-256 Digest | Status |
| :--- | :--- | :---: |
| `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | Frozen |
| `configs/source/snunet_vvvh_v1.json` | `ac89502b4859a19c5cff492a5437ea813aa55ee8dd58fe846bf4fc911b305282` | Frozen |
| `configs/source/snunet_vvvh_slope_v1.json` | `83907eb644aece2c6762319ef55fcf2161b9a918731d10e5d4cb0d8eb8761a5b` | Frozen |
| `docs/source_training_protocol_v1.md` | `5d6b47372d82bb68a04eb023778dfcb7cf5b602058b8ef95057b988f6a9e105b` | Frozen |
| `configs/splits/kuro_source_inner_split_proposed.json` | `16b1755a5e1eb87e03e658b2b90c4d501bf8afaaeeb73927d3885d64be1256da` | Preserved Historical |
| `configs/source/snunet_vvvh_draft.json` | `656d2ad2fad6f843821ad0c4144811923f8394a1af5dff5b83fce1f73dedf5c4` | Preserved Historical |
| `configs/source/snunet_vvvh_slope_draft.json` | `f9d48d0e9d0cf87361ed29ffc45fe8ae0d03e2d1c8a193b62820ce9f7bec6a83` | Preserved Historical |
| `outputs/week0/source_event_statistics.csv` | `bdf6deec1c48b6a05b01bf520f5e0af6e492894c9d7c347d85fc64de2396bf80` | Immutable Ground Truth |
| `outputs/week0/source_event_statistics.json` | `6eb9339b9a41e40f64f59eea3753efa074a78c95861a57dbbf2393c35dfb9b5d` | Immutable Ground Truth |
| `docs/MASTER_EXECUTION_PLAN.md` | `70fa605f4dadc09bfd98038bd164a08a6aaf36870c7578c34111dbdfef0b311f` | Immutable |
| `docs/two_week_pilot_complete_evidence_report.md` | `a6d6fa5c38e4ba64591bde82e6fe9d6a2ee20d6c1da87de5a6cf15301875adc3` | Immutable |
| `docs/week0_backup_record.md` | `ce8ef8beec5a21cf3ec8a5a5ea2e2b5e44a7d7ff9283a30dcb561da3a89f9f87` | Immutable |
| `docs/week0_decision_evidence.md` | `a1619b0d1e37bc902c1f9d5a86ce88775f0a3e87d825c9ea3fb758d4d8efbe95` | Verified |
| `docs/week0_review_for_user.md` | `c02f099c23d42c3d52d92d47da1f8ca1529a67feae05cb49f390050868f77341` | Corrected |
| `docs/source_data_pipeline_audit.md` | `8bd0971294c996cb50b9f56ffcd64e6cfd6471bc137cbb198c9408ab9fdd5100` | Corrected |
| `docs/source_training_protocol_draft.md` | `d1e31edd065249a890bb39b8a78adf9cb9af5f4830b97d8050b6e573a00e2533` | Corrected |

---

## 6. Test Suite Execution Summary

Executed using `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
**Results: 46 / 46 tests passed in 15.434s (`OK`).**

- `tests/test_completeness.py`: 5 tests passed
- `tests/test_metrics.py`: 6 tests passed
- `tests/test_preprocessing.py`: 7 tests passed
- `tests/test_protocol_rules.py`: 6 tests passed (NEW)
- `tests/test_risk_indicators.py`: 3 tests passed
- `tests/test_risk_probing_isolation.py`: 4 tests passed
- `tests/test_risk_schema.py`: 2 tests passed
- `tests/test_risk_thresholds.py`: 3 tests passed
- `tests/test_run_manifest.py`: 5 tests passed
- `tests/test_source_configs.py`: 11 tests passed (EXPANDED)
- `tests/test_source_inner_split.py`: 8 tests passed (EXPANDED)
- `tests/test_tta_isolation.py`: 6 tests passed

---

## 7. Research Integrity Certifications

1. **Test Manifest Access Clarification & Protocol Deviation:**
   - Exploratory inspection of `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` during preflight checks read only top-level manifest keys (event IDs).
   - Recorded as `DEVIATION-W0-01`.
   - Permanent prohibition against opening any test manifest is actively enforced.
2. **Event Data & Label Access Census:**
   - During Week 0, only labels and valid masks belonging to the **27 source events** were accessed:
     `[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]`.
3. **Quarantine Confirmations:**
   - Zero TTA-validation images or labels (`[279, 437, 520, 559, 1111003, 1111008]`) were loaded or accessed to design the split, select models, or tune hyperparameters.
   - Zero sealed-test images, labels, predictions, or metrics (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) were accessed in any way.
4. **Execution & Editing Certifications:**
   - Zero models were trained; zero forward or backward passes were executed.
   - Zero experimental CSV/JSON files were manually edited.
   - All files were produced deterministically by verified code.

---

## 8. Backup Verification

Two external backups remain verified in `/usr1/home/s126mdg22_06/FYP_SARCD/backups/`:
- `changeguard_refs_20261006_193120.bundle` (199 KB, SHA-256: `ccb68f97461a81108bc081b0b79bb6f789ff4ccd40e5e748a9991bc7742598cb`)
- `changeguard_worktree_20261006_193120.tar.gz` (192 KB, SHA-256: `e1d57d80a7da2364bc936002faa60aee30297083e1622fa9c84fddcb1e4e1f1a`)

---

## 9. Unresolved Items & Week 1 Entry Conditions

### Unresolved Items (Formally Block Formal Multi-Epoch Training)
1. **Physical Microbatch Size:** Must be experimentally determined via Week 1 memory smoke testing on single RTX 3090 GPU (evaluating `[2, 4, 8, 16]`).
2. **Train-Only Slope Normalization Statistics:** Must be generated strictly over the 22 inner-train events to produce `outputs/week1/source_train_slope_statistics.json` before Model B can launch.

### Week 1 Immediate Entry Protocol
1. Execute single-batch/single-epoch memory smoke test on GPU 0 to measure exact VRAM per microbatch.
2. Select optimal physical microbatch (e.g. 16 with accum 2, or 32 if VRAM permits).
3. Issue versioned Week 1 protocol amendment unlocking `"ready_for_formal_training": true`.
4. Generate and validate train-only slope statistics for Model B.
5. Launch controlled source training for Model A and Model B across seeds `[42, 43, 44]`.
