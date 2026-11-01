# ChangeGuard Checkpoint Registry

This registry documents the model architectures, training recipes, parameter counts, and cryptographic SHA-256 hashes for all trained checkpoints in ChangeGuard.

---

## 1. Full-Source Production Models (All-27 Kuro Siwo Events)

Trained on all 27 Kuro Siwo events (16 epochs, 10,736 optimizer steps). Used for deployment, historical Kuro Siwo target evaluation, and external zero-shot evaluation on GEOID-Flood.

| Model ID | Architecture | Recipe | Seed | Parameters | Checkpoint SHA-256 Hash |
|---|---|---|---|---|---|
| `R0_all27_s0` | SNUNet-ECAM | R0 (Baseline) | 0 | 12,034,819 | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| `R0_all27_s1` | SNUNet-ECAM | R0 (Baseline) | 1 | 12,034,819 | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` |
| `R0_all27_s2` | SNUNet-ECAM | R0 (Baseline) | 2 | 12,034,819 | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` |
| `R2_all27_s0` | SNUNet-ECAM | R2 (ChangeGuard) | 0 | 12,034,819 | `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7` |
| `R2_all27_s1` | SNUNet-ECAM | R2 (ChangeGuard) | 1 | 12,034,819 | `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2` |
| `R2_all27_s2` | SNUNet-ECAM | R2 (ChangeGuard) | 2 | 12,034,819 | `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893` |

---

## 2. Confirmatory 18-Event Cross-Validation Checkpoints (RST6)

Trained on 18 events, evaluated on 9 held-out events per fold (Fold A and Fold B across Seeds 0, 1, 2).

| Run ID | Architecture | Recipe | Fold | Seed | Checkpoint SHA-256 Hash |
|---|---|---|---|---|---|
| `rst6_r0_snunet_folda_s0_v1` | SNUNet-ECAM | R0 | Fold A | 0 | `3a597d4a43578668ceb457d4e9354ba54108c6402ac795fa1633db6dd10894be` |
| `rst6_r0_snunet_folda_s1_v1` | SNUNet-ECAM | R0 | Fold A | 1 | `77d243292410a823cfb84dfd0f283d5a42095dc765a0dbbb6f0b0fcb88a82d02` |
| `rst6_r0_snunet_folda_s2_v1` | SNUNet-ECAM | R0 | Fold A | 2 | `25bc77b94998495036fa780d6f224959dbbebcba50d75b3c3b036ca6d5cbe98c` |
| `rst6_r2_snunet_folda_s0_v2` | SNUNet-ECAM | R2 | Fold A | 0 | `231fb84ec2d5eec3a1d957ba15ec031a08ef48d8a6ff607833a697d812999e03` |
| `rst6_r2_snunet_folda_s1_v2` | SNUNet-ECAM | R2 | Fold A | 1 | `62b9a712f8a48efc6ee70a0d923053f3e2b4f98148b3014c6fb7e2467d0f9f30` |
| `rst6_r2_snunet_folda_s2_v2` | SNUNet-ECAM | R2 | Fold A | 2 | `c66d1ba5d023b3a62886c31bf047cfa2a88a0026a058da6f71d530616b60c382` |
| `rst6_r0_snunet_foldb_s0_v1` | SNUNet-ECAM | R0 | Fold B | 0 | `95a94ee732e796068e8334ddf4c6e913a48e89ea2dbd283fc3334ea4394038a8` |
| `rst6_r0_snunet_foldb_s1_v1` | SNUNet-ECAM | R0 | Fold B | 1 | `9bfa9dae4a500b5220c57173111f18fc47e17409c91f6e246b9a896ce19d8461` |
| `rst6_r0_snunet_foldb_s2_v1` | SNUNet-ECAM | R0 | Fold B | 2 | `e6e969d7a22d4f595df5bc4a413d964f43c3f915647572d42d386cbbd25f7716` |
| `rst6_r2_snunet_foldb_s0_v2` | SNUNet-ECAM | R2 | Fold B | 0 | `873efad4ef19f945037d2f95fcb6214ec0455db448cf10e8220f1807d957580b` |
| `rst6_r2_snunet_foldb_s1_v2` | SNUNet-ECAM | R2 | Fold B | 1 | `5dd7ea2b724fca240fbfad7340aa0e698bbfa970a2fbe61fa239a5ca2a188be4` |
| `rst6_r2_snunet_foldb_s2_v2` | SNUNet-ECAM | R2 | Fold B | 2 | `c4767ee93a02bbab0b4c6e54c8fe22003d159040cc247cf37ccaf8da2386927d` |

---

## 3. Weighting-Only Ablation Checkpoints (M2_ONLY)

| Run ID | Architecture | Recipe | Fold | Seed | Checkpoint SHA-256 Hash |
|---|---|---|---|---|---|
| `ablation_m2_only_snunet_folda_s0_v1` | SNUNet-ECAM | M2_ONLY | Fold A | 0 | `f2fc8c617b4fb24e3aa12df232bca605632616f7311c6d860d5b1ae310fa0770` |
| `ablation_m2_only_snunet_folda_s1_v1` | SNUNet-ECAM | M2_ONLY | Fold A | 1 | `9b3ff898fc67ecf48cb9219f5a770387e0fa0c19b67484df7ffea2c44814a09a` |
| `ablation_m2_only_snunet_folda_s2_v1` | SNUNet-ECAM | M2_ONLY | Fold A | 2 | `96a2f4da410f135b91a78882ca6c934f0dbeea333fa99feea883f360980c98f8` |
| `ablation_m2_only_snunet_foldb_s0_v1` | SNUNet-ECAM | M2_ONLY | Fold B | 0 | `dbd5872a0fc0bf24cf2a09564fbe3d21c327fc0ff7ec94589d8924b22db9c0a6` |
| `ablation_m2_only_snunet_foldb_s1_v1` | SNUNet-ECAM | M2_ONLY | Fold B | 1 | `b78a9c8b7aa1bb67fc9950dbcf2c0697e68dbb5514f7775ce5ea2c6a0c5c4046` |
| `ablation_m2_only_snunet_foldb_s2_v1` | SNUNet-ECAM | M2_ONLY | Fold B | 2 | `a5f25bf923055e8c148fc92e21e14e138a37910ff66046e7f8645e3ea3f6089e` |

---

## 4. Architecture Replication Checkpoints (SiamUnet_diff)

| Run ID | Architecture | Recipe | Fold | Seed | Checkpoint SHA-256 Hash | Status |
|---|---|---|---|---|---|---|
| `rst8_r0_siamunet_diff_folda_s0_v1` | SiamUnet_diff | R0 | Fold A | 0 | `db900d8b449fe4dbdd66d9c661ef4a8ea17b88ec25f235ad6ee9a2ef74e50849` | Completed |
| `rst8_r2_siamunet_diff_folda_s0_v2` | SiamUnet_diff | R2 | Fold A | 0 | `01aafe565a0890bfce87cfb274534dcdeaa82a3962dc2bb62a2faeb9c2c77d4c` | Completed |
| `rst8_r0_siamunet_diff_foldb_s0_v2` | SiamUnet_diff | R0 | Fold B | 0 | `732ae49dc976ff98d5c48b48f95c4e7d4bc8f0c5a2cbe5fc5be01625902bcce5` | Completed |
| `rst8_r2_siamunet_diff_foldb_s0_v2` | SiamUnet_diff | R2 | Fold B | 0 | `4292150b06ba677fc19056c70eb35a519d0a6428c9462c0199e46927a7c9dcfb` | Completed |
| `rst8_r0_siamunet_diff_folda_s1_v1` | SiamUnet_diff | R0 | Fold A | 1 | *Pending completion* | **Ongoing** |
| `rst8_r2_siamunet_diff_folda_s1_v1` | SiamUnet_diff | R2 | Fold A | 1 | *Pending completion* | **Ongoing** |
| `rst8_r0_siamunet_diff_foldb_s1_v1` | SiamUnet_diff | R0 | Fold B | 1 | *Pending completion* | **Ongoing** |
| `rst8_r2_siamunet_diff_foldb_s1_v1` | SiamUnet_diff | R2 | Fold B | 1 | *Pending completion* | **Ongoing** |
| `rst8_r0_siamunet_diff_folda_s2_v1` | SiamUnet_diff | R0 | Fold A | 2 | *Pending completion* | **Ongoing** |
| `rst8_r2_siamunet_diff_folda_s2_v1` | SiamUnet_diff | R2 | Fold A | 2 | *Pending completion* | **Ongoing** |
| `rst8_r0_siamunet_diff_foldb_s2_v1` | SiamUnet_diff | R0 | Fold B | 2 | *Pending completion* | **Ongoing** |
| `rst8_r2_siamunet_diff_foldb_s2_v1` | SiamUnet_diff | R2 | Fold B | 2 | *Pending completion* | **Ongoing** |

---

## 5. Checkpoint Availability & Release Decisions

### Availability Scope
- **Not Distributed in Git:** Because individual PyTorch checkpoint binaries range from ~50 MB to ~140 MB each (totalling $> 3$ GB across the experimental matrix), model weight files (`.pt`) are not tracked or hosted directly in this Git repository.
- **No Direct Download Mirror Currently Hosted:** There is currently no active public external download URL for pre-trained weights.
- **Inference Reproduction Boundary:** The repository is **not** immediately ready for out-of-the-box raw image inference reproduction until model weights are either retrained locally from scratch (using `tools/robust_source_training/run_rst2_infrastructure.py` or `tools/train_source_model.py` and the Kuro Siwo dataset) or made available through subsequent institutional release.
- **Immediate Auditability & Verification:** All compact canonical outputs, metrics, summary tables, and publication figures are tracked under `results/` and can be regenerated or audited immediately without model weights.

### Governance and Release Decisions
- **Checkpoint Distribution:** Formal release decisions regarding public weight hosting (e.g. Zenodo, Hugging Face, or institutional repository mirrors) remain under review by the author and institution.
- **Project Licensing:** Codebase licensing terms and intellectual property permissions remain reserved for determination by the author and institution. Permissions beyond research inspection are not implied.

### Cryptographic Verification
When checkpoint weights are present locally, verify their integrity using SHA-256:
```bash
sha256sum <path_to_checkpoint.pt>
```
Compare the resulting 64-character hexadecimal digest directly with the values documented in the tables above.
