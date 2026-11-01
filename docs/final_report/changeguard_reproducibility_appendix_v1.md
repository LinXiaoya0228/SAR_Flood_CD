# ChangeGuard Reproducibility and Audit Appendix

**Document ID:** `docs/final_report/changeguard_reproducibility_appendix_v1.md`  
**Date:** 2026-10-28  
**Repository Root:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Status:** `FROZEN REPRODUCIBILITY CONTRACT — RST11`

This document details the exact hardware, software, containerization, checkpoint digests, and execution commands required to independently audit and deterministically reproduce all empirical results in the ChangeGuard project.

---

## 1. Hardware and Execution Environment

- **Operating System:** Linux 5.15.0-generic x86_64
- **Host GPU Accelerator:** 4 × NVIDIA GeForce RTX 3090 (24,576 MiB VRAM each)
- **CUDA Driver / Runtime:** CUDA 12.1 / PyTorch 2.1.2+cu121
- **Host CPU:** AMD EPYC / Intel Xeon 64-thread workstation
- **System Memory:** 128 GB System RAM
- **Storage Subsystem:** Local NVMe SSD storage with direct posix I/O

### Python Environment Specification
- **Interpreter:** Python 3.10.13 (`/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python`)
- **Key Dependencies:**
  - `torch==2.1.2`
  - `torchvision==0.16.2`
  - `numpy==1.26.4`
  - `scipy==1.12.0`
  - `matplotlib==3.8.3`
  - `albumentations==1.4.0`

---

## 2. Directory Structure and Data Manifests

```text
ChangeGuard/
├── configs/robust_source_training/      # Frozen experimental configurations and ledgers
│   ├── rst10_evidence_manifest_v1.json  # Master 15-source evidence manifest
│   ├── rst10_reconstruction_config_v1.json # Deterministic reconstruction pipeline config
│   ├── rst9_ledger_v1.json             # Deployable model tracking and step budget
│   ├── rst5_confirmation_decision_rule_v1.json # Pre-registered P1/P2/P3 rules
│   └── rst1_source_event_inventory_v1.csv # 27-event source inventory
├── outputs/robust_source_training/     # Immutable archived experimental outputs
│   ├── rst4/analysis_v1/               # Group C discovery staircase outputs
│   ├── rst7/                           # Primary confirmatory outputs (108 rows)
│   ├── rst8/                           # SiamUnet_diff transfer outputs (36 rows)
│   ├── rst9/                           # Deployable model checkpoints (seeds 0, 1, 2)
│   ├── rst9_inference/                 # Historical target evaluation outputs (96 rows)
│   ├── rst9_analysis/                  # Retrospective historical analysis
│   ├── rst10/                          # Repaired canonical tables, figures, metadata
│   └── rst10_superseded_v1/            # Preserved preliminary attempt with incident note
├── tools/robust_source_training/       # Deterministic reconstruction and table/figure scripts
│   ├── run_rst10_pipeline.py           # Master end-to-end reconstruction entry point
│   ├── build_rst10_tables.py           # Canonical table generator
│   └── build_rst10_figures.py          # Publication figure generator (PDF + PNG)
├── tests/                              # Regression and integrity test suites
│   ├── test_rst10_pipeline.py          # 14 forensic consistency assertions
│   └── test_rst9_all27.py              # 10 architecture and source training assertions
└── docs/final_report/                  # Dissertation and manuscript documents
```

---

## 3. Cryptographic Checkpoint Ledger

All final deployable models are stored with verified SHA-256 digests in `outputs/robust_source_training/rst9/`:

| Run ID | Seed | Architecture | Epochs | Steps | Peak VRAM | Physical Path | Final Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `rst9_r2_snunet_all27_s0_v1` | 0 | SNUNet_ECAM | 16 | 10,736 | 4,913 MB | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1/final_checkpoint.pt` | `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7` |
| `rst9_r2_snunet_all27_s1_v1` | 1 | SNUNet_ECAM | 16 | 10,736 | 4,913 MB | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s1_v1/final_checkpoint.pt` | `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2` |
| `rst9_r2_snunet_all27_s2_v1` | 2 | SNUNet_ECAM | 16 | 10,736 | 4,913 MB | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s2_v1/final_checkpoint.pt` | `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893` |

Matched baseline checkpoints (R0, uniform patch sampling, unweighted loss):
- Seed 0: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
- Seed 1: `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a`
- Seed 2: `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd`

---

## 4. End-to-End Reconstruction Commands

To deterministically recompute all tables, figures, metadata, and validation checks from repository root:

```bash
cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard

# 1. Execute the master RST10 reconstruction pipeline
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python tools/robust_source_training/run_rst10_pipeline.py

# 2. Run the 14-test RST10 regression suite
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python tests/test_rst10_pipeline.py

# 3. Run the 10-test RST9 All-27 training and inference suite
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python tests/test_rst9_all27.py

# 4. Run the dissertation report verification tool
/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python tools/validate_rst11_report.py
```

### Determinism Controls
- Python hash seed fixed: `PYTHONHASHSEED=0`
- PyTorch determinism: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.backends.cudnn.deterministic = True`
- Bootstrap random seed fixed: `numpy.random.seed(42)`
