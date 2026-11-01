# Reproduction Guide: Environment Setup and Third-Party Dependencies

This document specifies the software environments, hardware requirements, and dependency installation procedures for ChangeGuard.

---

## 1. Hardware and OS Requirements

- **Operating System:** Linux (Ubuntu 20.04 / 22.04 LTS or compatible x86_64 distribution).
- **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM (e.g., RTX 3090, RTX 4090, A100, V100). Primary training benchmarks consumed $\le 5$ GB peak VRAM.
- **CUDA Toolchain:** CUDA 12.1+ driver support.
- **CPU & RAM:** Multi-core CPU ($\ge 8$ cores recommended for DataLoader workers) and $\ge 32$ GB system RAM.
- **Disk Storage:** $\ge 150$ GB free SSD storage for datasets, temporary cache, checkpoints, and evaluation predictions.

---

## 2. Python Environments

ChangeGuard utilizes two distinct, well-isolated Python environments:
1. `sar_cd`: The primary environment used for model training, optimization, losses, and core validation.
2. `vfm_env`: An auxiliary environment containing geospatial raster libraries (`rasterio`, `gdal`, `shapely`) used for dataset inspection, geographic reprojection, and external GEOID-Flood evaluation.

### Setting up `sar_cd` (Core Training & Evaluation)
```bash
# Create environment with Python 3.10
conda create -n sar_cd python=3.10 -y
conda activate sar_cd

# Install PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install core scientific and plotting libraries
pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.9.2 tqdm
```

### Setting up `vfm_env` (External Geospatial Evaluation)
```bash
# Create environment with Python 3.10
conda create -n vfm_env python=3.10 -y
conda activate vfm_env

# Install PyTorch and geospatial packages
pip install torch torchvision
pip install rasterio==1.4.4 numpy scipy matplotlib tqdm requests
```

---

## 3. Third-Party Dependencies: KuroSiwo

The benchmark SNUNet-ECAM architecture is loaded from the peer-reviewed Kuro Siwo repository.

### Acquisition Procedure:
```bash
# From repository root
git clone https://github.com/Orion-AI-Lab/KuroSiwo.git third_party/KuroSiwo
cd third_party/KuroSiwo
git checkout 4347ed1
cd ../..
```

If KuroSiwo is cloned elsewhere on your system, you may set the environment variable:
```bash
export KURO_ROOT=/path/to/KuroSiwo
```

---

## 4. Environment Verification

To verify that the environment is functional and imports succeed without GPU access, run the synthetic smoke test:

```bash
# From repository root
conda activate sar_cd
python tools/smoke_reproducibility.py
```

Expected output:
```text
[PASS] Environment packages verified.
[PASS] All frozen configurations validated.
[PASS] Synthetic tensor forward pass passed (SNUNet-ECAM & SiamUnet_diff).
[PASS] Metric calculations and table generation passed.
```

