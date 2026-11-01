# MLDA Cluster GPU Usage & Multi-GPU Operational Protocol

- **Document Version:** 1.0.0  
- **Host System:** `gpu22`  
- **Effective Date:** 2026-10-06  
- **Primary Audience:** ChangeGuard FYP Engineering & Modeling Workflows  

---

## 1. Physical Hardware State (Audited on 2026-10-06)

A read-only hardware inspection via `nvidia-smi` confirmed the physical GPU configuration on `gpu22`:
- **Physical GPU Model:** NVIDIA GeForce RTX 3090 (Ampere architecture, compute capability 8.6)
- **Total Physical GPU Count:** 4 independent GPU devices (Physical Bus IDs: `19:00.0`, `1A:00.0`, `67:00.0`, `68:00.0`)
- **Total VRAM per Card:** 24,576 MiB (24 GB GDDR6X) per physical GPU
- **Initial Verification Status:**
  - GPU 0: 0% compute utilization, 15 MiB memory allocated (idle)
  - GPU 1: 0% compute utilization, 15 MiB memory allocated (idle)
  - GPU 2: 0% compute utilization, 15 MiB memory allocated (idle)
  - GPU 3: 0% compute utilization, 38 MiB memory allocated (idle, display server processes)
  - No active compute or training processes detected from any user.

---

## 2. Core Operational Principles for Shared Server Usage

### 2.1 Shared Resource Etiquette
1. **Shared Multi-User Environment:** Physical GPUs are shared department resources. Under no circumstances should an individual user assume exclusive or permanent ownership of all cards.
2. **Independent Memory Spaces:** Four physical 24-GB GPUs do **NOT** combine automatically into a single unified 96-GB memory pool. Each card operates with an isolated 24-GB memory space. Models must be architected to fit within individual card limits.
3. **Pre-Launch Inspection:** Always inspect live resource utilization using `nvidia-smi` immediately prior to launching any job. Never launch blind background tasks.
4. **Absolute Prohibition on Process Interference:** Never kill, terminate, pause, or renice processes belonging to other users. If a GPU is occupied, select an idle GPU.
5. **Conservative Resource Allocation:** When initiating new training jobs, start by utilizing one or two idle GPUs rather than automatically monopolizing all four devices.

---

## 3. Execution Environment & Device Isolation

### 3.1 Python Environment Rules
- **PyTorch Training & Evaluation:** Always use the dedicated PyTorch environment:
  `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`
  Contains verified CUDA 12/13 PyTorch binaries with rasterio, richdem, albumentations, and OpenCV.
- **Git, Git Bundling & File Operations:** Use the system or base environment for standard shell tools, tar, git, and diff operations.

### 3.2 Single-GPU Process Isolation via `CUDA_VISIBLE_DEVICES`
To prevent cross-device memory fragmentation and accidental collisions, independent experiments must be bound explicitly to distinct physical cards:
- **Environment Variable Binding:**
  Set `CUDA_VISIBLE_DEVICES=N` (where $N \in \{0, 1, 2, 3\}$).
- **Logical Index Mapping:**
  Inside the Python process, the selected physical card $N$ will normally appear as `cuda:0` / `torch.device("cuda:0")`.
- **Invocation Example:**
  ```bash
  # Launch Model A on Physical GPU 0
  CUDA_VISIBLE_DEVICES=0 /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py --config configs/source/snunet_vvvh_draft.json

  # Launch Model B concurrently on Physical GPU 1
  CUDA_VISIBLE_DEVICES=1 /usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python tools/train_source_model.py --config configs/source/snunet_vvvh_slope_draft.json
  ```

---

## 4. Job Persistence, Checkpointing, and Logging

### 4.1 Long-Running Job Management via Tmux
- Long-running multi-epoch training jobs must be executed inside persistent `tmux` sessions to ensure survival across SSH drops, network timeouts, or IDE restarts.
- **Tmux is Not a Substitute for Checkpointing:** Disconnections do not cause data loss if models are regularly checkpointed to disk.

### 4.2 Mandatory Checkpoint Resume Support
- All production training scripts must implement robust checkpointing:
  - Save periodic state checkpoints (`last_checkpoint.pt`) after every completed epoch.
  - Save best validation checkpoints (`best_inner_val_checkpoint.pt`) based on the 5 inner-validation events.
  - Support automatic resume via `--resume_checkpoint=True`, restoring model weights, optimizer states, learning rate scheduler steps, and completed epoch counters.

### 4.3 Auditability and Manifest Logging
- Every launched job must write a run manifest (`run_manifest.json`) recording:
  - Exact command line invocation string;
  - Starting Git commit and dirty flag;
  - SHA-256 hash of the configuration file;
  - Assigned GPU device and driver metadata;
  - Standard output and standard error redirected to a dedicated log file under `outputs/`.

---

## 5. Security and Credentials Safeguards

- **Never** hardcode passwords, API tokens (e.g. Weights & Biases keys), internal usernames beyond the repository owner, or private credentials into source code, configuration files, Git commit history, or LLM chat prompts.
- All sensitive tokens must be managed via local environment variables outside version control.
