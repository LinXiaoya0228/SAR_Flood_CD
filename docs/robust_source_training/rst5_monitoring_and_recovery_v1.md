# RST6 Confirmatory Training: Monitoring, Recovery, and Operations Guide

## 1. Executive Summary

This guide provides operational procedures and copy-paste shell commands for managing, monitoring, inspecting, and recovering the 12 confirmatory training runs of **RST6** across the four NVIDIA GeForce RTX 3090 GPUs on host `gpu22`.

All operations adhere to the strict research-integrity protocol:
- **Zero Output Deletion**: Never delete or overwrite an output directory or log file.
- **Durable Failure Recording**: If a run fails, record its terminal status; do not purge the failure artifact.
- **Detached Execution**: All jobs must run in detached tmux sessions to ensure survival across SSH/VS Code disconnections.

---

## 2. Tmux Session & GPU Allocation Plan

To maximize throughput and prevent GPU out-of-memory or PCIe contention, at most **one training run** is assigned per GPU simultaneously.

The 12 confirmatory runs are executed across 3 sequential waves:

| Wave | GPU | Session Name | Run ID | Fold | Recipe | Seed |
|---|---|---|---|---|---|---|
| **Wave 1** | `cuda:0` | `cg_rst6_w1_g0` | `rst6_r0_snunet_folda_s0_v1` | Fold A | R0 | 0 |
| **Wave 1** | `cuda:1` | `cg_rst6_w1_g1` | `rst6_r2_snunet_folda_s0_v1` | Fold A | R2 | 0 |
| **Wave 1** | `cuda:2` | `cg_rst6_w1_g2` | `rst6_r0_snunet_foldb_s0_v1` | Fold B | R0 | 0 |
| **Wave 1** | `cuda:3` | `cg_rst6_w1_g3` | `rst6_r2_snunet_foldb_s0_v1` | Fold B | R2 | 0 |
| **Wave 2** | `cuda:0` | `cg_rst6_w2_g0` | `rst6_r0_snunet_folda_s1_v1` | Fold A | R0 | 1 |
| **Wave 2** | `cuda:1` | `cg_rst6_w2_g1` | `rst6_r2_snunet_folda_s1_v1` | Fold A | R2 | 1 |
| **Wave 2** | `cuda:2` | `cg_rst6_w2_g2` | `rst6_r0_snunet_foldb_s1_v1` | Fold B | R0 | 1 |
| **Wave 2** | `cuda:3` | `cg_rst6_w2_g3` | `rst6_r2_snunet_foldb_s1_v1` | Fold B | R2 | 1 |
| **Wave 3** | `cuda:0` | `cg_rst6_w3_g0` | `rst6_r0_snunet_folda_s2_v1` | Fold A | R0 | 2 |
| **Wave 3** | `cuda:1` | `cg_rst6_w3_g1` | `rst6_r2_snunet_folda_s2_v1` | Fold A | R2 | 2 |
| **Wave 3** | `cuda:2` | `cg_rst6_w3_g2` | `rst6_r0_snunet_foldb_s2_v1` | Fold B | R0 | 2 |
| **Wave 3** | `cuda:3` | `cg_rst6_w3_g3` | `rst6_r2_snunet_foldb_s2_v1` | Fold B | R2 | 2 |

---

## 3. Operational Monitoring Commands

### 3.1 List Active Tmux Sessions
```bash
tmux ls | grep cg_rst6 || echo "No active RST6 tmux sessions found."
```

### 3.2 Inspect Hardware & GPU Utilization
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
```

### 3.3 Tail Logs for Live Runs
```bash
# Tail a specific run log (e.g., Fold A, R2, Seed 0)
tail -f logs/robust_source_training/rst6/rst6_r2_snunet_folda_s0_v1.log

# View last 20 lines of all active RST6 logs
tail -n 20 logs/robust_source_training/rst6/*.log
```

### 3.4 Process Check by Run ID
```bash
ps aux | grep "run_rst2_infrastructure.py" | grep -v grep
```

---

## 4. Run State Auditing & Matrix Counting

### 4.1 Count Completed, Running, and Failed Runs
Run the following bash one-liner to query all 12 confirmatory runs:
```bash
python -c '
import json
from pathlib import Path

matrix = json.load(open("configs/robust_source_training/rst5_confirmation_matrix_v1.json"))
completed = 0
failed = 0
absent = 0

print(f"{'Run ID':<30} | {'Status':<25} | {'Final Checkpoint'}")
print("-" * 75)
for r in matrix["runs"]:
    run_id = r["run_id"]
    term_p = Path(f"outputs/robust_source_training/rst6/{run_id}/terminal_status.json")
    ckpt_p = Path(f"outputs/robust_source_training/rst6/{run_id}/final_checkpoint.pt")
    
    if term_p.exists():
        status = json.load(open(term_p)).get("status", "UNKNOWN")
        if status == "COMPLETED_SCIENTIFIC_RUN":
            completed += 1
        else:
            failed += 1
    else:
        status = "NOT_STARTED_OR_RUNNING"
        absent += 1
    
    ckpt_str = "EXISTS" if ckpt_p.exists() else "MISSING"
    print(f"{run_id:<30} | {status:<25} | {ckpt_str}")

print("-" * 75)
print(f"Summary: {completed}/12 Completed | {failed} Failed | {absent} In-Progress or Unlaunched")
'
```

---

## 5. Recovery & Fresh-Run Policy

### 5.1 Determining Run State
Before attempting recovery on any run, inspect `terminal_status.json`:
1. **`COMPLETED_SCIENTIFIC_RUN`**: The run finished successfully. **Do not re-run or overwrite**.
2. **`INTERRUPTED_RUN` / `FAILED` / Missing `terminal_status.json`**: An unexpected shutdown, system reboot, or process kill occurred.
   - Do NOT attempt `--resume` for confirmatory runs (`CONFIRMATORY_TRAINING`).
   - The runner enforces a fail-closed guard rejecting `--resume` for confirmatory training.
   - The interrupted directory is preserved permanently as failed evidence.

### 5.2 Recovering via Versioned Fresh Retry
Under the authoritative RST6 fresh-run recovery policy (`docs/robust_source_training/rst6_fresh_run_recovery_policy_v1.md`):
1. **Preserve Failed Attempt**: Leave `outputs/robust_source_training/rst6/<RUN_ID>/` untouched.
2. **Create Versioned Config**: Create `<RUN_ID>_v2.json` with identical scientific settings, seed, and fold, pointing to a fresh output root `outputs/robust_source_training/rst6/<RUN_ID>_v2/`.
3. **Launch Fresh Run**: Execute from Epoch 0 in a detached tmux session:
```bash
# 1. Activate environment
source ~/miniconda3/envs/sar_cd/bin/activate
cd /usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard

# 2. Launch fresh retry in detached tmux session
tmux new-session -d -s cg_rst6_<RUN_ID>_v2 \
  "python tools/robust_source_training/run_rst2_infrastructure.py \
     --config configs/robust_source_training/rst5_confirmation/<RUN_ID>_v2.json \
     --execute \
   >> logs/robust_source_training/rst6/<RUN_ID>_v2.log 2>&1"
```

The fresh retry guarantees bit-identical initialization and deterministic sample draw progression from Epoch 0 without partial state contamination.

---

## 6. Stability Under Client Disconnection

Because all runs are executed inside detached `tmux` sessions on host `gpu22`:
- Closing VS Code, closing terminal windows, or network drops **will not** kill or interrupt the background processes.
- Do not kill tmux sessions prematurely. Use `tmux attach -t <SESSION>` to inspect or `Ctrl-b d` to detach safely.
