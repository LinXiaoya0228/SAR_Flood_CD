#!/usr/bin/env bash
set -u

REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
cd "$REPO_ROOT" || exit 1

echo "Launching ablation M2_ONLY queues across 4 GPUs in detached tmux sessions..."

tmux new-session -d -s cg_ablation_gpu0 'bash scripts/robust_source_training/ablation_m2_only/run_gpu0.sh'
echo "Launched cg_ablation_gpu0"

tmux new-session -d -s cg_ablation_gpu1 'bash scripts/robust_source_training/ablation_m2_only/run_gpu1.sh'
echo "Launched cg_ablation_gpu1"

tmux new-session -d -s cg_ablation_gpu2 'bash scripts/robust_source_training/ablation_m2_only/run_gpu2.sh'
echo "Launched cg_ablation_gpu2"

tmux new-session -d -s cg_ablation_gpu3 'bash scripts/robust_source_training/ablation_m2_only/run_gpu3.sh'
echo "Launched cg_ablation_gpu3"

echo "All 4 tmux queues launched successfully."

