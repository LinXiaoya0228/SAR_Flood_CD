#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=3
REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
cd "$REPO_ROOT" || exit 1

PYTHON="/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python"
RUNNER="tools/robust_source_training/run_rst2_infrastructure.py"
CONF_DIR="configs/robust_source_training/ablation_m2_only"
LOG_DIR="logs/robust_source_training/ablation_m2_only"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue_gpu3.log"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Queue on GPU 3 (ablation M2_ONLY)" | tee -a "$QUEUE_LOG"

# 1. Fold B Seed 0
RUN_ID="ablation_m2_only_snunet_foldb_s0_v1"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] Launching $RUN_ID" | tee -a "$QUEUE_LOG"
PYTHONPATH=. $PYTHON $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] FAILED $RUN_ID with exit code $EXIT_CODE. Queue HALTED." | tee -a "$QUEUE_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] COMPLETED $RUN_ID successfully." | tee -a "$QUEUE_LOG"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Queue on GPU 3 FINISHED (1/1 runs complete)." | tee -a "$QUEUE_LOG"

