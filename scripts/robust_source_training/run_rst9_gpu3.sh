#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=3
REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
cd "$REPO_ROOT" || exit 1

PYTHON="/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python"
RUNNER="tools/robust_source_training/run_rst2_infrastructure.py"
CONF_DIR="configs/robust_source_training/rst9_all27"
LOG_DIR="logs/robust_source_training/rst9"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_rst9_gpu3.log"

RUN_ID="rst9_r2_snunet_all27_s2_v1"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] Launching $RUN_ID (All-27, R2, Seed 2)" | tee -a "$RUN_LOG"
PYTHONPATH=. $PYTHON $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] FAILED $RUN_ID with exit code $EXIT_CODE." | tee -a "$RUN_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 3] COMPLETED $RUN_ID successfully." | tee -a "$RUN_LOG"
