#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=0
REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
cd "$REPO_ROOT" || exit 1

PYTHON="/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python"
RUNNER="tools/robust_source_training/run_rst2_infrastructure.py"
CONF_DIR="configs/robust_source_training/rst8_siamunet_diff"
LOG_DIR="logs/robust_source_training/rst8"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_rst8_supp_gpu0.log"

# Job 1: Fold A, R0, Seed 1
RUN_ID="rst8_r0_siamunet_diff_folda_s1_v1"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] Launching $RUN_ID (Fold A, R0, Seed 1)" | tee -a "$RUN_LOG"
PYTHONPATH=. $PYTHON -u $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] FAILED $RUN_ID with exit code $EXIT_CODE." | tee -a "$RUN_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] COMPLETED $RUN_ID successfully." | tee -a "$RUN_LOG"

# Job 2: Fold A, R0, Seed 2
RUN_ID="rst8_r0_siamunet_diff_folda_s2_v1"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] Launching $RUN_ID (Fold A, R0, Seed 2)" | tee -a "$RUN_LOG"
PYTHONPATH=. $PYTHON -u $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] FAILED $RUN_ID with exit code $EXIT_CODE." | tee -a "$RUN_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 0] COMPLETED $RUN_ID successfully." | tee -a "$RUN_LOG"
