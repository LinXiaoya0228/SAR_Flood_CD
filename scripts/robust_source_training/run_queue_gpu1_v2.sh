#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
cd "$REPO_ROOT" || exit 1

PYTHON="/usr1/home/s126mdg22_06/miniconda3/envs/sar_cd/bin/python"
RUNNER="tools/robust_source_training/run_rst2_infrastructure.py"
CONF_DIR="configs/robust_source_training/rst5_confirmation"
LOG_DIR="logs/robust_source_training/rst6"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue_cg_rst6_folda_r2_v2.log"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Queue cg_rst6_folda_r2_v2 on GPU 1 (Fold A, R2 Retry v2)" | tee -a "$QUEUE_LOG"

# Seed 0 (Retry v2)
RUN_ID="rst6_r2_snunet_folda_s0_v2"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] Launching $RUN_ID" | tee -a "$QUEUE_LOG"
$PYTHON $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] FAILED $RUN_ID with exit code $EXIT_CODE. Queue HALTED." | tee -a "$QUEUE_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] COMPLETED $RUN_ID successfully." | tee -a "$QUEUE_LOG"

# Seed 1 (Retry v2)
RUN_ID="rst6_r2_snunet_folda_s1_v2"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] Launching $RUN_ID" | tee -a "$QUEUE_LOG"
$PYTHON $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] FAILED $RUN_ID with exit code $EXIT_CODE. Queue HALTED." | tee -a "$QUEUE_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] COMPLETED $RUN_ID successfully." | tee -a "$QUEUE_LOG"

# Seed 2 (Retry v2)
RUN_ID="rst6_r2_snunet_folda_s2_v2"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] Launching $RUN_ID" | tee -a "$QUEUE_LOG"
$PYTHON $RUNNER --config "$CONF_DIR/$RUN_ID.json" --execute >> "$LOG_DIR/$RUN_ID.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] FAILED $RUN_ID with exit code $EXIT_CODE. Queue HALTED." | tee -a "$QUEUE_LOG"
  exit $EXIT_CODE
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [GPU 1] COMPLETED $RUN_ID successfully." | tee -a "$QUEUE_LOG"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Queue cg_rst6_folda_r2_v2 FINISHED (3/3 runs complete)." | tee -a "$QUEUE_LOG"
