#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard"
PYTHON_BIN="/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python"
LOG_DIR="${REPO_ROOT}/outputs/external_evaluation"
mkdir -p "${LOG_DIR}"

echo "Launching ChangeGuard GEOID-Flood external evaluation across GPUs 0-3..."

# GPU 0: R0_s0 followed by R2_s1
tmux new-session -d -s eval_gpu0 "bash -c '
  echo \"Starting R0_s0 on GPU 0...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R0_s0 --device cuda:0 2>&1 | tee ${LOG_DIR}/eval_R0_s0.log && \
  echo \"Starting R2_s1 on GPU 0...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R2_s1 --device cuda:0 2>&1 | tee ${LOG_DIR}/eval_R2_s1.log && \
  echo \"GPU 0 evaluation queue finished successfully!\"
'"

# GPU 1: R0_s1 followed by R2_s2
tmux new-session -d -s eval_gpu1 "bash -c '
  echo \"Starting R0_s1 on GPU 1...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R0_s1 --device cuda:1 2>&1 | tee ${LOG_DIR}/eval_R0_s1.log && \
  echo \"Starting R2_s2 on GPU 1...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R2_s2 --device cuda:1 2>&1 | tee ${LOG_DIR}/eval_R2_s2.log && \
  echo \"GPU 1 evaluation queue finished successfully!\"
'"

# GPU 2: R0_s2
tmux new-session -d -s eval_gpu2 "bash -c '
  echo \"Starting R0_s2 on GPU 2...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R0_s2 --device cuda:2 2>&1 | tee ${LOG_DIR}/eval_R0_s2.log && \
  echo \"GPU 2 evaluation queue finished successfully!\"
'"

# GPU 3: R2_s0
tmux new-session -d -s eval_gpu3 "bash -c '
  echo \"Starting R2_s0 on GPU 3...\" && \
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/external_evaluation/run_geoid_evaluation.py --model_id R2_s0 --device cuda:3 2>&1 | tee ${LOG_DIR}/eval_R2_s0.log && \
  echo \"GPU 3 evaluation queue finished successfully!\"
'"

echo "All 4 tmux evaluation queues launched successfully."
tmux ls

