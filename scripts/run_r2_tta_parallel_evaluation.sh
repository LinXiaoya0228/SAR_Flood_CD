#!/usr/bin/env bash
# ChangeGuard Supplementary Evaluation: Parallel Execution Orchestrator
# Runs exactly 6 formal streams across seeds 0, 1, 2 on dedicated GPUs cuda:0, cuda:1, cuda:2.
# Enforces strictly at most one memory-heavy process per GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY_BIN="${PYTHON_BIN:-/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python}"
if ! command -v "${PY_BIN}" &>/dev/null; then
    PY_BIN="python"
fi
CONFIG="configs/supplementary_r2_tta/r2_tta_evaluation_matrix_v1.json"
LEDGER="configs/supplementary_r2_tta/r2_tta_evaluation_ledger_v1.json"
OUT_DIR="outputs/supplementary_r2_tta"
LOG_DIR="logs/supplementary_r2_tta"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "================================================================================"
echo "ChangeGuard Supplementary Evaluation: R2 + Tent / CoTTA Launch"
echo "Start Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Repository Root: ${REPO_ROOT}"
echo "Output Directory: ${OUT_DIR}"
echo "Log Directory: ${LOG_DIR}"
echo "================================================================================"

echo "Launching Seed 0 on cuda:0 (2 streams: r2_snunet_tent_seed0, r2_snunet_cotta_seed0)..."
PYTHONUNBUFFERED=1 ${PY_BIN} tools/supplementary_r2_tta/run_r2_tta_evaluation.py \
    --mode execute_formal \
    --config "${CONFIG}" \
    --ledger "${LEDGER}" \
    --output_dir "${OUT_DIR}" \
    --seed 0 \
    --device cuda:0 > "${LOG_DIR}/seed_0.log" 2>&1 &
PID_SEED0=$!

echo "Launching Seed 1 on cuda:1 (2 streams: r2_snunet_tent_seed1, r2_snunet_cotta_seed1)..."
PYTHONUNBUFFERED=1 ${PY_BIN} tools/supplementary_r2_tta/run_r2_tta_evaluation.py \
    --mode execute_formal \
    --config "${CONFIG}" \
    --ledger "${LEDGER}" \
    --output_dir "${OUT_DIR}" \
    --seed 1 \
    --device cuda:1 > "${LOG_DIR}/seed_1.log" 2>&1 &
PID_SEED1=$!

echo "Launching Seed 2 on cuda:2 (2 streams: r2_snunet_tent_seed2, r2_snunet_cotta_seed2)..."
PYTHONUNBUFFERED=1 ${PY_BIN} tools/supplementary_r2_tta/run_r2_tta_evaluation.py \
    --mode execute_formal \
    --config "${CONFIG}" \
    --ledger "${LEDGER}" \
    --output_dir "${OUT_DIR}" \
    --seed 2 \
    --device cuda:2 > "${LOG_DIR}/seed_2.log" 2>&1 &
PID_SEED2=$!

echo "Active PIDs: Seed 0=${PID_SEED0}, Seed 1=${PID_SEED1}, Seed 2=${PID_SEED2}"

# Wait for all three workers
STATUS_0=0
STATUS_1=0
STATUS_2=0

wait ${PID_SEED0} || STATUS_0=$?
wait ${PID_SEED1} || STATUS_1=$?
wait ${PID_SEED2} || STATUS_2=$?

echo "================================================================================"
echo "Worker Exit Statuses: Seed 0=${STATUS_0}, Seed 1=${STATUS_1}, Seed 2=${STATUS_2}"
echo "End Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "================================================================================"

if [ ${STATUS_0} -ne 0 ] || [ ${STATUS_1} -ne 0 ] || [ ${STATUS_2} -ne 0 ]; then
    echo "ERROR: One or more workers failed! Check logs in ${LOG_DIR}."
    exit 1
fi

echo "All 6 streams successfully completed. Running formal statistical analysis..."
PYTHONUNBUFFERED=1 ${PY_BIN} tools/supplementary_r2_tta/analyze_r2_tta_evaluation.py \
    --results_dir "${OUT_DIR}" \
    --output_dir "${OUT_DIR}/analysis" > "${LOG_DIR}/analysis.log" 2>&1

echo "Writing completion marker..."
cat << EOF > "${OUT_DIR}/EXECUTION_COMPLETE"
EXECUTION_STATUS: SUCCESS
TOTAL_STREAMS: 6
TOTAL_EVENT_ROWS: 60
COMPLETION_TIMESTAMP_UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
SEED_0_EXIT: ${STATUS_0}
SEED_1_EXIT: ${STATUS_1}
SEED_2_EXIT: ${STATUS_2}
ANALYSIS_OUTPUT: ${OUT_DIR}/analysis
EOF

echo "✓ ChangeGuard R2 + TTA Supplementary Evaluation and Analysis Pipeline Finished Successfully."

