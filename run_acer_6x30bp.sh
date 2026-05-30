#!/bin/bash
# run_acer_6x30bp.sh — train ACER on dataset1_6x30bp (50 tests)
# Usage: bash run_acer_6x30bp.sh [start] [end]
#   bash run_acer_6x30bp.sh        # runs tests 0-49
#   bash run_acer_6x30bp.sh 20     # resumes from test 20
#   bash run_acer_6x30bp.sh 20 35  # runs tests 20-35

set -uo pipefail

DATASET_DIR="datasets/fasta_files/dataset1_6x30bp"
EPISODES=5000
EVAL_INTERVAL=100
PATIENCE=0
ACER_ENTROPY=1.0
ACER_ENTROPY_END=0.01
ACER_INF_ROLLOUTS=10
RESULTS_CSV="results/acer_6x30bp_benchmark.csv"
FIGURES_DIR="figures/benchmark_6x30bp"
LOG_FILE="/tmp/acer_6x30bp.log"

START=${1:-0}
END=${2:-49}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results figures weights

PYTHON="$(which python)"

source "$SCRIPT_DIR/_ensure_mafft.sh" || true

echo "========================================================"
echo " ACER · dataset1_6x30bp · tests ${START}–${END}"
echo " episodes=${EPISODES}  entropy=${ACER_ENTROPY}  patience=${PATIENCE}"
echo " log → ${LOG_FILE}"
echo "========================================================"

FAILED=()

for i in $(seq "$START" "$END"); do
    FASTA="${DATASET_DIR}/test${i}.fasta"
    if [ ! -f "$FASTA" ]; then
        echo "[SKIP] ${FASTA} not found" | tee -a "$LOG_FILE"
        continue
    fi
    echo "" | tee -a "$LOG_FILE"
    echo "--- test${i} ---" | tee -a "$LOG_FILE"
    if "$PYTHON" main.py "$FASTA" \
        --algorithm acer \
        --scoring sp \
        --episodes "$EPISODES" \
        --save "acer_6x30bp_test${i}" \
        --acer-entropy "$ACER_ENTROPY" \
        --acer-entropy-end "$ACER_ENTROPY_END" \
        --acer-inference-rollouts "$ACER_INF_ROLLOUTS" \
        --results-csv "$RESULTS_CSV" \
        --figures-dir "$FIGURES_DIR" \
        --eval-interval "$EVAL_INTERVAL" \
        --patience "$PATIENCE" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[OK] test${i} done" | tee -a "$LOG_FILE"
    else
        echo "[FAIL] test${i} exited with error -- continuing" | tee -a "$LOG_FILE"
        FAILED+=("$i")
    fi
done

echo ""
echo "========================================================"
echo " Done.  Results → ${RESULTS_CSV}"
echo "         Figures → ${FIGURES_DIR}/"
echo "         Log     → ${LOG_FILE}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed tests: ${FAILED[*]}"
    echo " Resume with: bash run_acer_6x30bp.sh <first_failed_test>"
fi
echo "========================================================"
