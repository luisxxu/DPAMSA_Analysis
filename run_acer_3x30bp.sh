#!/bin/bash
# run_acer_3x30bp.sh — train ACER on all 50 dataset1_3x30bp test files
# Usage: bash run_acer_3x30bp.sh [start_index] [end_index]
#   start_index : first test index (default 0)
#   end_index   : last  test index (default 49)
#
# Examples
#   bash run_acer_3x30bp.sh          # run all 50 tests
#   bash run_acer_3x30bp.sh 0 9      # run test0–test9 only
#   bash run_acer_3x30bp.sh 10 19    # resume from test10

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR="datasets/fasta_files/dataset1_3x30bp"
EPISODES=5000
CHECKPOINT=1000
EVAL_INTERVAL=100
PATIENCE=5
RESULTS_CSV="results/acer_3x30bp_benchmark.csv"
FIGURES_DIR="figures/benchmark_3x30bp"
LOG_FILE="results/acer_3x30bp.log"

START=${1:-0}
END=${2:-49}

# ── Helpers ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results figures weights

echo "========================================================"
echo " ACER  ·  dataset1_3x30bp  ·  tests ${START}–${END}"
echo " episodes=${EPISODES}  patience=${PATIENCE}  eval_interval=${EVAL_INTERVAL}"
echo "========================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in $(seq "$START" "$END"); do
    FASTA="${DATASET_DIR}/test${i}.fasta"
    SAVE_NAME="acer_3x30bp_test${i}"

    if [ ! -f "$FASTA" ]; then
        echo "[SKIP] ${FASTA} not found"
        continue
    fi

    echo ""
    echo "--- test${i} ---"
    $(which python) main.py "$FASTA" \
        --algorithm acer \
        --scoring sp \
        --episodes "$EPISODES" \
        --save "$SAVE_NAME" \
        --checkpoint "$CHECKPOINT" \
        --results-csv "$RESULTS_CSV" \
        --figures-dir "$FIGURES_DIR" \
        --eval-interval "$EVAL_INTERVAL" \
        --patience "$PATIENCE"

done 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================"
echo " Done. Results → ${RESULTS_CSV}"
echo "        Figures → ${FIGURES_DIR}/"
echo "        Log     → ${LOG_FILE}"
echo "========================================================"
