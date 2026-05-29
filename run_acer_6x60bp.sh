#!/bin/bash
# run_acer_6x60bp.sh — train ACER on all 50 dataset1_6x60bp test files
# Usage: bash run_acer_6x60bp.sh [start_index] [end_index]
#
# Note: 6x60bp is the hardest dataset (longest sequences, 63-action space).
#       Entropy coef and episodes are raised accordingly.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR="datasets/fasta_files/dataset1_6x60bp"
EPISODES=8000
CHECKPOINT=1000
EVAL_INTERVAL=100
PATIENCE=8
ACER_ENTROPY=0.1
RESULTS_CSV="results/acer_6x60bp_benchmark.csv"
FIGURES_DIR="figures/benchmark_6x60bp"
LOG_FILE="results/acer_6x60bp.log"

START=${1:-0}
END=${2:-49}

# ── Helpers ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results figures weights

echo "========================================================"
echo " ACER  ·  dataset1_6x60bp  ·  tests ${START}–${END}"
echo " episodes=${EPISODES}  entropy=${ACER_ENTROPY}  patience=${PATIENCE}"
echo "========================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in $(seq "$START" "$END"); do
    FASTA="${DATASET_DIR}/test${i}.fasta"
    SAVE_NAME="acer_6x60bp_test${i}"

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
        --acer-entropy "$ACER_ENTROPY" \
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
