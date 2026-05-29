#!/bin/bash
# run_acer_6x60bp.sh — train ACER on dataset1_6x60bp (50 tests)
# Usage: bash run_acer_6x60bp.sh [start] [end]

set -euo pipefail

DATASET_DIR="datasets/fasta_files/dataset1_6x60bp"
EPISODES=8000
EVAL_INTERVAL=100
PATIENCE=8
ACER_ENTROPY=1.0
RESULTS_CSV="results/acer_6x60bp_benchmark.csv"
FIGURES_DIR="figures/benchmark_6x60bp"
LOG_FILE="/tmp/acer_6x60bp.log"

START=${1:-0}
END=${2:-49}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results figures weights

PYTHON="$(which python)"

# ── MAFFT: install static binary to /tmp if not already in PATH ───────────────
source "$SCRIPT_DIR/_ensure_mafft.sh" || true

echo "========================================================"
echo " ACER · dataset1_6x60bp · tests ${START}–${END}"
echo " episodes=${EPISODES}  entropy=${ACER_ENTROPY}  patience=${PATIENCE}"
echo " log → ${LOG_FILE}"
echo "========================================================"

for i in $(seq "$START" "$END"); do
    FASTA="${DATASET_DIR}/test${i}.fasta"
    if [ ! -f "$FASTA" ]; then
        echo "[SKIP] ${FASTA} not found"
        continue
    fi
    echo ""
    echo "--- test${i} ---"
    "$PYTHON" main.py "$FASTA" \
        --algorithm acer \
        --scoring sp \
        --episodes "$EPISODES" \
        --save "acer_6x60bp_test${i}" \
        --acer-entropy "$ACER_ENTROPY" \
        --results-csv "$RESULTS_CSV" \
        --figures-dir "$FIGURES_DIR" \
        --eval-interval "$EVAL_INTERVAL" \
        --patience "$PATIENCE"
done 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================"
echo " Done.  Results → ${RESULTS_CSV}"
echo "         Figures → ${FIGURES_DIR}/"
echo "         Log     → ${LOG_FILE}"
echo "========================================================"
