#!/bin/bash
# run_acer_3x30bp.sh — train ACER on dataset1_3x30bp (50 tests)
# Usage: bash run_acer_3x30bp.sh [start] [end]
#   bash run_acer_3x30bp.sh          # all tests (0–49)
#   bash run_acer_3x30bp.sh 10 19    # resume from test10

set -euo pipefail

DATASET_DIR="datasets/fasta_files/dataset1_3x30bp"
EPISODES=5000
EVAL_INTERVAL=100
PATIENCE=0
ACER_ENTROPY=0.5        # entropy annealing start (3 seqs: smaller action space)
ACER_ENTROPY_END=0.01   # entropy annealing end
ACER_INF_ROLLOUTS=10    # best-of-N stochastic rollouts at inference
RESULTS_CSV="results/acer_3x30bp_benchmark.csv"
FIGURES_DIR="figures/benchmark_3x30bp"
LOG_FILE="/tmp/acer_3x30bp.log"

START=${1:-0}
END=${2:-49}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results figures weights

PYTHON="$(which python)"

# ── MAFFT: install static binary to /tmp if not already in PATH ───────────────
source "$SCRIPT_DIR/_ensure_mafft.sh" || true

echo "========================================================"
echo " ACER · dataset1_3x30bp · tests ${START}–${END}"
echo " episodes=${EPISODES}  patience=${PATIENCE}"
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
        --save "acer_3x30bp_test${i}" \
        --acer-entropy "$ACER_ENTROPY" \
        --acer-entropy-end "$ACER_ENTROPY_END" \
        --acer-inference-rollouts "$ACER_INF_ROLLOUTS" \
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
