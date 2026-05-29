#!/bin/bash
# setup_env.sh — create (or update) the dpamsa conda environment
# Usage: bash setup_env.sh
#
# Run this once on a new machine before any of the run_acer_*.sh scripts.
# If the environment already exists it is updated in-place.

set -euo pipefail

ENV_NAME="dpamsa"
ENV_FILE="$(dirname "$0")/environment.yml"

# ── Detect conda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ── Create or update environment ─────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "[INFO] Updating existing '${ENV_NAME}' environment …"
    conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    echo "[INFO] Creating '${ENV_NAME}' environment …"
    conda env create --name "$ENV_NAME" --file "$ENV_FILE"
fi

# ── Verify key imports ────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying imports …"
conda run -n "$ENV_NAME" python - <<'PYCHECK'
import sys, torch
from Bio import SeqIO
import numpy, pandas, matplotlib, tqdm

print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}  (available={torch.cuda.is_available()})")
print(f"  BioPython installed ✓")
print(f"  numpy / pandas / matplotlib / tqdm installed ✓")
PYCHECK

# ── Verify MAFFT ─────────────────────────────────────────────────────────────
if conda run -n "$ENV_NAME" mafft --version &>/dev/null; then
    MAFFT_VER=$(conda run -n "$ENV_NAME" mafft --version 2>&1 | head -1)
    echo "  MAFFT   : ${MAFFT_VER} ✓"
else
    echo "  MAFFT   : [WARN] not found — MAFFT comparison will be skipped"
fi

echo ""
echo "[OK] Setup complete.  Activate with:  conda activate ${ENV_NAME}"
