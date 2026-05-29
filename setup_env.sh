#!/bin/bash
# setup_env.sh — install missing packages into the current Python environment
#
# DataHub / DSMLP already ships PyTorch, numpy, pandas, matplotlib, tqdm.
# This script installs only what is missing (biopython, ~5 MB).
# It never touches conda and never writes to a package cache.
#
# Usage:  bash setup_env.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"   # override with PYTHON=/path/to/python if needed

echo "=== DPAMSA dependency check ==="
echo "    Using: $($PYTHON --version 2>&1)"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
check_or_install() {
    local module=$1
    local pkg=${2:-$1}
    if $PYTHON -c "import ${module}" 2>/dev/null; then
        local ver
        ver=$($PYTHON -c "import ${module}; print(getattr(${module},'__version__','ok'))" 2>/dev/null || echo "ok")
        echo "  [OK]  ${module} ${ver}"
    else
        echo "  [--]  ${module} missing — installing ${pkg} (no cache) ..."
        pip install --no-cache-dir --quiet "${pkg}"
        echo "  [OK]  ${module} installed"
    fi
}

# ── Packages ─────────────────────────────────────────────────────────────────
check_or_install torch
check_or_install numpy
check_or_install pandas
check_or_install matplotlib
check_or_install tqdm
check_or_install Bio biopython

# ── GPU ───────────────────────────────────────────────────────────────────────
echo ""
$PYTHON - <<'PYCHECK'
import torch
ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if ok else "none"
print(f"  [GPU] CUDA={ok}  device={dev}")
PYCHECK

# ── MAFFT ────────────────────────────────────────────────────────────────────
echo ""
if command -v mafft &>/dev/null; then
    echo "  [OK]  mafft $(mafft --version 2>&1 | head -1)"
else
    echo "  [--]  mafft not found — MAFFT scores will be skipped"
    echo "        Install with (only ~3 MB):  conda install -c bioconda mafft --no-deps"
fi

echo ""
echo "=== Done. Now run:  bash run_acer_3x30bp.sh ==="
