#!/bin/bash
# setup_env.sh — install missing Python packages into the current environment
#
# On DataHub / DSMLP the base conda env already ships with PyTorch, numpy,
# pandas, matplotlib, tqdm, and scipy.  This script just fills in the gaps
# (biopython) and verifies everything is importable.
#
# Usage:  bash setup_env.sh
# No conda activate needed — run this in whatever shell/kernel you use.

set -euo pipefail

echo "=== DPAMSA dependency check ==="
echo ""

# ── Helper: try to import a module, install if missing ───────────────────────
check_or_install() {
    local module=$1
    local pkg=${2:-$1}   # pip package name (defaults to module name)
    if python -c "import ${module}" 2>/dev/null; then
        VER=$(python -c "import ${module}; print(getattr(${module}, '__version__', 'ok'))" 2>/dev/null || echo "ok")
        echo "  [OK]     ${module}==${VER}"
    else
        echo "  [MISSING] ${module} — installing ${pkg} ..."
        pip install --quiet "${pkg}"
        echo "  [OK]     ${module} installed"
    fi
}

# ── Core dependencies ─────────────────────────────────────────────────────────
check_or_install torch
check_or_install numpy
check_or_install pandas
check_or_install matplotlib
check_or_install tqdm
check_or_install Bio biopython

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
python - <<'PYCHECK'
import torch
cuda_ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
print(f"  [GPU]    CUDA available={cuda_ok}  device={dev}")
PYCHECK

# ── MAFFT check ───────────────────────────────────────────────────────────────
echo ""
if command -v mafft &>/dev/null; then
    MAFFT_VER=$(mafft --version 2>&1 | head -1)
    echo "  [OK]     mafft — ${MAFFT_VER}"
else
    echo "  [WARN]   mafft not found — MAFFT comparison will be skipped at runtime"
    echo "           To install:  conda install -c bioconda mafft"
    echo "           (only ~3 MB, safe to install even with a tight quota)"
fi

echo ""
echo "=== Setup complete. Run your dataset scripts: ==="
echo "   bash run_acer_3x30bp.sh"
echo "   bash run_acer_6x30bp.sh"
echo "   bash run_acer_6x60bp.sh"
