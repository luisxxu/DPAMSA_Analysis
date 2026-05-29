#!/bin/bash
# _ensure_mafft.sh — sourced by run_acer_*.sh scripts
#
# Guarantees that `mafft` is callable before training starts.
# Strategy:
#   1. Already in PATH → done.
#   2. Not found → download the official static Linux binary to /tmp.
#      No conda, no home-dir writes, no quota impact.
#
# After this file is sourced, either:
#   • `mafft` works from PATH as normal, OR
#   • $MAFFT_BIN is set to the full path of the downloaded binary
#     (main.py reads this env-var via os.environ.get("MAFFT_BIN","mafft"))

_ensure_mafft() {
    if command -v mafft &>/dev/null; then
        echo "[INFO] mafft: $(mafft --version 2>&1 | head -1)"
        return 0
    fi

    local MAFFT_VER="7.526"
    local MAFFT_URL="https://mafft.cbrc.jp/alignment/software/mafft-${MAFFT_VER}-linux.tgz"
    local MAFFT_DIR="/tmp/mafft_static"

    # Re-use a previous download if it already exists
    local CACHED
    CACHED=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" 2>/dev/null | head -1)
    if [ -n "$CACHED" ]; then
        export MAFFT_BIN="$CACHED"
        echo "[INFO] mafft (cached): $("$MAFFT_BIN" --version 2>&1 | head -1)"
        return 0
    fi

    echo "[INFO] mafft not found — downloading static binary v${MAFFT_VER} to /tmp ..."
    mkdir -p "$MAFFT_DIR"

    # --strip-components=1 removes the top-level mafft-linux64/ directory
    if ! curl -fsSL "$MAFFT_URL" | tar -xz -C "$MAFFT_DIR" --strip-components=1; then
        echo "[WARN] mafft download failed — MAFFT scores will be skipped"
        return 1
    fi

    local BIN
    BIN=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" | head -1)
    if [ -z "$BIN" ]; then
        echo "[WARN] mafft binary not found after extraction — MAFFT scores will be skipped"
        return 1
    fi

    chmod +x "$BIN"
    export MAFFT_BIN="$BIN"
    echo "[INFO] mafft installed: $("$MAFFT_BIN" --version 2>&1 | head -1)"
}

_ensure_mafft
