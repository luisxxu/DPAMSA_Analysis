#!/bin/bash
# _ensure_mafft.sh — sourced by run_acer_*.sh scripts
#
# Tries to make mafft available. Always exits 0 — mafft is optional;
# if it cannot be installed, main.py will skip the MAFFT comparison.
#
# On success, either:
#   • mafft is already in PATH, or
#   • $MAFFT_BIN is exported pointing to the downloaded binary

_ensure_mafft() {
    # Already available
    if command -v mafft &>/dev/null; then
        echo "[INFO] mafft: $(mafft --version 2>&1 | head -1)"
        return 0
    fi

    local MAFFT_VER="7.526"
    local MAFFT_URL="https://mafft.cbrc.jp/alignment/software/mafft-${MAFFT_VER}-linux.tgz"
    local MAFFT_DIR="/tmp/mafft_static"

    # Re-use a previous download
    local CACHED
    CACHED=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" 2>/dev/null | head -1)
    if [ -n "$CACHED" ]; then
        export MAFFT_BIN="$CACHED"
        echo "[INFO] mafft (cached): $("$MAFFT_BIN" --version 2>&1 | head -1)"
        return 0
    fi

    echo "[INFO] mafft not found — downloading static binary v${MAFFT_VER} ..."
    mkdir -p "$MAFFT_DIR"

    # Download and extract; suppress set -e by wrapping in if/else
    if curl -fsSL --max-time 60 "$MAFFT_URL" \
            | tar -xz -C "$MAFFT_DIR" --strip-components=1 2>/dev/null; then
        local BIN
        BIN=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" 2>/dev/null | head -1)
        if [ -n "$BIN" ]; then
            chmod +x "$BIN"
            export MAFFT_BIN="$BIN"
            echo "[INFO] mafft ready: $("$MAFFT_BIN" --version 2>&1 | head -1)"
            return 0
        fi
    fi

    echo "[WARN] mafft could not be installed — MAFFT scores will be skipped"
    return 0   # non-fatal: script continues without MAFFT
}

_ensure_mafft
