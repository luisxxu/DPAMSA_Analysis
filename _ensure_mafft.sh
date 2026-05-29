#!/bin/bash
# _ensure_mafft.sh — sourced by run_acer_*.sh scripts
#
# Downloads the official MAFFT static Linux binary to /tmp and exports
# MAFFT_BIN so main.py uses it directly (bypassing any broken system install).
# Always exits 0 — mafft is optional; training continues without it.

_ensure_mafft() {
    local MAFFT_VER="7.526"
    local MAFFT_URL="https://mafft.cbrc.jp/alignment/software/mafft-${MAFFT_VER}-linux.tgz"
    local MAFFT_DIR="/tmp/mafft_static"

    # Re-use a previous successful download
    local BIN
    BIN=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" 2>/dev/null | head -1)

    if [ -z "$BIN" ]; then
        echo "[INFO] Downloading mafft v${MAFFT_VER} to /tmp ..."
        mkdir -p "$MAFFT_DIR"
        # --strip-components=1 removes the top-level mafft-linux64/ folder
        if curl -fsSL --max-time 60 "$MAFFT_URL" \
                | tar -xz -C "$MAFFT_DIR" --strip-components=1 2>/dev/null; then
            BIN=$(find "$MAFFT_DIR" -name "mafft" -not -path "*/libexec/*" 2>/dev/null | head -1)
        fi
    fi

    if [ -z "$BIN" ]; then
        echo "[WARN] mafft download failed — MAFFT scores will be skipped"
        return 0
    fi

    chmod +x "$BIN"
    # Export the full path so main.py calls this binary directly,
    # bypassing any broken system mafft in PATH.
    export MAFFT_BIN="$BIN"
    # Unset stale MAFFT_BINARIES so the script auto-detects its own libexec.
    unset MAFFT_BINARIES
    echo "[INFO] mafft ready: $("$MAFFT_BIN" --version 2>&1 | head -1)"
}

_ensure_mafft
