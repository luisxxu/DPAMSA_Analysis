#!/bin/bash
# _ensure_mafft.sh — sourced by run_acer_*.sh scripts
#
# Downloads the MAFFT static Linux binary to /tmp and exports MAFFT_BIN.
# Always exits 0 — mafft is optional; training continues without it.
#
# The system mafft on some DataHub pods has a broken libexec/VERSION file,
# so we always use our own downloaded copy rather than whatever is in PATH.

_ensure_mafft() {
    local MAFFT_DIR="/tmp/mafft_static"
    local MAFFT_BIN_PATH=""

    # ── Re-use a previous successful download ─────────────────────────────────
    MAFFT_BIN_PATH=$(find "$MAFFT_DIR" -name "mafft" \
                         -not -path "*/libexec/*" 2>/dev/null | head -1)

    # ── Download if not cached ────────────────────────────────────────────────
    if [ -z "$MAFFT_BIN_PATH" ]; then
        echo "[INFO] Downloading MAFFT static binary to /tmp ..."
        mkdir -p "$MAFFT_DIR"

        # Try two mirror URLs; first official site, then backup
        local URLS=(
            "https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz"
            "https://mafft.cbrc.jp/alignment/software/mafft-7.505-linux.tgz"
        )
        local downloaded=0
        for url in "${URLS[@]}"; do
            echo "[INFO]   trying ${url} ..."
            if curl -fsSL --max-time 120 --retry 3 "$url" \
                    | tar -xz -C "$MAFFT_DIR" --strip-components=1 2>/dev/null; then
                downloaded=1
                break
            fi
        done

        if [ "$downloaded" -eq 0 ]; then
            echo "[WARN] MAFFT download failed — MAFFT scores will be skipped"
            return 0
        fi

        MAFFT_BIN_PATH=$(find "$MAFFT_DIR" -name "mafft" \
                             -not -path "*/libexec/*" 2>/dev/null | head -1)
    fi

    # ── Verify and export ─────────────────────────────────────────────────────
    if [ -z "$MAFFT_BIN_PATH" ]; then
        echo "[WARN] MAFFT binary not found after extraction — skipping"
        return 0
    fi

    chmod +x "$MAFFT_BIN_PATH"

    # Explicitly set MAFFT_BINARIES to OUR libexec so the mafft shell script
    # never falls back to a system/conda installation.  Unsetting is not enough:
    # DataHub's conda activation scripts re-inject MAFFT_BINARIES inside child
    # processes even when the interactive shell shows it as empty.
    local MAFFT_LIBEXEC
    MAFFT_LIBEXEC="$(dirname "$MAFFT_BIN_PATH")/libexec"
    export MAFFT_BINARIES="$MAFFT_LIBEXEC"
    export MAFFT_BIN="$MAFFT_BIN_PATH"

    # Run version check with MAFFT_BINARIES pinned so the output is reliable
    local VER
    VER=$(env MAFFT_BINARIES="$MAFFT_LIBEXEC" "$MAFFT_BIN" --version 2>&1 | head -1)
    echo "[INFO] MAFFT ready: ${VER}"
}

_ensure_mafft
