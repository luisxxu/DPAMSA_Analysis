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

    local MAFFT_LIBEXEC
    MAFFT_LIBEXEC="$(dirname "$MAFFT_BIN_PATH")/libexec"

    # ── Sync libexec/VERSION with the version embedded in the wrapper script ──
    # The portable tarball sometimes ships with a stale, missing, or v0.000
    # VERSION file, causing the mafft wrapper's internal check to fail.
    # We extract the version string from the script itself and write it to
    # libexec/VERSION so the two values always agree.
    local MAFFT_VER
    MAFFT_VER=$(grep '^version=' "$MAFFT_BIN_PATH" 2>/dev/null | head -1 | \
                sed 's/version="\([^"]*\)".*/\1/')
    if [ -n "$MAFFT_VER" ] && [ -d "$MAFFT_LIBEXEC" ]; then
        echo "$MAFFT_VER" > "$MAFFT_LIBEXEC/VERSION"
    fi

    # Pin MAFFT_BINARIES to OUR libexec (unsetting alone is not enough —
    # conda activation scripts can re-inject it inside child processes)
    export MAFFT_BINARIES="$MAFFT_LIBEXEC"
    export MAFFT_BIN="$MAFFT_BIN_PATH"

    local VER
    VER=$(env MAFFT_BINARIES="$MAFFT_LIBEXEC" "$MAFFT_BIN" --version 2>&1 | head -1)
    echo "[INFO] MAFFT ready: ${VER}"
}

_ensure_mafft
