#!/bin/bash
# _ensure_mafft.sh -- sourced by run_acer_*.sh scripts
#
# Downloads the MAFFT static Linux binary to /tmp and exports MAFFT_BIN.
# Always exits 0 -- mafft is optional; training continues without it.
#
# The system mafft on some DataHub pods has a broken libexec/VERSION file,
# so we always use our own downloaded copy rather than whatever is in PATH.

_ensure_mafft() {
    local MAFFT_DIR="/tmp/mafft_static"
    local MAFFT_BIN_PATH=""

    # Re-use a previous successful download
    MAFFT_BIN_PATH=$(find "$MAFFT_DIR" -name "mafft" \
                         -not -path "*/libexec/*" 2>/dev/null | head -1)

    # Download if not cached
    if [ -z "$MAFFT_BIN_PATH" ]; then
        echo "[INFO] Downloading MAFFT static binary to /tmp ..."
        mkdir -p "$MAFFT_DIR"

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
            echo "[WARN] MAFFT download failed -- MAFFT scores will be skipped"
            return 0
        fi

        MAFFT_BIN_PATH=$(find "$MAFFT_DIR" -name "mafft" \
                             -not -path "*/libexec/*" 2>/dev/null | head -1)
    fi

    if [ -z "$MAFFT_BIN_PATH" ]; then
        echo "[WARN] MAFFT binary not found after extraction -- skipping"
        return 0
    fi

    chmod +x "$MAFFT_BIN_PATH"

    local MAFFT_LIBEXEC
    MAFFT_LIBEXEC="$(dirname "$MAFFT_BIN_PATH")/libexec"

    # Fix libexec/VERSION
    # The portable tarball ships with a stale v0.000 VERSION file, causing the
    # wrapper script's internal version check to always fail.
    # Strategy: find the version string hard-coded in the wrapper script itself
    # using a permissive grep (handles leading whitespace) and write it to the
    # VERSION file so both values always agree.
    local MAFFT_VER
    MAFFT_VER=$(grep -m 1 'version="v' "$MAFFT_BIN_PATH" 2>/dev/null | \
                sed 's/.*version="\([^"]*\)".*/\1/')
    if [ -n "$MAFFT_VER" ] && [ -d "$MAFFT_LIBEXEC" ]; then
        echo "$MAFFT_VER" > "$MAFFT_LIBEXEC/VERSION"
        echo "[INFO]   patched libexec/VERSION -> ${MAFFT_VER}"
    else
        echo "[WARN]   could not extract version from script"
        echo "[WARN]   current VERSION: $(cat "$MAFFT_LIBEXEC/VERSION" 2>/dev/null || echo '<missing>')"
    fi

    # Pin MAFFT_BINARIES to OUR libexec.
    # Unsetting alone is not enough: DataHub conda activation scripts can
    # re-inject MAFFT_BINARIES inside child processes even when the
    # interactive shell reports it as empty.
    export MAFFT_BINARIES="$MAFFT_LIBEXEC"
    export MAFFT_BIN="$MAFFT_BIN_PATH"

    local VER
    VER=$(env MAFFT_BINARIES="$MAFFT_LIBEXEC" "$MAFFT_BIN" --version 2>&1 | head -1)
    echo "[INFO] MAFFT ready: ${VER}"
}

_ensure_mafft
