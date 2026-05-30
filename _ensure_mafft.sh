#!/bin/bash
# _ensure_mafft.sh -- sourced by run_acer_*.sh scripts
#
# Downloads the MAFFT static Linux binary to /tmp and exports MAFFT_BIN.
# Always exits 0 -- mafft is optional; training continues without it.

_ensure_mafft() {
    local MAFFT_DIR="/tmp/mafft_static"
    local MAFFT_BIN_PATH=""
    local downloaded_url=""

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
                downloaded_url="$url"
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

    # Fix the libexec/VERSION file.
    # The portable tarball ships with a stale or missing VERSION (shows v0.000),
    # causing the wrapper script's internal version check to always fail.
    # We try three strategies in order:
    #
    #  1. Bootstrap: run the binary, capture the mismatch error
    #     "v0.000 != v7.526 (2024/Apr/26)", extract the right-hand side.
    #  2. URL-based hardcode: derive version from the download URL.
    #  3. grep the wrapper script for the version string.
    local MAFFT_VER=""

    # Strategy 1: bootstrap from the version-mismatch error message
    if [ -z "$MAFFT_VER" ] && [ -d "$MAFFT_LIBEXEC" ]; then
        local bootstrap_out
        bootstrap_out=$(env MAFFT_BINARIES="$MAFFT_LIBEXEC" \
                            "$MAFFT_BIN_PATH" --version 2>&1 || true)
        MAFFT_VER=$(echo "$bootstrap_out" | grep ' != ' | \
                    sed 's/.* != //' | tr -d '\r' | head -1)
    fi

    # Strategy 2: hardcoded strings keyed to the download URL
    if [ -z "$MAFFT_VER" ]; then
        case "$downloaded_url" in
            *7.526*) MAFFT_VER="v7.526 (2024/Apr/26)" ;;
            *7.505*) MAFFT_VER="v7.505 (2022/Sep/16)" ;;
        esac
    fi

    # Strategy 3: grep the wrapper script
    if [ -z "$MAFFT_VER" ]; then
        MAFFT_VER=$(grep -m 1 'version=' "$MAFFT_BIN_PATH" 2>/dev/null | \
                    grep -oE 'v[0-9]+\.[0-9]+[^"]*' | head -1)
    fi

    # Write VERSION and pin MAFFT_BINARIES
    if [ -n "$MAFFT_VER" ] && [ -d "$MAFFT_LIBEXEC" ]; then
        echo "$MAFFT_VER" > "$MAFFT_LIBEXEC/VERSION"
        echo "[INFO]   wrote libexec/VERSION -> ${MAFFT_VER}"
    else
        echo "[WARN]   all version-detection strategies failed"
        echo "[WARN]   current VERSION: $(cat "$MAFFT_LIBEXEC/VERSION" 2>/dev/null || echo '<missing>')"
    fi

    export MAFFT_BINARIES="$MAFFT_LIBEXEC"
    export MAFFT_BIN="$MAFFT_BIN_PATH"

    local VER
    VER=$(env MAFFT_BINARIES="$MAFFT_LIBEXEC" "$MAFFT_BIN" --version 2>&1 | head -1)
    echo "[INFO] MAFFT ready: ${VER}"
}

_ensure_mafft
