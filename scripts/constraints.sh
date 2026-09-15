#!/bin/bash

# Regenerate the pinned constraints for both the core Undertale package and
# the inference server from a single, joint dependency resolution so the two
# constraint files can never diverge (e.g., pin conflicting torch versions).
#
# Run this from a Python 3.12 environment with pip-tools installed (it is
# included in both projects' `development` extras). Note that pip-compile
# resolves for the platform it runs on - by convention these files are
# compiled on macOS/arm64.
#
# After running this script, commit both constraint files together.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v pip-compile >/dev/null; then
    echo "[-] pip-compile not found - install the development extras first"
    exit 1
fi

compile() {
    local OUTPUT=$1

    echo "[ ] compiling $OUTPUT"

    # Note: --upgrade ignores pins in the existing output file - without it,
    # pip-compile reuses each file's existing pins and the two resolutions
    # diverge.
    pip-compile \
        --quiet \
        --upgrade \
        --extra=development \
        --strip-extras \
        --output-file="$OUTPUT" \
        pyproject.toml extras/inference-server/pyproject.toml

    echo "[+] compiled $OUTPUT"
}

compile constraints.txt
compile extras/inference-server/constraints.txt

echo "[+] constraint files regenerated - commit both files together"
