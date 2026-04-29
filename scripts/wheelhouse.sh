#!/bin/bash

download() {
    local NAME=$1
    local PLATFORM=$2
    local OUTPUT=$3

    echo "[ ] downloading $NAME wheels"

    pip download . -c constraints.txt -d $OUTPUT \
        --platform $PLATFORM \
        --only-binary=:all:


    if [ $? -ne 0 ]; then
        echo "[-] downloading $NAME wheels failed"
        exit $?
    fi

    echo "[+] downloaded $NAME wheels"
}

# Execute from the repo root, relative to this script.
cd "$(dirname "$0")/.."

# Local setup.
OUTPUT="$(pwd)/vendor"

# Download wheels.
echo "[ ] building wheelhouse at $OUTPUT"

download "Linux" "manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64" $OUTPUT
download "MacOS" "macosx_12_0_arm64" $OUTPUT
download "Windows" "win_amd64" $OUTPUT

echo "[+] wheelhouse complete at $OUTPUT"
