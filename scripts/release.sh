#!/bin/bash

# Build an offline release bundle for the core Undertale package: the
# undertale wheel plus per-platform wheelhouses containing its full
# dependency closure, installable on air-gapped machines with:
#
#     pip install --no-index --find-links wheelhouse/<platform> undertale
#
# See scripts/release/INSTALL.md (copied into the bundle) for the full
# installation instructions.

set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/lib/release.sh

check_prerequisites

VERSION="$(python -c 'import undertale; print(undertale.__version__)')"
NAME="undertale-release-$VERSION"

BUILD="build/release"
RELEASE="dist/$NAME"

echo "[ ] building $NAME"

rm -rf "$BUILD" "$RELEASE"
mkdir -p "$BUILD/prebuilt" "$RELEASE/wheelhouse"

build_wheel . "$BUILD"
prebuild_pure_wheels "$BUILD/prebuilt" constraints.txt
write_cpu_constraints "$BUILD" constraints.txt

WHEEL=("$BUILD"/undertale-"$VERSION"-py3-none-any.whl)

for PLATFORM in "${PLATFORMS[@]}"; do
    download_platform "$PLATFORM" "$RELEASE/wheelhouse" "$BUILD" constraints.txt "${WHEEL[@]}"
    check_wheelhouse "$PLATFORM" "$RELEASE/wheelhouse/$PLATFORM" undertale
done

cp scripts/release/INSTALL.md "$RELEASE/"
cp constraints.txt "$RELEASE/"

archive_release dist "$NAME"

echo "[+] built $NAME"
