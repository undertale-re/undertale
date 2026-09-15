#!/bin/bash

# Build an offline release bundle for the Undertale inference server: the
# undertale-inference and undertale wheels, per-platform wheelhouses
# containing their combined dependency closure, a pre-seeded HuggingFace
# cache, and deployment examples.
#
# See scripts/release/INSTALL.md (copied into the bundle) for the full
# installation instructions.
#
# Options:
#   --skip-cache    skip building the HuggingFace cache (~1 GB download) -
#                   useful when iterating on wheel problems.

set -euo pipefail

cd "$(dirname "$0")/.."

source ../../scripts/lib/release.sh

CACHE=yes
for ARGUMENT in "$@"; do
    case "$ARGUMENT" in
        --skip-cache)
            CACHE=no
            ;;
        *)
            echo "[-] unknown argument: $ARGUMENT"
            exit 1
            ;;
    esac
done

check_prerequisites

VERSION="$(python -c 'import inference; print(inference.__version__)')"
UNDERTALE_VERSION="$(python -c 'import undertale; print(undertale.__version__)')"
NAME="undertale-inference-release-$VERSION"

BUILD="build/release"
RELEASE="dist/$NAME"

echo "[ ] building $NAME"

rm -rf "$BUILD" "$RELEASE"
mkdir -p "$BUILD/prebuilt" "$RELEASE/wheelhouse"

build_wheel ../.. "$BUILD"
build_wheel . "$BUILD"
prebuild_pure_wheels "$BUILD/prebuilt" constraints.txt
write_cpu_constraints "$BUILD" constraints.txt

WHEELS=(
    "$BUILD/undertale-$UNDERTALE_VERSION-py3-none-any.whl"
    "$BUILD/undertale_inference-$VERSION-py3-none-any.whl"
)

for PLATFORM in "${PLATFORMS[@]}"; do
    download_platform "$PLATFORM" "$RELEASE/wheelhouse" "$BUILD" constraints.txt "${WHEELS[@]}"
    check_wheelhouse "$PLATFORM" "$RELEASE/wheelhouse/$PLATFORM" undertale undertale-inference
done

if [ "$CACHE" = "yes" ]; then
    echo "[ ] building HuggingFace cache"

    python -m undertale.utils.models.cache.build "$RELEASE/hf-cache"

    echo "[+] built HuggingFace cache"
else
    echo "[!] skipping HuggingFace cache - this bundle is incomplete"
fi

cp scripts/release/INSTALL.md "$RELEASE/"
cp constraints.txt "$RELEASE/"
cp -R examples "$RELEASE/"

cat > "$RELEASE/VERSIONS" <<EOF
undertale-inference $VERSION
undertale $UNDERTALE_VERSION
EOF

archive_release dist "$NAME"

echo "[+] built $NAME"
