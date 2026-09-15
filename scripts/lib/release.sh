# Shared functions for building offline release bundles.
#
# Source this file from a bash script running under `set -euo pipefail`. All
# functions expect to be called with absolute or caller-relative paths - none
# of them change the working directory.
#
# Note this file targets the system bash on macOS (3.2), so it avoids
# namerefs and uses the `${array[@]+...}` guard for possibly-empty arrays.

TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Wheelhouse platform labels, used as wheelhouse subdirectory names.
PLATFORMS=(macos-arm64 linux-x86_64 windows-amd64)

# Pinned packages that only publish sdists on PyPI. These must be built into
# wheels locally (and must be pure Python) because cross-platform `pip
# download --platform` forbids sdists. If a future constraint resolution pulls
# in a new sdist-only package, `pip download` will fail naming it - add it
# here.
SDIST_ONLY=(rouge-score)

platform_flags() {
    local LABEL=$1

    # Note: pip expands a macOS platform tag downward to older releases (and
    # universal2) automatically, but does not do the same for manylinux tags,
    # so the Linux glibc tags must be listed explicitly. The newest allowed
    # tag (manylinux_2_28) sets the glibc floor (2.28) for installation
    # targets.
    case "$LABEL" in
        macos-arm64)
            echo "--platform macosx_14_0_arm64"
            ;;
        linux-x86_64)
            echo "--platform manylinux1_x86_64" \
                "--platform manylinux2010_x86_64" \
                "--platform manylinux2014_x86_64" \
                "--platform manylinux_2_24_x86_64" \
                "--platform manylinux_2_28_x86_64"
            ;;
        windows-amd64)
            echo "--platform win_amd64"
            ;;
        *)
            echo "[-] unknown platform label: $LABEL" >&2
            return 1
            ;;
    esac
}

check_prerequisites() {
    if ! python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
        echo "[-] python 3.12 is required (matching requires-python)"
        return 1
    fi

    if ! pip --version | awk '{ split($2, version, "."); exit !(version[1] >= 23) }'; then
        echo "[-] pip >= 23 is required for cross-platform downloads"
        return 1
    fi
}

# Build a wheel from the given project directory, without dependencies.
build_wheel() {
    local SOURCE=$1
    local OUTPUT=$2

    echo "[ ] building wheel from $SOURCE"

    pip wheel --quiet --no-deps --wheel-dir "$OUTPUT" "$SOURCE"

    echo "[+] built wheel from $SOURCE"
}

# Build local wheels for pinned packages that only publish sdists on PyPI.
prebuild_pure_wheels() {
    local OUTPUT=$1
    local CONSTRAINTS=$2

    local NAME PIN
    for NAME in "${SDIST_ONLY[@]}"; do
        PIN="$(grep "^$NAME==" "$CONSTRAINTS")"

        echo "[ ] building wheel for sdist-only package $PIN"

        pip wheel --quiet --no-deps --wheel-dir "$OUTPUT" "$PIN"

        # A platform-specific wheel here would only be valid for the build
        # machine, not for every wheelhouse target.
        if ! compgen -G "$OUTPUT/${NAME//-/_}-*-py3-none-any.whl" >/dev/null; then
            echo "[-] $NAME did not build a pure (py3-none-any) wheel"
            return 1
        fi

        echo "[+] built wheel for sdist-only package $PIN"
    done
}

# Write a constraints overlay that forces the CPU-only torch build offered by
# the PyTorch package index (used for Linux and Windows targets, where the
# default PyPI wheels bundle CUDA).
write_cpu_constraints() {
    local BUILD=$1
    local CONSTRAINTS=$2

    grep '^torch==' "$CONSTRAINTS" | sed 's/$/+cpu/' > "$BUILD/constraints-cpu.txt"
}

# Download the full dependency closure of the given requirements for one
# target platform into $DESTINATION/$LABEL. Requirements are typically paths
# to locally built project wheels; pip copies them into the wheelhouse too,
# making it self-contained.
download_platform() {
    local LABEL=$1
    local DESTINATION=$2
    local BUILD=$3
    local CONSTRAINTS=$4
    shift 4

    local FLAGS
    FLAGS=($(platform_flags "$LABEL"))

    # On macOS/arm64 the default PyPI torch wheels are already CPU-only; the
    # CPU index and overlay are only needed where PyPI defaults to CUDA.
    local INDEX=()
    if [ "$LABEL" != "macos-arm64" ]; then
        INDEX=(-c "$BUILD/constraints-cpu.txt" --extra-index-url "$TORCH_CPU_INDEX")
    fi

    echo "[ ] downloading $LABEL wheels"

    pip download "$@" \
        -c "$CONSTRAINTS" \
        ${INDEX[@]+"${INDEX[@]}"} \
        --find-links "$BUILD/prebuilt" \
        --dest "$DESTINATION/$LABEL" \
        --only-binary=:all: \
        --implementation cp \
        --python-version 3.12 \
        "${FLAGS[@]}"

    echo "[+] downloaded $LABEL wheels"
}

# Verify that a wheelhouse is complete for its target platform by asking
# pip's resolver to (dry-run) install the given requirements from the
# wheelhouse alone. pip requires --target alongside --platform, so point it
# at a throwaway directory - --dry-run never writes to it.
check_wheelhouse() {
    local LABEL=$1
    local WHEELHOUSE=$2
    shift 2

    local FLAGS
    FLAGS=($(platform_flags "$LABEL"))

    echo "[ ] checking $LABEL wheelhouse completeness"

    pip install --dry-run --quiet \
        --ignore-installed \
        --no-index \
        --find-links "$WHEELHOUSE" \
        --only-binary=:all: \
        --implementation cp \
        --python-version 3.12 \
        "${FLAGS[@]}" \
        --target "$(mktemp -d)" \
        "$@"

    echo "[+] $LABEL wheelhouse is complete"
}

# Create $NAME.tar.gz next to the $PARENT/$NAME release directory and print a
# size summary.
archive_release() {
    local PARENT=$1
    local NAME=$2

    echo "[ ] archiving $NAME"

    tar -C "$PARENT" -czf "$PARENT/$NAME.tar.gz" "$NAME"

    echo "[+] release directory: $(du -sh "$PARENT/$NAME" | cut -f1) $PARENT/$NAME"
    echo "[+] release archive:   $(du -sh "$PARENT/$NAME.tar.gz" | cut -f1) $PARENT/$NAME.tar.gz"
}
