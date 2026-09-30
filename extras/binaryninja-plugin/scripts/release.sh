#!/usr/bin/env bash
#
# Build an installation bundle for the Binary Ninja plugin, containing the
# plugin directory and its documentation. The plugin is pure Python with no
# dependencies, so there is nothing to compile or vendor. See the Install from a
# Release Bundle section of README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

version=$("$PYTHON" -c \
    'import json; print(json.load(open("undertale/plugin.json"))["version"])')
name="undertale-binaryninja-plugin-${version}"

workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
bundle="$workspace/$name"
mkdir -p "$bundle" dist

cp -R undertale "$bundle/undertale"
find "$bundle/undertale" -name __pycache__ -type d -prune -exec rm -rf {} +
cp README.md "$bundle/README.md"

tar -C "$workspace" -czf "dist/$name.tar.gz" "$name"
echo "wrote dist/$name.tar.gz"
