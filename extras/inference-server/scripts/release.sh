#!/usr/bin/env bash
#
# Build an offline installation bundle for the inference server, containing
# wheels for undertale, undertale-inference, and all of their dependencies for
# the current platform. See the Offline Installation section of README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

# constraints.txt is compiled for Python 3.12 and wheels are
# interpreter-version-specific.
python_version=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$python_version" != "3.12" ]; then
    echo "error: Python 3.12 required, found $python_version" >&2
    exit 1
fi

version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' inference/__init__.py)
name="undertale-inference-${version}-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"

workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
bundle="$workspace/$name"
mkdir -p "$bundle" dist

"$PYTHON" -m pip wheel --wheel-dir "$bundle/wheelhouse" -c constraints.txt ../.. .

# Smoke-test an offline installation of the wheelhouse, then use it to seed
# the HuggingFace cache required by the function naming model (gpt2 tokenizer
# and config only - weights come from the checkpoint).
"$PYTHON" -m venv "$workspace/venv"
"$workspace/venv/bin/pip" install --no-index --find-links "$bundle/wheelhouse" \
    undertale undertale-inference
HF_HOME="$bundle/hf-cache" "$workspace/venv/bin/python" -c \
    'from transformers import GPT2Config, GPT2Tokenizer; GPT2Config.from_pretrained("gpt2"); GPT2Tokenizer.from_pretrained("gpt2")'

cp -R examples "$bundle/examples"
cp README.md "$bundle/README.md"

tar -C "$workspace" -czf "dist/$name.tar.gz" "$name"
echo "wrote dist/$name.tar.gz"
