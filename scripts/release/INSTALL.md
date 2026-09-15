# Undertale: Offline Installation

This bundle contains the `undertale` Python package and a complete set of
dependency wheels for the following platforms:

| `wheelhouse/` directory | Target |
| --- | --- |
| `macos-arm64` | macOS 14+ on Apple Silicon |
| `linux-x86_64` | Linux on x86-64 with glibc 2.28+ (e.g., RHEL 8+, Ubuntu 20.04+) |
| `windows-amd64` | Windows on x86-64 |

The Linux and Windows wheelhouses contain CPU-only builds of PyTorch.

## Requirements

- Python 3.12

## Installation

From the root of the extracted bundle, create a virtual environment and
install from the wheelhouse matching your platform - no network access or
package index is required:

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install --no-index --find-links wheelhouse/<platform> undertale
```

## HuggingFace Cache (optional)

Some model pipelines download resources (models, metrics) from the
HuggingFace Hub at runtime. For offline use, build a cache on a connected
machine and load it on the offline one:

```bash
# Connected machine:
python -m undertale.utils.models.cache.build <cache-directory>

# Offline machine (after transferring the cache directory):
python -m undertale.utils.models.cache.load <cache-directory>
export HF_HUB_OFFLINE=1
```

The `constraints.txt` included in this bundle records the exact dependency
versions the wheelhouses were resolved against.
