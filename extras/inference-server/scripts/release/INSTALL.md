# Undertale Inference Server: Offline Installation

This bundle contains everything needed to install the Undertale inference
server on a machine without network access:

- `wheelhouse/<platform>/` - the `undertale-inference` and `undertale` wheels
  plus their complete dependency closure
- `hf-cache/` - a pre-seeded HuggingFace cache (the inference worker loads
  the GPT2 tokenizer from it at startup)
- `examples/` - example gunicorn, nginx, and systemd configurations
- `constraints.txt` - the exact dependency versions the wheelhouses were
  resolved against
- `VERSIONS` - the bundled package versions

Supported platforms:

| `wheelhouse/` directory | Target |
| --- | --- |
| `macos-arm64` | macOS 14+ on Apple Silicon |
| `linux-x86_64` | Linux on x86-64 with glibc 2.28+ (e.g., RHEL 8+, Ubuntu 20.04+) |
| `windows-amd64` | Windows on x86-64 |

The Linux and Windows wheelhouses contain CPU-only builds of PyTorch.

## Requirements

- Python 3.12
- The model artifacts, obtained separately (they are too large to include in
  this bundle):
  - `maskedlm.ckpt` - masked language model checkpoint
  - `fnaming.ckpt` - function naming model checkpoint
  - `tokenizer.json` - the Undertale tokenizer

## Installation

### Packages

From the root of the extracted bundle, create a virtual environment and
install from the wheelhouse matching your platform - no network access or
package index is required. Note `undertale` is listed explicitly; it is a
runtime requirement of the server but not a declared package dependency:

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install --no-index --find-links wheelhouse/<platform> undertale undertale-inference
```

### Workspace

Create a workspace directory and copy in the model artifacts, named exactly
as follows (these are the default paths written by `inference initialize`):

```bash
export UNDERTALE_WORKSPACE=~/undertale-inference

mkdir -p "$UNDERTALE_WORKSPACE"
cp maskedlm.ckpt fnaming.ckpt tokenizer.json "$UNDERTALE_WORKSPACE/"
```

If `UNDERTALE_WORKSPACE` is not set, the server uses
`/etc/undertale-inference/` (appropriate for systemd deployments).

Initialize the configuration file and migrate the database:

```bash
inference initialize
inference migrate
```

For a local, unauthenticated deployment (e.g., over a Unix socket), disable
authentication in `$UNDERTALE_WORKSPACE/settings.ini`:

```ini
authentication = no
```

### HuggingFace Cache

Load the bundled cache into the workspace and force offline mode so the
worker never attempts to reach the HuggingFace Hub:

```bash
export HF_HOME="$UNDERTALE_WORKSPACE/huggingface"

python -m undertale.utils.models.cache.load hf-cache

export HF_HUB_OFFLINE=1
```

`HF_HOME` and `HF_HUB_OFFLINE` must be set in every shell (or service unit)
that runs an inference worker.

## Running

### Local Unix Socket Service

Start the API bound to a Unix socket:

```bash
gunicorn --bind unix:./undertale-inference.sock inference.api:app
```

Start the inference worker(s):

```bash
inference worker --parallelism 2
```

On macOS, gunicorn and the workers need
`export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to survive `fork()`.

### Systemd Service

Follow the systemd/nginx instructions in the server README using the bundled
`examples/`, and add the offline environment to both units, e.g.:

```ini
[Service]
Environment=UNDERTALE_WORKSPACE=/etc/undertale-inference/
Environment=HF_HOME=/etc/undertale-inference/huggingface
Environment=HF_HUB_OFFLINE=1
```

## Smoke Test

Verify the API responds over the socket:

```bash
curl --unix-socket ./undertale-inference.sock http://localhost/
```

Submit a completion and confirm a worker processes it (with authentication
disabled, all API requests run as an automatically created admin user named
`default`):

```bash
curl --unix-socket ./undertale-inference.sock \
    -H 'Content-Type: application/json' \
    -d '{"input": "xor rax [MASK]"}' \
    http://localhost/maskedlm/completion/

inference completions
```
