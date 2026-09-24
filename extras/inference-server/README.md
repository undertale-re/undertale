# Undertale: Inference Server

Undertale inference server.

## Description

A lightweight inference REST API that collects optional feedback and telemetry
from users.

## Installation

### Prerequisites

- [nginx][nginx] as a reverse proxy
- [gunicorn][gunicorn] as the WSGI server
- The core Undertale python package, installed

[nginx]: https://nginx.org/
[gunicorn]: https://gunicorn.org/

### Installing

Install the Python package:

```bash
pip install undertale-inference
```

Initialize the configuration file:

```bash
inference initialize
```

By default the configuration file is written to
`/etc/undertale-inference/settings.ini`.

The `maskedlm-checkpoint` and `function-naming-checkpoint` settings are
optional; leave a setting empty (or remove its line) to disable that model.
A worker with a model disabled logs a warning at startup and fails
completion requests of that type.

Migrate the database:

```bash
inference migrate
```

#### Offline Installation

For deployment targets without internet access, build a release bundle on an
internet-connected machine with the same OS, architecture, and Python version
(3.12) as the target:

```bash
./scripts/release.sh
```

This writes `dist/undertale-inference-<version>-<os>-<arch>.tar.gz` containing
a `wheelhouse/` of all required packages, a pre-seeded HuggingFace cache
(`hf-cache/`), the `examples/` directory, and this README.

On the target system, extract the bundle and run the installation from the root
directory. (note `undertale` is named explicitly - it is a runtime requirement
of the inference worker):

```bash
pip install --no-index --find-links wheelhouse undertale undertale-inference
```

If the function naming model is enabled, load the bundled HuggingFace cache
into `HF_HOME` (defaults to `~/.cache/huggingface`):

```bash
python -m undertale.utils.models.cache.load hf-cache
```

Then set `HF_HUB_OFFLINE=1` (and `HF_HOME`, if customized) in the worker's
environment (e.g., with `Environment=` lines in
`undertale-inference-worker.service`).

Then continue with the `inference initialize` and `inference migrate` steps
above.

#### Authenticated Systemd Service

Configure the settings to point to your LDAP instance for authentication.

Install the API systemd service using the example as a reference:

```bash
cp examples/undertale-inference.service /etc/systemd/system/
systemctl enable --now undertale-inference
```

Configure nginx to proxy `/api/` to gunicorn using the example configuration as
a reference:

```bash
cp examples/nginx.conf /etc/nginx/conf.d/undertale-inference.conf
nginx -s reload
```

Install the inference worker systemd service using the example as a reference:

```bash
cp examples/undertale-inference-worker.service /etc/systemd/system/
systemctl enable --now undertale-inference-worker
```

#### Unauthenticated Local Service

Authentication may be disabled for simple, co-located inference service
deployments (e.g., `authentication = False` in the configuration).

Start the inference server bound to e.g., a Unix socket:

```bash
gunicorn --bind unix:./undertale-inference.sock inference.api:app
```

Start the inference worker(s):

```bash
inference worker --parallelism 2
```

## Usage

### Authenticated Systemd Service

Use `systemctl` to manage the services:

```bash
systemctl start undertale-inference
systemctl stop undertale-inference
systemctl restart undertale-inference
systemctl status undertale-inference

# Enable or disable auto-start on boot
systemctl enable undertale-inference
systemctl disable undertale-inference
```

The above command can also be used with the `undertale-inference-worker`
service.

### Management

The `inference` CLI provides commands for managing the server:

```bash
# Grant or revoke admin privileges for a user
inference admin --promote <username>
inference admin --demote <username>

# Reset running completions to queued to recover from worker failure
inference purge

# List users and their completion / feedback counts
inference users
inference users --sorted          # sort by completion count (descending)

# Force authentication of a given user by username
# (errors when authentication is disabled)
inference authenticate username

# List completions (default limit: 10)
inference completions
inference completions --user <username>
inference completions --date YYYY-MM-DD
inference completions --input <substring>
inference completions --limit <n>

# Submit a completion from the CLI
inference submit -u username -t MaskedLM "xor rax [MASK]"
inference submit -u username -t FunctionNaming "push rbp\nmov rbp, rsp\n..."

# Delete a completion
inference delete 42
inference delete 42 --confirm

# Export completions to Parquet
inference export completions.parquet
inference export --start-date YYYY-MM-DD completions.parquet
```

## Contributing

### Prerequisites

The main Undertale conda environment must be set up before developing the
inference server. See the `Installation` section of the documentation for more
details.

### Development Environment

Update the existing `undertale` conda environment with the inference server's
development dependencies:

```bash
conda env update -f environment.development.yml
conda activate undertale
```

You might also find the environment file `environments/development.env` useful
for development purposes. This file sets environment variables for the project
to a useful configuration for development. To activate it, run:

```bash
source environments/development.env
```

### Development Server

After initialization and migration, you can start the Flask development server
directly:

```bash
flask --app inference.api run
```

Start a development inference worker:

```bash
inference worker --parallelism 1
```

### Building a Release Bundle

To build an offline installation bundle for the current platform (requires
Python 3.12 and internet access):

```bash
./scripts/release.sh
```

See the [Offline Installation](#offline-installation) section for details on
the bundle contents and installation.
