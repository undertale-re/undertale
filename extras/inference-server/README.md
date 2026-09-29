# Undertale: Inference Server

Undertale inference server.

## Description

A lightweight inference REST API that collects optional feedback and telemetry
from users.

## Installation

### Prerequisites

- The core Undertale python package, installed
- Optional: [nginx][nginx] as a reverse proxy

[nginx]: https://nginx.org/

### Installing

#### Online Installation

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

The worker then needs `HF_HUB_OFFLINE=1` (and `HF_HOME`, if customized) in its
environment - the example worker unit has commented-out `Environment=` lines
for this.

Then continue with the `inference initialize` and `inference migrate` steps
above.

#### Services

Every deployment runs the same two systemd services: the API server and the
inference worker. Install both from the examples:

```bash
cp examples/undertale-inference.service /etc/systemd/system/
cp examples/undertale-inference-worker.service /etc/systemd/system/
systemctl enable --now undertale-inference undertale-inference-worker
```

The example units carry the default deployment configuration: the API listens
on loopback TCP at `127.0.0.1:8000` and the worker runs with a parallelism of
4. Edit the copies in `/etc/systemd/system/` to change either, then run
`systemctl daemon-reload` and restart the affected service.

That default is a local, unauthenticated deployment, so set `authentication =
no` in `/etc/undertale-inference/settings.ini` (`inference initialize` writes
`authentication = yes`). To serve other users, see
[Authentication](#authentication); to expose the API over a Unix domain socket
instead of TCP, see [Sockets](#sockets).

## Configuration

### Authentication

Authentication is off in the default deployment. To authenticate users against
an LDAP instance, set the following in
`/etc/undertale-inference/settings.ini`:

```ini
[undertale-inference]
authentication = yes
jwtsecret = <random hex string>
ldaphost = ad.example.com
ldapport = 636
ldapdomain = example.com
```

`inference initialize` generates a `jwtsecret` for you; keep it secret and
stable, since rotating it invalidates every issued token. All four settings are
required when `authentication = yes` - the server refuses to start otherwise.

Restart the services to pick up the change:

```bash
systemctl restart undertale-inference undertale-inference-worker
```

Grant admin privileges to the users who need them:

```bash
inference admin --promote <username>
```

An authenticated deployment is typically fronted by nginx, terminating TLS and
proxying `/api/` to the API service. Use the example configuration as a
reference:

```bash
cp examples/nginx.conf /etc/nginx/conf.d/undertale-inference.conf
nginx -s reload
```

The example proxies to the default `127.0.0.1:8000` bind and sets the
`X-Forwarded-*` headers the API needs to generate correct URLs.

### Sockets

By default the API binds a loopback TCP socket, which is what nginx and any
other host-local reverse proxy expect:

```
ExecStart=gunicorn --workers=4 --bind 127.0.0.1:8000 inference.api:app
```

For a co-located deployment - where every client runs on the same machine as
the server, e.g. the Binary Ninja plugin - a Unix domain socket avoids opening
a port entirely. Edit `/etc/systemd/system/undertale-inference.service`:

```
RuntimeDirectory=undertale-inference
ExecStart=gunicorn --workers=4 --bind unix:/run/undertale-inference/undertale-inference.sock inference.api:app
```

`RuntimeDirectory=` creates `/run/undertale-inference/` owned by the service
user on start and removes it on stop. Apply the change with:

```bash
systemctl daemon-reload
systemctl restart undertale-inference
```

Clients must be on the same machine to reach a Unix domain socket - it is not
reachable over the network. If nginx fronts a socket-bound server, point
`proxy_pass` at the socket instead of the TCP address:

```nginx
proxy_pass http://unix:/run/undertale-inference/undertale-inference.sock:/;
```

## Usage

### Services

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
