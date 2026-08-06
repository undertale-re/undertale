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

Migrate the database:

```bash
inference migrate
```

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
