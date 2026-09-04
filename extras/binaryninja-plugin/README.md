# Undertale: Binary Ninja Plugin

A Binary Ninja plugin that renames a function by sending its disassembly to
an [Undertale Inference Server](../inference-server) and applying the
predicted name.

## Workflow

1. Select a function in Binary Ninja and run **Undertale > Generate Function
   Name** (right-click in the disassembly/IL view, or the Tools menu).
2. The plugin reads the function's disassembly and pretokenizes it into the
   form the model was trained on.
3. The pretokenized disassembly is sent to the inference server for analysis.
4. The plugin polls the inference server for the function name prediction.
5. The predicted name is applied to the function as a single undoable
   action.

Naming runs in the background, so Binary Ninja stays responsive while
waiting on the inference server.

## Installation

The installable plugin lives in
[`undertale/`](undertale). Symlink (or copy)
that directory into Binary Ninja's user plugin folder:

```bash
# In Linux
ln -s "/path/to/undertale" ~/.binaryninja/plugins/undertale

# In macOS
ln -s "/path/to/undertale" ~/Library/Application\ Support/Binary\ Ninja/plugins/undertale  # macOS
```

Then restart Binary Ninja to register the plugin.

## Configuration

On first use, the plugin prompts you to configure the inference server connection, as either:

* a TCP address (`host:port`), or
* a Unix domain socket path

This configuration is saved in Binary Ninja's user settings and persists across restarts. To update it, select **Undertale > Reconfigure Inference Server Connection**. This clears the saved connection and immediately prompts you to configure a new one.

The plugin also exposes an **Inference Completion Poll Timeout** setting (default 60 seconds), which controls how long it waits for the inference server to finish naming a function before giving up. To update it, select **Undertale > Reconfigure Inference Completion Poll Timeout**.

Which one to pick depends on how the [inference server](../inference-server) is deployed:

### Authenticated Systemd Service

This deployment uses NGINX as a frontend for Gunicorn, with LDAP authentication. Configure the plugin to use the TCP option and point it to the endpoint exposed by NGINX.

The first time the plugin talks to an authenticated server, it prompts for your LDAP username and password, logs in, and caches the resulting token in Binary Ninja's user settings (usually at `~/.binaryninja/settings.json`) so you aren't prompted again until the token is rejected (e.g. it expires) or you reconfigure the connection.

> [!WARNING]
> **Credential and token handling is not hardened yet:**
> - The login token is stored **unencrypted** in Binary Ninja's user settings file (`settings.json` in the user directory), the same place the plugin caches your connection info. Anyone with read access to that file (or that user account) can read the token and use it to call the inference server as you until it expires.
> - Tokens are long-lived (14 days server-side, by default) and there is no dedicated "log out" action — the only ways to discard a cached token are **Undertale > Reconfigure Inference Server Connection**, or manually clearing `undertale.inferenceServerToken` from Binary Ninja's settings.
> - There is no token refresh: once a token expires or is revoked server-side, the plugin re-prompts for credentials on the next request.

### Unauthenticated Local Service

This deployment runs Gunicorn directly, with no NGINX and no authentication,
for simple co-located setups. Configure the plugin with the Unix domain
socket option, pointed at the same path passed to gunicorn's `--bind
unix:...` flag. This only works when Binary Ninja runs on the same machine
as the server — Unix domain sockets aren't reachable over the network.

## Requirements

- Binary Ninja build 4000 or newer, with Python 3 scripting enabled.
- Network or filesystem access to a running Undertale Inference Server (see
  [`extras/inference-server`](../inference-server)).

## Development

> [!NOTE]
> To keep the plugin independent of Undertale's much heavier dependency set, the disassembly code is duplicated here. `undertale/utils/disassembly.py` is a byte-for-byte copy of [`undertale/pipeline/disassembly.py`](https://github.com/undertale-re/undertale/blob/binaryninja-plugin/undertale/pipeline/disassembly.py).
>
> If either copy changes, update the other as well. `tests/unit.py` (`TestPipelineDisassemblyCodeConsistency`) verifies that the two remain identical.

