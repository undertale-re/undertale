"""Undertale's Inference Server connection: configuring, caching, and persisting it.

The connection (TCP host:port or a Unix domain socket path) is configured
once and persists across Binary Ninja restarts until explicitly reconfigured
via the command named by :py:data:`RECONFIGURE_COMMAND_NAME`.

Two layers of caching, for two different lifetimes:
    1. An attribute parked on the `binaryninja` module. `binaryninja` is not
       re-imported on "Reload Plugins", so this survives reloads and avoids
       re-reading settings on every call, for the life of the process.
    2. A Binary Ninja user-scoped Setting, written to
       <User Directory>/settings.json, which survives closing Binary Ninja
       entirely.
"""

import json
import os
import socket
import stat
from typing import Any, Dict, List, Optional

import binaryninja
from binaryninja import (
    BinaryView,
    ChoiceField,
    Settings,
    TextLineField,
    get_form_input,
    log_error,
    log_info,
)
from binaryninja.enums import SettingsScope

SETTINGS_GROUP = "undertale"
SETTINGS_KEY = "undertale.inferenceServerConnection"
RECONFIGURE_COMMAND_NAME = "Undertale\\Reconfigure Inference Server Connection"

CONNECTION_FORM_TITLE = "Inference Server Connection"
CONNECTION_KIND_CHOICES = ["TCP (host:port)", "Unix Domain Socket"]
CONNECTION_KIND_TCP, CONNECTION_KIND_UNIX = range(len(CONNECTION_KIND_CHOICES))
CONNECTION_ATTR = "_undertale_inference_connection"

INFERENCE_DEFAULT_HOST = "127.0.0.1"
INFERENCE_DEFAULT_PORT = "5000"

UNIX_SOCKET_VALIDATION_TIMEOUT = 10

Connection = Dict[str, Any]


def _register_settings() -> None:
    """Register the persisted-connection setting.

    Unlike a PluginCommand registration, this is idempotent, so it can run on
    every module load (including "Reload Plugins") without a once-per-process
    guard.
    """
    settings = Settings()
    settings.register_group(SETTINGS_GROUP, "Undertale")
    settings.register_setting(
        SETTINGS_KEY,
        json.dumps(
            {
                "title": "Inference Server Connection",
                "type": "string",
                "isSerialized": True,
                "default": "",
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"],
                "description": (
                    "Cached Inference Server Connection (JSON), configured "
                    "via the plugin's connection dialog. Cleared by "
                    f"{RECONFIGURE_COMMAND_NAME!r}."
                ),
            }
        ),
    )


def _load_connection() -> Optional[Connection]:
    """Read the persisted connection from Binary Ninja's user settings.

    Returns None if nothing has been saved yet.
    """
    connection_data = Settings().get_string(SETTINGS_KEY)

    if not connection_data:
        return None

    try:
        return json.loads(connection_data)
    except ValueError:
        log_error(
            f"Invalid stored connection for the Undertale's Inference Server: {connection_data!r}"
        )
        return None


def _validate_unix_socket_path(path: str) -> bool:
    """Check that ``path`` exists, is a Unix domain socket, and accepts a
    connection.

    Logs an error and returns False if any of these checks fail.
    """
    if not os.path.exists(path):
        log_error(f"Unix domain socket does not exist: {path!r}")
        return False

    if not stat.S_ISSOCK(os.stat(path).st_mode):
        log_error(f"Not a Unix domain socket: {path!r}")
        return False

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(UNIX_SOCKET_VALIDATION_TIMEOUT)
            connection.connect(path)
    except OSError as error:
        log_error(f"Unable to connect to Unix domain socket {path!r}: {error}")
        return False

    return True


def _run_form(fields: List[Any], title: str = CONNECTION_FORM_TITLE) -> bool:
    """Run a Binary Ninja form.

    It logs and returns False if the user cancels it."""
    if get_form_input(fields, title):
        return True

    log_error("Operation cancelled: Inference server connection was not configured.")
    return False


def _prompt_tcp() -> Optional[Connection]:
    """Ask the user for a TCP host and port.

    Returns None if the user cancels or gives an invalid answer.
    """
    host_field = TextLineField("Host", INFERENCE_DEFAULT_HOST)
    port_field = TextLineField("Port", INFERENCE_DEFAULT_PORT)

    if not _run_form([host_field, port_field]):
        return None

    host = (host_field.result or "").strip()
    port = (port_field.result or "").strip()
    if not host or not port.isdigit():
        log_error(f"Invalid host:port: {host!r}:{port!r}")
        return None
    return {"kind": "tcp", "host": host, "port": int(port)}


def _prompt_unix() -> Optional[Connection]:
    """Ask the user for a Unix domain socket path.

    Returns None if the user cancels or gives an invalid answer.
    """
    path_field = TextLineField("Path", "/path/to/undertale-inference.sock")

    if not _run_form([path_field]):
        return None

    path = (path_field.result or "").strip()
    if not path:
        log_error("No path given")
        return None
    if not _validate_unix_socket_path(path):
        return None
    return {"kind": "unix", "path": path}


def _prompt_for_connection() -> Optional[Connection]:
    """Ask the user for a new inference server connection.

    First asks for the connection type, then delegates to the prompt
    specific to that type (host and port for TCP, a single path for a Unix
    domain socket).

    Returns None if the user cancels or gives an invalid answer at either
    step.
    """
    kind_field = ChoiceField("Connection Type", CONNECTION_KIND_CHOICES)
    if not _run_form([kind_field]):
        return None

    if kind_field.result == CONNECTION_KIND_TCP:
        return _prompt_tcp()
    return _prompt_unix()


def _save_connection(connection: Connection) -> None:
    """Persist the connection so it survives closing Binary Ninja."""
    Settings().set_string(
        SETTINGS_KEY, json.dumps(connection), scope=SettingsScope.SettingsUserScope
    )


def _clear_saved_connection() -> None:
    """Remove the persisted connection so the user is prompted again."""
    Settings().reset(SETTINGS_KEY, scope=SettingsScope.SettingsUserScope)


def get_connection() -> Optional[Connection]:
    """Return the inference server connection.

    First it will try the in-process cache if present, else the persisted
    setting if present, else prompt the user and persist their answer.
    """
    connection = getattr(binaryninja, CONNECTION_ATTR, None)
    if connection is not None:
        return connection

    connection = _load_connection()
    if connection is not None:
        setattr(binaryninja, CONNECTION_ATTR, connection)
        return connection

    connection = _prompt_for_connection()
    if connection is None:
        return None

    setattr(binaryninja, CONNECTION_ATTR, connection)
    _save_connection(connection)

    log_info(f"Undertale's Inference Server connection configured: {connection}")

    return connection


def reconfigure_connection(bv: BinaryView) -> None:
    """Recofingures a connection to the Inference Server.

    Discard the cached and persisted connection, then immediately prompt
    for a new one.
    """
    if hasattr(binaryninja, CONNECTION_ATTR):
        delattr(binaryninja, CONNECTION_ATTR)

    _clear_saved_connection()
    get_connection()


_register_settings()

__all__ = [
    "Connection",
    "RECONFIGURE_COMMAND_NAME",
    "get_connection",
    "reconfigure_connection",
]
