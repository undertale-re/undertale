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
SETTINGS_KEY_POLL_TIMEOUT = "undertale.inferencePollTimeout"
SETTINGS_KEY_TOKEN = "undertale.inferenceServerToken"
RECONFIGURE_COMMAND_NAME = "Undertale\\Reconfigure Inference Server Connection"
RECONFIGURE_POLL_TIMEOUT_COMMAND_NAME = (
    "Undertale\\Reconfigure Inference Completion Poll Timeout"
)

DEFAULT_POLL_TIMEOUT = 60

CONNECTION_FORM_TITLE = "Inference Server Connection"
POLL_TIMEOUT_FORM_TITLE = "Inference Completion Poll Timeout"
CREDENTIALS_FORM_TITLE = "Inference Server Credentials"
CONNECTION_KIND_CHOICES = ["TCP (host:port)", "Unix Domain Socket"]
CONNECTION_KIND_TCP, CONNECTION_KIND_UNIX = range(len(CONNECTION_KIND_CHOICES))
CONNECTION_ATTR = "_undertale_inference_connection"
TOKEN_ATTR = "_undertale_inference_token"

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
    settings.register_setting(
        SETTINGS_KEY_POLL_TIMEOUT,
        json.dumps(
            {
                "title": "Inference Completion Poll Timeout",
                "type": "number",
                "default": DEFAULT_POLL_TIMEOUT,
                "minValue": 1,
                "description": (
                    "Seconds to wait for the Inference Server to finish "
                    "naming a function before giving up. Reconfigurable via "
                    f"{RECONFIGURE_POLL_TIMEOUT_COMMAND_NAME!r}."
                ),
            }
        ),
    )
    settings.register_setting(
        SETTINGS_KEY_TOKEN,
        json.dumps(
            {
                "title": "Inference Server Token",
                "type": "string",
                "isSerialized": True,
                "default": "",
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"],
                "description": (
                    "Cached authentication token (JWT) for the Inference "
                    "Server, obtained by logging in. Cleared by "
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


def _run_form(
    fields: List[Any],
    title: str = CONNECTION_FORM_TITLE,
    cancel_message: str = "Operation cancelled: Inference server connection was not configured.",
) -> bool:
    """Run a Binary Ninja form.

    It logs and returns False if the user cancels it."""
    if get_form_input(fields, title):
        return True

    log_error(cancel_message)
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


def prompt_credentials() -> Optional[Dict[str, str]]:
    """Ask the user for their inference server username and password.

    NOTE: Binary Ninja's form API has no masked/password field, so the
    password is entered and displayed as plain text.

    Returns None if the user cancels or leaves either field blank.
    """
    username_field = TextLineField("Username")
    password_field = TextLineField("Password")

    if not _run_form(
        [username_field, password_field],
        CREDENTIALS_FORM_TITLE,
        cancel_message="Operation cancelled: no credentials provided.",
    ):
        return None

    username = (username_field.result or "").strip()
    password = password_field.result or ""
    if not username or not password:
        log_error("Username and password are both required")
        return None
    return {"username": username, "password": password}


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


def get_token() -> Optional[str]:
    """Return the cached authentication token for the Inference Server, if
    one has been obtained by logging in.

    First it will try the in-process cache if present, else the persisted
    setting if present, else None.
    """
    token = getattr(binaryninja, TOKEN_ATTR, None)
    if token is not None:
        return token

    token = Settings().get_string(SETTINGS_KEY_TOKEN) or None
    if token is not None:
        setattr(binaryninja, TOKEN_ATTR, token)

    return token


def save_token(token: str) -> None:
    """Cache and persist an authentication token obtained by logging in."""
    setattr(binaryninja, TOKEN_ATTR, token)
    Settings().set_string(
        SETTINGS_KEY_TOKEN, token, scope=SettingsScope.SettingsUserScope
    )


def clear_token() -> None:
    """Discard the cached and persisted authentication token."""
    if hasattr(binaryninja, TOKEN_ATTR):
        delattr(binaryninja, TOKEN_ATTR)
    Settings().reset(SETTINGS_KEY_TOKEN, scope=SettingsScope.SettingsUserScope)


def _prompt_poll_timeout() -> Optional[int]:
    """Ask the user for a new Inference Completion Poll Timeout, in seconds.

    Returns None if the user cancels or gives an invalid answer.
    """
    timeout_field = TextLineField("Poll Timeout (seconds)", str(DEFAULT_POLL_TIMEOUT))

    if not get_form_input([timeout_field], POLL_TIMEOUT_FORM_TITLE):
        log_error(
            "Operation cancelled: Inference Completion Poll Timeout was not configured."
        )
        return None

    timeout = (timeout_field.result or "").strip()
    if not timeout.isdigit() or int(timeout) < 1:
        log_error(f"Invalid poll timeout: {timeout!r}")
        return None
    return int(timeout)


def reconfigure_poll_timeout(bv: BinaryView) -> None:
    """Reconfigures the Inference Completion Poll Timeout.

    Prompts for a new timeout value and persists it, replacing the current
    one.
    """
    timeout = _prompt_poll_timeout()
    if timeout is None:
        return

    Settings().set_integer(
        SETTINGS_KEY_POLL_TIMEOUT, timeout, scope=SettingsScope.SettingsUserScope
    )
    log_info(f"Undertale's Inference Completion Poll Timeout configured: {timeout}s")


def get_poll_timeout() -> int:
    """Return the configured timeout, in seconds, to wait for the Inference
    Server to finish naming a function."""
    return Settings().get_integer(SETTINGS_KEY_POLL_TIMEOUT)


def reconfigure_connection(bv: BinaryView) -> None:
    """Recofingures a connection to the Inference Server.

    Discard the cached and persisted connection, then immediately prompt
    for a new one.
    """
    if hasattr(binaryninja, CONNECTION_ATTR):
        delattr(binaryninja, CONNECTION_ATTR)

    _clear_saved_connection()
    clear_token()
    get_connection()


_register_settings()
