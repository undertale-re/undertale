"""Undertale's Inference Server connection: configuring, caching, and persisting it.

The connection (TCP host:port or a Unix domain socket path) is configured
once and persists across Binary Ninja restarts until explicitly reconfigured
via the command named by :py:data:`CONFIGURE_COMMAND_NAME`.

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
    execute_on_main_thread_and_wait,
    get_form_input,
    log_error,
    log_info,
    log_warn,
)
from binaryninja.enums import SettingsScope
from binaryninjaui import UIContext
from PySide6.QtWidgets import QInputDialog, QLineEdit

from .notify import alert_user

SETTINGS_GROUP = "undertale"
SETTINGS_KEY = "undertale.inferenceServerConnection"
SETTINGS_KEY_POLL_TIMEOUT = "undertale.inferencePollTimeout"
SETTINGS_KEY_TOKEN = "undertale.inferenceServerToken"
CONFIGURE_COMMAND_NAME = "Undertale\\Configure Plugin"
CONFIGURE_MENU_PATH = CONFIGURE_COMMAND_NAME.replace("\\", " > ")  # used for logging

DEFAULT_POLL_TIMEOUT = 60
MIN_POLL_TIMEOUT = 1
MAX_POLL_TIMEOUT = 3600

CONNECTION_FORM_TITLE = "Inference Server Connection"
CONFIGURE_FORM_TITLE = "Configure Undertale Plugin"
CREDENTIALS_FORM_TITLE = "Inference Server Credentials"
CONNECTION_KIND_CHOICES = ["TCP (host:port)", "Unix Domain Socket"]
CONNECTION_KIND_TCP, CONNECTION_KIND_UNIX = range(len(CONNECTION_KIND_CHOICES))
CLEAR_TOKEN_CHOICES = ["No", "Yes"]
CLEAR_TOKEN_NO, CLEAR_TOKEN_YES = range(len(CLEAR_TOKEN_CHOICES))
CONNECTION_ATTR = "_undertale_inference_connection"
TOKEN_ATTR = "_undertale_inference_token"

INFERENCE_DEFAULT_HOST = "127.0.0.1"
INFERENCE_DEFAULT_PORT = "5000"
INFERENCE_DEFAULT_SOCKET_PATH = "/path/to/undertale-inference.sock"

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
                    "via the plugin's connection dialog. Reconfigured by "
                    f"{CONFIGURE_COMMAND_NAME!r}."
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
                "minValue": MIN_POLL_TIMEOUT,
                "maxValue": MAX_POLL_TIMEOUT,
                "description": (
                    "Seconds to wait for the Inference Server to finish "
                    "naming a function before giving up. Reconfigurable via "
                    f"{CONFIGURE_COMMAND_NAME!r}."
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
                    f"{CONFIGURE_COMMAND_NAME!r}."
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
        alert_user(
            f"Unix domain socket does not exist: {path!r}.\n\nThe Inference Server "
            "may not be running, or the path is wrong.\n\nStart the server, then "
            f"re-run {CONFIGURE_MENU_PATH} and enter the correct socket path."
        )
        return False

    if not stat.S_ISSOCK(os.stat(path).st_mode):
        alert_user(
            f"Not a Unix domain socket: {path!r}.\n\nThis path points to a regular "
            f"file or directory.\n\nRe-run {CONFIGURE_MENU_PATH} "
            "and enter the Inference Server's socket path."
        )
        return False

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(UNIX_SOCKET_VALIDATION_TIMEOUT)
            connection.connect(path)
    except OSError as error:
        alert_user(
            f"Unable to connect to Unix domain socket {path!r}: {error}.\n\nThe "
            "socket file exists but nothing is accepting connections there — the "
            "Inference Server is likely not running.\n\nStart it, then re-run "
            f"{CONFIGURE_MENU_PATH}."
        )
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

    log_warn(cancel_message)
    return False


def _build_tcp(host: str, port: str) -> Optional[Connection]:
    """Validate a host and port and build a TCP connection dict.

    Logs an error and returns None if either is invalid.
    """
    if not host or not port.isdigit():
        alert_user(
            f"Invalid host:port: {host!r}:{port!r}.\n\nRe-run "
            f"{CONFIGURE_MENU_PATH} and enter a hostname or IP address (e.g. "
            "127.0.0.1) for Host, and a numeric port (e.g. 5000) for Port."
        )
        return None
    return {"kind": "tcp", "host": host, "port": int(port)}


def _build_unix(path: str) -> Optional[Connection]:
    """Validate a Unix domain socket path and build a connection dict.

    Logs an error and returns None if the path is empty or unusable.
    """
    if not path:
        alert_user(
            f"No path given.\n\nRe-run {CONFIGURE_MENU_PATH} and enter the "
            "Inference Server's Unix domain socket path."
        )
        return None
    if not _validate_unix_socket_path(path):
        return None
    return {"kind": "unix", "path": path}


def _parse_timeout(timeout: str) -> Optional[int]:
    """Validate a poll timeout entered as a string, in seconds.

    Logs an error and returns None if it is not a whole number of at least 1.
    """
    if not timeout.isdigit() or int(timeout) < 1:
        alert_user(
            f"Invalid poll timeout: {timeout!r}.\n\nEnter a whole number of "
            f"seconds (1 or greater).\n\nRe-run {CONFIGURE_MENU_PATH} to try again."
        )
        return None
    return int(timeout)


def _prompt_tcp() -> Optional[Connection]:
    """Ask the user for a TCP host and port.

    Returns None if the user cancels or gives an invalid answer.
    """
    host_field = TextLineField("Host", INFERENCE_DEFAULT_HOST)
    port_field = TextLineField("Port", INFERENCE_DEFAULT_PORT)

    if not _run_form([host_field, port_field]):
        return None

    return _build_tcp(
        (host_field.result or "").strip(), (port_field.result or "").strip()
    )


def _prompt_unix() -> Optional[Connection]:
    """Ask the user for a Unix domain socket path.

    Returns None if the user cancels or gives an invalid answer.
    """
    path_field = TextLineField("Path", INFERENCE_DEFAULT_SOCKET_PATH)

    if not _run_form([path_field]):
        return None

    return _build_unix((path_field.result or "").strip())


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

    Uses Qt dialogs with a masked password field. The dialogs run on the UI
    thread because this is called from a background task.

    Returns None if the user cancels or leaves either field blank.
    """
    captured: Dict[str, str] = {}

    def prompt() -> None:
        context = UIContext.activeContext()
        if context is None:
            return
        parent = context.mainWindow()
        username, ok = QInputDialog.getText(
            parent, CREDENTIALS_FORM_TITLE, "Username:", QLineEdit.Normal
        )
        if not ok:
            return
        password, ok = QInputDialog.getText(
            parent, CREDENTIALS_FORM_TITLE, "Password:", QLineEdit.Password
        )
        if not ok:
            return
        captured["username"] = username
        captured["password"] = password

    execute_on_main_thread_and_wait(prompt)

    if not captured:  # user cancelled either dialog
        return None
    username = captured["username"].strip()
    password = captured["password"]
    if not username or not password:
        alert_user("Username and password are both required")
        return None
    return {"username": username, "password": password}


def _save_connection(connection: Connection) -> None:
    """Persist the connection so it survives closing Binary Ninja."""
    Settings().set_string(
        SETTINGS_KEY, json.dumps(connection), scope=SettingsScope.SettingsUserScope
    )


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


def get_poll_timeout() -> int:
    """Return the configured timeout, in seconds, to wait for the Inference
    Server to finish naming a function."""
    return Settings().get_integer(SETTINGS_KEY_POLL_TIMEOUT)


def configure_plugin(bv: BinaryView) -> None:
    """Configure the Undertale Plugin.

    Presents the current connection and poll timeout for editing, and offers
    to clear a saved login token. Because Binary Ninja forms are static (no
    conditional fields), the form shows both TCP and Unix connection fields;
    only the fields for the chosen connection type are validated.

    Nothing is persisted unless every value validates, so a cancel or a bad
    entry leaves the existing configuration untouched. The saved token is
    discarded when the connection target changes or when the user asks for it,
    so stale credentials never carry over to a different server.
    """
    current = getattr(binaryninja, CONNECTION_ATTR, None) or _load_connection()

    current_kind = CONNECTION_KIND_TCP
    host_default = INFERENCE_DEFAULT_HOST
    port_default = INFERENCE_DEFAULT_PORT
    path_default = INFERENCE_DEFAULT_SOCKET_PATH
    if current is not None:
        if current.get("kind") == "unix":
            current_kind = CONNECTION_KIND_UNIX
            path_default = current.get("path", path_default)
        else:
            host_default = current.get("host", host_default)
            port_default = str(current.get("port", port_default))

    kind_field = ChoiceField("Connection Type", CONNECTION_KIND_CHOICES)
    kind_field.result = current_kind
    host_field = TextLineField("Host", host_default)
    port_field = TextLineField("Port", port_default)
    path_field = TextLineField("Unix Socket Path", path_default)
    timeout_field = TextLineField("Poll Timeout (seconds)", str(get_poll_timeout()))

    fields = [kind_field, host_field, port_field, path_field, timeout_field]

    clear_token_field = None
    if get_token() is not None:
        clear_token_field = ChoiceField("Clear saved login token", CLEAR_TOKEN_CHOICES)
        fields.append(clear_token_field)

    if not get_form_input(fields, CONFIGURE_FORM_TITLE):
        log_warn("Operation cancelled: the Undertale plugin was not configured.")
        return

    timeout = _parse_timeout((timeout_field.result or "").strip())
    if timeout is None:
        return

    if kind_field.result == CONNECTION_KIND_TCP:
        connection = _build_tcp(
            (host_field.result or "").strip(), (port_field.result or "").strip()
        )
    else:
        connection = _build_unix((path_field.result or "").strip())
    if connection is None:
        return

    setattr(binaryninja, CONNECTION_ATTR, connection)
    _save_connection(connection)
    Settings().set_integer(
        SETTINGS_KEY_POLL_TIMEOUT, timeout, scope=SettingsScope.SettingsUserScope
    )

    clear_requested = (
        clear_token_field is not None and clear_token_field.result == CLEAR_TOKEN_YES
    )
    if clear_requested or (current is not None and current != connection):
        clear_token()

    log_info(
        f"Undertale plugin configured: connection={connection}, "
        f"poll timeout={timeout}s"
    )


_register_settings()
