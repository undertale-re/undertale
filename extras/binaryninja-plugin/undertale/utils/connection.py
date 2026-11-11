"""Undertale's Inference Server connection: configuring, caching, and persisting it.

The connection (HTTP or HTTPS host:port, or a Unix domain socket path, plus a
base URL) is configured once and persists across Binary Ninja restarts until
explicitly reconfigured via the command named by :py:data:`CONFIGURE_COMMAND_NAME`.

A connection is a normalized dict that fully identifies a server, so two
connections are the same server exactly when they compare equal::

    {"kind": "tcp", "scheme": "https", "host": "example.com", "port": 443, "base_url": "/api"}
    {"kind": "unix", "socket": "/run/undertale/inference.sock", "base_url": "/"}

The trusted CA certificate file is kept in a separate setting, because it
decides which certificates to trust rather than which server to talk to.

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
import ssl
import stat
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import binaryninja
from binaryninja import (
    BinaryView,
    ChoiceField,
    Settings,
    TextLineField,
    execute_on_main_thread_and_wait,
    get_form_input,
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
SETTINGS_KEY_CA_FILE = "undertale.inferenceServerCAFile"
CONFIGURE_COMMAND_NAME = "Undertale\\Configure Plugin"
CONFIGURE_MENU_PATH = CONFIGURE_COMMAND_NAME.replace("\\", " > ")  # used for logging

DEFAULT_POLL_TIMEOUT = 60
MIN_POLL_TIMEOUT = 1
MAX_POLL_TIMEOUT = 3600

DEFAULT_BASE_URL = "/"

CONNECTION_FORM_TITLE = "Inference Server Connection"
CONFIGURE_FORM_TITLE = "Configure Undertale Plugin"
CREDENTIALS_FORM_TITLE = "Inference Server Credentials"
CONNECTION_KIND_CHOICES = ["HTTP", "HTTPS", "Unix Domain Socket"]
CONNECTION_KIND_HTTP, CONNECTION_KIND_HTTPS, CONNECTION_KIND_UNIX = range(
    len(CONNECTION_KIND_CHOICES)
)
CLEAR_TOKEN_CHOICES = ["No", "Yes"]
CLEAR_TOKEN_NO, CLEAR_TOKEN_YES = range(len(CLEAR_TOKEN_CHOICES))
CONNECTION_ATTR = "_undertale_inference_connection"
TOKEN_ATTR = "_undertale_inference_token"

# The keys every stored connection of each kind has, besides "kind".
CONNECTION_KEYS = {
    "tcp": {"scheme", "host", "port", "base_url"},
    "unix": {"socket", "base_url"},
}
SCHEME_DEFAULT_PORTS = {"http": 80, "https": 443}

INFERENCE_DEFAULT_HOST = "127.0.0.1"
INFERENCE_DEFAULT_PORT = "8000"
INFERENCE_DEFAULT_TLS_PORT = "443"
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
    settings.register_setting(
        SETTINGS_KEY_CA_FILE,
        json.dumps(
            {
                "title": "Inference Server CA Certificate File",
                "type": "string",
                "default": "",
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"],
                "description": (
                    "PEM file with the certificate authority (or the "
                    "self-signed certificate) to trust for HTTPS connections "
                    "to the Undertale Inference Server. Leave empty to use the system "
                    f"trust store. Reconfigurable via {CONFIGURE_COMMAND_NAME!r}."
                ),
            }
        ),
    )


def _load_connection() -> Optional[Connection]:
    """Read the persisted connection from Binary Ninja's user settings.

    Returns None if nothing has been saved yet, or if the saved value is not a
    valid connection. An invalid connection is discarded along with its token,
    so the user is prompted for a new one.
    """
    connection_data = Settings().get_string(SETTINGS_KEY)

    if not connection_data:
        return None

    try:
        connection = json.loads(connection_data)
    except ValueError:
        connection = None
    if not is_valid_connection(connection):
        log_warn(
            "Discarding invalid stored Undertale Inference Server connection: "
            f"{connection_data!r}"
        )
        Settings().reset(SETTINGS_KEY, scope=SettingsScope.SettingsUserScope)
        clear_token()
        return None
    return connection


def is_valid_connection(value: Any) -> bool:
    """Determine whether ``value`` is a valid connection."""
    if not isinstance(value, dict):
        return False

    keys = CONNECTION_KEYS.get(str(value.get("kind")))
    if keys is None or set(value) - {"kind"} != keys:
        return False

    # Only accept a base URL already in the form the configure form saves.
    base_url = value["base_url"]
    if not isinstance(base_url, str) or _normalize_base_url(base_url) != base_url:
        return False

    if value["kind"] == "unix":
        return isinstance(value["socket"], str) and bool(value["socket"])

    port = value["port"]
    return (
        value["scheme"] in SCHEME_DEFAULT_PORTS
        and isinstance(value["host"], str)
        and bool(value["host"])
        and isinstance(port, int)
        and not isinstance(port, bool)
        and 0 < port < 65536
    )


def _cached_connection() -> Optional[Connection]:
    """Return the in-process cached connection, discarding it if it is not
    valid."""
    connection = getattr(binaryninja, CONNECTION_ATTR, None)
    if connection is None or is_valid_connection(connection):
        return connection

    delattr(binaryninja, CONNECTION_ATTR)
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


def _build_tcp(
    scheme: str, host: str, port: str, base_url: str
) -> Optional[Connection]:
    """Validate a host, port, and base URL and build an HTTP or HTTPS
    connection dict.

    A bracketed IPv6 address (e.g. ``[::1]``) is stored without its brackets,
    since http.client expects the bare address.

    Logs an error and returns None if any of them is invalid.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        alert_user(
            f"Invalid host:port: {host!r}:{port!r}.\n\nRe-run "
            f"{CONFIGURE_MENU_PATH} and enter a hostname or IP address (e.g. "
            "127.0.0.1) for Host, and a numeric port (e.g. 8000) for Port."
        )
        return None

    parsed = _parse_base_url(base_url)
    if parsed is None:
        return None

    return {
        "kind": "tcp",
        "scheme": scheme,
        "host": host.lower(),
        "port": int(port),
        "base_url": parsed,
    }


def _build_unix(path: str, base_url: str) -> Optional[Connection]:
    """Validate a Unix domain socket path and base URL and build a connection
    dict.

    Logs an error and returns None if either is empty or unusable.
    """
    if not path:
        alert_user(
            f"No path given.\n\nRe-run {CONFIGURE_MENU_PATH} and enter the "
            "Inference Server's Unix domain socket path."
        )
        return None
    parsed = _parse_base_url(base_url)
    if parsed is None:
        return None

    if not _validate_unix_socket_path(path):
        return None

    return {"kind": "unix", "socket": path, "base_url": parsed}


def _parse_ca_file(path: str) -> Optional[str]:
    """Validate a CA certificate file entered by the user.

    An empty value means the system trust store. Otherwise the file must
    exist and hold at least one PEM certificate.

    Alerts the user and returns None if the file cannot be loaded; otherwise
    returns the path, or "" for the system trust store.
    """
    if not path:
        return ""

    path = os.path.expanduser(path)
    try:
        ssl.create_default_context(cafile=path)
    except OSError as error:  # includes ssl.SSLError
        alert_user(
            f"Unable to load the CA certificate file {path!r}: {error}.\n\n"
            "Enter the path to a PEM file with the certificate authority (or "
            "the self-signed certificate) the Undertale Inference Server's certificate "
            "is signed by, or leave it empty to use the system trust store."
            f"\n\nRe-run {CONFIGURE_MENU_PATH} to try again."
        )
        return None

    return path


def describe_server(connection: Connection) -> str:
    """Describe an Inference Server endpoint for user-facing messages.

    Args:
        connection: The server's connection.

    Returns:
        A user friendly description of the endpoint: a URL for HTTP and
        HTTPS, or the socket path for a Unix domain socket.
    """
    base_url = connection["base_url"]
    if connection["kind"] == "unix":
        description = f"the Unix domain socket {connection['socket']!r}"
        if base_url != DEFAULT_BASE_URL:
            description += f" with base URL {base_url!r}"
        return description

    scheme = connection["scheme"]
    host = connection["host"]
    if ":" in host:  # an IPv6 address
        host = f"[{host}]"
    port = connection["port"]
    address = host if port == SCHEME_DEFAULT_PORTS[scheme] else f"{host}:{port}"
    path = "" if base_url == DEFAULT_BASE_URL else base_url
    return f"{scheme}://{address}{path}"


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


def _kind_choice(connection: Connection) -> int:
    """Return the index into :py:data:`CONNECTION_KIND_CHOICES` of a
    connection's type."""
    if connection["kind"] == "unix":
        return CONNECTION_KIND_UNIX

    if connection["scheme"] == "https":
        return CONNECTION_KIND_HTTPS

    return CONNECTION_KIND_HTTP


def _prompt_tcp(
    scheme: str, current: Optional[Connection], form: Dict[str, str]
) -> Optional[Tuple[Connection, Optional[str]]]:
    """Ask the user for an HTTP or HTTPS host, port, and base URL, plus a CA
    certificate file for HTTPS.

    Args:
        scheme: "http" or "https".
        current: The TCP connection to prefill the fields from, or None for
            the defaults. If its scheme differs from ``scheme``, only its
            host and base URL are kept, since its port belongs to the other
            scheme.
        form: The ``title`` and ``cancel_message`` for :py:func:`_run_form`.

    Returns:
        ``(connection, ca_file)``, where ``ca_file`` is the CA certificate
        file to save ("" for the system trust store), or None for HTTP. None
        if the user cancels or gives an invalid answer.
    """
    host = INFERENCE_DEFAULT_HOST
    port = INFERENCE_DEFAULT_TLS_PORT if scheme == "https" else INFERENCE_DEFAULT_PORT
    base_url = DEFAULT_BASE_URL
    if current is not None:
        host, base_url = current["host"], current["base_url"]
        if current["scheme"] == scheme:
            port = str(current["port"])

    host_field = TextLineField("Host", host)
    port_field = TextLineField("Port", port)
    base_url_field = TextLineField("Base URL", base_url)
    fields = [host_field, port_field, base_url_field]
    ca_field = None
    if scheme == "https":
        ca_field = TextLineField("CA Certificate File (optional)", get_ca_file() or "")
        fields.append(ca_field)

    if not _run_form(fields, **form):
        return None

    connection = _build_tcp(
        scheme,
        (host_field.result or "").strip(),
        (port_field.result or "").strip(),
        base_url_field.result or "",
    )
    if connection is None:
        return None

    if ca_field is None:
        return connection, None

    ca_file = _parse_ca_file((ca_field.result or "").strip())
    if ca_file is None:
        return None
    return connection, ca_file


def _prompt_unix(
    current: Optional[Connection], form: Dict[str, str]
) -> Optional[Tuple[Connection, Optional[str]]]:
    """Ask the user for a Unix domain socket path and base URL.

    Args:
        current: The connection to prefill the fields from, or None for the
            defaults.
        form: The ``title`` and ``cancel_message`` for :py:func:`_run_form`.

    Returns:
        ``(connection, None)``, or None if the user cancels or gives an
        invalid answer.
    """
    path, base_url = INFERENCE_DEFAULT_SOCKET_PATH, DEFAULT_BASE_URL
    if current is not None:
        path, base_url = current["socket"], current["base_url"]

    path_field = TextLineField("Socket Path", path)
    base_url_field = TextLineField("Base URL", base_url)

    if not _run_form([path_field, base_url_field], **form):
        return None

    connection = _build_unix(
        (path_field.result or "").strip(), base_url_field.result or ""
    )
    if connection is None:
        return None
    return connection, None


def _prompt_details(
    kind: int, current: Optional[Connection], form: Dict[str, str]
) -> Optional[Tuple[Connection, Optional[str]]]:
    """Ask the user for the details of a connection of the chosen type.

    This is the second step of the connection dialogs: it only shows the
    fields that apply to ``kind``, since Binary Ninja forms cannot hide
    fields.

    Args:
        kind: An index into :py:data:`CONNECTION_KIND_CHOICES`.
        current: The current connection. Its values prefill the fields only
            if it is of the same kind (Unix domain socket, or HTTP/HTTPS).
        form: The ``title`` and ``cancel_message`` for :py:func:`_run_form`.

    Returns:
        ``(connection, ca_file)``, as returned by :py:func:`_prompt_tcp` and
        :py:func:`_prompt_unix`.
    """
    if current is not None and (current["kind"] == "unix") != (
        kind == CONNECTION_KIND_UNIX
    ):
        current = None
    if kind == CONNECTION_KIND_UNIX:
        return _prompt_unix(current, form)
    scheme = "https" if kind == CONNECTION_KIND_HTTPS else "http"
    return _prompt_tcp(scheme, current, form)


def _prompt_for_connection() -> Optional[Tuple[Connection, Optional[str]]]:
    """Ask the user for a new inference server connection.

    First asks for the connection type, then for the details of that type.

    Returns:
        ``(connection, ca_file)``, as returned by :py:func:`_prompt_details`.
        None if the user cancels or gives an invalid answer at either step.
    """
    kind_field = ChoiceField("Connection Type", CONNECTION_KIND_CHOICES)
    if not _run_form([kind_field]):
        return None

    return _prompt_details(kind_field.result, None, {})


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


def _save_connection(connection: Connection, ca_file: Optional[str]) -> None:
    """Cache the connection and persist it so it survives closing Binary
    Ninja.

    Args:
        connection: The connection to save.
        ca_file: The CA certificate file to save ("" for the system trust
            store), or None to leave the saved one unchanged.
    """
    setattr(binaryninja, CONNECTION_ATTR, connection)
    Settings().set_string(
        SETTINGS_KEY, json.dumps(connection), scope=SettingsScope.SettingsUserScope
    )
    if ca_file is not None:
        Settings().set_string(
            SETTINGS_KEY_CA_FILE, ca_file, scope=SettingsScope.SettingsUserScope
        )


def get_connection() -> Optional[Connection]:
    """Return the inference server connection.

    First it will try the in-process cache if present, else the persisted
    setting if present, else prompt the user and persist their answer.
    """
    connection = _cached_connection()
    if connection is not None:
        return connection

    connection = _load_connection()
    if connection is not None:
        setattr(binaryninja, CONNECTION_ATTR, connection)
        return connection

    details = _prompt_for_connection()
    if details is None:
        return None

    connection, ca_file = details

    _save_connection(connection, ca_file)
    clear_token()

    log_info(
        "Undertale's Inference Server connection configured: "
        f"{describe_server(connection)}"
    )

    return connection


def get_ca_file() -> Optional[str]:
    """Return the configured CA certificate file for HTTPS connections, or
    None to use the system trust store.
    """
    ca_file = Settings().get_string(SETTINGS_KEY_CA_FILE)
    return os.path.expanduser(ca_file) if ca_file else None


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


def _parse_base_url(base_url: str) -> Optional[str]:
    """Validate a base URL path prefix entered by the user and normalize it.

    A base URL is a bare path such as '/' or '/api', not a full URL. A value
    that carries a scheme or host (e.g. 'http://host/api'), or a query or
    fragment, is rejected rather than silently mangled into a nonsensical
    prefix.

    Normalizing guarantees a leading slash and no trailing slash so it can be
    joined directly with request paths that begin with '/'. An empty value or
    '/' normalizes to '/', meaning no prefix.

    Alerts the user and returns None on invalid input; otherwise returns the
    normalized prefix.
    """
    normalized = _normalize_base_url(base_url)
    if normalized is None:
        alert_user(
            f"Invalid base URL: {base_url.strip()!r}.\n\nEnter a bare path "
            "prefix such as '/' or '/api', not a full URL (no scheme, host, "
            f"query, or fragment).\n\nRe-run {CONFIGURE_MENU_PATH} to try again."
        )
    return normalized


def _normalize_base_url(base_url: str) -> Optional[str]:
    """Normalize a base URL path prefix, as :py:func:`_parse_base_url` does,
    without alerting the user.

    Returns None if ``base_url`` is not a bare path.
    """
    base_url = base_url.strip()
    parts = urlsplit(base_url)
    if parts.scheme or parts.netloc or parts.query or parts.fragment:
        return None
    if not base_url.startswith("/"):
        base_url = "/" + base_url
    return base_url.rstrip("/") or DEFAULT_BASE_URL


def join_base_url(base_url: str, path: str) -> str:
    """Prefix a request ``path`` (which must begin with '/') with a normalized
    base URL. When ``base_url`` is '/', the path is returned unchanged."""
    prefix = "" if base_url == DEFAULT_BASE_URL else base_url
    return f"{prefix}{path}"


def configure_plugin(bv: BinaryView) -> None:
    """Configure the Undertale Plugin.

    Runs in two steps, because Binary Ninja forms are static (no conditional
    fields): the first form asks for the connection type, the poll timeout,
    and whether to clear a saved login token; the second asks only for the
    details of the chosen connection type, prefilled from the current
    connection when it is of the same type.

    Nothing is persisted unless every value validates, so a cancel or a bad
    entry in either form leaves the existing configuration untouched. The
    saved token is discarded when the connection target (including its base
    URL) changes, when there was no valid saved connection to compare
    against, or when the user asks for it, so stale credentials never
    carry over to a different server.
    """
    current = _cached_connection() or _load_connection()

    kind_field = ChoiceField(
        "Connection Type",
        CONNECTION_KIND_CHOICES,
        default=CONNECTION_KIND_HTTP if current is None else _kind_choice(current),
    )
    timeout_field = TextLineField("Poll Timeout (seconds)", str(get_poll_timeout()))
    fields = [kind_field, timeout_field]

    clear_token_field = None
    if get_token() is not None:
        clear_token_field = ChoiceField("Clear saved login token", CLEAR_TOKEN_CHOICES)
        fields.append(clear_token_field)

    form = {
        "title": CONFIGURE_FORM_TITLE,
        "cancel_message": "Operation cancelled: the Undertale plugin was not configured.",
    }
    if not _run_form(fields, **form):
        return

    timeout = _parse_timeout((timeout_field.result or "").strip())
    if timeout is None:
        return

    details = _prompt_details(kind_field.result, current, form)
    if details is None:
        return
    connection, ca_file = details

    _save_connection(connection, ca_file)
    Settings().set_integer(
        SETTINGS_KEY_POLL_TIMEOUT, timeout, scope=SettingsScope.SettingsUserScope
    )

    clear_requested = (
        clear_token_field is not None and clear_token_field.result == CLEAR_TOKEN_YES
    )
    if clear_requested or current != connection:
        clear_token()

    log_info(
        f"Undertale plugin configured: server={describe_server(connection)}, "
        f"CA file={get_ca_file()!r}, poll timeout={timeout}s"
    )


_register_settings()
