"""Configuring the plugin: the Inference Server connection, the CA
certificate file, the poll timeout, and the saved login token.

The connection is asked for on first use by :py:func:`get_connection`, and can
be reconfigured at any time via the command named by
:py:data:`~.settings.CONFIGURE_COMMAND_NAME`, which runs
:py:func:`configure_plugin`.
"""

import os
import socket
import ssl
import stat
from typing import Any, Dict, List, Optional, Tuple

from binaryninja import (
    BinaryView,
    ChoiceField,
    TextLineField,
    get_form_input,
    log_info,
    log_warn,
)

from .connection import (
    DEFAULT_BASE_URL,
    Connection,
    describe_server,
    normalize_base_url,
    save_connection,
    stored_connection,
)
from .notify import alert_user
from .settings import (
    CONFIGURE_MENU_PATH,
    MAX_POLL_TIMEOUT,
    MIN_POLL_TIMEOUT,
    clear_token,
    get_ca_file,
    get_poll_timeout,
    get_token,
    save_ca_file,
    save_poll_timeout,
)

CONNECTION_FORM_TITLE = "Inference Server Connection"
CONFIGURE_FORM_TITLE = "Configure Undertale Plugin"
CONNECTION_KIND_CHOICES = ["HTTP", "HTTPS", "Unix Domain Socket"]
CONNECTION_KIND_HTTP, CONNECTION_KIND_HTTPS, CONNECTION_KIND_UNIX = range(
    len(CONNECTION_KIND_CHOICES)
)
CLEAR_TOKEN_CHOICES = ["No", "Yes"]
CLEAR_TOKEN_NO, CLEAR_TOKEN_YES = range(len(CLEAR_TOKEN_CHOICES))

INFERENCE_DEFAULT_HOST = "127.0.0.1"
INFERENCE_DEFAULT_PORT = "8000"
INFERENCE_DEFAULT_TLS_PORT = "443"
INFERENCE_DEFAULT_SOCKET_PATH = "/path/to/undertale-inference.sock"

UNIX_SOCKET_VALIDATION_TIMEOUT = 10

# A connection and the CA certificate file to save with it ("" for the system
# trust store), or None to leave the saved one unchanged.
Details = Tuple[Connection, Optional[str]]


def _validate_unix_socket_path(path: str) -> bool:
    """Check that ``path`` exists, is a Unix domain socket, and accepts a
    connection.

    Alerts the user and returns False if any of these checks fail.
    """
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        alert_user(
            f"Unix domain socket does not exist: {path!r}.\n\nThe Inference Server "
            "may not be running, or the path is wrong.\n\nStart the server, then "
            f"re-run {CONFIGURE_MENU_PATH} and enter the correct socket path."
        )
        return False
    except OSError as error:
        alert_user(
            f"Unable to access Unix domain socket {path!r}: {error}.\n\nCheck the "
            f"path's permissions, then re-run {CONFIGURE_MENU_PATH}."
        )
        return False

    if not stat.S_ISSOCK(mode):
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


def _parse_base_url(base_url: str) -> Optional[str]:
    """Validate a base URL path prefix entered by the user and normalize it.

    A value that carries a scheme or host (e.g. 'http://host/api'), or a query
    or fragment, is rejected rather than silently mangled into a nonsensical
    prefix. See :py:func:`~.connection.normalize_base_url`.

    Alerts the user and returns None on invalid input; otherwise returns the
    normalized prefix.
    """
    normalized = normalize_base_url(base_url)
    if normalized is None:
        alert_user(
            f"Invalid base URL: {base_url.strip()!r}.\n\nEnter a bare path "
            "prefix such as '/' or '/api', not a full URL (no scheme, host, "
            f"query, or fragment).\n\nRe-run {CONFIGURE_MENU_PATH} to try again."
        )
    return normalized


def _build_tcp(
    scheme: str, host: str, port: str, base_url: str
) -> Optional[Connection]:
    """Validate a host, port, and base URL and build an HTTP or HTTPS
    connection dict.

    A bracketed IPv6 address (e.g. ``[::1]``) is stored without its brackets,
    since http.client expects the bare address.

    Alerts the user and returns None if any of them is invalid.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if not host or not port.isdecimal() or not 0 < int(port) < 65536:
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

    Alerts the user and returns None if either is empty or unusable.
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
    returns the path as entered (:py:func:`~.settings.get_ca_file` expands
    it), or "" for the system trust store.
    """
    if not path:
        return ""

    try:
        ssl.create_default_context(cafile=os.path.expanduser(path))
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


def _parse_timeout(timeout: str) -> Optional[int]:
    """Validate a poll timeout entered as a string, in seconds.

    Alerts the user and returns None if it is not a whole number between
    :py:data:`~.settings.MIN_POLL_TIMEOUT` and
    :py:data:`~.settings.MAX_POLL_TIMEOUT`.
    """
    if (
        not timeout.isdecimal()
        or not MIN_POLL_TIMEOUT <= int(timeout) <= MAX_POLL_TIMEOUT
    ):
        alert_user(
            f"Invalid poll timeout: {timeout!r}.\n\nEnter a whole number of "
            f"seconds from {MIN_POLL_TIMEOUT} to {MAX_POLL_TIMEOUT}.\n\nRe-run "
            f"{CONFIGURE_MENU_PATH} to try again."
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
) -> Optional[Details]:
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
) -> Optional[Details]:
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
) -> Optional[Details]:
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


def _prompt_for_connection() -> Optional[Details]:
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


def _save_details(connection: Connection, ca_file: Optional[str]) -> None:
    """Save a connection, and its CA certificate file unless ``ca_file`` is
    None."""
    save_connection(connection)
    if ca_file is not None:
        save_ca_file(ca_file)


def get_connection() -> Optional[Connection]:
    """Return the inference server connection.

    First it will try the saved connection if present, else prompt the user
    and persist their answer.
    """
    connection = stored_connection()
    if connection is not None:
        return connection

    details = _prompt_for_connection()
    if details is None:
        return None

    connection, ca_file = details

    _save_details(connection, ca_file)
    clear_token()

    log_info(
        "Undertale's Inference Server connection configured: "
        f"{describe_server(connection)}"
    )

    return connection


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
    current = stored_connection()

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

    _save_details(connection, ca_file)
    save_poll_timeout(timeout)

    clear_requested = (
        clear_token_field is not None and clear_token_field.result == CLEAR_TOKEN_YES
    )
    if clear_requested or current != connection:
        clear_token()

    log_info(
        f"Undertale plugin configured: server={describe_server(connection)}, "
        f"CA file={get_ca_file()!r}, poll timeout={timeout}s"
    )
