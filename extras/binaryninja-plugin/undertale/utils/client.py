"""A minimal HTTP client for the Undertale Inference Server's JSON API.

Supports plain HTTP, HTTPS, and Unix domain sockets (see
:py:mod:`.connection`), and logs in when the server requires authentication.

Design note: the Unix domain socket is handled by hand on top of the standard
library's http.client. httpx was explored as an alternative, since it supports
Unix sockets, but it brings additional dependencies. Those would have to be
vendored into the plugin, or installed into the Python interpreter Binary Ninja
uses, which can conflict with other plugins. The socket logic here is about 40
lines; httpx would shrink it to roughly 10-20. That isn't worth the extra
dependencies, so this stays as is. Revisit the vendoring strategy if the
plugin's dependencies grow.

HTTPS follows the same reasoning: it uses the standard library's ssl and
http.client.HTTPSConnection. The one optional extra is truststore, used only
if it is already importable, so the OS trust store (macOS Keychain, Linux
system store) works without a CA file. The plugin never vendors or installs
it; a configured CA certificate file is the dependency-free fix whenever the
default trust store does not work.
"""

import http.client
import ipaddress
import json
import os
import socket
import ssl
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Set, Tuple

try:
    import truststore  # optional; uses the OS trust store when installed
except ImportError:
    truststore = None  # type: ignore[assignment]

from binaryninja import log_info

from .connection import Connection, join_base_url
from .notify import alert_user
from .prompts import prompt_credentials
from .settings import (
    CONFIGURE_MENU_PATH,
    clear_token,
    get_ca_file,
    get_poll_timeout,
    get_token,
    save_token,
)

INFERENCE_CONNECT_TIMEOUT = 10
INFERENCE_POLL_INTERVAL = 1

UNAUTHORIZED_STATUSES = (401, 422)  # to handle missing or expired tokens


class UnixHTTPConnection(http.client.HTTPConnection):
    """An HTTPConnection that dials a Unix domain socket.

    Matches how the Undertale Inference Server's gunicorn deployment is
    bound (see its "Unauthenticated Local Service" documentation).
    """

    def __init__(
        self, path: str, timeout: Optional[float] = INFERENCE_CONNECT_TIMEOUT
    ) -> None:
        super().__init__("localhost", timeout=timeout)
        self.unix_socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None:
            self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_socket_path)


def trust_source() -> str:
    """Describe where HTTPS certificates are verified against, for
    user-facing messages. Follows the same precedence as
    :py:func:`_tls_context`."""
    ca_file = get_ca_file()
    if ca_file is not None:
        return f"the CA certificate file {ca_file!r}"
    if truststore is not None:
        return "the OS trust store (via truststore)"
    return "Python's default trust store"


_tls_cache: Optional[Tuple[Tuple[Optional[str], float], ssl.SSLContext]] = None


def _tls_context() -> ssl.SSLContext:
    """Return the TLS context for HTTPS connections to the inference server.

    The context is cached until the configured CA certificate file, or its
    modification time, changes.
    """
    global _tls_cache
    ca_file = get_ca_file()
    try:
        key = (ca_file, os.stat(ca_file).st_mtime if ca_file is not None else 0.0)
    except OSError:  # let _build_tls_context report the unreadable file
        return _build_tls_context(ca_file)
    if _tls_cache is None or _tls_cache[0] != key:
        _tls_cache = (key, _build_tls_context(ca_file))
    return _tls_cache[1]


def _build_tls_context(ca_file: Optional[str]) -> ssl.SSLContext:
    """Build the TLS context for HTTPS connections to the inference server.

    Certificates and hostnames are always verified. The trust source is, in
    order of precedence: the CA certificate file ``ca_file``, the OS trust
    store if truststore is importable, and Python's default trust store.
    """
    if ca_file is not None:
        try:
            context = ssl.create_default_context(cafile=ca_file)
        except OSError as error:  # includes ssl.SSLError
            raise RuntimeError(
                f"Unable to load the CA certificate file {ca_file!r}: {error}. "
                f"Fix or clear it via {CONFIGURE_MENU_PATH}"
            ) from error
    elif truststore is not None:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def _open_connection(connection: Connection) -> http.client.HTTPConnection:
    if connection["kind"] == "unix":
        return UnixHTTPConnection(connection["socket"])
    if connection["scheme"] == "https":
        return http.client.HTTPSConnection(
            connection["host"],
            connection["port"],
            timeout=INFERENCE_CONNECT_TIMEOUT,
            context=_tls_context(),
        )
    return http.client.HTTPConnection(
        connection["host"], connection["port"], timeout=INFERENCE_CONNECT_TIMEOUT
    )


def _is_loopback(host: str) -> bool:
    """Determine whether ``host`` names this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AuthenticationRequired(RuntimeError):
    """Raised when the inference server requires authentication and the user
    declined to provide credentials for it.

    This is a reason to stop rather than an error to surface as a failure.
    """


class Unauthorized(RuntimeError):
    """Raised when the inference server rejects a request's token (see
    :py:data:`UNAUTHORIZED_STATUSES`)."""


class NotFound(RuntimeError):
    """Raised when the inference server responds with 404."""


class ResponseError(RuntimeError):
    """Raised when the inference server responds with any other error status.

    Attributes:
        status: The HTTP status code.
        structured: Whether the body was a JSON object, as the inference
            server's own errors are. False suggests something else answered,
            such as a reverse proxy.
    """

    def __init__(self, message: str, status: int, structured: bool) -> None:
        super().__init__(message)
        self.status = status
        self.structured = structured


class Redirected(RuntimeError):
    """Raised when the inference server responds with a redirect (3xx).

    The plugin does not follow redirects: the inference server never sends
    them, so one means something else answered, such as a reverse proxy
    redirecting plain HTTP to HTTPS.

    Attributes:
        status: The HTTP status code.
        location: The redirect target from the Location header, if any.
    """

    def __init__(self, message: str, status: int, location: Optional[str]) -> None:
        super().__init__(message)
        self.status = status
        self.location = location


# Hosts already warned about receiving a token or credentials over plain HTTP.
_warned_hosts: Set[str] = set()


def _warn_plaintext(conn: http.client.HTTPConnection) -> None:
    """Warn, once per host per session, when a login token or credentials
    are about to be sent unencrypted to another machine."""
    if isinstance(conn, (http.client.HTTPSConnection, UnixHTTPConnection)):
        return
    if _is_loopback(conn.host) or conn.host in _warned_hosts:
        return
    _warned_hosts.add(conn.host)
    alert_user(
        f"The Undertale Inference Server at {conn.host} is configured over "
        "plain HTTP, so your login token and credentials are sent "
        "unencrypted.\n\nUse an HTTPS connection instead, via "
        f"{CONFIGURE_MENU_PATH}.",
        warning=True,
    )


def request(
    conn: http.client.HTTPConnection,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """One JSON request/response against the inference server API, over an
    already-open connection."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token is not None:
        _warn_plaintext(conn)
        headers["Authorization"] = f"Bearer {token}"
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()

    raw = response.read()

    if response.status >= 400:
        text = raw.decode("utf-8", "replace")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        structured = isinstance(parsed, dict)
        error = parsed.get("error", text) if structured else text
        message = f"{method} {path} -> {response.status}: {error}"
        if response.status in UNAUTHORIZED_STATUSES:
            raise Unauthorized(message)
        if response.status == 404:
            raise NotFound(message)
        raise ResponseError(message, response.status, structured)

    if 300 <= response.status < 400:
        location = response.getheader("Location")
        raise Redirected(
            f"{method} {path} -> {response.status}: redirected to {location!r}",
            response.status,
            location,
        )

    return json.loads(raw.decode("utf-8")) if raw else {}


def _login(conn: http.client.HTTPConnection, connection: Connection) -> str:
    """Prompt for credentials and log in to the inference server.

    Persists the resulting token so future requests skip the prompt until
    it's rejected or the connection is reconfigured.

    Raises AuthenticationRequired if the user cancels the credentials
    prompt.
    """
    credentials = prompt_credentials()
    if credentials is None:
        raise AuthenticationRequired("Login cancelled: no credentials provided")

    _warn_plaintext(conn)

    response = request(
        conn, "POST", join_base_url(connection["base_url"], "/login/"), credentials
    )
    token = response["token"]
    save_token(token)

    log_info("Logged in to the Undertale Inference Server")

    return token


def _authenticate(
    conn: http.client.HTTPConnection, connection: Connection
) -> Optional[str]:
    """Ensure requests on this connection are authenticated, if the
    inference server requires it.

    Uses the given, already-open connection, via :py:func:`request`. Tries a
    previously saved token first; if there isn't one, or the server rejects
    it, prompts for credentials and logs in.

    Returns the bearer token to attach to subsequent requests, or None if
    the server does not require authentication.
    """
    token = get_token()

    try:
        body = request(
            conn, "GET", join_base_url(connection["base_url"], "/"), token=token
        )
    except Unauthorized:
        if token is not None:
            clear_token()
        return _login(conn, connection)

    if not body.get("authentication"):
        return None
    return token


@contextmanager
def session(
    connection: Connection,
) -> Generator[Tuple[http.client.HTTPConnection, Optional[str]], None, None]:
    """Open an authenticated connection to the inference server.

    The connection is closed on exit.

    Args:
        connection: The inference server to connect to.

    Yields:
        ``(conn, token)``: the open connection and the bearer token (None if
        the server does not require authentication).
    """
    conn = _open_connection(connection)
    try:
        token = _authenticate(conn, connection)
        yield conn, token
    finally:
        conn.close()


def send(
    connection: Connection,
    token: Optional[str],
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send one request on a fresh connection, reusing an existing token.

    Skips the authentication check :py:func:`session` makes. Only if the
    server rejects ``token`` (for example, it expired) does this
    authenticate again and retry once.

    Args:
        connection: The inference server to connect to.
        token: A bearer token from an earlier :py:func:`session`, or None.
        method: The HTTP method.
        path: The request path, already joined with the base URL.
        body: The JSON body to send, if any.

    Returns:
        The decoded JSON response.
    """
    conn = _open_connection(connection)
    try:
        return request(conn, method, path, body, token=token)
    except Unauthorized:
        pass
    finally:
        conn.close()

    with session(connection) as (conn, token):
        return request(conn, method, path, body, token=token)


def request_name(connection: Connection, disassembly: str) -> Tuple[int, str]:
    """POST a function-naming completion, then poll until it's complete.

    Reuses a single connection across the POST and the entire poll loop
    rather than opening a fresh one for every request.

    Args:
        connection: The inference server to request the name from.
        disassembly: The function's pretokenized disassembly.

    Returns:
        ``(completion_id, name)``.
    """
    base_url = connection["base_url"]
    with session(connection) as (conn, token):
        created = request(
            conn,
            "POST",
            join_base_url(base_url, "/fnaming/completion/"),
            {"input": disassembly},
            token=token,
        )
        completion_id = created["id"]

        poll_timeout = get_poll_timeout()
        deadline = time.monotonic() + poll_timeout
        while True:
            completion = request(
                conn,
                "GET",
                join_base_url(base_url, f"/fnaming/completion/{completion_id}/"),
                token=token,
            )
            if completion.get("failed"):
                error = (
                    completion.get("output") or ""
                ).strip() or "the server reported an error"
                raise RuntimeError(
                    f"Inference server failed on completion {completion_id}: {error}"
                )
            if completion["completed"]:
                if not completion["output"]:
                    raise RuntimeError(
                        f"Completion {completion_id} finished with no output"
                    )
                return completion_id, completion["output"].strip()
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Completion {completion_id} did not finish within {poll_timeout}s"
                )
            time.sleep(INFERENCE_POLL_INTERVAL)
