"""Undertale: Binary Ninja Plugin

Rename functions by performing inference on disassembly using the Undertale-trained model.

The plugin's workflow consists of the following steps
    1. After the user selects a function in Binary Ninja
    2. The plugin reads the function's disassembly and pretokenizes it into the
        form the model was trained on (see utils/disassembly.py)
    3. The pretokenized disassembly is sent to the inference server for analysis
    4. The plugin will poll the inference server for the function name prediction
    5. Finally, the predicted name will be applied to the function
    6. Optionally, the user rates the generated name (see "Undertale > Rate
        Function Name"), and the rating is sent back to the inference server

The inference server connection is configured once and persists across
Binary Ninja restarts until explicitly reconfigured (see "Undertale >
Configure Plugin"). The supported endpoints include
    - HTTP connections specified as host:port
    - HTTPS connections specified as host:port, verified against the system
        trust store or a configured CA certificate file
    - Unix domain sockets specified by filesystem path
"""

import http.client
import ipaddress
import json
import os
import socket
import ssl
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import truststore  # optional; uses the OS trust store when installed
except ImportError:
    truststore = None  # type: ignore[assignment]

from binaryninja import (
    BackgroundTaskThread,
    BinaryView,
    ChoiceField,
    Function,
    LabelField,
    MultilineTextField,
    PluginCommand,
    execute_on_main_thread_and_wait,
    get_form_input,
    log_error,
    log_info,
)

from .utils import (
    CONFIGURE_COMMAND_NAME,
    CONFIGURE_MENU_PATH,
    Connection,
    alert_user,
    clear_token,
    configure_plugin,
    describe_server,
    get_ca_file,
    get_completion,
    get_connection,
    get_poll_timeout,
    get_token,
    is_same_server,
    join_base_url,
    pretokenize_disassembly,
    prompt_credentials,
    remove_completion,
    renamed,
    save_token,
    store_completion,
)

FNAMING_COMMAND_NAME = "Undertale\\Generate Function Name"
FNAMING_MENU_PATH = FNAMING_COMMAND_NAME.replace("\\", " > ")  # used for logging
RATE_COMMAND_NAME = "Undertale\\Rate Function Name"

RATE_FORM_TITLE = "Rate Function Name"
RATING_CHOICES = ["Bad", "Fair", "Average", "Good", "Great"]

INFERENCE_CONNECT_TIMEOUT = 10
INFERENCE_POLL_INTERVAL = 1

UNAUTHORIZED_STATUSES = (401, 422)  # to handle missing or expired tokens


# Design note: the Unix domain socket is handled by hand on top of the standard
# library's http.client. httpx was explored as an alternative, since it supports
# Unix sockets, but it brings additional dependencies. Those would have to be
# vendored into the plugin, or installed into the Python interpreter Binary Ninja
# uses, which can conflict with other plugins. The socket logic here is about 40
# lines; httpx would shrink it to roughly 10-20. That isn't worth the extra
# dependencies, so this stays as is. Revisit the vendoring strategy if the
# plugin's dependencies grow.
#
# HTTPS follows the same reasoning: it uses the standard library's ssl and
# http.client.HTTPSConnection. The one optional extra is truststore, used only
# if it is already importable, so the OS trust store (macOS Keychain, Linux
# system store) works without a CA file. The plugin never vendors or installs
# it; a configured CA certificate file is the dependency-free fix whenever the
# default trust store does not work.
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


def _trust_source() -> str:
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
_warned_hosts: set = set()


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
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            parsed = None
        structured = isinstance(parsed, dict)
        if structured:
            error = parsed.get("error", raw)
        else:
            error = raw.decode("utf-8", "replace")
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


def function_disassembly(bv: BinaryView, func: Function) -> str:
    """Render and formats the function's disassembly.

    The disassembly is pretokenized in the form Undertale's model was trained
    on.

    NOTE: a function is not necessarily one contiguous range. Compilers move
    cold paths elsewhere, so walking func.start..func.highest_address can pull
    in instructions belonging to other functions. Iterating basic blocks
    avoids that.

    To match training (see undertale/pipeline/binary.py), we undefine all data
    variables before disassembling so operands render as real addresses instead
    of expressions like ``jmp data_var[42]``. Leaving them defined would emit
    different immediate values than the model was trained on.
    """
    cached_data_vars = {
        address: variable.type for address, variable in bv.data_vars.items()
    }
    for address in cached_data_vars:
        bv.undefine_data_var(address, blacklist=True)
    try:
        blocks = sorted(func.basic_blocks, key=lambda b: b.start)
        tokens = pretokenize_disassembly(blocks)
    finally:
        for address, variable_type in cached_data_vars.items():
            bv.define_data_var(address, variable_type)
    return " ".join(tokens)


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


def rename(bv: BinaryView, func: Function, name: str) -> None:
    """Rename as a single undoable action so one Ctrl+Z reverts it."""
    old = func.name
    state = bv.begin_undo_actions()
    try:
        func.name = name
    finally:
        try:
            bv.commit_undo_actions(state)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Failed to commit undo action for rename of {old}: {exc}")
    log_info(f"Renamed {old} -> {func.name} @ {hex(func.start)}")


def report_failure(exc: Exception, action: str, connection: Connection) -> None:
    """Surface a failure talking to the inference server to the user.

    Shared by the background tasks so naming and rating report connection
    problems identically.

    Args:
        exc: The exception that stopped the task.
        action: What the task was doing, such as ``"Naming sub_401000"``.
        connection: The server's connection.
    """
    endpoint = describe_server(connection)
    plain_http = connection.get("scheme") == "http"
    if isinstance(exc, AuthenticationRequired):
        log_info(str(exc))
    elif isinstance(exc, ConnectionRefusedError):
        alert_user(
            f"Could not reach the Undertale Inference Server at {endpoint}: "
            "connection refused.\n\nMake sure the server is running and "
            "listening there.\n\nIf the address is wrong, run "
            f"{CONFIGURE_MENU_PATH}."
        )
    elif isinstance(exc, FileNotFoundError):
        alert_user(
            f"Could not reach the Undertale Inference Server at {endpoint}: "
            "the socket file does not exist.\n\nThe server may not be running, "
            "or the socket path is wrong.\n\nStart the server or run "
            f"{CONFIGURE_MENU_PATH}."
        )
    elif isinstance(exc, (TimeoutError, socket.timeout)):
        alert_user(
            f"Timed out after {INFERENCE_CONNECT_TIMEOUT}s connecting to the "
            f"Undertale Inference Server at {endpoint}.\n\nCheck that the server "
            "is reachable and not overloaded, or reconfigure the connection "
            f"via {CONFIGURE_MENU_PATH}."
        )
    # ssl.SSLError and RemoteDisconnected are OSErrors, so these come first.
    elif isinstance(exc, ssl.SSLCertVerificationError):
        alert_user(
            "Could not verify the TLS certificate of the Undertale Inference "
            f"Server at {endpoint}: {exc.verify_message}.\n\nThe certificate "
            f"was checked against {_trust_source()}.\n\nIf the server uses a "
            "self-signed certificate or a private certificate authority, set "
            f"its CA Certificate File via {CONFIGURE_MENU_PATH}. The "
            "certificate must also list the configured host in its "
            "subjectAltName."
        )
    elif isinstance(exc, ssl.SSLError):
        alert_user(
            "TLS handshake with the Undertale Inference Server at "
            f"{endpoint} failed: {exc.reason or exc}.\n\nIs the server actually "
            "serving HTTPS on this port?\n\nIf it serves plain HTTP, switch the "
            f"connection type to HTTP via {CONFIGURE_MENU_PATH}."
        )
    elif plain_http and isinstance(exc, ConnectionResetError):
        # A TLS server resets (or, via a proxy, closes) a plain HTTP connection,
        # which includes http.client.RemoteDisconnected. So does a server that
        # crashed or restarted mid-request, so this is only a hint.
        alert_user(
            f"The Undertale Inference Server at {endpoint} closed the "
            f"connection: {exc}.\n\nThe server may have crashed or restarted; "
            "check its logs.\n\nIf it serves HTTPS on this port, switch the "
            f"connection type to HTTPS via {CONFIGURE_MENU_PATH}."
        )
    elif (
        plain_http
        # A proxy's (not the inference server's) response to plain HTTP sent to
        # its HTTPS port, such as nginx's or Caddy's.
        and isinstance(exc, ResponseError)
        and exc.status == 400
        and not exc.structured
    ):
        alert_user(
            f"The Undertale Inference Server at {endpoint} did not answer the "
            "plain HTTP request.\n\nIf it serves HTTPS on this port, switch the "
            f"connection type to HTTPS via {CONFIGURE_MENU_PATH}."
        )
    elif (
        plain_http
        and isinstance(exc, Redirected)
        and (exc.location or "").lower().startswith("https://")
    ):
        alert_user(
            f"The server at {endpoint} redirects to HTTPS ({exc.location}).\n\n"
            "Switch the connection type to HTTPS via "
            f"{CONFIGURE_MENU_PATH}."
        )
    elif isinstance(exc, Redirected):
        target = exc.location or "an unknown location"
        alert_user(
            f"The server at {endpoint} redirected the request to {target} "
            f"(HTTP {exc.status}), and the plugin does not follow redirects."
            "\n\nThe Undertale Inference Server does not normally redirect, "
            "so something else may be answering. Check the host, port, and base URL via "
            f"{CONFIGURE_MENU_PATH}."
        )
    elif isinstance(exc, OSError):
        alert_user(
            f"Network error talking to the Undertale Inference Server at "
            f"{endpoint}: {exc}.\n\nVerify the connection, and reconfigure it "
            f"via {CONFIGURE_MENU_PATH} if needed."
        )
    else:
        alert_user(f"{action} failed (server at {endpoint}): {exc}")


class NameFunctionTask(BackgroundTaskThread):
    """Runs off the UI thread.

    Blocking HTTP calls on the main thread would freeze Binary Ninja for the
    duration.
    """

    def __init__(self, bv: BinaryView, func: Function, connection: Connection) -> None:
        BackgroundTaskThread.__init__(self, f"Naming {func.name}...", True)
        self.bv = bv
        self.func = func
        self.connection = connection

    def run(self) -> None:
        try:
            disassembly = function_disassembly(self.bv, self.func)
            if not disassembly:
                alert_user(f"No disassembly available for {self.func.name}")
                return
            completion_id, name = request_name(self.connection, disassembly)
            # Forget any earlier completion first, so it can't be rated against
            # the new name if recording this one fails.
            remove_completion(self.func)
            rename(self.bv, self.func, name)
        except Exception as exc:  # noqa: BLE001
            report_failure(exc, f"Naming {self.func.name}", self.connection)
            return

        # The rename already happened, so a failure here only affects rating.
        try:
            store_completion(self.func, completion_id, name, self.connection)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Failed to record completion {completion_id}: {exc}")
            alert_user(
                f"Renamed {self.func.name}, but could not record the completion, "
                f"so it can't be rated: {exc}"
            )


def prompt_feedback(
    generated: str, rating: Optional[int], comments: str
) -> Optional[Tuple[int, str]]:
    """Ask the user to rate a generated name.

    The form runs on the UI thread because this is called from a background
    task.

    Args:
        generated: The name the server generated.
        rating: The rating already saved on the server, if any, as an index
            into ``RATING_CHOICES``.
        comments: The comments already saved on the server.

    Returns:
        ``(rating, comments)``, or None if the user cancels.
    """
    rating_field = ChoiceField("Rating", RATING_CHOICES, default=rating)
    comments_field = MultilineTextField("Comments", comments)
    fields: List[Any] = [
        LabelField(f"Generated name: {generated}"),
        rating_field,
        comments_field,
    ]

    submitted = []
    execute_on_main_thread_and_wait(
        lambda: submitted.append(get_form_input(fields, RATE_FORM_TITLE))
    )

    if not submitted or not submitted[0]:
        return None
    return int(rating_field.result), comments_field.result or ""


class RateFunctionTask(BackgroundTaskThread):
    """Sends the user's rating of a generated name to the inference server.

    Runs off the UI thread, like :py:class:`NameFunctionTask`. Feedback is only
    sent to the server that generated the name, and only after confirming the
    completion there still matches the name it produced.
    """

    def __init__(self, func: Function, connection: Connection) -> None:
        BackgroundTaskThread.__init__(self, f"Rating {func.name}...", True)
        self.func = func
        self.connection = connection

    def run(self) -> None:
        try:
            self.rate()
        except Exception as exc:  # noqa: BLE001
            report_failure(exc, f"Rating {self.func.name}", self.connection)

    def rate(self) -> None:
        entry = get_completion(self.func)
        if entry is None:
            alert_user(f"Undertale did not name {self.func.name}, so it can't be rated")
            return
        if renamed(self.func, entry):
            alert_user(
                f"{self.func.name} was renamed or its rename was undone after "
                f"Undertale named it {entry['name']!r}, so it can't be rated."
            )
            return

        if not is_same_server(entry, self.connection):
            alert_user(
                "The name of this function was generated by the Undertale "
                f"Inference Server at {describe_server(entry['server'])}, "
                "but the plugin is configured for "
                f"{describe_server(self.connection)}.\n\nRatings can only "
                "be sent to the server that generated the name.\n\nTo rate it, "
                f"point the plugin back at that server via {CONFIGURE_MENU_PATH}."
            )
            return

        path = join_base_url(
            self.connection["base_url"], f"/fnaming/completion/{entry['id']}/"
        )
        feedback_path = path + "feedback/"
        with session(self.connection) as (conn, token):
            try:
                completion = request(conn, "GET", path, token=token)
            except NotFound:
                completion = None
        if (
            completion is None
            or (completion.get("output") or "").strip() != entry["name"]
        ):
            alert_user(
                f"The completion that named this function (id {entry['id']}) "
                "could not be found on the Undertale Inference Server at "
                f"{describe_server(self.connection)}, or it no longer "
                "matches the generated name.\n\nNo rating was sent.\n\nTo rate "
                f"this function, run {FNAMING_MENU_PATH} again to get a new "
                "name, then rate it."
            )
            return

        feedback = prompt_feedback(
            entry["name"],
            completion.get("rating"),
            completion.get("comments") or "",
        )
        if feedback is None:
            log_info(f"Rating cancelled for {self.func.name}")
            return
        rating, comments = feedback

        # The form may have been open for a while, so reconnect.
        send(
            self.connection,
            token,
            "POST",
            feedback_path,
            {"rating": rating, "comments": comments.strip() or None},
        )
        log_info(
            f"Rated {self.func.name} @ {hex(self.func.start)} as "
            f"{RATING_CHOICES[rating]!r} (completion {entry['id']})"
        )


def name_function(bv: BinaryView, func: Function) -> None:
    """The real plugin entry point."""
    connection = get_connection()
    if connection is None:
        return
    NameFunctionTask(bv, func, connection).start()


def rate_function(bv: BinaryView, func: Function) -> None:
    """Plugin entry point for rating a name Undertale generated."""
    connection = get_connection()
    if connection is None:
        return
    RateFunctionTask(func, connection).start()


def is_rateable(bv: BinaryView, func: Function) -> bool:
    """Only functions that still have the name Undertale gave them can be
    rated."""
    entry = get_completion(func)
    return entry is not None and not renamed(func, entry)


PluginCommand.register_for_function(
    FNAMING_COMMAND_NAME,
    "Send the function's disassembly to the Undertale Inference Server and apply the predicted name",
    name_function,
)
PluginCommand.register_for_function(
    RATE_COMMAND_NAME,
    "Rate the name the Undertale Inference Server generated for this function",
    rate_function,
    is_valid=is_rateable,
)
PluginCommand.register(
    CONFIGURE_COMMAND_NAME,
    "Configure the Inference Server connection, poll timeout, and saved login token",
    configure_plugin,
)
