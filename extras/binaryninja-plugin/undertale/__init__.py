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
    - TCP connections specified as host:port
    - Unix domain sockets specified by filesystem path
"""

import http.client
import json
import socket
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

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
    get_base_url,
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


def _open_connection(connection: Connection) -> http.client.HTTPConnection:
    if connection["kind"] == "unix":
        return UnixHTTPConnection(connection["path"])
    return http.client.HTTPConnection(
        connection["host"], connection["port"], timeout=INFERENCE_CONNECT_TIMEOUT
    )


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
        headers["Authorization"] = f"Bearer {token}"
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()

    raw = response.read()

    if response.status >= 400:
        try:
            error = json.loads(raw.decode("utf-8")).get("error", raw)
        except ValueError:
            error = raw.decode("utf-8", "replace")
        if response.status in UNAUTHORIZED_STATUSES:
            raise Unauthorized(f"{method} {path} -> {response.status}: {error}")
        if response.status == 404:
            raise NotFound(f"{method} {path} -> 404")
        raise RuntimeError(f"{method} {path} -> {response.status}: {error}")

    return json.loads(raw.decode("utf-8")) if raw else {}


def _login(conn: http.client.HTTPConnection, base_url: str) -> str:
    """Prompt for credentials and log in to the inference server.

    Persists the resulting token so future requests skip the prompt until
    it's rejected or the connection is reconfigured.

    Raises AuthenticationRequired if the user cancels the credentials
    prompt.
    """
    credentials = prompt_credentials()
    if credentials is None:
        raise AuthenticationRequired("Login cancelled: no credentials provided")

    response = request(conn, "POST", join_base_url(base_url, "/login/"), credentials)
    token = response["token"]
    save_token(token)

    log_info("Logged in to the Undertale Inference Server")

    return token


def _authenticate(conn: http.client.HTTPConnection, base_url: str) -> Optional[str]:
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
        body = request(conn, "GET", join_base_url(base_url, "/"), token=token)
    except Unauthorized:
        if token is not None:
            clear_token()
        return _login(conn, base_url)

    if not body.get("authentication"):
        return None
    return token


@contextmanager
def session(
    connection: Connection,
) -> Generator[Tuple[http.client.HTTPConnection, str, Optional[str]], None, None]:
    """Open an authenticated connection to the inference server.

    The connection is closed on exit.

    Args:
        connection: The inference server to connect to.

    Yields:
        ``(conn, base_url, token)``: the open connection, the configured base
        URL, and the bearer token (None if the server does not require
        authentication).
    """
    base_url = get_base_url()
    conn = _open_connection(connection)
    try:
        token = _authenticate(conn, base_url)
        yield conn, base_url, token
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

    with session(connection) as (conn, _, token):
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


def request_name(connection: Connection, disassembly: str) -> Tuple[int, str, str]:
    """POST a function-naming completion, then poll until it's complete.

    Reuses a single connection across the POST and the entire poll loop
    rather than opening a fresh one for every request.

    Args:
        connection: The inference server to request the name from.
        disassembly: The function's pretokenized disassembly.

    Returns:
        ``(completion_id, name, base_url)``, where ``base_url`` is the one the
        completion was requested under.
    """
    with session(connection) as (conn, base_url, token):
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
                return completion_id, completion["output"].strip(), base_url
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


def report_failure(exc: Exception, action: str, endpoint: str) -> None:
    """Surface a failure talking to the inference server to the user.

    Shared by the background tasks so naming and rating report connection
    problems identically.

    Args:
        exc: The exception that stopped the task.
        action: What the task was doing, such as ``"Naming sub_401000"``.
        endpoint: A description of the server, from
            :py:func:`describe_server`.
    """
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
        endpoint = describe_server(self.connection)
        try:
            disassembly = function_disassembly(self.bv, self.func)
            if not disassembly:
                alert_user(f"No disassembly available for {self.func.name}")
                return
            completion_id, name, base_url = request_name(self.connection, disassembly)
            # Forget any earlier completion first, so it can't be rated against
            # the new name if recording this one fails.
            remove_completion(self.func)
            rename(self.bv, self.func, name)
        except Exception as exc:  # noqa: BLE001
            report_failure(exc, f"Naming {self.func.name}", endpoint)
            return

        # The rename already happened, so a failure here only affects rating.
        try:
            store_completion(self.func, completion_id, name, self.connection, base_url)
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
        endpoint = describe_server(self.connection)
        try:
            self.rate()
        except Exception as exc:  # noqa: BLE001
            report_failure(exc, f"Rating {self.func.name}", endpoint)

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

        base_url = get_base_url()
        if not is_same_server(entry, self.connection, base_url):
            server = entry["server"]
            alert_user(
                "The name of this function was generated by the Undertale "
                "Inference Server at "
                f"{describe_server(server['connection'], server['base_url'])}, "
                "but the plugin is configured for "
                f"{describe_server(self.connection, base_url)}.\n\nRatings can only "
                "be sent to the server that generated the name.\n\nTo rate it, "
                f"point the plugin back at that server via {CONFIGURE_MENU_PATH}."
            )
            return

        path = join_base_url(base_url, f"/fnaming/completion/{entry['id']}/")
        feedback_path = path + "feedback/"
        with session(self.connection) as (conn, _, token):
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
                f"{describe_server(self.connection, base_url)}, or it no longer "
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
