"""Undertale Function Namer Plugin

Rename functions by performing inference on disassembly using the Undertale-trained model.

The plugin's workflow consists of the following steps
    1. After the user selects a function in Binary Ninja
    2. The plugin reads the function's disassembly and pretokenizes it into the
        form the model was trained on (see utils/disassembly.py)
    3. The pretokenized disassembly is sent to the inference server for analysis
    4. The plugin will poll the inference server for the function name prediction
    5. Finally, the predicted name will be applied to the function

The inference server connection is configured once and persists across
Binary Ninja restarts until explicitly reconfigured (see "Undertale >
Reconfigure Inference Server Connection"). The supported endpoints include
    - TCP connections specified as host:port
    - Unix domain sockets specified by filesystem path
"""

import http.client
import json
import socket
import time
from typing import Any, Dict, Optional

from binaryninja import (
    BackgroundTaskThread,
    BinaryView,
    Function,
    PluginCommand,
    log_error,
    log_info,
)

from .utils import (
    RECONFIGURE_COMMAND_NAME,
    RECONFIGURE_POLL_TIMEOUT_COMMAND_NAME,
    Connection,
    get_connection,
    get_poll_timeout,
    pretokenize_disassembly,
    reconfigure_connection,
    reconfigure_poll_timeout,
)

FNAMING_COMMAND_NAME = "Undertale\\Infer and Rename Function"

INFERENCE_CONNECT_TIMEOUT = 10
INFERENCE_POLL_INTERVAL = 1


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


def request(
    conn: http.client.HTTPConnection,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One JSON request/response against the inference server API, over an
    already-open connection."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()
    raw = response.read()

    if response.status >= 400:
        try:
            error = json.loads(raw.decode("utf-8")).get("error", raw)
        except ValueError:
            error = raw.decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} -> {response.status}: {error}")

    return json.loads(raw.decode("utf-8")) if raw else {}


def function_disassembly(func: Function) -> str:
    """Render and formats the function's disassembly.

    The disassembly is pretokenized in the form Undertale's model was trained
    on.

    NOTE: a function is not necessarily one contiguous range. Compilers move
    cold paths elsewhere, so walking func.start..func.highest_address can pull
    in instructions belonging to other functions. Iterating basic blocks
    avoids that.
    """
    blocks = sorted(func.basic_blocks, key=lambda b: b.start)
    tokens = pretokenize_disassembly(blocks)
    return " ".join(tokens)


def request_name(connection: Connection, disassembly: str) -> str:
    """POST a function-naming completion, then poll until it's complete.

    Reuses a single connection across the POST and the entire poll loop
    rather than opening a fresh one for every request.
    """
    conn = _open_connection(connection)
    try:
        created = request(conn, "POST", "/fnaming/completion/", {"input": disassembly})
        completion_id = created["id"]

        poll_timeout = get_poll_timeout()
        deadline = time.monotonic() + poll_timeout
        while True:
            completion = request(conn, "GET", f"/fnaming/completion/{completion_id}/")
            if completion["completed"]:
                if not completion["output"]:
                    raise RuntimeError(
                        f"Completion {completion_id} finished with no output"
                    )
                return completion["output"].strip()
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Completion {completion_id} did not finish within {poll_timeout}s"
                )
            time.sleep(INFERENCE_POLL_INTERVAL)
    finally:
        conn.close()


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
            log_error(f"failed to commit undo action for rename of {old}: {exc}")
    log_info(f"renamed {old} -> {func.name} @ {hex(func.start)}")


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
            disassembly = function_disassembly(self.func)
            if not disassembly:
                log_error(f"No disassembly available for {self.func.name}")
                return
            name = request_name(self.connection, disassembly)
            rename(self.bv, self.func, name)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Naming failed for {self.func.name}: {exc}")


def name_function(bv: BinaryView, func: Function) -> None:
    """The real entry point."""
    connection = get_connection()
    if connection is None:
        return
    NameFunctionTask(bv, func, connection).start()


PluginCommand.register_for_function(
    FNAMING_COMMAND_NAME,
    "Send the function's disassembly to the Undertale Inference Server and apply the predicted name",
    name_function,
)
PluginCommand.register(
    RECONFIGURE_COMMAND_NAME,
    "Discard the saved Inference Server connection and prompt for a new one",
    reconfigure_connection,
)
PluginCommand.register(
    RECONFIGURE_POLL_TIMEOUT_COMMAND_NAME,
    "Prompt for a new Inference Completion Poll Timeout",
    reconfigure_poll_timeout,
)
