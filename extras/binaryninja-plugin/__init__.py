"""Undertale Function Namer Plugin

Rename functions by performing inference on disassembly using the Undertale-trained model.

The plugin's workflow consists of the following steps
    1. After the user selects a function in Binary Ninja
    2. The plugin reads the function's disassembly and pretokenizes it into the
        form the model was trained on (see _disassembly.py)
    3. The pretokenized disassembly is sent to the inference server for analysis
    4. The plugin will poll the inference server for the function name prediction
    5. Finally, the predicted name will be applied to the function

The inference server connection is configured once and persists across
Binary Ninja restarts until explicitly reconfigured (see "Undertale >
Reconfigure inference server connection"). The supported endpoints include
    - TCP connections specified as host:port
    - Unix domain sockets specified by filesystem path
"""

import http.client
import json
import socket
import sys
import time

import binaryninja
from binaryninja import (
    BackgroundTaskThread,
    PluginCommand,
    log_debug,
    log_error,
    log_info,
)

# Only reached dynamically via `module.reconfigure_connection` in the
# reload-safe trampoline below, never referenced by name in this file.
from ._connection import reconfigure_connection  # noqa: F401
from ._connection import RECONFIGURE_COMMAND_NAME, get_connection
from ._disassembly import pretokenize_disassembly

FNAMING_COMMAND_NAME = "Undertale\\Infer and Rename Function"

INFERENCE_CONNECT_TIMEOUT = 10
INFERENCE_POLL_INTERVAL = 1
INFERENCE_POLL_TIMEOUT = 60


# --- HTTP transport, over TCP or a Unix domain socket ------------------------


class UnixHTTPConnection(http.client.HTTPConnection):
    """An HTTPConnection that dials a Unix domain socket instead of TCP.

    Matches how the inference server's gunicorn deployment is bound (see
    extras/inference-server/README.md, "Unauthenticated Local Service").
    """

    def __init__(self, path, timeout=INFERENCE_CONNECT_TIMEOUT):
        super().__init__("localhost", timeout=timeout)
        self.unix_socket_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None:
            self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_socket_path)


def _open_connection(connection):
    if connection["kind"] == "unix":
        return UnixHTTPConnection(connection["path"])
    return http.client.HTTPConnection(
        connection["host"], connection["port"], timeout=INFERENCE_CONNECT_TIMEOUT
    )


def request(connection, method, path, body=None):
    """One JSON request/response against the inference server API."""
    conn = _open_connection(connection)
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read()
    finally:
        conn.close()

    parsed = json.loads(raw.decode("utf-8")) if raw else {}
    if response.status >= 400:
        raise RuntimeError(
            f"{method} {path} -> {response.status}: {parsed.get('error', raw)}"
        )
    return parsed


# --- Step 1: read the function's disassembly --------------------------------


def function_disassembly(func):
    """Render the function's disassembly in the pretokenized form the model
    was trained on: one whitespace-separated token per mnemonic, register, or
    immediate, with commas and other formatting tokens dropped. This is what
    gets sent as the model input.

    Note: a function is not necessarily one contiguous range. Compilers move
    cold paths elsewhere, so walking func.start..func.highest_address can pull
    in instructions belonging to other functions. Iterating basic blocks
    avoids that.
    """
    blocks = sorted(func.basic_blocks, key=lambda b: b.start)
    tokens = pretokenize_disassembly(blocks)
    return " ".join(tokens)


# --- Steps 2-3: request a name from the inference server --------------------


def request_name(connection, disassembly):
    """POST a function-naming completion, then poll until it's complete."""
    created = request(
        connection, "POST", "/fnaming/completion/", {"input": disassembly}
    )
    completion_id = created["id"]

    deadline = time.monotonic() + INFERENCE_POLL_TIMEOUT
    while True:
        completion = request(connection, "GET", f"/fnaming/completion/{completion_id}/")
        if completion["completed"]:
            if not completion["output"]:
                raise RuntimeError(
                    f"completion {completion_id} finished with no output"
                )
            return completion["output"].strip()
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"completion {completion_id} did not finish within {INFERENCE_POLL_TIMEOUT}s"
            )
        time.sleep(INFERENCE_POLL_INTERVAL)


# --- Step 4: rename ---------------------------------------------------------


def rename(bv, func, name):
    """Rename as a single undoable action so one Ctrl+Z reverts it."""
    old = func.name
    try:
        state = bv.begin_undo_actions()
    except TypeError:
        state = None
    try:
        func.name = name
    finally:
        try:
            (
                bv.commit_undo_actions(state)
                if state is not None
                else bv.commit_undo_actions()
            )
        except Exception:  # noqa: BLE001
            pass
    log_info(f"renamed {old} -> {func.name} @ {hex(func.start)}")


# --- Wiring -----------------------------------------------------------------


class NameFunctionTask(BackgroundTaskThread):
    """Runs off the UI thread. Blocking HTTP calls on the main thread would
    freeze Binary Ninja for the duration."""

    def __init__(self, bv, func, connection):
        BackgroundTaskThread.__init__(self, f"Naming {func.name}...", True)
        self.bv = bv
        self.func = func
        self.connection = connection

    def run(self):
        try:
            disassembly = function_disassembly(self.func)
            if not disassembly:
                log_error(f"no disassembly available for {self.func.name}")
                return
            name = request_name(self.connection, disassembly)
            rename(self.bv, self.func, name)
        except (OSError, RuntimeError, ValueError) as exc:
            log_error(f"naming failed for {self.func.name}: {exc}")


def name_function(bv, func):
    """The real entry point. Edit freely — reloads pick this up (see below)."""
    connection = get_connection()
    if connection is None:
        return
    NameFunctionTask(bv, func, connection).start()


# --- Registration: exactly once per process ---------------------------------
#
# Every PluginCommand.register_* variant is documented as leaking the original
# plugin when called twice with the same name. "Reload Plugins" re-executes
# this module, so naive top-level registration leaks once per reload.
#
# A module-level `_registered = False` flag does NOT help: reloading re-runs
# the module and resets it. The `binaryninja` module object, however, is not
# reloaded, so an attribute parked on it survives and gives a genuine
# once-per-process guard.
#
# That alone would break the reload workflow, since the command would stay
# bound to the *original* callback and your edits would never take effect. The
# fix is to register a permanent shim that resolves the implementation by name
# at call time. Register once, dispatch late: no leak, and reloads still work.

_SENTINEL = "_undertale_inference_namer_registered"
_MODULE_NAME = __name__


def _dispatch_name_function(bv, func):
    """Late-bound trampoline. Looks up the current module on every invocation
    so a reloaded `name_function` is what actually runs."""
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        log_error(f"{_MODULE_NAME} is not loaded")
        return
    module.name_function(bv, func)


def _dispatch_reconfigure_connection(bv):
    """Late-bound trampoline for `reconfigure_connection`, for the same
    reload-safety reason as `_dispatch_name_function`."""
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        log_error(f"{_MODULE_NAME} is not loaded")
        return
    module.reconfigure_connection(bv)


def _register_once():
    if getattr(binaryninja, _SENTINEL, False):
        log_debug(f"{_MODULE_NAME}: already registered, skipping (reload)")
        return False
    PluginCommand.register_for_function(
        FNAMING_COMMAND_NAME,
        "Send the function's disassembly to the inference server and apply the returned name",
        _dispatch_name_function,
    )
    PluginCommand.register(
        RECONFIGURE_COMMAND_NAME,
        "Discard the saved inference server connection and prompt for a new one",
        _dispatch_reconfigure_connection,
    )
    setattr(binaryninja, _SENTINEL, True)
    log_debug(
        f"{_MODULE_NAME}: registered {FNAMING_COMMAND_NAME!r} and {RECONFIGURE_COMMAND_NAME!r}"
    )
    return True


_register_once()
