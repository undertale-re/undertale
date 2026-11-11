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

from binaryninja import (
    BackgroundTaskThread,
    BinaryView,
    Function,
    PluginCommand,
    log_error,
    log_info,
)

from .utils import (
    CONFIGURE_COMMAND_NAME,
    CONFIGURE_MENU_PATH,
    RATING_CHOICES,
    Connection,
    NotFound,
    alert_user,
    configure_plugin,
    describe_server,
    get_completion,
    get_connection,
    is_same_server,
    join_base_url,
    pretokenize_disassembly,
    prompt_feedback,
    remove_completion,
    renamed,
    report_failure,
    request,
    request_name,
    send,
    session,
    store_completion,
)

FNAMING_COMMAND_NAME = "Undertale\\Generate Function Name"
FNAMING_MENU_PATH = FNAMING_COMMAND_NAME.replace("\\", " > ")  # used for logging
RATE_COMMAND_NAME = "Undertale\\Rate Function Name"


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
