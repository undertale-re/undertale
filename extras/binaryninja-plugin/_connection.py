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
from typing import Any, Dict, Optional

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

INFERENCE_DEFAULT_ADDRESS = "127.0.0.1:5000"

RECONFIGURE_COMMAND_NAME = "Undertale\\Reconfigure Inference Server Connection"

CONNECTION_ATTR = "_undertale_inference_connection"
SETTINGS_GROUP = "undertale"
SETTINGS_KEY = "undertale.inferenceServerConnection"

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
            f"Invalid stored connection for the Undertale's inference server: {connection_data!r}"
        )
        return None


def _prompt_for_connection() -> Optional[Connection]:
    """Ask the user for a new inference server connection.

    Returns None if the user cancels or gives an invalid address.
    """
    kind_field = ChoiceField(
        "Connection type", ["TCP (host:port)", "Unix domain socket"]
    )
    prompt = f"Address, e.g. {INFERENCE_DEFAULT_ADDRESS} or /path/to/undertale-inference.sock"
    try:
        address_field = TextLineField(prompt, INFERENCE_DEFAULT_ADDRESS)
    except TypeError:
        address_field = TextLineField(prompt)

    if not get_form_input([kind_field, address_field], "Inference Server Connection"):
        log_error("inference server connection not configured; cancelled")
        return None

    address = (address_field.result or "").strip()
    if not address:
        log_error("no address given")
        return None

    if kind_field.result == 0:
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            log_error(f"invalid host:port: {address!r}")
            return None
        return {"kind": "tcp", "host": host, "port": int(port)}
    return {"kind": "unix", "path": address}


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


def reconfigure_connection(bv: BinaryView) -> None:
    """Discard the cached and persisted connection, then immediately prompt
    for a new one."""
    if hasattr(binaryninja, CONNECTION_ATTR):
        delattr(binaryninja, CONNECTION_ATTR)
    _clear_saved_connection()
    get_connection()


_register_settings()

__all__ = ["RECONFIGURE_COMMAND_NAME", "get_connection", "reconfigure_connection"]
