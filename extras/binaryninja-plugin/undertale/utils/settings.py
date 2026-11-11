"""The plugin's persisted settings: registering them, reading, and writing.

Settings are user-scoped, written to <User Directory>/settings.json, so they
survive closing Binary Ninja entirely. They are reconfigured by the command
named by :py:data:`CONFIGURE_COMMAND_NAME`.

The login token is also cached as an attribute parked on the `binaryninja`
module. `binaryninja` is not re-imported on "Reload Plugins", so this survives
reloads and avoids re-reading settings on every call, for the life of the
process.
"""

import json
import os
from typing import Optional

import binaryninja
from binaryninja import Settings
from binaryninja.enums import SettingsScope

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

TOKEN_ATTR = "_undertale_inference_token"


def _register_settings() -> None:
    """Register the plugin's settings.

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


def get_ca_file() -> Optional[str]:
    """Return the configured CA certificate file for HTTPS connections, or
    None to use the system trust store.
    """
    ca_file = Settings().get_string(SETTINGS_KEY_CA_FILE)
    return os.path.expanduser(ca_file) if ca_file else None


def save_ca_file(ca_file: str) -> None:
    """Persist the CA certificate file for HTTPS connections ("" for the
    system trust store)."""
    Settings().set_string(
        SETTINGS_KEY_CA_FILE, ca_file, scope=SettingsScope.SettingsUserScope
    )


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


def save_poll_timeout(timeout: int) -> None:
    """Persist the timeout, in seconds, to wait for the Inference Server to
    finish naming a function."""
    Settings().set_integer(
        SETTINGS_KEY_POLL_TIMEOUT, timeout, scope=SettingsScope.SettingsUserScope
    )


_register_settings()
