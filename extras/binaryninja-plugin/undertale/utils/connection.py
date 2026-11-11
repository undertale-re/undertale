"""Undertale's Inference Server connection: validating, describing, caching,
and persisting it.

The connection (HTTP or HTTPS host:port, or a Unix domain socket path, plus a
base URL) is configured once and persists across Binary Ninja restarts until
explicitly reconfigured via the command named by
:py:data:`~.settings.CONFIGURE_COMMAND_NAME`.

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
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import binaryninja
from binaryninja import Settings, log_warn
from binaryninja.enums import SettingsScope

from .settings import SETTINGS_KEY, clear_token

DEFAULT_BASE_URL = "/"

CONNECTION_ATTR = "_undertale_inference_connection"

# The keys every stored connection of each kind has, besides "kind".
CONNECTION_KEYS = {
    "tcp": {"scheme", "host", "port", "base_url"},
    "unix": {"socket", "base_url"},
}
SCHEME_DEFAULT_PORTS = {"http": 80, "https": 443}

Connection = Dict[str, Any]


def is_valid_connection(value: Any) -> bool:
    """Determine whether ``value`` is a valid connection."""
    if not isinstance(value, dict):
        return False

    keys = CONNECTION_KEYS.get(str(value.get("kind")))
    if keys is None or set(value) - {"kind"} != keys:
        return False

    # Only accept a base URL already in the form the configure form saves.
    base_url = value["base_url"]
    if not isinstance(base_url, str) or normalize_base_url(base_url) != base_url:
        return False

    if value["kind"] == "unix":
        return isinstance(value["socket"], str) and bool(value["socket"])

    port = value["port"]
    return (
        isinstance(value["scheme"], str)
        and value["scheme"] in SCHEME_DEFAULT_PORTS
        and isinstance(value["host"], str)
        and bool(value["host"])
        and isinstance(port, int)
        and not isinstance(port, bool)
        and 0 < port < 65536
    )


def normalize_base_url(base_url: str) -> Optional[str]:
    """Normalize a base URL path prefix.

    A base URL is a bare path such as '/' or '/api', not a full URL.
    Normalizing guarantees a leading slash and no trailing slash so it can be
    joined directly with request paths that begin with '/'. An empty value or
    '/' normalizes to '/', meaning no prefix.

    Returns None if ``base_url`` is not a bare path (it carries a scheme,
    host, query, or fragment).
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


def stored_connection() -> Optional[Connection]:
    """Return the saved connection, without prompting the user.

    First it will try the in-process cache if present, else the persisted
    setting if present (caching it), else None.
    """
    connection = _cached_connection()
    if connection is not None:
        return connection

    connection = _load_connection()
    if connection is not None:
        setattr(binaryninja, CONNECTION_ATTR, connection)
    return connection


def _cached_connection() -> Optional[Connection]:
    """Return the in-process cached connection, discarding it if it is not
    valid."""
    connection = getattr(binaryninja, CONNECTION_ATTR, None)
    if connection is None or is_valid_connection(connection):
        return connection

    delattr(binaryninja, CONNECTION_ATTR)
    return None


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


def save_connection(connection: Connection) -> None:
    """Cache the connection and persist it so it survives closing Binary
    Ninja."""
    setattr(binaryninja, CONNECTION_ATTR, connection)
    Settings().set_string(
        SETTINGS_KEY, json.dumps(connection), scope=SettingsScope.SettingsUserScope
    )
