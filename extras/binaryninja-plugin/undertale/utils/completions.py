"""Track which Inference Server completion named each function.

When Undertale names a function, it records the completion ID and the name the
server suggested. These records live in the BinaryView's metadata under
``METADATA_KEY``, keyed by the function's start address, so they are saved in
the ``.bndb`` along with the rest of the analysis.

Each record also stores the server that produced the completion. Feedback on a
name is only ever sent to that server, since completion IDs mean nothing to any
other server.
"""

from typing import Any, Dict, Optional

from binaryninja import BinaryView, Function

from .connection import DEFAULT_BASE_URL, Connection

METADATA_KEY = "undertale.completions"


def _load_completions(bv: BinaryView) -> Dict[str, Any]:
    try:
        completions = bv.query_metadata(METADATA_KEY)
    except KeyError:
        return {}
    return dict(completions) if isinstance(completions, dict) else {}


def store_completion(
    bv: BinaryView,
    func: Function,
    completion_id: int,
    name: str,
    connection: Connection,
    base_url: str,
) -> None:
    """Record the completion that named ``func``, replacing any earlier one.

    ``name`` is the server's output rather than ``func.name``, since Binary
    Ninja may sanitize the name it applies.
    """
    completions = _load_completions(bv)
    completions[hex(func.start)] = {
        "id": completion_id,
        "name": name,
        "server": {"connection": connection, "base_url": base_url},
    }
    bv.store_metadata(METADATA_KEY, completions)


def get_completion(bv: BinaryView, func: Function) -> Optional[Dict[str, Any]]:
    """Return the completion that named ``func``, or None if Undertale never
    named it."""
    return _load_completions(bv).get(hex(func.start))


def is_same_server(
    entry: Dict[str, Any], connection: Connection, base_url: str
) -> bool:
    """Determines whether ``entry`` came from the server at ``connection`` and
    ``base_url``."""
    server = entry.get("server", {})
    return server.get("connection") == connection and server.get("base_url") == base_url


def describe_server(connection: Connection, base_url: Optional[str] = None) -> str:
    """User friendly description of an Inference Server endpoint, for
    messages."""
    if connection["kind"] == "unix":
        description = f"the Unix domain socket {connection['path']!r}"
    else:
        description = f"{connection['host']}:{connection['port']} (TCP)"
    if base_url is not None and base_url != DEFAULT_BASE_URL:
        description += f" with base URL {base_url!r}"
    return description
