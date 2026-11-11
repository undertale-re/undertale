"""Explaining failures talking to the Inference Server to the user, with hints
on how to fix them."""

import socket
import ssl

from binaryninja import log_info

from .client import (
    INFERENCE_CONNECT_TIMEOUT,
    AuthenticationRequired,
    Redirected,
    ResponseError,
    trust_source,
)
from .connection import Connection, describe_server
from .notify import alert_user
from .settings import CONFIGURE_MENU_PATH


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
            f"Timed out after {INFERENCE_CONNECT_TIMEOUT}s connecting to, or "
            "waiting for a response from, the Undertale Inference Server at "
            f"{endpoint}.\n\nCheck that the server "
            "is reachable and not overloaded, or reconfigure the connection "
            f"via {CONFIGURE_MENU_PATH}."
        )
    # ssl.SSLError and RemoteDisconnected are OSErrors, so these come first.
    elif isinstance(exc, ssl.SSLCertVerificationError):
        alert_user(
            "Could not verify the TLS certificate of the Undertale Inference "
            f"Server at {endpoint}: {exc.verify_message}.\n\nThe certificate "
            f"was checked against {trust_source()}.\n\nIf the server uses a "
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
