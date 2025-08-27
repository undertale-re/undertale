"""Miscellaneous utilities."""

import logging
import signal

logger = logging.getLogger(__name__)


DEFAULT_SIGINT = signal.getsignal(signal.SIGINT)


def suppress_sigint() -> None:
    """Replace the usual SIGINT (Ctrl-C) to exit gracefully.

    This just suppresses the stack trace and prints an error message instead.
    """

    def handle(signal, frame):
        logger.critical("interrupted, exiting")
        exit(1)

    signal.signal(signal.SIGINT, handle)


def allow_sigint() -> None:
    """Enable the default SIGINT (Ctrl-C) handler.

    This may be called after `suppress_sigint()` to enable the default handler
    again.
    """

    signal.signal(signal.SIGINT, DEFAULT_SIGINT)
