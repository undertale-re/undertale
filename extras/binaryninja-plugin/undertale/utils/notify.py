"""Surfacing plugin failures to the user."""

from binaryninja import execute_on_main_thread, log_error
from binaryninja.enums import MessageBoxButtonSet, MessageBoxIcon
from binaryninja.interaction import show_message_box


def alert_user(message: str) -> None:
    """Show alerts both in the log and as a focus-stealing modal.

    A Qt message box must be created on the UI thread.

    NOTE: Sentences in ``message`` are separated by blank lines so the modal reads as
    stacked paragraphs; the log entry collapses them back to a single line.
    """
    log_error(message.replace("\n\n", " "))
    execute_on_main_thread(
        lambda: show_message_box(
            "Undertale",
            message,
            MessageBoxButtonSet.OKButtonSet,
            MessageBoxIcon.ErrorIcon,
        )
    )
