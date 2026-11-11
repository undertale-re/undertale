"""Surfacing plugin failures and warnings to the user."""

from binaryninja import execute_on_main_thread, log_error, log_warn
from binaryninja.enums import MessageBoxButtonSet, MessageBoxIcon
from binaryninja.interaction import show_message_box


def alert_user(message: str, warning: bool = False) -> None:
    """Show alerts both in the log and as a focus-stealing modal.

    A Qt message box must be created on the UI thread. The caller carries on
    without waiting for the user to dismiss it.

    NOTE: Sentences in ``message`` are separated by blank lines so the modal reads as
    stacked paragraphs; the log entry collapses them back to a single line.

    Args:
        message: The text to show.
        warning: Show a warning (warning icon, logged as a warning) instead
            of an error.
    """
    log = log_warn if warning else log_error
    log(message.replace("\n\n", " "))
    icon = MessageBoxIcon.WarningIcon if warning else MessageBoxIcon.ErrorIcon
    execute_on_main_thread(
        lambda: show_message_box(
            "Undertale",
            message,
            MessageBoxButtonSet.OKButtonSet,
            icon,
        )
    )
