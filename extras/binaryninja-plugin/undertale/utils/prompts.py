"""Dialogs that ask the user for input from a background task.

They all run their dialogs on the UI thread, since they are called from
background tasks.
"""

from typing import Any, Dict, List, Optional, Tuple

from binaryninja import (
    ChoiceField,
    LabelField,
    MultilineTextField,
    execute_on_main_thread_and_wait,
    get_form_input,
)
from binaryninja.enums import (
    MessageBoxButtonResult,
    MessageBoxButtonSet,
    MessageBoxIcon,
)
from binaryninja.interaction import show_message_box
from binaryninjaui import UIContext
from PySide6.QtWidgets import QInputDialog, QLineEdit

from .notify import alert_user

CREDENTIALS_FORM_TITLE = "Inference Server Credentials"
RATE_FORM_TITLE = "Rate Function Name"
RATING_CHOICES = ["Bad", "Fair", "Average", "Good", "Great"]


def prompt_credentials() -> Optional[Dict[str, str]]:
    """Ask the user for their inference server username and password.

    Uses Qt dialogs with a masked password field. The dialogs run on the UI
    thread because this is called from a background task.

    Returns None if the user cancels or leaves either field blank.
    """
    captured: Dict[str, str] = {}

    def prompt() -> None:
        context = UIContext.activeContext()
        if context is None:
            return
        parent = context.mainWindow()
        username, ok = QInputDialog.getText(
            parent, CREDENTIALS_FORM_TITLE, "Username:", QLineEdit.Normal
        )
        if not ok:
            return
        password, ok = QInputDialog.getText(
            parent, CREDENTIALS_FORM_TITLE, "Password:", QLineEdit.Password
        )
        if not ok:
            return
        captured["username"] = username
        captured["password"] = password

    execute_on_main_thread_and_wait(prompt)

    if not captured:  # user cancelled either dialog
        return None
    username = captured["username"].strip()
    password = captured["password"]
    if not username or not password:
        alert_user("Username and password are both required")
        return None
    return {"username": username, "password": password}


def confirm_plaintext(message: str) -> bool:
    """Ask the user whether to go ahead despite the warning in ``message``.

    Blocks until the user answers. The message box runs on the UI thread
    because this is called from a background task.

    Returns True only if the user answers Yes.
    """
    answers = []
    execute_on_main_thread_and_wait(
        lambda: answers.append(
            show_message_box(
                "Undertale",
                message,
                MessageBoxButtonSet.YesNoButtonSet,
                MessageBoxIcon.WarningIcon,
            )
        )
    )
    return bool(answers) and answers[0] == MessageBoxButtonResult.YesButton


def prompt_feedback(
    generated: str, rating: Optional[int], comments: str
) -> Optional[Tuple[int, str]]:
    """Ask the user to rate a generated name.

    The form runs on the UI thread because this is called from a background
    task.

    Args:
        generated: The name the server generated.
        rating: The rating already saved on the server, if any, as an index
            into ``RATING_CHOICES``.
        comments: The comments already saved on the server.

    Returns:
        ``(rating, comments)``, or None if the user cancels.
    """
    rating_field = ChoiceField("Rating", RATING_CHOICES, default=rating)
    comments_field = MultilineTextField("Comments", comments)
    fields: List[Any] = [
        LabelField(f"Generated name: {generated}"),
        rating_field,
        comments_field,
    ]

    submitted = []
    execute_on_main_thread_and_wait(
        lambda: submitted.append(get_form_input(fields, RATE_FORM_TITLE))
    )

    if not submitted or not submitted[0]:
        return None
    return int(rating_field.result), comments_field.result or ""
