import re
from typing import Optional

__all__ = ["sanitize"]

SPECIAL_TOKEN = re.compile(r"(\[[A-Z]+\])")


def sanitize(text: Optional[str]) -> str:
    """Sanitize a model input string.

    Normalizes whitespace, lowercases plain text, and preserves special tokens
    (e.g. [MASK], [CLS]) in their original uppercase form.

    Args:
        text: The raw user-supplied input string.

    Returns:
        The sanitized input string.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Strip leading/trailing whitespace, collapse internal whitespace to single spaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Lowercase everything except special tokens surrounded by brackets
    segments = SPECIAL_TOKEN.split(text)
    return "".join(
        segment if SPECIAL_TOKEN.fullmatch(segment) else segment.lower()
        for segment in segments
    )
