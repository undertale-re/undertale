import sys
import typing
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    FileHandler,
    Filter,
    Formatter,
    StreamHandler,
    getLogger,
)

LEVELS = {
    "debug": DEBUG,
    "info": INFO,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
}


class CharacterLevelFilter(Filter):
    """Adds logging level as a single character for formatting."""

    CHARACTERS = {
        DEBUG: "-",
        INFO: "+",
        WARNING: "!",
        ERROR: "*",
        CRITICAL: "#",
    }

    def filter(self, record):
        record.levelchar = self.CHARACTERS.get(record.levelno, " ")
        return True


class ColorLevelFilter(Filter):
    """Adds logging level as a color for formatting."""

    WHITE_DIM = "\x1b[37;2m"
    WHITE = "\x1b[37m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    RED_BOLD = "\x1b[31;1m"
    END = "\x1b[0m"
    NULL = END

    COLORS = {
        DEBUG: WHITE_DIM,
        INFO: WHITE,
        WARNING: YELLOW,
        ERROR: RED,
        CRITICAL: RED_BOLD,
    }

    def filter(self, record):
        record.levelcolor = self.COLORS.get(record.levelno, self.NULL)
        return True


def setup_logging(
    level: int = INFO,
    colors: bool = True,
    stream: bool = True,
    file: typing.Optional[str] = None,
    clear: bool = True,
) -> None:
    """Setup log handling.

    Note: this should only be called once.

    Arguments:
        level: Logging level (from `logging` module).
        colors: Enable logging colors (if supported).
        stream: Enable stream logging.
        file: If provided, enable file logging to the path provided.
        clear: Remove all other handlers from the root logger.
    """

    root = getLogger()
    root.setLevel(level)

    if clear:
        handlers = list(root.handlers)
        for handler in handlers:
            root.removeHandler(handler)

    if stream:
        format = "[%(levelchar)s] %(message)s"

        if colors and sys.stderr.isatty():
            format = f"%(levelcolor)s{format}{ColorLevelFilter.END}"

        formatter = Formatter(format)

        handler = StreamHandler()
        handler.setLevel(level)
        handler.addFilter(CharacterLevelFilter())
        handler.addFilter(ColorLevelFilter())

        handler.setFormatter(formatter)

        root.addHandler(handler)

    if file:
        format = "[%(asctime)s %(levelname)-8s]: %(message)s"

        formatter = Formatter(format)

        handler = FileHandler(file)
        handler.setLevel(level)

        handler.setFormatter(formatter)

        root.addHandler(handler)


def get_logger(*args, **kwargs):
    """Passthrough to `logging.getLogger()`."""

    return getLogger(*args, **kwargs)


__all__ = ["setup_logging", "get_logger"]
