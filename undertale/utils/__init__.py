"""Utilities and helper scripts."""

import hashlib
import os
import subprocess
from datetime import datetime
from os import makedirs
from os.path import abspath, exists, expanduser, isfile, splitext
from typing import Iterable, Optional, Tuple

from ..exceptions import EnvironmentError as LocalEnvironmentError
from ..exceptions import PathDoesNotExist
from ..logging import get_logger

logger = get_logger(__name__)


def hash(data: bytes) -> str:
    """Compute the cryptographic hash of the given bytes.

    This is the common hashing algorithm used for uniqueness everywhere in this
    project.

    Arguments:
        data: The data from which to compute a hash.

    Returns:
        A string representing the hexdigest of the hash of ``data``.
    """

    return hashlib.sha256(data).hexdigest()


def timestamp(time: Optional[datetime] = None) -> str:
    """Generate a timestamp string in the standard format.

    Arguments:
        time: An optional datetime for which to generate a timestamp. If not
            provided the current date and time will be used.

    Returns:
        A string timestamp in the standard format.
    """

    time = time or datetime.now()
    return time.strftime("%Y%m%d-%H%M%S")


def enforce_extension(path: str, extension: str) -> str:
    """Set the file extension of the given path.

    This logs a message if the file extension was changed, possibly
    unexpectedly.

    Arguments:
        path: The path to validate.
        extension: The extension to enforce.

    Returns:
        The modified path with the given extension applied.
    """

    base, found = splitext(path)
    if found != extension:
        if found != "":
            logger.info(f"changing extension {found!r} to {extension!r} for {path!r}")
        else:
            logger.info(f"adding extension {extension!r} to {path!r}")

        path = f"{base}{extension}"

    return path


def assert_path_exists(path: str) -> str:
    """Assert that a given path exists.

    Arguments:
        path: The path to check.

    Returns:
        The absolute version of the given path.

    Raises:
        PathDoesNotExist: If the given path does not exist.
    """

    absolute = abspath(expanduser(path))

    if not exists(absolute):
        raise PathDoesNotExist(f"{path}: path does not exist")

    return absolute


def get_or_create_file(path: str) -> Tuple[str, bool]:
    """Get the given file, creating it if it doesn't already exist.

    Arguments:
        path: The path to the file to get or create.

    Returns:
        The absolute version of the given path and a boolean indicating if it
        was created by this function.
    """

    absolute = abspath(expanduser(path))

    if exists(absolute):
        logger.warning(f"{path!r} already exists")
        return absolute, False

    with open(absolute, "w"):
        pass

    return absolute, True


def get_or_create_directory(path: str) -> Tuple[str, bool]:
    """Get the given directory, creating it if it doesn't already exist.

    Arguments:
        path: The path to the directory to get or create.

    Returns:
        The absolute version of the given path and a boolean indicating if it
        was created by this function.
    """

    absolute = abspath(expanduser(path))

    if exists(absolute):
        logger.warning(f"{path!r} already exists")
        return absolute, False

    makedirs(absolute)

    return absolute, True


def find(
    name: str,
    environment: Optional[str] = None,
    guesses: Optional[Iterable[str]] = None,
) -> str:
    """Finds a particular binary on this system.

    Attempts to find the binary given by ``name``, first checking the value of
    the environment variable named ``environment`` (if provided), then by
    checking the system path, then finally checking hardcoded paths in
    ``guesses`` (if provided). This function is cross-platform compatible - it
    works on Windows, Linux, and Mac. If there are spaces in the path found,
    this function will wrap its return value in double quotes.

    Args:
        name: Binary name.
        environment: An optional environment variable to check.
        guesses: An optional list of hardcoded paths to check.

    Returns:
        A string with the absolute path to the binary if found, otherwise
        ``None``.
    """

    def sanitize(path):
        quotes = ("'", "'")
        if " " in path and path[0] not in quotes and path[-1] not in quotes:
            path = '"{}"'.format(path)

        return path

    if environment:
        path = os.environ.get(environment)
        if path is not None:
            path = abspath(expanduser(path))
            if isfile(path):
                return sanitize(path)

    if os.name == "posix":
        search = "which"
    elif os.name == "nt":
        search = "where.exe"
    else:
        raise EnvironmentError("unknown platform: {}".format(os.name))

    try:
        with open(os.devnull, "w") as output:
            path = subprocess.check_output([search, name], stderr=output).decode(
                "utf-8"
            )

        return sanitize(abspath(path.strip()))
    except subprocess.CalledProcessError:
        pass

    if guesses:
        for path in guesses:
            if isfile(path):
                return sanitize(path)

    message = f"could not find {name!r} or it is not installed"

    if environment is not None:
        message = (
            f"{message} (hint: did you set the {environment} environment variable?)"
        )

    raise LocalEnvironmentError(message)


def write_parquet(frame, path: str, **kwargs) -> None:
    """Write parquet to the given path.

    This helper method enforces some defaults on a typical
    ``frame.to_parquet()`` operation.

    Arguments:
        frame: A ``DataFrame``-like object to write (e.g.,
            ``pandas.DataFrame``, ``polars.DataFrame``, etc.)
        path: The path where the parquet file should be written.
        **kargs: Additional kwargs to pass to the underlying ``to_parquet``
            method, or override from defaults.

    Raises:
        ValueError: If the given ``frame`` object does not support
        ``.to_parquet()``.
    """

    if not hasattr(frame, "to_parquet"):
        raise ValueError(
            f"{frame.__class__.__module__}.{frame.__class__.__name__} does not support `.to_parquet()`"
        )

    defaults = {
        "compression": None,
        "schema": None,
    }
    defaults.update(kwargs)

    frame.to_parquet(path, **defaults)  # noqa: UT001


__all__ = [
    "timestamp",
    "assert_path_exists",
    "get_or_create_file",
    "get_or_create_directory",
    "find",
    "write_parquet",
]
