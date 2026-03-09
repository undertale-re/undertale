"""Utilities and helper scripts."""

import hashlib
import os
from datetime import datetime
from multiprocessing import Queue, get_context
from os import makedirs, stat, walk
from os.path import (
    abspath,
    basename,
    exists,
    expanduser,
    isfile,
    join,
    relpath,
    splitext,
)
from queue import Empty
from shutil import copy2
from subprocess import CalledProcessError, check_output
from traceback import format_exc
from typing import Callable, Iterable, Optional, Tuple

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
            path = check_output([search, name], stderr=output).decode("utf-8")

        return sanitize(abspath(path.strip()))
    except CalledProcessError:
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


def cache_path(path: str) -> str:
    """Copy a file or directory to the cache and return the cache path.

    Checks the ``UNDERTALE_CACHE`` environment variable. If set, copies
    ``path`` to ``$UNDERTALE_CACHE/basename(path)``, logging each file
    copied. If the destination already exists, verifies each file's size
    and modification time (similar to rsync) and re-copies stale files.
    If ``UNDERTALE_CACHE`` is not set, logs a warning and returns the
    original ``path`` unchanged.

    Args:
        path: Path to the file or directory to cache.

    Returns:
        The cache path if ``UNDERTALE_CACHE`` is set, otherwise ``path``.
    """

    cache_root = os.environ.get("UNDERTALE_CACHE")
    if cache_root is None:
        logger.warning(f"UNDERTALE_CACHE is not set - skipping cache for {path!r}")
        return path

    if not exists(cache_root):
        raise FileNotFoundError(f"UNDERTALE_CACHE path does not exist {cache_root!r}")

    if not exists(path):
        raise FileNotFoundError(f"source path does not exist {path!r}")

    destination = join(cache_root, basename(path))

    def copy_if_stale(source: str, dest: str) -> None:
        if exists(dest):
            source_stat = stat(source)
            dest_stat = stat(dest)
            if (
                source_stat.st_size == dest_stat.st_size
                and source_stat.st_mtime == dest_stat.st_mtime
            ):
                logger.info(f"path already exists in cache {dest!r}")
                return
        copy2(source, dest)
        logger.info(f"cached {source!r} to {dest!r}")

    if isfile(path):
        copy_if_stale(path, destination)
    else:
        for source_directory, _, filenames in walk(path):
            relative = relpath(source_directory, path)
            destination_directory = join(destination, relative)
            makedirs(destination_directory, exist_ok=True)
            for filename in filenames:
                copy_if_stale(
                    join(source_directory, filename),
                    join(destination_directory, filename),
                )

    return destination


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


class RemoteException(Exception):
    """A wrapper around an exception raised by a remote process."""

    def __init__(self, type: str, representation: str, traceback: str):
        super().__init__()

        self.type = type
        self.representation = representation
        self.traceback = traceback

    def __str__(self):
        return f"{self.type}: {self.representation}\n------ Remote Traceback ------\n{self.traceback}"

    @classmethod
    def from_exception(cls, exception: BaseException):
        return cls(exception.__class__.__name__, str(exception), traceback=format_exc())


def subprocess(
    function: Optional[Callable] = None, timeout: Optional[float] = None
) -> Callable:
    """A decorator that runs the given function in a subprocess.

    Arguments:
        timeout: A timeout (seconds) after which to raise an exception. If
            ``None``, then the subprocess will be allowed to run indefinitely.

    Raises:
        TimeoutError: If ``timeout`` seconds have passed and ``function`` has
            not returned.
        Exception: If an exception is raised by the decorated function.
    """

    context = get_context("fork")

    def inner(function: Callable):
        def wrapper(*args, **kwargs):
            queue = Queue(maxsize=1)

            def target(q: Queue, *args, **kwargs) -> None:
                try:
                    q.put(function(*args, **kwargs))
                except BaseException as e:
                    q.put(RemoteException.from_exception(e))

            process = context.Process(
                target=target, args=[queue, *args], kwargs=kwargs, daemon=True
            )
            process.start()

            try:
                result = queue.get(timeout=timeout)
            except Empty:
                if process.is_alive():
                    process.terminate()

                process.join()
                queue.cancel_join_thread()
                queue.close()

                raise TimeoutError(
                    f"{function.__name__} subprocess timeout after {timeout}s"
                )

            process.join()

            if isinstance(result, RemoteException):
                raise result

            return result

        return wrapper

    if function is None:
        return inner
    return inner(function)


__all__ = [
    "timestamp",
    "assert_path_exists",
    "cache_path",
    "get_or_create_file",
    "get_or_create_directory",
    "find",
    "write_parquet",
    "subprocess",
    "RemoteException",
]
