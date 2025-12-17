class Error(Exception):
    """Base class for all exceptions."""


class EnvironmentError(Error):
    """Raisd when there are issued with dependencies/environment/etc."""


class PathError(Error):
    """Errors with paths."""


class PathDoesNotExist(PathError):
    """A requested path does not exist when it should."""


class PathExists(PathError):
    """A requested already exists when it should not."""


class InvalidFileType(PathError):
    """Raised when a file is not of the expected type."""


class SchemaError(Error):
    """Errors with schema validation."""


__all__ = [
    "Error",
    "EnvironmentError",
    "PathError",
    "PathDoesNotExist",
    "PathExists",
    "SchemaError",
]
