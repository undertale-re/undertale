class Error(Exception):
    """Base class for all exceptions raised by this package."""


class CommandError(Error):
    """Raised when there is an issue running a CLI command."""


class ConfigurationError(Error):
    """Raised when there are issues with configuration."""


__all__ = ["Error", "CommandError", "ConfigurationError"]
