import os
from abc import ABC, abstractmethod
from configparser import ConfigParser, NoOptionError
from os.path import exists
from secrets import token_hex
from typing import Dict

from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)


WORKSPACE_ENVIRONMENT = "UNDERTALE_WORKSPACE"
WORKSPACE_DEFAULT = "/etc/undertale-inference/"
SETTINGS_PATH = "{workspace}/settings.ini"
SETTINGS_SECTION = "undertale-inference"


def get_workspace() -> str:
    path = os.environ.get(WORKSPACE_ENVIRONMENT, WORKSPACE_DEFAULT)

    if not exists(path):
        raise ConfigurationError(f"workspace path does not exist: {path}")

    return path


def get_settings_path() -> str:
    return SETTINGS_PATH.format(workspace=get_workspace())


class Setting(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        """The configuration key required by this setting."""

    def parse(self, value: str):
        """Value parsing.

        By default this does variable substitutions for supported parameters -
        if this behavior is desired, overwriters should call
        ``super().parse(value)`` before adding their own logic.
        """

        return value.format(workspace=get_workspace())

    @property
    @abstractmethod
    def default(self) -> str:
        """The default value to store in the settings file."""


class Database(Setting):
    key = "database"
    default = "{workspace}/database.sqlite3"


class Tokenizer(Setting):
    key = "tokenizer"
    default = "{workspace}/tokenizer.json"


class MaskedLMCheckpoint(Setting):
    key = "maskedlm-checkpoint"
    default = "{workspace}/maskedlm.ckpt"


class JWTSecret(Setting):
    key = "jwtsecret"
    default = token_hex(32)


class LDAPHost(Setting):
    key = "ldaphost"
    default = "ad.example.com"


class LDAPPort(Setting):
    key = "ldapport"
    default = "636"

    def parse(self, value: str) -> int:
        return int(value)


class LDAPDomain(Setting):
    key = "ldapdomain"
    default = "example.com"


def initialize() -> None:
    path = get_settings_path()

    if exists(path):
        raise ConfigurationError(f"settings file already exists: {path}")

    settings = {}
    for setting in Setting.__subclasses__():
        s = setting()  # type: ignore
        settings[s.key] = s.default

    parser = ConfigParser()
    parser.read_dict({SETTINGS_SECTION: settings})

    with open(path, "w") as f:
        parser.write(f)

    logger.info(f"wrote initial configuration file to {path}")


def fetch() -> Dict[str, str]:
    """Fetch the current settings.

    Returns:
        A dictionary mapping settings to their configured values.
    """

    path = get_settings_path()

    if not exists(path):
        raise ConfigurationError(f"settings file not found: {path}")

    parser = ConfigParser()
    parser.read(path)

    if SETTINGS_SECTION not in parser.sections():
        raise ConfigurationError(
            f"settings file {path} missing section {SETTINGS_SECTION!r}"
        )

    settings = {}
    for setting in Setting.__subclasses__():
        s = setting()  # type: ignore

        try:
            settings[s.key] = s.parse(parser.get(SETTINGS_SECTION, s.key))
        except NoOptionError as e:
            raise ConfigurationError(str(e))

    return settings


__all__ = ["initialize", "fetch"]
