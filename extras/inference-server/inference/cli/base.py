from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Sequence, Type

from ..exceptions import CommandError
from ..logging import get_logger

logger = get_logger(__name__)


class Command(metaclass=ABCMeta):
    """A CLI command."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this command."""

    @property
    @abstractmethod
    def help(self) -> str:
        """Help content."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments to this command (optional).

        Arguments:
            parser: The argument parser to which arguments should be added.
        """

        pass

    @abstractmethod
    def handle(self, arguments: Namespace) -> None:
        """Execute the command.

        Arguments:
            arguments: The parsed arguments object from argparse.

        Raises:
            CommandError: For known error types.
        """

    def execute(self, arguments: Namespace) -> None:
        """Wrapped command handling.

        Handles raised errors gracefully.

        Arguments:
            arguments: The parsed arguments object from argparse.
        """

        try:
            self.handle(arguments)
        except CommandError as e:
            logger.critical(e)
            exit(1)


def build_parser(parser: ArgumentParser, commands: Sequence[Type[Command]]) -> None:
    """Augment an argument parser with available CLI commands.

    Arguments:
        parser: The argument parser to augment.
        commands: A list of commands to add.
    """

    subparsers = parser.add_subparsers(dest="command", required=True)

    for Command in commands:
        command = Command()

        command_parser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(command_parser)
        command_parser.set_defaults(function=command.execute)
