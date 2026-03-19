from argparse import ArgumentParser

from . import __description__, __title__
from .cli import __commands__, build_parser
from .logging import setup_logging


def main() -> None:
    setup_logging()

    parser = ArgumentParser(description=f"{__title__}: {__description__}")

    build_parser(parser, __commands__)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
