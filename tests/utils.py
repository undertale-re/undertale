import sys
from argparse import ArgumentParser
from os.path import dirname, join
from unittest import TestLoader, TextTestRunner

from undertale import __title__
from undertale.logging import CRITICAL, setup_logging


def load_resource(path: str) -> bytes:
    """Load a testing resource by path.

    Arguments:
        path: The path to the testing resource relative to the root of the
            test directory.

    Returns:
        The content of the requested resource as bytes.
    """

    with open(join(dirname(__file__), path), "rb") as f:
        return f.read()


def main(name: str) -> None:
    setup_logging(level=CRITICAL)

    parser = ArgumentParser(description=f"{__title__}: {name} tests")

    parser.add_argument(
        "test", nargs="?", help="a test case or individual test specifier"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="run in verbose mode"
    )

    arguments = parser.parse_args()

    loader = TestLoader()
    module = sys.modules["__main__"]
    if arguments.test:
        suite = loader.loadTestsFromName(arguments.test, module=module)
    else:
        suite = loader.loadTestsFromModule(module)

    runner = TextTestRunner(verbosity=2 if arguments.verbose else 1)
    result = runner.run(suite)

    if not result.wasSuccessful():
        exit(1)
