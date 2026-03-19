"""Shells for inspecting datasets locally.

By default, calling this script loads the ``pandas`` shell.
"""

from argparse import ArgumentParser
from code import interact
from typing import Any, Callable

from .... import logging

logger = logging.get_logger(__name__)


def main(load: Callable[[str], Any], mode: str):
    """Generic dataset shell implementation."""

    logging.setup_logging()

    parser = ArgumentParser(
        description=f"load a dataset and start a shell for inspection ({mode})"
    )

    parser.add_argument("dataset", help="path to the dataset to load")

    arguments = parser.parse_args()

    logger.info(f"loading dataset ({mode}): {arguments.dataset!r}")

    dataset = load(arguments.dataset)

    logger.info(f"{arguments.dataset!r} is available in the `dataset` variable")

    interact(local={"dataset": dataset})
