"""Load a pre-built HuggingFace cache."""

import os
from argparse import ArgumentParser
from shutil import copytree

from ....logging import get_logger, setup_logging

logger = get_logger(__name__)


def load(path: str) -> None:
    """Copy a pre-built HuggingFace cache into the current HF_HOME.

    Arguments:
        path: Path to the pre-built cache directory.
    """

    destination = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    logger.info(f"loading cache from {path!r} to {destination!r}")

    copytree(path, destination, dirs_exist_ok=True)

    logger.info("cache loaded")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="load a pre-built HuggingFace cache into HF_HOME"
    )

    parser.add_argument("input", help="path to the pre-built cache directory")

    arguments = parser.parse_args()

    setup_logging()

    load(arguments.input)
