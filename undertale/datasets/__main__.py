import argparse
import code
import logging

import datasets

from .. import logging as undertale_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="utilities for managing datasets")

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="dataset utility to run"
    )

    shell_parser = subparsers.add_parser(
        "shell",
        help="load a dataset and open a python shell for exploration",
    )

    shell_parser.add_argument(
        "path",
        help="path to a dataset file to load (or the name of one on the HuggingFace hub)",
    )

    arguments = parser.parse_args()

    undertale_logging.setup_logging()

    if arguments.command == "shell":
        try:
            dataset = datasets.load_dataset(arguments.path)
        except Exception as e:
            logger.critical(e)
            exit(1)

        logger.info("the loaded dataset is available in the `dataset` variable")
        code.interact(local={"dataset": dataset})
