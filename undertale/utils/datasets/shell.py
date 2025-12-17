from argparse import ArgumentParser
from code import interact

import pandas

from ... import logging

logger = logging.get_logger(__name__)


if __name__ == "__main__":
    logging.setup_logging()

    parser = ArgumentParser(
        description="load a dataset and start a shell for inspection"
    )

    parser.add_argument("dataset", help="path to the dataset to load")

    arguments = parser.parse_args()

    logger.info(f"loading dataset: {arguments.dataset!r}")

    dataset = pandas.read_parquet(arguments.dataset)

    logger.info(f"{arguments.dataset!r} is available in the `dataset` variable")

    interact(local={"dataset": dataset})
