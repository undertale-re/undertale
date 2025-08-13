import argparse
import code
import logging

import polars

from ...logging import setup_logging
from ..base import Dataset

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load a dataset and open a python shell for exploration"
    )

    parser.add_argument(
        "-l",
        "--lazy",
        action="store_true",
        help="load the dataset as a polars `LazyFrame`",
    )

    parser.add_argument("input", help="input location")

    arguments = parser.parse_args()
    setup_logging()

    try:
        if arguments.lazy:
            dataset = polars.scan_parquet(arguments.input)
        else:
            dataset = Dataset.load(arguments.input)
    except Exception as e:
        logger.critical(e)
        exit(1)

    logger.info(f"{arguments.input!r} is available in the `dataset` variable")

    code.interact(local={"dataset": dataset})
