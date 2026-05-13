"""Cast columns to a new dtype in a parquet dataset."""

import argparse
from typing import Tuple

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Cast, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)


def parse_cast(value: str) -> Tuple[str, str]:
    """Parse a column:dtype cast specification.

    Arguments:
        value: A string in ``"column:dtype"`` format.

    Returns:
        A tuple of ``(column, dtype)``.

    Raises:
        argparse.ArgumentTypeError: If the value is not in the expected format.
    """

    parts = value.split(":")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"cast must be in column:dtype format, got {value!r}"
        )

    column, dtype = parts
    return column, dtype


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="cast columns in a parquet dataset")

    parser.add_argument(
        "--cast",
        nargs="+",
        required=True,
        metavar="COLUMN:DTYPE",
        type=parse_cast,
        help="column cast pairs in column:dtype format",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("casting columns")

        source = assert_path_exists(arguments.input)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[Cast(dict(arguments.cast))],
        )

        flush(client)

    logger.info("cast complete")
