"""Filter rows in a parquet dataset."""

import argparse
from typing import Tuple

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Filter, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)


def parse_clause(value: str) -> Tuple[str, str]:
    """Parse a column:value filter clause.

    Arguments:
        value: A string in ``"column:value"`` format.

    Returns:
        A tuple of ``(column, substring)``.

    Raises:
        argparse.ArgumentTypeError: If the value is not in the expected format.
    """

    parts = value.split(":", 1)

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"clause must be in column:value format, got {value!r}"
        )

    column, substring = parts
    return column, substring


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="filter rows in a parquet dataset")

    parser.add_argument(
        "--filter",
        nargs="+",
        required=True,
        metavar="COLUMN:VALUE",
        type=parse_clause,
        help="column:value pairs; keeps rows matching all clauses (case-sensitive)",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("filtering rows")

        source = assert_path_exists(arguments.input)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[Filter(dict(arguments.filter))],
        )

        flush(client)

    logger.info("filter complete")
