"""Rename columns in a parquet dataset."""

import argparse
from typing import Tuple

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Rename, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)


def parse_rename(value: str) -> Tuple[str, str]:
    """Parse an old:new column rename specification.

    Arguments:
        value: A string in ``"old:new"`` format.

    Returns:
        A tuple of ``(old, new)``.

    Raises:
        argparse.ArgumentTypeError: If the value is not in the expected format.
    """

    parts = value.split(":")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"rename must be in old:new format, got {value!r}"
        )

    old, new = parts
    return old, new


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="rename columns in a parquet dataset")

    parser.add_argument(
        "--rename",
        nargs="+",
        required=True,
        metavar="OLD:NEW",
        type=parse_rename,
        help="column rename pairs in old:new format",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("renaming columns")

        source = assert_path_exists(arguments.input)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[Rename(dict(arguments.rename))],
        )

        flush(client)

    logger.info("rename complete")
