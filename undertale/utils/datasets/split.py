"""Split a dataset into configurable named splits."""

import argparse
from typing import List, Tuple

from dask.dataframe import read_parquet as dask_read_parquet

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.utils import assert_path_exists, get_or_create_directory, write_parquet

logger = get_logger(__name__)


def parse_split(value: str) -> Tuple[str, float]:
    """Parse a name:percentage split specification.

    Args:
        value: A string in ``"name:percentage"`` format.

    Returns:
        A tuple of ``(name, percentage)``.

    Raises:
        argparse.ArgumentTypeError: If the value is not in the expected format.
    """

    parts = value.split(":")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"split must be in name:percentage format, got {value!r}"
        )

    name, percentage_str = parts

    try:
        percentage = float(percentage_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"percentage must be a number, got {percentage_str!r}"
        )
    return name, percentage


def split(
    source: str,
    output: str,
    splits: List[Tuple[str, float]],
    seed: int = 42,
) -> None:
    """Split a dataset into named partitions.

    Args:
        source: Path to the input dataset.
        output: Base path for output directories. Each split is written to
            ``"{output}-{name}"``.
        splits: A list of ``(name, percentage)`` tuples. Percentages must sum
            to 100.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If split percentages do not sum to 100.
    """
    total = sum(percentage for _, percentage in splits)
    if abs(total - 100) > 1e-6:
        raise ValueError(f"split percentages must sum to 100, got {total}")

    fractions = [percentage / 100 for _, percentage in splits]

    outputs = [get_or_create_directory(f"{output}-{name}") for name, _ in splits]

    if all(created for _, created in outputs):
        frame = dask_read_parquet(source)
        frames = frame.random_split(fractions, random_state=seed)

        for (output_path, _), (name, _), split_frame in zip(outputs, splits, frames):
            logger.info(f"writing {name} split to {output_path!r}")
            write_parquet(split_frame, output_path, write_index=False)


if __name__ == "__main__":
    parser = DatasetArgumentParser(
        description="split a dataset into configurable named splits"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=[("training", 90.0), ("validation", 10.0)],
        metavar="NAME:PERCENTAGE",
        type=parse_split,
        help="splits as name:percentage pairs",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    total = sum(percentage for _, percentage in arguments.splits)
    if abs(total - 100) > 1e-6:
        parser.error(f"split percentages must sum to 100, got {total}")

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("splitting dataset")

        source = assert_path_exists(arguments.input)
        split(source, arguments.output, arguments.splits, arguments.seed)

        flush(client)

    logger.info("split complete")
