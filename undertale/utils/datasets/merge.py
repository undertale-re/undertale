"""Merge two or more parquet dataset directories."""

from os import listdir
from os.path import join
from typing import List

from undertale.logging import get_logger
from undertale.parsers import ArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.dask import CLUSTER_TYPES
from undertale.pipeline.parquet import Repartition, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="merge two or more parquet datasets into one")

    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="two or more input dataset directories to merge",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output dataset directory",
    )
    parser.add_argument(
        "-p", "--parallelism", type=int, default=1, help="degree of parallelism"
    )
    parser.add_argument(
        "-c",
        "--cluster",
        choices=CLUSTER_TYPES,
        default="local",
        help="cluster type",
    )

    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument(
        "--chunks",
        type=int,
        help="number of chunk files to generate",
    )
    sizing.add_argument(
        "--size",
        help='maximum chunk size in bytes or string (e.g. "25MB")',
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    if len(arguments.inputs) < 2:
        parser.error("at least two input datasets are required")

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("merging datasets")

        chunks: List[str] = []
        for directory in arguments.inputs:
            source = assert_path_exists(directory)
            chunks.extend(join(source, f) for f in listdir(source))

        modify_parquet(
            input=chunks,
            output=arguments.output,
            operations=[Repartition(chunks=arguments.chunks, size=arguments.size)],
        )

        flush(client)

    logger.info("merge complete")
