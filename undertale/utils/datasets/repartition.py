"""Repartition a parquet dataset."""

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Repartition, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = DatasetArgumentParser(
        description="repartition a parquet dataset into configurable chunks"
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

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("repartitioning dataset")

        source = assert_path_exists(arguments.input)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[Repartition(chunks=arguments.chunks, size=arguments.size)],
        )

        flush(client)

    logger.info("repartition complete")
