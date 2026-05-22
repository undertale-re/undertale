"""Randomly shuffle rows in a parquet dataset."""

from dask.dataframe import read_parquet as dask_read_parquet

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Repartition, Shuffle, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="shuffle rows in a parquet dataset")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="random seed for reproducibility",
    )

    parser.add_argument(
        "-n",
        "--partitions",
        type=int,
        default=None,
        metavar="N",
        help="if provided, re-partition to N chunks before shuffling for a closer approximation to a global shuffle",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("shuffling dataset")

        source = assert_path_exists(arguments.input)

        if arguments.partitions is not None:
            original = dask_read_parquet(source).npartitions
            operations = [
                Repartition(chunks=arguments.partitions),
                Shuffle(seed=arguments.seed),
                Repartition(chunks=original),
            ]
        else:
            operations = [Shuffle(seed=arguments.seed)]

        modify_parquet(
            input=source,
            output=arguments.output,
            operations=operations,
        )

        flush(client)

    logger.info("shuffle complete")
