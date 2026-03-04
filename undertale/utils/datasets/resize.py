"""Resize a parquet dataset."""

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import resize_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = DatasetArgumentParser(
        description="resize a parquet dataset into configurable chunks"
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

    parser.add_argument(
        "--deduplicate",
        nargs="+",
        metavar="COLUMN",
        help="deduplicate dataset by these column names",
    )

    columns = parser.add_mutually_exclusive_group()
    columns.add_argument(
        "--drop",
        nargs="+",
        metavar="COLUMN",
        help="column names to drop",
    )
    columns.add_argument(
        "--keep",
        nargs="+",
        metavar="COLUMN",
        help="column names to keep (mutually exclusive with --drop)",
    )

    parser.add_argument(
        "--compression",
        help='compression algorithm (e.g. "snappy")',
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("resizing dataset")

        source = assert_path_exists(arguments.input)
        resize_parquet(
            input=source,
            output=arguments.output,
            chunks=arguments.chunks,
            size=arguments.size,
            deduplicate=arguments.deduplicate,
            drop=arguments.drop,
            keep=arguments.keep,
            compression=arguments.compression,
        )

        flush(client)

    logger.info("resize complete")
