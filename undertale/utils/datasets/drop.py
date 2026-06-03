"""Drop or keep columns in a parquet dataset."""

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Drop, Keep, modify_parquet
from undertale.utils import assert_path_exists

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = DatasetArgumentParser(
        description="drop or keep columns in a parquet dataset"
    )

    columns = parser.add_mutually_exclusive_group(required=True)
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

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("dropping/keeping columns")

        source = assert_path_exists(arguments.input)
        operation = Drop(arguments.drop) if arguments.drop else Keep(arguments.keep)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[operation],
        )

        flush(client)

    logger.info("drop/keep complete")
