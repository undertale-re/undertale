"""Exclude rows from a parquet dataset."""

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.pipeline.parquet import Exclude, modify_parquet
from undertale.utils import assert_path_exists
from undertale.utils.datasets.filter import parse_clause

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="exclude rows from a parquet dataset")

    parser.add_argument(
        "--exclude",
        nargs="+",
        required=True,
        metavar="COLUMN:VALUE",
        type=parse_clause,
        help="column:value pairs; removes rows matching all clauses (case-sensitive)",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("excluding rows")

        source = assert_path_exists(arguments.input)
        modify_parquet(
            input=source,
            output=arguments.output,
            operations=[Exclude(dict(arguments.exclude))],
        )

        flush(client)

    logger.info("exclude complete")
