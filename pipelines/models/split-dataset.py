from dask.dataframe import read_parquet as dask_read_parquet

from undertale.logging import get_logger
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, flush
from undertale.utils import assert_path_exists, get_or_create_directory, write_parquet

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(
        description="split a dataset into training and validation"
    )

    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.9,
        help="fraction of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("splitting dataset")

        source = assert_path_exists(arguments.input)
        training_output, training_created = get_or_create_directory(
            f"{arguments.output}-training"
        )
        validation_output, validation_created = get_or_create_directory(
            f"{arguments.output}-validation"
        )

        if training_created and validation_created:
            frame = dask_read_parquet(source)
            training, validation = frame.random_split(
                [arguments.fraction, 1 - arguments.fraction],
                random_state=arguments.seed,
            )

            logger.info(f"writing training split to {training_output!r}")
            write_parquet(training, training_output, write_index=False)

            logger.info(f"writing validation split to {validation_output!r}")
            write_parquet(validation, validation_output, write_index=False)

        flush(client)

    logger.info("split complete")
