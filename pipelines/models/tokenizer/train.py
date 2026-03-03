from undertale.logging import get_logger
from undertale.models.tokenizer import (
    merge_preprocessed_tokens,
    preprocess_tokens,
    train_tokenizer,
)
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.parquet import resize_parquet

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="tokenizer training")

    parser.add_argument(
        "-b",
        "--progress",
        action="store_true",
        help="display tokenizer training progress",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("training tokenizer")

        chunks = client.submit(
            resize_parquet,
            arguments.input,
            f"{arguments.output}-resized",
            chunks=arguments.parallelism,
        )
        preprocessed = fanout(
            client, preprocess_tokens, chunks, f"{arguments.output}-preprocessed"
        )
        merged = client.submit(
            merge_preprocessed_tokens, preprocessed, f"{arguments.output}-merged"
        )
        trained = client.submit(
            train_tokenizer,
            merged,
            f"{arguments.output}",
            silent=not arguments.progress,
        )

        trained.result()

        flush(client)

    logger.info("training complete")
