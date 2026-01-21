from undertale.logging import get_logger
from undertale.models.tokenizer import tokenize
from undertale.parsers import PipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.parquet import resize_parquet

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = PipelineArgumentParser(description="dataset tokenization")

    parser.add_argument(
        "-t", "--tokenizer", type=str, required=True, help="path to a trained tokenizer"
    )
    parser.add_argument(
        "-m",
        "--minimize",
        action="store_true",
        help="remove all columns from the tokenized dataset except for tokens",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("tokenizing dataset")

        chunks = client.submit(
            resize_parquet,
            arguments.input,
            f"{arguments.output}-resized",
            chunks=arguments.parallelism,
        )
        tokenized = fanout(
            client,
            tokenize,
            chunks,
            f"{arguments.output}-processed",
            tokenizer=arguments.tokenizer,
        )
        merged = client.submit(
            resize_parquet,
            tokenized,
            arguments.output,
            size="100MB",
            keep=["input_ids", "attention_mask"] if arguments.minimize else None,
        )

        merged.result()

        flush(client)

    logger.info("tokenization complete")
