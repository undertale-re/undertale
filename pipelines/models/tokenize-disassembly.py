from undertale.logging import get_logger
from undertale.models.tokenizer import tokenize
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.parquet import (
    Keep,
    ParquetOperation,
    Repartition,
    modify_parquet,
)

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="dataset tokenization")

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
            modify_parquet,
            arguments.input,
            f"{arguments.output}-repartitioned",
            [Repartition(chunks=arguments.parallelism)],
        )
        tokenized = fanout(
            client,
            tokenize,
            chunks,
            f"{arguments.output}-processed",
            tokenizer=arguments.tokenizer,
        )

        merge_operations: list[ParquetOperation] = [Repartition(size="100MB")]
        if arguments.minimize:
            merge_operations = [Keep(["id", "tokens", "mask"]), *merge_operations]

        merged = client.submit(
            modify_parquet,
            tokenized,
            arguments.output,
            merge_operations,
        )

        merged.result()

        flush(client)

    logger.info("tokenization complete")
