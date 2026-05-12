from undertale.logging import get_logger
from undertale.models.summarization import tokenize_summaries_gpt2
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.parquet import Repartition, modify_parquet

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="summary tokenization")

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("tokenizing summaries")

        chunks = client.submit(
            modify_parquet,
            arguments.input,
            f"{arguments.output}-repartitioned",
            [Repartition(chunks=arguments.parallelism)],
        )
        tokenized = fanout(
            client,
            tokenize_summaries_gpt2,
            chunks,
            f"{arguments.output}-processed",
        )

        merged = client.submit(
            modify_parquet,
            tokenized,
            arguments.output,
            [Repartition(size="100MB")],
        )

        merged.result()

        flush(client)

    logger.info("tokenization complete")
