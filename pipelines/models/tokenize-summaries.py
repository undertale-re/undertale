import os

from undertale.logging import get_logger
from undertale.models.summarization import tokenize_summaries_gpt2
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush, read_directory
from undertale.pipeline.parquet import Repartition, modify_parquet
from undertale.utils.models.cache.load import load as load_hf_cache

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="summary tokenization")

    parser.add_argument(
        "-f",
        "--cache",
        help="path to a HuggingFace cache directory - if not provided, models will be downloaded as necessary",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        if arguments.cache:
            client.wait_for_workers(arguments.parallelism)

            logger.info("loading HuggingFace cache")

            client.run(load_hf_cache, arguments.cache)
            client.run(lambda: os.environ.update({"HF_HUB_OFFLINE": "1"}))

        logger.info("tokenizing summaries")

        chunks = read_directory(client, arguments.input)
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
