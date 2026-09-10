"""Evaluate a trained summarization model."""

from undertale.logging import get_logger
from undertale.models.summarization import evaluate_summarized, summarize_tokenized
from undertale.parsers import DatasetArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush, read_directory
from undertale.pipeline.json import average_json

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = DatasetArgumentParser(description="evaluate a trained summarization model")

    parser.add_argument(
        "-t", "--tokenizer", type=str, required=True, help="path to a trained tokenizer"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("evaluating model")

        chunks = read_directory(client, arguments.input)
        summarized = fanout(
            client,
            summarize_tokenized,
            chunks,
            f"{arguments.output}-summarized",
            tokenizer=arguments.tokenizer,
            checkpoint=arguments.checkpoint,
        )
        evaluated = fanout(
            client,
            evaluate_summarized,
            summarized,
            f"{arguments.output}-evaluated",
        )
        averaged = client.submit(average_json, evaluated, arguments.output)

        averaged.result()

        flush(client)

    logger.info("evaluation complete")
