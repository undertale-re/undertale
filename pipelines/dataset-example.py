import json
from os.path import join
from random import randint
from time import sleep
from typing import List

from undertale.logging import get_logger
from undertale.parsers import DatasetPipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout
from undertale.pipeline.json import merge_json
from undertale.utils import assert_path_does_not_exist, assert_path_exists

logger = get_logger(__name__)


def generate_numbers(output: str, size: int = 64) -> List[str]:
    """Generate random numbers and save them as JSON.

    Each number will be saved as its own JSON file.

    Arguments:
        output: The output directory where each of the numbers will be saved.
        size: The number of random numbers to generate.

    Returns:
        A list of paths to generated files.
    """

    output = assert_path_does_not_exist(output, create=True)

    logger.info(f"generating {size} inputs")

    results = []
    for i in range(size):
        number = randint(1, 4096)
        path = join(output, f"part_{i:04d}.json")

        with open(path, "w") as f:
            json.dump(number, f)

        results.append(path)

    return results


def process_number(input: str, output: str, delay: int = 0) -> str:
    """Process numbers.

    A simple example of some computation over a single file.

    Arguments:
        input: A path to a JSON number file to process.
        output: The path where the processed number should be written.
        delay: An artifical delay to simulate processing time.

    Returns:
        The path to the processed output file.
    """

    input = assert_path_exists(input)
    output = assert_path_does_not_exist(output)

    logger.info(f"processing {input}")

    with open(input, "r") as f:
        number = json.load(f)

    processed = 1 / number

    sleep(delay)

    with open(output, "w") as f:
        json.dump(processed, f)

    return output


if __name__ == "__main__":
    parser = DatasetPipelineArgumentParser(description="example dataset")
    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(
            type=arguments.cluster,
            parallelism=arguments.parallelism,
        ) as cluster,
        Client(cluster) as client,
    ):
        logger.info("processing dataset")

        generated = client.submit(generate_numbers, f"{arguments.output}-generated")
        processed = fanout(
            client, process_number, generated, f"{arguments.output}-processed"
        )
        merged = client.submit(merge_json, processed, f"{arguments.output}.json")

        result = merged.result()

    logger.info(f"processing complete: {result}")
