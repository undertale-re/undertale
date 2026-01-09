import json
from os.path import exists, join

from pandas import DataFrame

from undertale.exceptions import PathDoesNotExist
from undertale.logging import get_logger
from undertale.parsers import DatasetPipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.pipeline.cpp import compile_cpp
from undertale.pipeline.parquet import resize_parquet
from undertale.pipeline.tarfile import extract_tarfile
from undertale.utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def parse_samples(input: str, output: str) -> str:
    """Extract sample files from unpacked HumanEval-X source.

    Arguments:
        input: A directory of HumanEval-X source.
        output: Output path where samples will be saved.

    Returns:
        The path to extracted sample output.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_directory(output)

    if not created:
        return output

    logger.info(f"parsing HumanEval-X samples from {input!r} to {output!r}")

    cpp = "cpp/data/humaneval.jsonl"
    source = join(input, cpp)

    if not exists(source):
        raise PathDoesNotExist(
            f"malformed HumanEval-X dataset (missing source file: {cpp!r})"
        )

    data = []
    with open(source, "r") as f:
        for line in f:
            raw = json.loads(line)
            data.append(
                {
                    "id": raw["task_id"],
                    "source": f"{raw['declaration']}{raw['canonical_solution']}",
                    "summary": raw["prompt"],
                }
            )

    frame = DataFrame(data)

    logger.info(f"parsed {len(frame)} samples")

    frame.to_parquet(join(output, "dataset.parquet"))

    return output


if __name__ == "__main__":
    parser = DatasetPipelineArgumentParser(description="humaneval-x dataset")
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

        extracted = client.submit(
            extract_tarfile, arguments.input, f"{arguments.output}-extracted"
        )
        parsed = client.submit(parse_samples, extracted, f"{arguments.output}-parsed")
        chunks = client.submit(
            resize_parquet,
            parsed,
            f"{arguments.output}-resized",
            chunks=arguments.parallelism,
        )
        compiled = fanout(client, compile_cpp, chunks, f"{arguments.output}-compiled")
        disassembled = fanout(
            client,
            segment_and_disassemble_binary,
            compiled,
            f"{arguments.output}-disassembled",
        )
        merged = client.submit(
            resize_parquet, disassembled, f"{arguments.output}", size="100MB"
        )

        merged.result()

    logger.info("processing complete")
