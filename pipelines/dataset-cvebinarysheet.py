import os
from os.path import join
from typing import Union

import pandas as pd

from undertale.logging import get_logger
from undertale.parsers import DatasetPipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def parse_samples(input: str, output: str, max_size: int = int(1e6)) -> list[str]:
    """Process cvebinarysheet into a parquet file

    Arguments:
        input: A path to a JSON number file to process.
        output: The path where the processed number should be written.

    Returns:
        The path to the processed output file.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_directory(output)

    if not created:
        logger.info("loading dataset")
        return [join(output, fname) for fname in os.listdir(output)]

    logger.info(f"parsing cvebinarysheet samples from {input!r} to {output!r}")
    id = 0
    outputs = []
    for project in os.listdir(input):
        documents = []
        for version in os.listdir(join(input, project)):
            if os.path.isdir(join(input, project, version)):
                for floc in os.listdir(join(input, project, version)):
                    file_path = join(input, project, version, floc)
                    if (
                        os.path.isfile(file_path)
                        and os.path.getsize(file_path) < max_size
                    ):
                        with open(file_path, "rb") as f:
                            document: dict[str, Union[str, bytes]] = {
                                "binary": f.read(max_size),
                                "project": project,
                                "version": version,
                                "filename": floc,
                                "id": str(id),
                            }
                        documents.append(document)
                        id += 1
        if len(documents) > 0:
            data = pd.DataFrame.from_dict(documents)
            outputs.append(join(output, f"{project}.parquet"))
            data.to_parquet(join(output, f"{project}.parquet"))
        # data.to_parquet(join(output, "dataset.parquet"))
    return outputs


if __name__ == "__main__":
    parser = DatasetPipelineArgumentParser(description="CVEBinarySheet")
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

        parsed = client.submit(
            parse_samples, arguments.input, f"{arguments.output}-parsed"
        )
        # parsed.result()
        # chunks = client.submit(
        #     resize_parquet,
        #     parsed,
        #     f"{arguments.output}-resized",
        #     chunks=arguments.parallelism,
        # )
        # chunks.result()
        # compiled = fanout(client, compile_cpp, chunks, f"{arguments.output}-compiled")
        disassembled = fanout(
            client,
            segment_and_disassemble_binary,
            parsed,
            f"{arguments.output}-disassembled",
        )
        disassembled.result()
        # merged = client.submit(
        #     resize_parquet, disassembled, f"{arguments.output}", size="100MB"
        # )

        # merged.result()
        flush(client)

    logger.info("processing complete")
