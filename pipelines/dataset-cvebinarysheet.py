import os
from os.path import join
from typing import Union

import pandas as pd

from undertale.logging import get_logger
from undertale.parsers import DatasetPipelineArgumentParser
from undertale.pipeline import Client, Cluster
from undertale.utils import assert_path_exists

logger = get_logger(__name__)


def parse_samples(input: str, output: str) -> str:
    """Process cvebinarysheet into a parquet file

    Arguments:
        input: A path to a JSON number file to process.
        output: The path where the processed number should be written.

    Returns:
        The path to the processed output file.
    """

    input = assert_path_exists(input)
    if not os.path.isdir(output):
        os.mkdir(output)

    logger.info(f"parsing cvebinarysheet samples from {input!r} to {output!r}")

    for project in os.listdir(input):
        documents = []
        for version in os.listdir(join(input, project)):
            if os.path.isdir(join(input, project, version)):
                for floc in os.listdir(join(input, project, version)):
                    if os.path.isfile(join(input, project, version, floc)):
                        with open(join(input, project, version, floc), "rb") as f:
                            document: dict[str, Union[str, bytes]] = {
                                "code": f.read(int(1e6)),
                                "project": project,
                                "version": version,
                                "filename": floc,
                            }
                        documents.append(document)
        data = pd.DataFrame.from_dict(documents)
        data.to_parquet(join(output, f"{project}.par"))

    return output


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

        parsed.result()

    logger.info("processing complete")
