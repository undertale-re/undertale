import os
from os.path import join
from typing import Dict, List, Union

import pandas as pd
from pandas import DataFrame, Series, read_parquet

from undertale.logging import get_logger
from undertale.parsers import PipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.pipeline.parquet import (
    hash_parquet_column,
    resize_parquet,
)
from undertale.utils import (
    assert_path_exists,
    get_or_create_directory,
    get_or_create_file,
    write_parquet,
)

logger = get_logger(__name__)


def parse_samples(input: str, output: str, max_size: int = int(1e7)) -> List[str]:
    """Process cvebinarysheet into a parquet file

    Arguments:
        input: A path to a JSON number file to process.
        output: The path where the processed number should be written.

    Returns:
        The path to the processed output file.
    """
    included_architectures = ["i686", "x86_64"]
    skip_packages = ["libpcap", "coreutils", "ffmpeg", "libtiff"]
    input = assert_path_exists(input)
    output, created = get_or_create_directory(output)

    if not created:
        logger.info("loading dataset")
        return [join(output, fname) for fname in os.listdir(output)]

    logger.info(f"parsing cvebinarysheet samples from {input!r} to {output!r}")
    id = 0
    outputs = []
    for project in [proj for proj in os.listdir(input) if proj not in skip_packages]:
        documents = []
        for version in os.listdir(join(input, project)):
            if os.path.isdir(join(input, project, version)):
                for floc in os.listdir(join(input, project, version)):
                    file_path = join(input, project, version, floc)
                    if (
                        os.path.isfile(file_path)
                        and os.path.getsize(file_path) < max_size
                        and floc.split("-")[1] in included_architectures
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
            write_parquet(data, join(output, f"{project}.parquet"))
    return outputs


def associate_cve_row(row: Series, CVEs: DataFrame) -> Dict[str, str | bytes]:
    fname = row["name"].lower().strip()
    project = row["project"]
    version = row["version"]
    cve_subset = CVEs[
        (CVEs["project"] == project)
        & (CVEs["fname"] == fname)
        & ((CVEs["version"] == version) | (CVEs["version"] == "-"))
    ]
    if len(cve_subset) > 0:
        vuln = "-".join(list(cve_subset["cve_name"].unique()))
    else:
        vuln = ""

    linked_cve = {
        **row.to_dict(),
        "vulnerability": vuln,
    }
    return linked_cve


def associate_cve(input: str, output: str) -> str:
    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    frame = read_parquet(input)

    associated = []
    CVEs = pd.read_csv(
        "/home/gridsan/pa27879/undertale_shared/dask/datasets/cvebinsheet/CVES.csv"
    )
    for _, row in frame.iterrows():
        logger.info(f"Now associating {row['id']}")

        new_row = associate_cve_row(row, CVEs)

        associated.append(new_row)
    associated = DataFrame.from_dict(associated)

    logger.info(
        f"successfully segmented and disassembled {len(associated)} functions from {len(input)} binaries"
    )

    write_parquet(associated, output)

    return output


if __name__ == "__main__":
    parser = PipelineArgumentParser(description="CVEBinarySheet")
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
        chunks = client.submit(
            resize_parquet,
            parsed,
            f"{arguments.output}-resized",
            size="20MB",
        )
        # chunks.result()
        # compiled = fanout(client, compile_cpp, chunks, f"{arguments.output}-compiled")
        disassembled = fanout(
            client,
            segment_and_disassemble_binary,
            chunks,
            f"{arguments.output}-disassembled",
        )

        linked = fanout(
            client,
            associate_cve,
            disassembled,
            f"{arguments.output}-linked",
        )
        # disassembled.result()

        hashed = fanout(
            client,
            hash_parquet_column,
            linked,
            f"{arguments.output}-hashed",
            column="binary",
            target="binary_hash",
        )

        merged = client.submit(
            resize_parquet,
            hashed,
            arguments.output,
            size="100MB",
            deduplicate=["binary_hash"],
            drop=["binary_hash"],
        )
        # merged = client.submit(
        #     resize_parquet, disassembled, f"{arguments.output}", size="100MB"
        # )

        merged.result()
        flush(client)

    logger.info("processing complete")
