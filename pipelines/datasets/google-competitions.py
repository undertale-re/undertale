import json
import os
import sqlite3
import zlib
from os.path import basename, dirname, isdir, join, splitext
from pathlib import Path
from typing import List

from pandas import DataFrame

from undertale.logging import get_logger
from undertale.parsers import PipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.pipeline.cpp import compile_cpp
from undertale.pipeline.dask import merge
from undertale.pipeline.parquet import hash_parquet_column, resize_parquet
from undertale.pipeline.zip import unzip_file
from undertale.utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = ["C", "CPP"]


def unpack_sqlar(input: str, output: str):
    """Unpack a SQLite Archiver archive.

    SQLite Archiver (https://sqlite.org/sqlar/doc/trunk/README.md) is a zip
    alternative for building compressed archives using zlib and sqlite.

    Arguments:
        input: The path to the SQLite Archiver archive to unpack.
        output: The output directory where files should be unpacked.
    """
    connection = sqlite3.connect(input)
    cursor = connection.cursor()

    cursor.execute("SELECT name, mode, mtime, sz, data FROM sqlar")

    parents = set()
    for name, mode, mtime, sz, data in cursor:
        target = join(output, name)
        parent = dirname(target)

        if parent not in parents:
            parents.add(parent)
            parent, _ = get_or_create_directory(parent)

        # Directory.
        if data is None:
            target, created = get_or_create_directory(target)
            if not created:
                continue
            os.utime(target, (mtime, mtime))
            os.chmod(target, mode)

        # Symlink.
        elif sz == -1:
            to = data.decode("utf-8")
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            os.symlink(to, target)

        # File.
        else:
            if len(data) < sz:
                data = zlib.decompress(data)
            with open(target, "wb") as f:
                f.write(data)
            os.utime(target, (mtime, mtime))
            os.chmod(target, mode)

    connection.close()


def extract_competition(input: str, output: str) -> str:
    """Extracts solutions from a Google Competition.

    Arguments:
        input: Path to the competition directory.
        output: Path to the output directory.

    Returns:
        The path to the directory with the relevant competition data.
    """
    input = assert_path_exists(input)
    competition = Path(input)

    # Recursively match everything in the input directory.
    paths = list(competition.rglob("*"))

    # Filter only directories in the root of the archive.
    challenges = [
        path
        for path in paths
        if path.is_dir()
        and path.parent == competition
        and path.relative_to(competition) != Path("static")
    ]

    # Extract challenge data.
    for challenge in challenges:
        data = join(challenge, "raw_data.sqlar")
        solutions = join(challenge, "solutions.sqlar")

        # Missing solution data, skip this challenge.
        if Path(solutions) not in paths:
            logger.warning(f"solutions missing for {challenge}")
            continue

        # Process SQLite Archive Files from challenges with solutions.
        logger.info(f"extracting {challenge}")

        cwd = join(output, basename(challenge))
        unpack_sqlar(data, cwd)
        unpack_sqlar(solutions, cwd)

        logger.info(f"succesfully extracted to {cwd}")

    return output


def get_challenges(input) -> List[str]:
    """Returns paths to challenges that contain at least one competitor submission.

    Arguments:
        input: Path to the directories of each competition.

    Returns:
        A list of paths to competition challenges with one or more competitor submissions.
    """
    challenges = list()

    for competition in input:
        challenges.extend(
            [
                join(competition, challenge)
                for challenge in os.listdir(competition)
                if isdir(join(competition, challenge, "raw_data", "attempts"))
            ]
        )

    return challenges


def parse_challenge(input: str, output: str) -> str:
    """Parses a competition challenge into a dataset.

    Arguments:
        input: Path to a challenge from a competition.
        output: Path to the output directory.

    Returns:
        Path to the directory with the parsed challenge data as parquet.
    """
    logger.info(f"parsing challenge {input}")

    input = assert_path_exists(input)

    output, created = get_or_create_directory(output)
    if not created:
        return output

    rows = []

    # Load language and task information.
    with open(join(input, "raw_data", "index.json"), "r") as f:
        index = json.load(f)

        tasks = {
            task["id"]: f"<h2>{task['title']}</h2>\n{task['statement']}"
            for task in index["challenge"]["tasks"]
        }

        # Parse competitors attempts.
        attempts_path = join(input, "raw_data", "attempts")

        total = 0
        for file in os.listdir(attempts_path):
            competitor = splitext(file)[0]
            with open(join(attempts_path, file), "r", errors="ignore") as j:
                attempts = json.load(j)["attempts"]

            for i, attempt in enumerate(attempts):
                language = attempt["src_language__str"]

            # Attempt is not in a supported language.
            if language not in SUPPORTED_LANGUAGES:
                continue

            # Attempt failed at least one test.
            if any(
                [
                    test.get("verdict", 0) != 1
                    for test in attempt["judgement"]["results"]
                ]
            ):
                continue

            # Fetch source.
            source_name = f"{competitor}.{i}.{language.lower()}"
            source_path = join(input, "solutions", source_name)
            try:
                with open(source_path, "r") as s:
                    source = s.read()
            except FileNotFoundError:
                logger.warning(f"solution not found: {source_name}")
                continue

            # Finally, return a row.
            rows.append(
                {
                    "id": attempt["id"],
                    "source": source,
                    "competitor": competitor,
                    "task": tasks[attempt["task_id"]],
                }
            )

            total += 1

        logger.info(f"found {total} solutions to {len(tasks)} tasks from {input}")

        frame = DataFrame(rows)
        frame.to_parquet(join(output, "dataset.parquet"))

        return output


if __name__ == "__main__":
    parser = PipelineArgumentParser(description="google competitions dataset")
    arguments = parser.parse_args()
    parser.setup(arguments)

    with (
        Cluster(type=arguments.cluster, parallelism=arguments.parallelism) as cluster,
        Client(cluster) as client,
    ):
        logger.info("processing dataset")

        # Pre-process
        ## Gather competitions on disk as individual Zip files.
        inputs = merge(
            client, [join(arguments.input, f) for f in os.listdir(arguments.input)]
        )

        ## Unzip competitions.
        unzipped = fanout(client, unzip_file, inputs, f"{arguments.output}-unzipped")

        ## Filter relevant solution data from each competition.
        extracted = fanout(
            client, extract_competition, unzipped, f"{arguments.output}-extracted"
        )

        ## Parsed challenges from each competition into parquet.
        challenges = client.submit(get_challenges, extracted)
        parsed = fanout(
            client, parse_challenge, challenges, f"{arguments.output}-parsed"
        )
        parsed = parsed.result()
        parquets = merge(
            client,
            [
                join(challenge, parquet)
                for challenge in parsed
                for parquet in os.listdir(challenge)
            ],
        )
        chunks = client.submit(
            resize_parquet,
            parquets,
            f"{arguments.output}-resized",
            chunks=arguments.parallelism,
        )

        # Post-process
        compiled = fanout(client, compile_cpp, chunks, f"{arguments.output}-compiled")
        disassembled = fanout(
            client,
            segment_and_disassemble_binary,
            compiled,
            f"{arguments.output}-disassembled",
        )
        hashed = fanout(
            client,
            hash_parquet_column,
            disassembled,
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

        merged.result()

        flush(client)

    logger.info("processing complete")
