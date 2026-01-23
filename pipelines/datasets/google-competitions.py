import os
import sqlite3
import zlib
from os.path import join
from pathlib import Path

from undertale.logging import get_logger
from undertale.parsers import PipelineArgumentParser
from undertale.pipeline import Client, Cluster, fanout
from undertale.pipeline.dask import merge
from undertale.pipeline.zip import unzip_file
from undertale.utils import get_or_create_directory

logger = get_logger(__name__)


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
        target = os.path.join(output, name)
        parent = os.path.dirname(target)

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


def extract_competitions(input: str, output: str) -> str:
    """Extracts solutions to Google Competitions.

    Arguments:
        input: Path to the Google Competition directory.
        output: Path to the output directory to create.

    Returns:
        The output directory created.
    """
    paths = list(Path(input).rglob("*"))

    # Filter only directories in the root of the archive.
    competitions = [
        path
        for path in paths
        if path.is_dir()
        and len(os.path.normpath(path.relative_to(input)).split(os.sep)) == 1
        and path.relative_to(input) != Path("static")
    ]

    # Extract competition data.
    for competition in competitions:
        data = os.path.join(competition, "raw_data.sqlar")
        solutions = os.path.join(input, competition, "solutions.sqlar")

        # Missing solution data, skip this competition.
        if Path(solutions) not in paths:
            logger.warning(f"solutions missing for {competition}")
            continue

        # Process SQLite Archive Files from competitions with solutions.
        logger.info(f"extracting {competition}")

        unpack_sqlar(data, join(output, os.path.basename(competition)))
        unpack_sqlar(solutions, join(output, os.path.basename(competition)))

        logger.info(f"{competition} extracted succesfully")

    return output


if __name__ == "__main__":
    parser = PipelineArgumentParser(description="google competitions dataset")
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

        inputs = merge(
            client, [join(arguments.input, f) for f in os.listdir(arguments.input)]
        )
        unzipped = fanout(client, unzip_file, inputs, f"{arguments.output}-unzipped")
        extracted = fanout(
            client, extract_competitions, unzipped, f"{arguments.output}-extracted"
        )

        # Debugging ------------------------
        r = extracted.result()

        import code

        code.interact(local=locals())
