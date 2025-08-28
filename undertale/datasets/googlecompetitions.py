"""Challenges from Google Code Jam, Kick Start, and Hash Code competitions.

Solutions in C and C++, compiled.

Data: https://zibada.guru/gcj/.
"""

import json
import logging
import os
import sqlite3
import tempfile
import zipfile
import zlib

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers.base import BaseDiskReader

from .base import Dataset, main

# from .pipeline.compilers import CppCompiler
# from .pipeline.disassemblers import GhidraDisassembler

logger = logging.getLogger(__name__)


def unpack_sqlar(path: str, output: str) -> None:
    """Unpack a SQLite Archiver archive.

    SQLite Archiver (https://sqlite.org/sqlar/doc/trunk/README.md) is a zip
    alternative for building compressed archives using zlib and sqlite.

    Arguments:
        path: The path to the SQLite Archiver archive to unpack.
        output: The output directory where files should be unpacked.
    """

    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    cursor.execute("SELECT name, mode, mtime, sz, data FROM sqlar")

    for name, mode, mtime, sz, data in cursor:
        target = os.path.join(output, name)
        parent = os.path.dirname(target)

        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Directory.
        if data is None:
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)
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


SUPPORTED_LANGUAGES = ["C", "CPP"]


class GoogleCompetitionReader(BaseDiskReader):
    name = "💻 Google Competition"

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with zipfile.ZipFile(self.data_folder.path, "r") as f:
            for info in f.infolist():
                path = os.path.normpath(info.filename)

                # Only directories in the root of the archive.
                if (
                    not info.is_dir()
                    or len(path.split(os.sep)) != 1
                    or path in ["static"]
                ):
                    continue

                competition = path

                logger.info(f"extracting {competition}")

                working = tempfile.TemporaryDirectory()

                # Extract competition data.
                data = os.path.join(path, "raw_data.sqlar")
                solutions = os.path.join(path, "solutions.sqlar")

                unpack_sqlar(f.extract(data, working.name), working.name)
                unpack_sqlar(f.extract(solutions, working.name), working.name)

                logger.info(f"parsing {competition}")

                # Load language and task information.
                with open(
                    os.path.join(working.name, "raw_data", "index.json"), "r"
                ) as j:
                    index = json.load(j)

                tasks = {
                    t["id"]: f"<h2>{t['title']}</h2>\n{t['statement']}"
                    for t in index["challenge"]["tasks"]
                }

                # Parse competitor attempts.
                attempts_path = os.path.join(working.name, "raw_data", "attempts")
                for file in os.listdir(attempts_path):
                    competitor = os.path.splitext(file)[0]
                    with open(
                        os.path.join(attempts_path, file), "r", errors="ignore"
                    ) as j:
                        attempts = json.load(j)["attempts"]

                    for i, attempt in enumerate(attempts):
                        language = attempt["src_language__str"]

                        # Attempt is not in a supported language.
                        if language not in SUPPORTED_LANGUAGES:
                            continue

                        # Attempt failed at least one test.
                        if any(
                            [
                                t.get("verdict", 0) != 1
                                for t in attempt["judgement"]["results"]
                            ]
                        ):
                            continue

                        # Fetch source.
                        source_path = os.path.join(
                            working.name,
                            "solutions",
                            f"{competitor}.{i}.{language.lower()}",
                        )
                        with open(source_path, "r") as s:
                            source = s.read()

                        # Finally, yield a Document.
                        data = {
                            "id": attempt["id"],
                            "competitor": competitor,
                            "task": tasks[attempt["task_id"]],
                            "source": source,
                        }

                        yield self.get_document_from_dict(data, filepath, attempt["id"])


def adapt_googlecompetition_from_raw(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data["id"],
        "text": data["source"],
        "metadata": {"competitor": data["competitor"], "task": data["task"]},
    }


class GoogleCompetitions(Dataset):
    def get_pipeline(self, input, writer, parallelism):
        """"""
        steps = [
            GoogleCompetitionReader(
                os.path.join(input, "gcj-archive-2023.zip"),
                adapter=adapt_googlecompetition_from_raw,
            ),
            # CppCompiler(),
            # GhidraDisassembler(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(GoogleCompetitions)
