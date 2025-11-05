"""Challenges from Google Code Jam, Kick Start, and Hash Code competitions.

Solutions in C and C++, compiled.

Data: https://zibada.guru/gcj/.
"""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.readers.base import BaseDiskReader

from .base import WRITERS, Dataset, main
from .pipeline.compilers import CppCompiler

# from .pipeline.disassemblers import GhidraDisassembler


class GoogleCompetitions(Dataset):
    class GoogleCompetitionReader(BaseDiskReader):
        """"""

        name = "💻 Google Competition"

        SUPPORTED_LANGUAGES = ["C", "CPP"]

        @staticmethod
        def unpack_sqlar(path: str, output: str) -> None:
            """Unpack a SQLite Archiver archive.

            SQLite Archiver (https://sqlite.org/sqlar/doc/trunk/README.md) is a zip
            alternative for building compressed archives using zlib and sqlite.

            Arguments:
                path: The path to the SQLite Archiver archive to unpack.
                output: The output directory where files should be unpacked.
            """
            import os
            import sqlite3
            import zlib

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

        def read_file(self, filepath: str) -> DocumentsPipeline:
            import json
            import logging
            import os
            import tempfile
            import zipfile

            logger = logging.getLogger(__name__)

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

                    # Missing solution data, skip this competition.
                    if solutions not in f.namelist():
                        logger.warning(f"solutions missing for {competition}")
                        continue

                    self.unpack_sqlar(f.extract(data, working.name), working.name)
                    self.unpack_sqlar(f.extract(solutions, working.name), working.name)

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

                    # Missing attempt data, skip this competition.
                    if not os.path.isdir(attempts_path):
                        logger.warning(f"attempts missing for {competition}")
                        continue

                    total = 0
                    for file in os.listdir(attempts_path):
                        competitor = os.path.splitext(file)[0]
                        with open(
                            os.path.join(attempts_path, file), "r", errors="ignore"
                        ) as j:
                            attempts = json.load(j)["attempts"]

                        for i, attempt in enumerate(attempts):
                            language = attempt["src_language__str"]

                            # Attempt is not in a supported language.
                            if language not in self.SUPPORTED_LANGUAGES:
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
                            source_name = f"{competitor}.{i}.{language.lower()}"
                            source_path = os.path.join(
                                working.name, "solutions", source_name
                            )
                            try:
                                with open(source_path, "r") as s:
                                    source = s.read()
                            except FileNotFoundError:
                                logger.warning(f"solution not found: {source_name}")
                                continue

                            # Finally, yield a Document.
                            data = {
                                "id": attempt["id"],
                                "competitor": competitor,
                                "task": tasks[attempt["task_id"]],
                                "source": source,
                            }

                            yield self.get_document_from_dict(
                                data, filepath, attempt["id"]
                            )

                            total += 1

                    logger.info(
                        f"found {total} solutions to {len(tasks)} tasks from {competition}"
                    )

    @staticmethod
    def adapt_googlecompetition_from_raw(
        self, data: dict, path: str, id_in_file: int | str
    ) -> dict:
        return {
            "id": data["id"],
            "text": data["source"],
            "metadata": {"competitor": data["competitor"], "task": data["task"]},
        }

    @staticmethod
    def adapt_googlecompetition_from_parquet(
        self, data: dict, path: str, id_in_file: int | str
    ) -> dict:
        return {
            "id": data.pop("id", id_in_file),
            "text": data.pop("code"),
            "metadata": data,
        }

    def get_pipeline(self, input, writer, parallelism):
        """"""

        def googlecompetitions_reader_factory(archive: str):
            import os

            return self.GoogleCompetitionReader(
                os.path.join(input, archive),
                adapter=self.adapt_googlecompetition_from_raw,
            )

        intermediate = input.rsplit("-raw", 1)[0]
        intermediate = f"{intermediate}-source"

        parse = self.get_executor(
            [
                googlecompetitions_reader_factory("gcj-archive-2008-2017.zip"),
                googlecompetitions_reader_factory("gcj-archive-2018.zip"),
                googlecompetitions_reader_factory("gcj-archive-2019.zip"),
                googlecompetitions_reader_factory("gcj-archive-2020.zip"),
                googlecompetitions_reader_factory("gcj-archive-2021.zip"),
                googlecompetitions_reader_factory("gcj-archive-2022.zip"),
                googlecompetitions_reader_factory("gcj-archive-2023.zip"),
                googlecompetitions_reader_factory("kickstart-archive-2013-2018.zip"),
                googlecompetitions_reader_factory("kickstart-archive-2019.zip"),
                googlecompetitions_reader_factory("kickstart-archive-2020.zip"),
                googlecompetitions_reader_factory("kickstart-archive-2021.zip"),
                googlecompetitions_reader_factory("kickstart-archive-2022.zip"),
                googlecompetitions_reader_factory("hashcode-archive-2014-2022.zip"),
            ]
            + WRITERS["parquet"](intermediate, size=1024 * 1024),
            env_command="""
        eval \"$(conda shell.bash hook)\"
        conda activate undertale
        """,
            time="1:00:00",
            cpus_per_task=4,
            mem_per_cpu_gb=4,
            tasks=1,
            job_name="googlecompetitions-parse",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )

        return self.get_executor(
            [
                ParquetReader(
                    intermediate, adapter=self.adapt_googlecompetition_from_parquet
                ),
                CppCompiler(),
            ]
            + writer,
            env_command="""
        eval \"$(conda shell.bash hook)\"
        conda activate undertale
        """,
            depends=parse,
            time="48:00:00",
            cpus_per_task=1,
            mem_per_cpu_gb=4,
            tasks=parallelism,
            job_name="googlecompetitions-compile",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )


if __name__ == "__main__":
    main(GoogleCompetitions)
