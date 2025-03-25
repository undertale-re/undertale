import json
import logging
import os
import sqlite3
import sys

import chardet
import datasets

from . import dataset, schema
from .transforms import compile
from .transforms.disassemble import capstone, ghidra

logger = logging.getLogger(__name__)


def unpack_sqlite(path: str):
    """Unpack a given sqlite file into a dict of file names and content.

    This parses the Google Code Jam archive sqlite file format.

    Arguemnts:
        path: The path to the sqlite file to unpack.

    Returns:
        A dictionary mapping file name to parsed content.
    """

    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    files = {}
    for name, _, _, _, content in cursor.execute("select * from sqlar;"):
        files[name] = content
        logging.debug(f"unpacked {name!r}")

    return files


class GoogleCodeJam(dataset.Dataset):
    url = "https://zibada.guru/gcj/"
    description = "Unofficial arhive of Google Cod Jam from 2008 to 2023"
    path = "google-code-jam"

    @classmethod
    def loaddata(cls, path: str):  # path: full path
        logging.info(f"loading .sqlar files from: {path}")

        competition = unpack_sqlite(f"{path}/raw_data.sqlar")

        solutions = unpack_sqlite(f"{path}/solutions.sqlar")

        index = json.loads(competition["raw_data/index.json"])
        return competition, solutions, index

    @classmethod
    def sqlar2tasks(cls, competition, solutions, index):
        logging.info("parsing .sqlar files")

        tasks = {}
        for task in index["challenge"]["tasks"]:
            tasks[task["id"]] = {
                "title": task["title"],
                "statement": task["statement"],
                "analysis": task["analysis"],
                "solutions": [],
            }
        for key, content in competition.items():
            if not key.startswith("raw_data/attempts/"):
                continue

            author, _ = os.path.splitext(os.path.basename(key))

            ## Detect encoding
            detected = chardet.detect(content)
            encoding = detected["encoding"]
            ##confidence = detected['confidence']
            content = content.decode(encoding, errors="replace").encode("utf-8")

            attempts = json.loads(content)
            for index, attempt in enumerate(attempts["attempts"]):
                task_id = attempt["task_id"]
                language = attempt["src_language__str"]

                # Filter source language.
                if language not in ["C", "CPP"]:
                    # - logging.warning(
                    # -     f"unsupported language {language}: attempt at {task_id} by {author}"
                    # - )
                    continue

                # Filter incorrect solutions.
                correct = True
                for judgement in attempt["judgement"]["results"]:
                    if judgement.get("verdict__str") == "WRONG_ANSWER":
                        correct = False
                        break

                if not correct:
                    # - logging.debug(f"failed attempt at {task_id} by {author}")
                    continue

                # - logging.info(f"problem {task_id} solved by {author}")

                _, extension = os.path.splitext(attempt["source_file"]["filename"])
                solution = os.path.join("solutions", f"{author}.{index}{extension}")

                tasks[task_id]["solutions"].append(
                    {"author": author, "source": solutions[solution].decode()}
                )

        return tasks

    @classmethod
    def tasks2rows(cls, tasks):
        logging.info("converting tasks to rows")
        logging.info(f"number of tasks: {len(tasks.keys()):,}")

        competition_solutions = 0
        rows = []
        for task, problem in tasks.items():
            num = len(problem["solutions"])
            logging.info(f"task - number of solutions: {num:,}")
            competition_solutions += len(problem["solutions"])

            for sol in problem["solutions"]:
                rows.append(
                    {
                        "id": f"{task}.{problem['title']}.{sol['author']}",
                        "summary": problem["statement"],
                        "source": sol["source"],
                    }
                )

        logging.info(f"Number of solutions in all tasks = {competition_solutions:,}")

        return rows, competition_solutions

    @classmethod
    def parse(cls, path: str, processes=None):
        """
        parse unpack:
            Unpack .sqlar file into json files
        parse fuse:
            Combine json files and compile dataset
        parse {dir}:
            Directory dir has raw_data.sqlar and solutions.sqlar file pair to compile dataset
            Example: parse y2023/2023a

        """
        if path == "unpack":
            home = os.path.expanduser("~")
            raw = "undertale_shared/datasets/raw/google-code-jam"
            path = os.path.join(home, raw)

            # find directories with both files
            dir_list = []
            for dirpath, dirnames, filenames in os.walk(path):
                if "raw_data.sqlar" in filenames and "solutions.sqlar" in filenames:
                    dir_list.append(dirpath)
            logging.info(f"{len(dir_list)} directories have both file pairs")

            staging = f"{path}/staging"
            if not os.path.exists(staging):
                os.makedirs(staging)

            total_number = 0
            for idx, dir in enumerate(dir_list):
                logging.info(f"processing dir #{idx:04} : {dir}")
                try:
                    competition, solutions, index = cls.loaddata(dir)
                    tasks = cls.sqlar2tasks(competition, solutions, index)
                    rows, number_solutions = cls.tasks2rows(tasks)
                except Exception as e:
                    logging.exception(f"in unpack;idx={idx} Execption: {e}")

                total_number += number_solutions
                fname = f"{staging}/rows-{idx:04}.json"
                with open(fname, "w") as f:
                    json.dump(rows, f)

            logging.info("finished unpacking .sqlar files")
            logging.info(f"Total number of solutions: {total_number}")
            sys.exit()

        elif path == "fuse":

            home = os.path.expanduser("~")
            raw = "undertale_shared/datasets/raw/google-code-jam"
            staging = os.path.join(home, raw, "staging")
            logging.info(f"collecting .json files in dir: {staging}")

            rows = []
            for fname in os.listdir(staging):
                if fname.endswith(".json"):
                    full_path = os.path.join(staging, fname)
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        rows.extend(data)
            logging.info(f"items in dataset: {len(rows)}")

            dataset = datasets.Dataset.from_list(rows)
            logging.info("compile json files to dataset")

        else:  # path has both competition and solutions files
            home = os.path.expanduser("~")
            raw = "undertale_shared/datasets/raw/google-code-jam"
            path = os.path.join(home, raw, path)

            competition, solutions, index = cls.loaddata(path)
            tasks = cls.sqlar2tasks(competition, solutions, index)
            logging.info("tasks variable avaiable")

            rows = cls.tasks2rows(tasks)
            logging.info("rows variable avaiable")

            dataset = datasets.Dataset.from_list(rows)
            logging.info("dataset variable avaiable")

        dataset.__class__ = cls

        return dataset


class GoogleCodeJamCompiled(GoogleCodeJam):
    path = "google-code-jam-compiled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
    ]


class GoogleCodeJamCompiledDisassembled(GoogleCodeJam):
    path = "google-code-jam-compiled-disassembled"
    schema = schema.SummarizedFunction

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


class GoogleCodeJamCompiledGhidraDisassembled(GoogleCodeJam):
    path = "google-code-jam-compiled-ghidra-disassembled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
        ghidra.GhidraDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main(
        [
            GoogleCodeJamCompiledDisassembled,
            GoogleCodeJamCompiledGhidraDisassembled,
            GoogleCodeJamCompiled,
            GoogleCodeJam,
        ]
    )
