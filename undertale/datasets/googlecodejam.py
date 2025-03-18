import logging
import chardet
import code
import datasets
import json
import os
import sqlite3
import sys

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
    def loaddata(cls, path: str):
        logging.info(f"loading .sqlar files")

        home = os.path.expanduser("~")
        raw = 'undertale_shared/datasets/raw/google-code-jam'

        competition = os.path.join(home, raw, f'{path}/raw_data.sqlar')
        competition = unpack_sqlite(competition)

        solutions = os.path.join(home, raw, f'{path}/solutions.sqlar')
        solutions = unpack_sqlite(solutions)

        index = json.loads(competition["raw_data/index.json"])
        return competition, solutions, index


    @classmethod
    def sqlar2tasks(cls, competition, solutions, index):
        logging.info(f"parsing .sqlar files")

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
            encoding = detected['encoding']
            ##confidence = detected['confidence']
            content = content.decode(encoding,errors="replace").encode('utf-8')
    
            attempts = json.loads(content)
            for index, attempt in enumerate(attempts["attempts"]):
                task_id = attempt["task_id"]
                language = attempt["src_language__str"]
    
                # Filter source language.
                if language not in ["C", "CPP"]:
                    #- logging.warning(
                    #-     f"unsupported language {language}: attempt at {task_id} by {author}"
                    #- )
                    continue
    
                # Filter incorrect solutions.
                correct = True
                for judgement in attempt["judgement"]["results"]:
                    if judgement.get("verdict__str") == "WRONG_ANSWER":
                        correct = False
                        break
    
                if not correct:
                    #- logging.debug(f"failed attempt at {task_id} by {author}")
                    continue
    
                #- logging.info(f"problem {task_id} solved by {author}")
    
                _, extension = os.path.splitext(attempt["source_file"]["filename"])
                solution = os.path.join("solutions", f"{author}.{index}{extension}")
    
                tasks[task_id]["solutions"].append(
                    {"author": author, "source": solutions[solution].decode()}
                )

        return tasks


    @classmethod
    def tasks2rows(cls, tasks):
        logging.info(f"converting tasks to rows")
        logging.info(f"number of tasks: {len(tasks.keys()):,}")

        total_solutions = 0
        rows = []
        for task, problem in tasks.items():
            num = len(problem["solutions"])
            logging.info(f"task - number of solutions: {num:,}")
            total_solutions += len(problem["solutions"])

            for sol in problem['solutions']:
                rows.append({
                    'id': f"{task}.{problem['title']}.{sol['author']}",
                    'summary': problem['statement'],
                    'source': sol['source']
            })

        logging.info(f'total - number of solutions = {total_solutions:,}')

        return rows


    @classmethod
    def parse(cls, path: str, processes=None):
        if path == "download":
            print('error: download not supported')
            sys.exit()
        else:
            competition, solutions, index = cls.loaddata(path)
            tasks = cls.sqlar2tasks(competition, solutions, index)
            logging.info(f"tasks variable avaiable")

            rows = cls.tasks2rows(tasks)
            logging.info(f"rows variable avaiable")

            dataset = datasets.Dataset.from_list(rows)
            logging.info(f"dataset variable avaiable")

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

