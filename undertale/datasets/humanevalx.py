import logging

import datasets

from . import dataset, schema
from .transforms import compile
from .transforms.disassemble import capstone, ghidra, rizin

logger = logging.getLogger(__name__)


class HumanEvalX(dataset.Dataset):
    url = "https://huggingface.co/datasets/THUDM/humaneval-x"
    description = "natural language descriptions of programming problems and source code implmentations, compiled"
    path = "humaneval-x"

    @classmethod
    def download(cls, processes=None):
        """Download this dataset from the HuggingFace Hub.

        Also do a little pre-processing.
        """

        def parse(sample):
            annotation = sample["prompt"]

            if annotation[:2] != "/*":
                return {"id": sample["task_id"], "summary": "SKIP", "source": ""}

            annotation = annotation.split(">>>")[0]
            annotation = annotation[2:].strip()
            annotation = annotation.replace("\n", "")

            source = f"{sample['declaration']}{sample['canonical_solution']}"

            return {
                "id": sample["task_id"],
                "summary": annotation,
                "source": source,
            }

        dataset = datasets.load_dataset("THUDM/humaneval-x", "cpp")["test"]
        dataset = dataset.map(
            parse,
            remove_columns=[
                "task_id",
                "prompt",
                "declaration",
                "canonical_solution",
                "test",
                "example_test",
            ],
            num_proc=processes,
        )
        dataset = dataset.filter(lambda d: d["summary"] != "SKIP", num_proc=processes)

        return dataset

    @classmethod
    def parse(cls, path: str, processes=None):
        if path == "download":
            logger.info(f"downloading {cls.__name__} from the HuggingFace Hub")

            dataset = cls.download(processes=processes)
        else:
            dataset = datasets.load_from_disk(path)

        dataset.__class__ = cls

        return dataset


class HumanEvalXCompiled(HumanEvalX):
    path = "humaneval-x-compiled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
    ]


class HumanEvalXCompiledDisassembled(HumanEvalX):
    path = "humaneval-x-compiled-disassembled"
    schema = schema.SummarizedFunction

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


class HumanEvalXCompiledGhidraDisassembled(HumanEvalX):
    path = "humaneval-x-compiled-ghidra-disassembled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
        ghidra.GhidraDisassemble(),
    ]


class HumanEvalXCompiledRZDisassembled(HumanEvalXCompiled):
    path = "humaneval-x-compiled-rz-disassembled"

    transforms = [rizin.RizinDisassemble()]


if __name__ == "__main__":
    dataset.main(
        [
            HumanEvalXCompiledDisassembled,
            HumanEvalXCompiledGhidraDisassembled,
            HumanEvalXCompiledRZDisassembled,
            HumanEvalXCompiled,
            HumanEvalX,
        ]
    )
