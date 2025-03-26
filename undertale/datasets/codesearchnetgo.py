import logging

import datasets
from datasets import concatenate_datasets

from . import dataset, schema
from .transforms import compile
from .transforms.disassemble import capstone

logger = logging.getLogger(__name__)


class CodeSearchNetGo(dataset.Dataset):
    url = "https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text"
    description = (
        "natural language and code pairs from open sourced libraries hosted on GitHub"
    )
    notes = r"""
Functions from one/six programming languages:
  [x] Go
  [ ] Java
  [ ] JavaScript
  [ ] PHP
  [ ] Python
  [ ] Ruby
From the paper: https://arxiv.org/pdf/1909.09436
"""
    path = "codesearchnet-go"

    @classmethod
    def download(cls, processes=None):
        """Download this dataset from the HuggingFace Hub.

        Also do a little pre-processing.
        """

        def parse(sample):
            annotation = sample["docstring"]
            annotation = annotation.replace("\n", "")
            annotation = annotation.replace("// ", " ")
            annotation = annotation.replace("//", " ")

            return {
                "summary": annotation,
                "source": sample["code"],
            }

        dataset = datasets.load_dataset("google/code_x_glue_ct_code_to_text", "go")
        dataset = concatenate_datasets(
            [dataset["train"], dataset["test"], dataset["validation"]]
        )

        dataset = dataset.map(
            parse,
            remove_columns=[
                "id",
                "repo",
                "path",
                "func_name",
                "original_string",
                "code",
                "code_tokens",
                "docstring",
                "docstring_tokens",
                "sha",
                "url",
                "language",
            ],
            num_proc=processes,
        )
        id_column = list(map(str, range(len(dataset))))
        dataset = dataset.add_column("id", id_column)

        return dataset

    @classmethod
    def parse(cls, path: str, processes=None):
        if path == "download":
            logger.info(f"downloading {cls.__name__} from the HuggingFace Hub")

            dataset = cls.download()

        else:
            dataset = datasets.load_from_disk(path)

        dataset.__class__ = cls

        return dataset


class CodeSearchNetGoCompiled(CodeSearchNetGo):
    path = "codesearchnet-go-compiled"

    transforms = [compile.CompileGo(), compile.CompileErrorsFilter()]


class CodeSearchNetGoCompiledWindowsARM(CodeSearchNetGo):
    path = "codesearchnet-go-compiled-windows-arm"

    transforms = [
        compile.CompileGo(os="windows", arch="arm"),
        compile.CompileErrorsFilter(),
    ]


class CodeSearchNetGoCompiledWindowsARMUnoptimized(CodeSearchNetGo):
    path = "codesearchnet-go-compiled-windows-arm-unoptimized"

    transforms = [
        compile.CompileGo(os="windows", arch="arm", compile_flags=["-N"]),
        compile.CompileErrorsFilter(),
    ]


class CodeSearchNetGoCompiledDisassembled(CodeSearchNetGo):
    path = "codesearchnet-go-compiled-disassembled"
    schema = schema.SummarizedFunction

    transforms = [
        compile.CompileGo(),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


class CodeSearchNetGoCompiledWindowsARMDisassembled(CodeSearchNetGo):
    path = "codesearchnet-go-compiled-windows-arm-disassembled"

    transforms = [
        compile.CompileGo(os="windows", arch="arm"),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main(
        [
            CodeSearchNetGoCompiledDisassembled,
        ]
    )
