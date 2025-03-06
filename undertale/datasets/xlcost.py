import logging

import datasets

from . import dataset, schema
from .transforms import compile
from .transforms.disassemble import capstone

logger = logging.getLogger(__name__)


class XLCost(dataset.Dataset):
    url = "https://huggingface.co/datasets/codeparrot/xlcost-text-to-code"
    description = "a text-to-code generation benchmark dataset, compiled"
    path = "xlcost"

    @classmethod
    def download(cls, processes=None):
        """Download this dataset from the HuggingFace Hub.

        Also do a little pre-processing.
        """

        def parse(sample):
            return {
                "source": sample["code"],
                "summary": sample["text"],
            }

        dataset = datasets.load_dataset(
            "codeparrot/xlcost-text-to-code", "C++-program-level"
        )
        dataset = datasets.concatenate_datasets(
            [
                dataset["train"],
                dataset["test"],
                dataset["validation"],
            ]
        )

        keep = {"id", "source", "summary"}
        dataset = dataset.map(
            parse, remove_columns=set(dataset.features) - keep, num_proc=processes
        )

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


class XLCostCompiled(XLCost):
    path = "xlcost-compiled"

    transforms = [compile.CompileCpp(), compile.CompileErrorsFilter()]


class XLCostCompiledDisassembled(XLCost):
    path = "xlcost-compiled-disassembled"
    schema = schema.SummarizedFunction

    transforms = [
        compile.CompileCpp(),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main([XLCostCompiledDisassembled, XLCostCompiled, XLCost])
