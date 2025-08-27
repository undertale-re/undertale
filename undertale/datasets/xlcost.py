import hashlib
import os

from datatrove.pipeline.readers import JsonlReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler


def adapt_xlcost_from_raw(self, data: dict, path: str, id_in_file: int | str) -> dict:

    source = data["code"]
    identifier = hashlib.sha256(source.encode("utf-8")).hexdigest()

    return {
        "id": identifier,
        "text": source,
        "metadata": {"summary": data["text"]},
    }


class XLCost(Dataset):
    """The XLCost text-to-code generation benchmark.

    C and C++ language subsets, compiled.

    Data: https://huggingface.co/datasets/codeparrot/xlcost-text-to-code.
    """

    def get_pipeline(self, input, writer, parallelism):
        def xlcost_dataset_reader_split_factory(language: str, split: str):
            return JsonlReader(
                os.path.join(input, f"{language}-program-level", f"{split}.json"),
                adapter=adapt_xlcost_from_raw,
            )

        def xlcost_dataset_reader_language_factory(language: str):
            for split in ("train", "test", "valid"):
                yield xlcost_dataset_reader_split_factory(language, split)

        steps = [
            *xlcost_dataset_reader_language_factory("C"),
            *xlcost_dataset_reader_language_factory("C++"),
            CppCompiler(),
            GhidraDisassembler(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(XLCost)
