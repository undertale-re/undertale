import hashlib

from datatrove.pipeline.readers import HuggingFaceDatasetReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler


def adapt_xlcost_from_huggingface(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    source = data["code"]
    identifier = hashlib.sha256(source.encode("utf-8")).hexdigest()

    return {
        "id": identifier,
        "text": source,
        "metadata": {"summary": data["text"]},
    }


class XLCost(Dataset):
    name = "xlcost"

    def get_pipeline(self, input, writer, parallelism):
        def xlcost_dataset_reader_split_factory(language: str, split: str):
            return HuggingFaceDatasetReader(
                "codeparrot/xlcost-text-to-code",
                {"name": f"{language}-program-level", "split": split},
                adapter=adapt_xlcost_from_huggingface,
            )

        def xlcost_dataset_reader_language_factory(language: str):
            for split in ("train", "test", "validation"):
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
