from datatrove.pipeline.readers import ParquetReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler


def adapt_googlecodejam(self, data: dict, path: str, id_in_file: int | str) -> dict:
    return {
        "id": data["row"],
        "text": data["source"],
        "metadata": {"summary": data["statement"]},
    }


class GoogleCodeJam(Dataset):
    def get_pipeline(self, input, writer, parallelism):
        steps = [
            ParquetReader(
                data_folder="/home/st25587/undertale_shared/datasets/gcj_testset",
                adapter=adapt_googlecodejam,
            ),
            CppCompiler(),
            GhidraDisassembler(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(GoogleCodeJam)
