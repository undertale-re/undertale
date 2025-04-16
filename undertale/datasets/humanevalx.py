from datatrove.pipeline.readers import HuggingFaceDatasetReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler


def adapt_humanevalx_from_huggingface(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data["task_id"],
        "text": f"{data['declaration']}{data['canonical_solution']}",
        "metadata": {"summary": data["prompt"]},
    }


class HumanEvalX(Dataset):
    name = "humaneval-x"

    def build(self, input, output=None, settings=None):
        config = self.get_configuration(settings)

        executor = self.get_executor_local(
            [
                HuggingFaceDatasetReader(
                    "THUDM/humaneval-x",
                    {"name": "cpp", "split": "test"},
                    adapter=adapt_humanevalx_from_huggingface,
                ),
                CppCompiler(),
                GhidraDisassembler(),
                self.get_writer(output),
            ],
            tasks=config.parallelism,
        )

        executor.run()


if __name__ == "__main__":
    main(HumanEvalX)
