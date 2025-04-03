from datatrove.pipeline.readers import HuggingFaceDatasetReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler, RadareDisassembler


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

    def get_pipeline(self, input, writer, parallelism):
        steps = [
            HuggingFaceDatasetReader(
                "THUDM/humaneval-x",
                {"name": "cpp", "split": "test"},
                adapter=adapt_humanevalx_from_huggingface,
            ),
            CppCompiler(),
            RadareDisassembler(),
            GhidraDisassembler(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(HumanEvalX)
