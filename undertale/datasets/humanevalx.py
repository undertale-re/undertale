"""The HumanEval-X multilingual code benchmark.

C++ language subset, compiled.

Data: https://huggingface.co/datasets/zai-org/humaneval-x.
"""

import os

from datatrove.pipeline.readers import JsonlReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import RizinDisassembler


def adapt_humanevalx_from_raw(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data["task_id"],
        "text": f"{data['declaration']}{data['canonical_solution']}",
        "metadata": {"summary": data["prompt"]},
    }


class HumanEvalX(Dataset):
    def get_pipeline(self, input, writer, parallelism):
        """"""
        steps = [
            JsonlReader(
                os.path.join(input, "cpp/data/humaneval.jsonl"),
                adapter=adapt_humanevalx_from_raw,
            ),
            CppCompiler(),
            RizinDisassembler(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(HumanEvalX)
