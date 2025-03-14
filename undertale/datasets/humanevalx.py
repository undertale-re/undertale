from datatrove.pipeline.readers import HuggingFaceDatasetReader

from . import utils
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


humanevalx = [
    HuggingFaceDatasetReader(
        "THUDM/humaneval-x",
        {"name": "cpp", "split": "test"},
        adapter=adapt_humanevalx_from_huggingface,
    ),
    CppCompiler(),
    GhidraDisassembler(),
]

if __name__ == "__main__":
    utils.main(pipelines={"humanevalx": humanevalx}, default="humanevalx")
