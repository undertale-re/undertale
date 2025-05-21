from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter

from .base import DEFAULT_DATASETS_DIRECTORY, Dataset, main
from .pipeline.compilers import CppCompiler

# from .pipeline.disassemblers import GhidraDisassembler
from .pipeline.segmenters.rizin import RizinFunctionSegmentAndDisassemble
from .pipeline.disassemblers import GhidraDisassembler
from .pipeline.formatters import ITEMPretokenizer


def adapt_humanevalx_from_huggingface(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data["task_id"],
        "text": f"{data['declaration']}{data['canonical_solution']}",
        "metadata": {"summary": data["prompt"]},
    }


class HumanEvalX(Dataset):
    def get_pipeline(self, input, writer, parallelism):
        steps = [
            HuggingFaceDatasetReader(
                "THUDM/humaneval-x",
                {"name": "cpp", "split": "test"},
                adapter=adapt_humanevalx_from_huggingface,
            ),
            CppCompiler(),
            RizinFunctionSegmentAndDisassemble(),
            ParquetWriter(
                output_folder=f"{DEFAULT_DATASETS_DIRECTORY}humaneval-x-dt-segmented-disassembled-rz",
                adapter=lambda self, doc: doc.metadata,
                max_file_size=50 * 1024 * 1024,
            ),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(HumanEvalX)
