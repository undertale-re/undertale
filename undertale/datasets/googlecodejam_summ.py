from datatrove.pipeline.readers import ParquetReader

from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import GhidraDisassembler
# from .pipeline.summariziers.codestral import CodestralSummarizier
from .pipeline.summariziers.openai import OpenAISummarizier


def adapt_googlecodejam(self, data: dict, path: str, id_in_file: int | str) -> dict:
    return {
        "id": data["row"],
        "text": data["source"],
        "metadata": {"summary": data["statement"]},
    }

def basic_adapter(self, data:dict, path: str, id_in_file: int | str) -> dict:
    return {
        "id": id_in_file,
        "text": data["source"],
        "metadata": data
    }

class GoogleCodeJamSumm(Dataset):
    def get_pipeline(self, input, writer, parallelism):
        steps = [
            # ParquetReader(
            #     data_folder="/home/st25587/undertale_shared/datasets/gcj_testset",
            #     adapter=adapt_googlecodejam,
            # ),
            # CppCompiler(),
            # GhidraDisassembler(),
            ParquetReader(data_folder="~/undertale_shared/datasets/google-code-jam",
                          adapter=basic_adapter),
            OpenAISummarizier(),
        ]
        steps.extend(writer)

        return self.get_executor(steps, tasks=parallelism)


if __name__ == "__main__":
    main(GoogleCodeJamSumm)
