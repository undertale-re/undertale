import logging

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter

logger = logging.getLogger(__name__)


def adapt_humanevalx_from_huggingface(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data["task_id"],
        "text": f"{data['declaration']}{data['canonical_solution']}",
        "metadata": {"summary": data["prompt"]},
    }


def adapt_humanevalx_to_parquet(self, document: Document) -> dict:
    sample = {}

    sample["id"] = document.id
    sample["code"] = document.text

    for k, v in document.metadata.items():
        if k in ["dataset"]:
            continue
        sample[k] = v

    return sample


class CompileCpp(PipelineStep):
    type = "⚙️ - PROCESS"
    name = "🏗️ Compile C++"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import os
        import subprocess
        import tempfile

        from datatrove.data import Document

        if not data:
            return

        for document in data:
            with self.track_time():
                source = document.text

                working = tempfile.TemporaryDirectory()

                sourcefile = os.path.join(working.name, "source.cpp")

                with open(sourcefile, "w") as f:
                    f.write(source)

                objectfile = os.path.join(working.name, "source.o")

                process = subprocess.run(
                    f"g++ -c {sourcefile} -o {objectfile}",
                    cwd=working.name,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if process.returncode == 0:
                    with open(objectfile, "rb") as f:
                        code = f.read()

                    metadata = document.metadata.copy()
                    metadata["source"] = document.text

                    yield Document(id=document.id, text=code, metadata=metadata)

                    self.stat_update("succeeded")
                else:
                    message = "failed to compile source:\n"
                    message += "=" * 80 + "\n"
                    message += source.strip() + "\n"
                    message += "-" * 36 + " stdout " + "-" * 36 + "\n"
                    message += process.stdout.decode().strip() + "\n"
                    message += "-" * 36 + " stderr " + "-" * 36 + "\n"
                    message += process.stderr.decode().strip() + "\n"
                    message += "=" * 80

                    logger.warning(message)

                    self.stat_update("failed")


humanevalx = [
    HuggingFaceDatasetReader(
        "THUDM/humaneval-x",
        {"name": "cpp", "split": "test"},
        adapter=adapt_humanevalx_from_huggingface,
    ),
    CompileCpp(),
    ParquetWriter("output", adapter=adapt_humanevalx_to_parquet),
]


if __name__ == "__main__":
    executor = LocalPipelineExecutor(
        pipeline=humanevalx,
    )
    executor.run()
