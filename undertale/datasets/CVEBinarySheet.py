"""A dataset harvested from packages available via Advanced Package Tool.

``apt`` is a popular package management tool that ships with Ubuntu - this
dataset is built by downloading packages from a mirror, unpacking them, and
scanning for executable files.
"""

import logging
import os

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from .base import Dataset
from .pipeline.disassemblers.ghidra import GhidraDisassembler

logger = logging.getLogger(__name__)


def adapt_cvebinarysheet_from_dict(data: dict) -> dict:
    return {
        "id": data["filename"],
        "text": data["code"][0],
        "metadata": {"metadata": data["metadata"]},
    }


class LoadCVEBinarySheet(PipelineStep):
    _requires_dependencies = [
        "os",
        "shutil",
        "datasets",
        "datatrove",
    ]

    name = "CVEBinarySheet"
    type = "📖 - READER"

    def __init__(
        self,
        build_options: list[str] = ["debug"],
        wrapper_args: list[str] | None = None,
        data_loc: str = "~/undertale_shared/people/paul/data/test_binaries",
    ):

        self.data_loc = data_loc

    def run(self, data=None, rank=0, world_size=0):
        """"""
        import os

        from datatrove.data import Document

        ds = []
        for floc in os.listdir(self.data_loc):
            document = {}
            with open(os.path.join(self.data_loc, floc), "rb") as f:
                document["code"] = f.read()
            document["filename"] = floc
            document["metadata"] = ""
            ds.append(document)

        for row in ds:
            f_id = row["filename"]
            yield Document(
                id=f"fid={f_id}",
                text=row["code"],
                metadata={
                    "binary": row["code"],
                    "text": row["code"],
                    "metadata": {"value": row["metadata"]},
                },
            )


class CVEBinarySheet(Dataset):
    name = "apt-pkg"
    DEFAULT_DATASETS_DIRECTORY = "./"

    def get_pipeline(self, input, writer, parallelism):
        """"""
        from datatrove.utils.logging import logger

        if input == "binaries":
            executor = self.get_my_executor(input)
            executor.pipeline.append(
                ParquetWriter(
                    output_folder=f"{self.DEFAULT_DATASETS_DIRECTORY}apt-pkg",
                    adapter=lambda self, doc: doc.metadata,
                    max_file_size=50 * 1024 * 1024,
                )
            )
            logger.info("get_pipeline binaries")
            return executor

        return None

    def get_my_executor(self, input, ghidra_install_dir, venv_path, partition="RTX-24"):
        # Stage 0: Parse function bytes and metadata
        from datatrove.utils.logging import logger

        os.environ["GHIDRA_INSTALL_DIR"] = ghidra_install_dir

        logger.info("get_my_executor")

        slurm_parse = SlurmPipelineExecutor(
            pipeline=[
                LoadCVEBinarySheet(),
            ],
            venv_path=venv_path,
            logging_dir="~/undertale/logs",
            time="48:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=40,
            tasks=10,
            job_name="parse_CVEBinarySheet",
            partition=partition,
            sbatch_args={
                "distribution": "cyclic:cyclic",
                "chdir": "~/",
            },
        )

        # Stage 1: Disassemble binaries in parallel
        slurm_disassemble = SlurmPipelineExecutor(
            depends=slurm_parse,
            pipeline=[
                ParquetReader(f"{self.DEFAULT_DATASETS_DIRECTORY}apt-pkg"),
                # LoadAPTPackages(),
                GhidraDisassembler(),
            ],
            venv_path=venv_path,
            logging_dir="~/undertale/logs",
            time="48:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=40,
            tasks=10,
            job_name="disassemble_aptpkg",
            partition=partition,
            sbatch_args={
                "distribution": "cyclic:cyclic",
                "chdir": "~/",
            },
        )

        if input == "binaries":
            # return parse
            return slurm_parse
        elif input == "r2":
            # return disassemble
            return slurm_disassemble
        return None


if __name__ == "__main__":
    os.environ["GHIDRA_INSTALL_DIR"] = ""  # fill in with ghidra install directory
