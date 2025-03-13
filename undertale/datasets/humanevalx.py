import argparse
import logging

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
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
]


def adapt_to_flatten(self, document: Document) -> dict:
    sample = {}

    sample["id"] = document.id
    sample["code"] = document.text

    for k, v in document.metadata.items():
        sample[k] = v

    return sample


writers = {
    "parquet": lambda output: ParquetWriter(output, adapter=adapt_to_flatten),
}


executors = {
    "local": lambda pipeline, options: LocalPipelineExecutor(pipeline),
    "slurm": lambda pipeline, options: SlurmPipelineExecutor(
        pipeline,
        tasks=options.get("slurm_tasks"),
        time=options.get("slurm_time"),
        job_name=options.get("slurm_job_name"),
        mem_per_cpu_gb=options.get("slurm_mem_per_cpu"),
        cpus_per_task=options.get("slurm_cpus_per_task"),
        max_array_launch_parallel=True,
        partition=options.get("slurm_partition"),
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process the HumanEval-X dataset")

    parser.add_argument(
        "-e",
        "--executor",
        choices=executors,
        default="local",
        help="executor on which to run the given pipeline",
    )

    parser.add_argument(
        "--slurm-tasks", type=int, default=32, help="number of parallel tasks to run"
    )
    parser.add_argument(
        "--slurm-time",
        default="02:00:00",
        help="maximum time a job can be allowed to run",
    )
    parser.add_argument(
        "--slurm-job-name", default="build-dataset", help="a name for this job"
    )
    parser.add_argument(
        "--slurm-mem-per-cpu", type=int, default=1, help="maximum memory per cpu (GB)"
    )
    parser.add_argument(
        "--slurm-cpus-per-task", type=int, default=4, help="CPUs allocated to each task"
    )
    parser.add_argument(
        "--slurm-partition", help="partition to which this job should be submitted"
    )

    parser.add_argument(
        "-w",
        "--writer",
        choices=writers,
        default="parquet",
        help="output writer (format)",
    )

    parser.add_argument("output", help="output location")

    arguments = parser.parse_args()

    pipeline = humanevalx.copy()
    pipeline.append(writers[arguments.writer](arguments.output))

    executor = executors[arguments.executor](pipeline, arguments.__dict__)

    executor.run()
