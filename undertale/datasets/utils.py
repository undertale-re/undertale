import argparse
import code
import logging
from typing import Callable, Dict, List

import datasets
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from .. import logging as undertale_logging

logger = logging.getLogger(__name__)


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


def main(
    readers: Dict[str, Callable[[str], List[PipelineStep]]],
    default_reader: str,
    pipelines: Dict[str, List[PipelineStep]],
    default_pipeline: str,
) -> None:
    """The CLI entrypoint for parsing a dataset.

    This should be called in `__main__` for dataset modules.

    Arguments:
        readers: A mapping of reader name to lists of pipeline steps.
        default_reader: The default reader name.
        pipelines: A mapping of pipeline name to lists of pipeline steps.
        default_pipeline: The default pipeline name.
    """

    parser = argparse.ArgumentParser(
        description="process this dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-r",
        "--reader",
        choices=readers,
        default=default_reader,
        help="the reader to use",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        choices=pipelines,
        default=default_pipeline,
        help="the pipeline to run",
    )

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

    parser.add_argument("input", help="input location")
    parser.add_argument("output", help="output location")

    arguments = parser.parse_args()

    pipeline = readers[arguments.reader](arguments.input).copy()
    pipeline += pipelines[arguments.pipeline].copy()
    pipeline.append(writers[arguments.writer](arguments.output))
    executor = executors[arguments.executor](pipeline, arguments.__dict__)

    executor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="utilities for managing datasets")

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="dataset utility to run"
    )

    shell_parser = subparsers.add_parser(
        "shell",
        help="load a dataset and open a python shell for exploration",
    )

    shell_parser.add_argument(
        "path",
        help="path to a dataset file to load (or the name of one on the HuggingFace hub)",
    )

    arguments = parser.parse_args()

    undertale_logging.setup_logging()

    if arguments.command == "shell":
        try:
            dataset = datasets.load_dataset(arguments.path)
        except Exception as e:
            logger.critical(e)
            exit(1)

        logger.info("the loaded dataset is available in the `dataset` variable")
        code.interact(local={"dataset": dataset})
