import argparse
import datetime
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import datasets
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from .. import logging as undertale_logging

logger = logging.getLogger(__name__)

Pipeline = List[PipelineStep]


DEFAULT_DATASETS_DIRECTORY = "~/undertale_shared/datasets/"
"""The default directory for dataset caching."""


class DatasetAlreadyExistsError(Exception):
    """Raised when attempting to commit a dataset that already exists."""


def adapt_to_flatten(self, document: Document) -> dict:
    sample = {}

    sample["id"] = document.id
    sample["code"] = document.text

    for k, v in document.metadata.items():
        sample[k] = v

    return sample


writers = {
    "parquet": lambda output: [ParquetWriter(output, adapter=adapt_to_flatten)],
}

default_writer = "parquet"

executors = {
    "local": lambda pipeline, options: LocalPipelineExecutor(
        pipeline,
        tasks=options.get("executor_tasks"),
        logging_dir=options.get("executor_logging_directory"),
    ),
    "slurm": lambda pipeline, options: SlurmPipelineExecutor(
        pipeline,
        tasks=options.get("executor_tasks"),
        time=options.get("slurm_time"),
        job_name=options.get("slurm_job_name"),
        mem_per_cpu_gb=options.get("slurm_mem_per_cpu"),
        cpus_per_task=options.get("slurm_cpus_per_task"),
        max_array_launch_parallel=True,
        partition=options.get("slurm_partition"),
        logging_dir=options.get("executor_logging_directory"),
    ),
}

default_executor = "local"


class Dataset(metaclass=ABCMeta):
    """The base class for all Undertale datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this dataset.

        This should be lowercase, kebab-case.
        """

    @property
    @abstractmethod
    def readers(self) -> Dict[str, Callable[[str], Pipeline]]:
        """A mapping of names to reader pipelines factories.

        These pipelines should handle all of the steps for reading a raw
        dataset, but not yet process it.

        Factories take a single argument - an input string provided by the user
        that can be used to indicate from where data should be read.

        These allow you to specify multiple possible sources of input to your
        pipeline.
        """

    @property
    @abstractmethod
    def default_reader(self) -> str:
        """The default reader pipeline to use."""

    @property
    @abstractmethod
    def pipelines(self) -> Dict[str, Pipeline]:
        """A mapping of names to processing pipelines.

        These pipelines should handle only data processing, not parsing.
        """

    @property
    @abstractmethod
    def default_pipeline(self) -> str:
        """The default processing pipeline to use."""

    def process(
        self,
        input: str,
        output: str,
        reader: Optional[str] = None,
        pipeline: Optional[str] = None,
        executor: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        writer: Optional[str] = None,
    ) -> None:
        """Process this dataset from raw datas files or input.

        Arguments:
            input: Input value (path, name, etc.).
            output: Output value (path, name, etc.).
            reader: Name of the reader pipeline to use.
            pipeline: Name of the processing pipeline to run.
            executor: Name of the executor to use.
            options: A dictionary of options to provide to the executor.
            writer: Name of the writer pipeline to use.
        """

        reader = reader or self.default_reader
        pipeline = pipeline or self.default_pipeline
        executor = executor or default_executor
        options = options or {}
        writer = writer or default_writer

        workflow = self.readers[reader](input).copy()
        workflow += self.pipelines[pipeline].copy()
        workflow += writers[writer](output)
        executor = executors[executor](workflow, options)

        executor.run()

    @property
    def path(self) -> str:
        """The path within the cache directory where this is located."""

        datasets = os.environ.get("UNDERTALE_DATASETS_DIRECTORY")

        if datasets is None:
            logger.warning(
                f"UNDERTALE_DATASETS_DIRECTORY environment variable is not set - assuming {DEFAULT_DATASETS_DIRECTORY!r}"
            )

            datasets = DEFAULT_DATASETS_DIRECTORY

        path = os.path.join(datasets, self.name)
        path = os.path.abspath(os.path.expanduser(path))

        return path

    @staticmethod
    def load(path: str) -> datasets.Dataset:
        """Load a dataset from the given path.

        Arguments:
            path: The path from which this dataset should be loaded.

        Returns:
            A dataset loaded from the given path.
        """

        logger.debug(f"loading dataset from {path!r}")

        return datasets.load_dataset(path)

    @staticmethod
    def store(dataset: datasets.Dataset, path: str) -> None:
        """Save a dataset to the given path.

        Arguments:
            path: The path where this dataset should be saved.
        """

        dataset.save_to_disk(path)

        logger.debug(f"wrote dataset to {path!r}")

    def fetch(self) -> datasets.Dataset:
        """Fetch this dataset from the datasets directory.

        Returns:
            A dataset object loaded from the datasets directory.
        """

        return self.load(self.path)

    def commit(self, dataset: datasets.Dataset, force=False) -> None:
        """Commit this dataset to the datasets directory.

        Attempt to save this dataset to the datasets directory path.

        Arguments:
            dataset: The dataset to commit.
            force: If `True` overwrite any existing datasets in the commit
                location. Otherwise raise an exception if something exists
                there.

        Raises:
            DatasetAlreadyExistsError: if the dataset already exists and
                `force` is `False`.
        """

        path = self.path

        if os.path.exists(path):
            if force:
                logger.info(f"overwriting {self.__class__.__name__} at {path!r}")

                shutil.rmtree(path)
            else:
                message = f"failed to save {self.__class__.__name__} - {self.path!r} already exists in the dataset directory"

                logger.error(message)

                raise DatasetAlreadyExistsError(message)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        dataset._info.dataset_name = self.__class__.__name__
        dataset._info.version = datasets.Version("0.0.0", description=f"{timestamp}")

        self.store(dataset, path)


def main(dataset_class: Dataset) -> None:
    """The CLI entrypoint for parsing a dataset.

    This should be called in `__main__` for dataset modules.

    Arguments:
        dataset_class: A dataset class to interact with.
    """

    undertale_logging.setup_logging()

    dataset = dataset_class()

    parser = argparse.ArgumentParser(
        description=f"process {dataset.name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-r",
        "--reader",
        choices=dataset.readers,
        default=dataset.default_reader,
        help="the reader to use",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        choices=dataset.pipelines,
        default=dataset.default_pipeline,
        help="the pipeline to run",
    )

    parser.add_argument(
        "-e",
        "--executor",
        choices=executors,
        default=default_executor,
        help="executor on which to run the given pipeline",
    )

    parser.add_argument(
        "--executor-logging-directory",
        default="./logs/",
        help="logging directory for executor logs",
    )
    parser.add_argument(
        "--executor-tasks", type=int, default=1, help="number of parallel tasks to run"
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
        default=default_writer,
        help="output writer (format)",
    )

    parser.add_argument("input", help="input location")

    parser.add_argument("-o", "--output", default=dataset.path, help="output location")

    arguments = parser.parse_args()

    dataset.process(
        input=arguments.input,
        output=arguments.output,
        reader=arguments.reader,
        pipeline=arguments.pipeline,
        executor=arguments.executor,
        options=arguments.__dict__,
        writer=arguments.writer,
    )
