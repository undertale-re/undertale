import argparse
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import datasets
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.executor.base import PipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from .. import logging as undertale_logging
from .schema import Schema

logger = logging.getLogger(__name__)

Pipeline = List[PipelineStep]


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
    "local": LocalPipelineExecutor,
    "slurm": SlurmPipelineExecutor,
}

default_executor = "local"


class Dataset(metaclass=ABCMeta):
    """The base class for all Undertale datasets.

    Arguments:
        writer: The name of the dataset writer to use.
        executor: The name of the dataset executor to use.
        logging_directory: A path to the directory to use for logging.
    """

    def __init__(
        self,
        writer: str = default_writer,
        executor: str = default_executor,
        logging_directory: Optional[str] = None,
    ):
        self.writer = writer
        self.executor = executor
        self.logging_directory = logging_directory or f"{self.name}-logs"

    schema: Optional[Schema] = None
    """The schema class that this dataset implements.

    This should be the literal class from the `schema` module.
    """

    def get_executor(self, pipeline: List[PipelineStep], **kwargs) -> PipelineExecutor:
        """Returns an executor for the current pipeline.

        Arguments:
            pipeline: A list of pipeline steps.
        """

        return executors[self.executor](
            pipeline, logging_dir=self.logging_directory, **kwargs
        )

    @abstractmethod
    def get_pipeline(
        self, input: str, writer: List[PipelineStep], parallelism: int = 1
    ) -> PipelineExecutor:
        """Build and return the dataset processing pipeline.

        This should make use of the `get_executor` method to wrap the
        configured executor.

        Arguments:
            input: Some input data from the user (path, name, etc.).
            writer: A series of output writer steps to add to the pipeline.
            parallelism: The degree of parallelism; dataset authors can choose
                to implement this however they want.
        """

    def build(self, input: str, output: str, parallelism: int = 1) -> None:
        writer = writers[self.writer](output)
        executor = self.get_pipeline(input, writer, parallelism)

        executor.run()

    @staticmethod
    def load(path: str) -> datasets.Dataset:
        """Load a dataset from the given path.

        Arguments:
            path: The path from which this dataset should be loaded.

        Returns:
            A dataset loaded from the given path.
        """

        logger.debug(f"loading dataset from {path!r}")

        return datasets.load_dataset(path, split="train")

    @staticmethod
    def store(dataset: datasets.Dataset, path: str) -> None:
        """Save a dataset to the given path.

        Arguments:
            path: The path where this dataset should be saved.
        """

        dataset.save_to_disk(path)

        logger.debug(f"wrote dataset to {path!r}")


def main(dataset_class: Dataset) -> None:
    """The CLI entrypoint for parsing a dataset.

    This should be called in `__main__` for dataset modules.

    Arguments:
        dataset_class: A dataset class to interact with.
    """

    undertale_logging.setup_logging()

    parser = argparse.ArgumentParser(
        description=f"parsing utilities for {dataset_class.__name__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--executor",
        choices=executors,
        default=default_executor,
        help="executor on which to run the given pipeline",
    )

    parser.add_argument(
        "-w",
        "--writer",
        choices=writers,
        default=default_writer,
        help="output writer (format)",
    )

    parser.add_argument(
        "-l", "--logging-directory", help="override logging directory path"
    )

    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        default=1,
        help="degree of parallelism (dataset implementation dependent)",
    )

    parser.add_argument("input", help="input location")
    parser.add_argument("output", help="output location")

    arguments = parser.parse_args()

    dataset = dataset_class(writer=arguments.writer, executor=arguments.executor)
    dataset.build(
        input=arguments.input,
        output=arguments.output,
        parallelism=arguments.parallelism,
    )
