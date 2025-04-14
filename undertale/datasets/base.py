import argparse
import code
import datetime
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import datasets
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.executor.base import PipelineExecutor
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

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this dataset.

        This should be lowercase, kebab-case.
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
        """Build and reutrn the dataset processing pipeline.

        This should make use of the `executor` method to wrape the configured
        executor.

        Arguments:
            input: Some input data from the user (path, name, etc.).
            writer: A series of output writer steps to add to the pipeline.
            parallelism: The degree of parallelism; dataset authors can choose
                to implement this however they want.
        """

    def build(
        self, input: str, output: Optional[str] = None, parallelism: int = 1
    ) -> None:
        output = output or self.path

        writer = writers[self.writer](output)
        executor = self.get_pipeline(input, writer, parallelism)

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

    parser = argparse.ArgumentParser(
        description=f"parsing utilities for {dataset_class.__name__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="dataset operation to perform"
    )

    parse_parser = subparsers.add_parser("parse", help="parse the dataset")

    parse_parser.add_argument(
        "-e",
        "--executor",
        choices=executors,
        default=default_executor,
        help="executor on which to run the given pipeline",
    )

    parse_parser.add_argument(
        "-w",
        "--writer",
        choices=writers,
        default=default_writer,
        help="output writer (format)",
    )

    parse_parser.add_argument(
        "-l", "--logging-directory", help="override logging directory path"
    )

    parse_parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        default=1,
        help="degree of parallelism (dataset implementation dependent)",
    )

    parse_parser.add_argument("input", help="input location")

    parse_parser.add_argument("-o", "--output", help="override output location")

    shell_parser = subparsers.add_parser(
        "shell",
        help="load the dataset and open a pyhton shell for exploration",
    )

    shell_parser.add_argument("-i", "--input", help="override input location")

    arguments = parser.parse_args()

    if arguments.command == "parse":
        dataset = dataset_class(writer=arguments.writer, executor=arguments.executor)
        dataset.build(
            input=arguments.input,
            output=arguments.output,
            parallelism=arguments.parallelism,
        )
    elif arguments.command == "shell":
        dataset = dataset_class(
            writer=writers[default_writer], executor=executors[default_executor]
        )
        path = arguments.input or dataset.path

        logger.info(f"loading {dataset_class.__name__} from {path!r}")

        try:
            dataset = datasets.load_dataset(path)
        except Exception as e:
            logger.critical(e)
            exit(1)

        logger.info(f"{dataset_class.__name__} is available in the `dataset` variable")

        code.interact(local={"dataset": dataset})
