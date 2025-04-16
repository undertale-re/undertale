import argparse
import code
import configparser
import datetime
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import datasets
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from .. import logging as undertale_logging
from .schema import Schema

logger = logging.getLogger(__name__)

Pipeline = List[PipelineStep]


class DatasetAlreadyExistsError(Exception):
    """Raised when attempting to commit a dataset that already exists."""


@dataclass
class BuildConfiguration:
    """Common configuration settings for building datasets.

    Arguments:
        parallelism: The degree of parallelism - interpreting this is up to the
            dataset author, but broadly it should correspond to the maximum
            number of parallel tasks to execute at any point in the full
            pipeline.
    """

    parallelism: int = 1

    @classmethod
    def from_dict(cls, settings: Dict):
        return cls(**settings)

    @classmethod
    def from_filepath(cls, path: str):
        config = configparser.ConfigParser()

        with open(path, "r") as f:
            config.read_file(f)

        settings = {
            "parallelism": int(config.get("undertale", "parallelism", fallback=None))
        }
        settings = {k: v for k, v in settings.items() if v}

        return cls(**settings)


class Dataset(metaclass=ABCMeta):
    """The base class for all Undertale datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this dataset.

        This should be lowercase, kebab-case.
        """

    schema: Optional[Schema] = None
    """The schema class that this dataset implements.

    This should be the literal class from the `schema` module.
    """

    DEFAULT_DATASETS_DIRECTORY = "~/undertale_shared/datasets/"
    """The default directory for dataset caching."""

    def get_path(self) -> str:
        """The path within the cache directory where this is located."""

        datasets = os.environ.get("UNDERTALE_DATASETS_DIRECTORY")

        if datasets is None:
            logger.warning(
                f"UNDERTALE_DATASETS_DIRECTORY environment variable is not set - assuming {self.DEFAULT_DATASETS_DIRECTORY!r}"
            )

            datasets = self.DEFAULT_DATASETS_DIRECTORY

        path = os.path.join(datasets, self.name)
        path = os.path.abspath(os.path.expanduser(path))

        return path

    def get_writer(self, output: Optional[str] = None) -> ParquetWriter:
        """A helper to get a writer step for the given output path.

        Arguments:
            output: Output location.

        Returns:
            A writer step that can be added to a pipeline.
        """

        output = output or self.get_path()

        def adapt_to_flatten(self, document: Document) -> dict:
            sample = {}

            sample["id"] = document.id
            sample["code"] = document.text

            for k, v in document.metadata.items():
                sample[k] = v

            return sample

        return ParquetWriter(output, adapter=adapt_to_flatten)

    def get_logging_directory(self) -> str:
        """Get the configured logging directory."""

        return f"{self.name}-logs"

    def get_executor_local(
        self, pipeline: List[PipelineStep], **kwargs
    ) -> LocalPipelineExecutor:
        """A helper to get a `LocalPipelineExecutor`.

        Arguments:
            pipeline: A list of `PipelineSteps`.
            **kwargs: Passed through to the `LocalPipelineExecutor`.

        Returns:
            A `LocalPipelineExecutor` for the given pipeline, with the given
            configuration parameters and defaults.
        """

        return LocalPipelineExecutor(
            pipeline, logging_dir=self.get_logging_directory(), **kwargs
        )

    def get_executor_slurm(
        self, pipeline: List[PipelineStep], **kwargs
    ) -> SlurmPipelineExecutor:
        """A helper to get a `SlurmPipelineExecutor`.

        Arguments:
            pipeline: A list of `PipelineSteps`.
            **kwargs: Passed through to the `SlurmPipelineExecutor`.

        Returns:
            A `SlurmPipelineExecutor` for the given pipeline, with the given
            configuration parameters and defaults.
        """

        return SlurmPipelineExecutor(
            pipeline, logging_dir=self.get_logging_directory(), **kwargs
        )

    def get_configuration_path(self) -> str:
        return f"{self.name}.ini"

    def get_configuration(self, settings: Optional[Dict] = None):
        """A helper to get a `BuildConfiguration`.

        Arguments:
            settings: Configuration settings.

        Returns:
            A `BuildConfiguration` build from the given settings.
        """

        if settings is not None:
            return BuildConfiguration.from_dict(settings)

        try:
            path = self.get_configuration_path()
            logger.warning(f"build configuration not provided - assuming {path!r}")
            return BuildConfiguration.from_filepath(path)
        except Exception as e:
            logger.warning(
                f"{path!r}: invalid configuration file - using defaults ({e})"
            )

        return BuildConfiguration()

    @abstractmethod
    def build(
        self, input: str, output: Optional[str] = None, settings: Optional[Dict] = None
    ) -> None:
        """Build this dataset.

        Implementations should make use of the helpers above for consistency.

        Arguments:
            input: Input location or data.
            output: Output location.
            settings: Configuration settings.
        """

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

        return self.load(self.get_path())

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

        path = self.get_path()

        if os.path.exists(path):
            if force:
                logger.info(f"overwriting {self.__class__.__name__} at {path!r}")

                shutil.rmtree(path)
            else:
                message = f"failed to save {self.__class__.__name__} - {self.get_path()!r} already exists in the dataset directory"

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
    parse_parser.add_argument("input", help="input location")
    parse_parser.add_argument("-o", "--output", help="override output location")
    parse_parser.add_argument(
        "-s",
        "--settings",
        help="configuration file path",
    )

    shell_parser = subparsers.add_parser(
        "shell",
        help="load the dataset and open a pyhton shell for exploration",
    )
    shell_parser.add_argument("-i", "--input", help="override input location")

    arguments = parser.parse_args()

    if arguments.command == "parse":
        settings = None
        if arguments.settings:
            settings = BuildConfiguration.from_filepath(arguments.settings).__dict__

        dataset = dataset_class()
        dataset.build(input=arguments.input, output=arguments.output, settings=settings)
    elif arguments.command == "shell":
        dataset = dataset_class()
        path = arguments.input or dataset.path

        logger.info(f"loading {dataset_class.__name__} from {path!r}")

        try:
            dataset = datasets.load_dataset(path)
        except Exception as e:
            logger.critical(e)
            exit(1)

        logger.info(f"{dataset_class.__name__} is available in the `dataset` variable")

        code.interact(local={"dataset": dataset})
