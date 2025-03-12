import abc
import argparse
import code
import datetime
import logging
import os
import shutil
import typing

import datasets

from .. import logging as undertale_logging
from .. import utils

logger = logging.getLogger(__name__)


# DEFAULT_DATASETS_DIRECTORY = "~/undertale_shared/datasets/"
DEFAULT_DATASETS_DIRECTORY = "/scratch/pa27879/"
"""The default directory for dataset caching.

On LLSC this will map into the shared project directory.
"""


class DatasetAlreadyExistsError(Exception):
    """Raised when attempting to commit a dataset that already exists."""


class Dataset(datasets.Dataset, metaclass=abc.ABCMeta):
    """The base class for all Undertale datasets.

    This class encapsulates a few different requirements for implemented
    datasets:
        1. *Metadata* - information about what the dataset is and where it came
            from.
        2. *Scripted Paring* - all the code necessary to produce the dataset so
            we can reproduce it later if we need to.
        3. *Caching* - storage and retrieval from a (possibly shared) directory.
    """

    url = ""
    """The URL where this raw dataset was downloaded."""

    description = ""
    """A short (one sentence) description of this dataset."""

    path = ""
    """Path in the datasets directory where this dataset may be found.

    This should be the path relative to the datasets directory.
    """

    schema = None
    """The schema class that this dataset implements.

    This should be the literal class from the `schema` module.
    """

    @classmethod
    def get_path(cls) -> str:
        """Get the path within the cache directory where this is located.

        Returns:
            The absolute path to the cache location of this dataset.
        """

        datasets = os.environ.get("UNDERTALE_DATASETS_DIRECTORY")

        if datasets is None:
            logger.warning(
                f"UNDERTALE_DATASETS_DIRECTORY environment variable is not set - assuming {DEFAULT_DATASETS_DIRECTORY!r}"
            )

            datasets = DEFAULT_DATASETS_DIRECTORY

        path = os.path.join(datasets, cls.path)
        path = os.path.abspath(os.path.expanduser(path))

        return path

    @classmethod
    @abc.abstractmethod
    def parse(cls, path: str, processes: typing.Optional[int] = None):
        """Parse a given raw dataset file or directory.

        Arguments:
            path: The path to a file or directory to process into a dataset.
            processes: The number of parallel processes to use.

        Returns:
            A Dataset object created from the given file.
        """

        pass

    transforms: typing.List[typing.Any] = []
    """A list of transforms that should be applied to this dataset.

    Concrete classes should define this - it can be an empty list.
    """

    def transform(self, processes=None):
        """Apply configured transforms to this dataset.

        Arguments:
            processes: The number of parallel processes to use.

        Returns:
            The final, transformed version of this dataset after all configured
            transforms have been applied.
        """

        dataset = self

        if len(self.transforms) == 0:
            logger.warning(f"{self.__class__.__name__} does not include any transforms")

        for transform in self.transforms:
            dataset = transform.apply(dataset, processes=processes)

        return dataset

    @classmethod
    def load(cls, path: str):
        """Load this dataset from the given path.

        Arguments:
            path: The path from which this dataset should be loaded.
        """

        logger.info(f"loading {cls.__name__} from {path!r}")

        dataset = cls.load_from_disk(path)
        dataset.__class__ = cls

        return dataset

    def store(self, path: str) -> None:
        """Save this dataset to the given path.

        Arguments:
            path: The path where this dataset should be saved.
        """

        self.save_to_disk(path)

        logger.info(f"wrote {self.__class__.__name__} to {path!r}")

    @classmethod
    def fetch(cls):
        """Fetch this dataset from the datasets directory."""

        path = cls.get_path()

        return cls.load(path)

    def commit(self, force=False) -> None:
        """Commit this dataset to the datasets directory.

        Attempt to save this dataset to the datasets directory path.

        Arguments:
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
                message = f"failed to save {self.__class__.__name__} - {self.path!r} already exists in the dataset directory"

                logger.error(message)
                raise DatasetAlreadyExistsError(message)

        self._info.dataset_name = self.__class__.__name__
        self._info.description = self.description
        self._info.homepage = self.url

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        self._info.version = datasets.Version("0.0.0", description=f"{timestamp}")

        self.store(path)


def main(classes):
    options = {c.__name__: c for c in classes}
    default = classes[0].__name__

    parser = argparse.ArgumentParser(
        description=f"parsing/loading utilities for {', '.join(options)}"
    )

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-v",
        "--variant",
        choices=options,
        default=default,
        help="dataset variant (default: %(default)s)",
    )
    parent.add_argument(
        "-l",
        "--logging-level",
        choices=undertale_logging.LEVELS,
        default="info",
        help="logging level (default: %(default)s)",
    )
    parent.add_argument(
        "--logging-file", default=None, help="logging file (default: %(default)s)"
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="dataset operation to perform"
    )

    parse_parser = subparsers.add_parser(
        "parse", help="parse and commit the dataset", parents=[parent]
    )
    parse_parser.add_argument("path", help="path to the raw file to parse")
    parse_parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing dataset"
    )
    parse_parser.add_argument(
        "-p",
        "--processes",
        default=1,
        type=int,
        help="number of parallel processes to use (default: %(default)s)",
    )

    subparsers.add_parser(
        "shell",
        help="load the dataset and open a python shell for exploration",
        parents=[parent],
    )

    arguments = parser.parse_args()

    undertale_logging.setup_logging(
        level=undertale_logging.LEVELS[arguments.logging_level],
        file=arguments.logging_file,
    )

    cls = options[arguments.variant]

    if arguments.command == "parse":
        utils.suppress_sigint()

        logger.info(f"parsing {cls.__name__} from {arguments.path!r}")
        dataset = cls.parse(arguments.path, processes=arguments.processes)
        logger.info(f"transforming {cls.__name__}")
        dataset = dataset.transform(processes=arguments.processes)
        logger.info(f"processed {cls.__name__} with {len(dataset)} samples")
        logger.info(f"committing {cls.__name__}")
        dataset.commit(force=arguments.force)
    elif arguments.command == "shell":
        dataset = cls.fetch()
        logger.info(f"{cls.__name__} is available in the `dataset` variable")
        code.interact(local={"dataset": dataset})
