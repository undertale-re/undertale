from argparse import ArgumentParser as ArgparseArgumentParser
from argparse import Namespace

from ..logging import setup_logging
from .dask import CLUSTER_TYPES


class ArgumentParser(ArgparseArgumentParser):
    """A custom argument parser providing some helpers and utilities."""

    def setup(self, arguments: Namespace) -> None:
        """Setup necessary services and configuration."""

        setup_logging()


class DatasetPipelineArgumentParser(ArgumentParser):
    """A custom argument parser for dataset pipelines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("input", help="input name")
        self.add_argument("output", help="output name")

        self.add_argument(
            "-p", "--parallelism", type=int, default=1, help="degree of parallelism"
        )
        self.add_argument(
            "-c",
            "--cluster",
            choices=CLUSTER_TYPES,
            default="local",
            help="cluster type",
        )
