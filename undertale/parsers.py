"""Argument parsers."""

from argparse import ArgumentParser as ArgparseArgumentParser
from argparse import Namespace

from .logging import setup_logging
from .pipeline.dask import CLUSTER_TYPES


class ArgumentParser(ArgparseArgumentParser):
    """A custom argument parser providing some helpers and utilities."""

    def setup(self, arguments: Namespace) -> None:
        """Setup necessary services and configuration."""

        setup_logging()


class DatasetArgumentParser(ArgumentParser):
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


class ModelArgumentParser(ArgumentParser):
    """A custom argument parser for model training pipelines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("dataset", help="training dataset path")
        self.add_argument("output", help="model output directory")

        self.add_argument("-v", "--validation", help="validation dataset path")

        self.add_argument(
            "-c",
            "--checkpoint",
            help="trained model checkpoint from which to resume training",
        )
        self.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
        self.add_argument(
            "-a", "--accelerator", default="auto", help="accelerator to use"
        )
        self.add_argument(
            "-d",
            "--devices",
            type=int,
            default=1,
            help="number of accelerator devices per node",
        )
        self.add_argument(
            "-n", "--nodes", type=int, default=1, help="number of nodes to use"
        )
        self.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=48,
            help="maximum number of training epochs",
        )
        self.add_argument(
            "-l",
            "--learning-rate",
            type=float,
            default=None,
            help="learning rate (model-defined default)",
        )
        self.add_argument(
            "-w",
            "--warmup",
            type=float,
            default=None,
            help="learning rate warmup percentage (model-defined default)",
        )
        self.add_argument("-m", "--name", help="training run version name")


__all__ = ["ArgumentParser", "DatasetArgumentParser", "ModelArgumentParser"]
