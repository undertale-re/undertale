"""Parquet parsing."""

from abc import ABC, abstractmethod
from os import listdir
from os.path import join
from typing import Dict, List, Optional

from dask.dataframe import DataFrame
from dask.dataframe import read_parquet as dask_read_parquet
from pandas.api.types import is_string_dtype, pandas_dtype

from ..exceptions import SchemaError
from ..logging import get_logger
from ..utils import (
    assert_path_exists,
    get_or_create_directory,
    hash,
    write_parquet,
)

logger = get_logger(__name__)


class ParquetOperation(ABC):
    """Abstract base class for parquet DataFrame transformations."""

    @abstractmethod
    def __call__(self, frame: DataFrame) -> DataFrame:
        """Apply this operation to a Dask DataFrame.

        Arguments:
            frame: The input DataFrame.

        Returns:
            The transformed DataFrame.
        """


class HashColumn(ParquetOperation):
    """Add a column containing the hash of an existing column.

    Arguments:
        column: Name of the column to hash.
        target: Name of the new hash column to create.
    """

    def __init__(self, column: str, target: str):
        self.column = column
        self.target = target

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"hashing column {self.column!r} into {self.target!r}")

        if self.column not in frame.columns:
            raise SchemaError(f"dataset does not include the column {self.column!r}")

        return frame.assign(
            **{self.target: frame[self.column].apply(hash, meta=(self.target, str))}
        )


class Deduplicate(ParquetOperation):
    """Remove duplicate rows by a set of columns.

    Arguments:
        columns: Column names to deduplicate by (unique together).
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"deduplicating dataset by column(s): {', '.join(self.columns)}")

        for column in self.columns:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")

        return frame.drop_duplicates(subset=self.columns, keep="first")


class Drop(ParquetOperation):
    """Drop specific columns from the dataset.

    Arguments:
        columns: Column names to drop.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"dropping dataset column(s): {', '.join(self.columns)}")

        for column in self.columns:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")

        return frame.drop(columns=self.columns)


class Keep(ParquetOperation):
    """Keep only specific columns from the dataset.

    Arguments:
        columns: Column names to keep.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"keeping only dataset column(s): {', '.join(self.columns)}")

        for column in self.columns:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")

        return frame[self.columns]


class Filter(ParquetOperation):
    """Keep rows where every column contains its target substring.

    Arguments:
        clauses: Mapping of column names to substrings that must be present.
    """

    def __init__(self, clauses: Dict[str, str]):
        if not clauses:
            raise ValueError("clauses must not be empty")
        self.clauses = clauses

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"filtering dataset by column(s): {', '.join(self.clauses)}")

        for column in self.clauses:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")
            if not is_string_dtype(frame[column].dtype):
                raise ValueError(
                    f"column {column!r} is not a string column"
                    f" (got {frame[column].dtype})"
                )

        checks = [
            frame[column].str.contains(value, regex=False, na=False)
            for column, value in self.clauses.items()
        ]
        mask = checks[0]
        for check in checks[1:]:
            mask = mask & check

        return frame[mask]


class Exclude(ParquetOperation):
    """Remove rows where every column contains its target substring.

    Arguments:
        clauses: Mapping of column names to substrings that must be present
            for a row to be excluded.
    """

    def __init__(self, clauses: Dict[str, str]):
        if not clauses:
            raise ValueError("clauses must not be empty")
        self.clauses = clauses

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"excluding from dataset by column(s): {', '.join(self.clauses)}")

        for column in self.clauses:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")
            if not is_string_dtype(frame[column].dtype):
                raise ValueError(
                    f"column {column!r} is not a string column"
                    f" (got {frame[column].dtype})"
                )

        checks = [
            frame[column].str.contains(value, regex=False, na=False)
            for column, value in self.clauses.items()
        ]
        mask = checks[0]
        for check in checks[1:]:
            mask = mask & check

        return frame[~mask]


class Rename(ParquetOperation):
    """Rename columns in the dataset.

    Arguments:
        mapping: A mapping of old column names to new column names.
    """

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"renaming dataset column(s): {', '.join(self.mapping)}")

        for column in self.mapping:
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")

        return frame.rename(columns=self.mapping)


class Shuffle(ParquetOperation):
    """Randomly shuffle rows in the dataset.

    Note:
        This is a per-partition shuffle. In most cases, this is good enough,
        but if you want to get closer to a global shuffle, consider
        repartitioning to a smaller number of shards before shuffling. For
        example, repartitioning to a single shard is equivalent to global
        shuffle, but requires the full dataset to fit within the memory of a
        single worker.

    Arguments:
        seed: Random seed for reproducibility. If ``None``, the shuffle is
            non-deterministic.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info("shuffling dataset")
        return frame.sample(frac=1, random_state=self.seed)


class Cast(ParquetOperation):
    """Cast column types in the dataset.

    Arguments:
        mapping: A mapping of column names to new column type strings.
    """

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def __call__(self, frame: DataFrame) -> DataFrame:
        logger.info(f"casting dataset column(s): {', '.join(self.mapping)}")

        for column, dtype in self.mapping.items():
            if column not in frame.columns:
                raise SchemaError(f"dataset does not include the column {column!r}")

            try:
                pandas_dtype(dtype)
            except TypeError:
                raise ValueError(f"invalid dtype {dtype!r}")

        return frame.astype(self.mapping)


class Repartition(ParquetOperation):
    """Repartition the dataset by number of chunks or target chunk size.

    Arguments:
        chunks: Number of chunk files to generate.
        size: The maximum chunk size in bytes or string representation
            (e.g., "25MB").

    Raises:
        ValueError: If not exactly one of ``chunks`` or ``size`` is specified.
    """

    def __init__(
        self,
        chunks: Optional[int] = None,
        size: Optional[int | str] = None,
    ):
        if chunks is None and size is None:
            raise ValueError("exactly one of `chunks` or `size` must be specified")
        if chunks is not None and size is not None:
            raise ValueError("only one of `chunks` or `size` may be specified")

        self.chunks = chunks
        self.size = size

    def __call__(self, frame: DataFrame) -> DataFrame:
        if self.chunks is not None:
            logger.info(f"repartitioning dataset to {self.chunks} chunk(s)")
        else:
            logger.info(f"repartitioning dataset to chunk size {self.size}")

        return frame.repartition(npartitions=self.chunks, partition_size=self.size)


def modify_parquet(
    input: str | List[str],
    output: str,
    operations: List[ParquetOperation],
    compression: Optional[str] = None,
) -> List[str]:
    """Modify a parquet dataset by applying a sequence of operations.

    This method is memory-efficient and supports larger-than-memory parquet
    datasets using Dask.

    Note:
        When using :class:`Repartition` with ``chunks``, the number of chunks
        is guaranteed but the number of rows per chunk may not be exactly the
        same. If ``chunks`` exceeds the number of rows in the dataset,
        ``chunks`` parquet files will still be created, but some of them will
        be empty.

    Arguments:
        input: Path to the parquet dataset directory or a list of paths to
            each chunk of the dataset.
        output: Path to the target directory.
        operations: A list of :class:`ParquetOperation` instances to apply
            in order.
        compression: If provided, name of the algorithm to use (e.g.,
            'snappy'). By default no compression will be used. See the
            ``pyarrow`` documentation for a list of supported compression
            methods.

    Returns:
        A list of paths to the generated files.
    """

    if isinstance(input, str):
        input = assert_path_exists(input)
    else:
        input = [assert_path_exists(i) for i in input]

    output, created = get_or_create_directory(output)

    if created:
        frame = dask_read_parquet(input)

        for operation in operations:
            frame = operation(frame)

        logger.info(f"writing dataset to {output!r}")

        write_parquet(frame, output, write_index=False, compression=compression)

    return [join(output, f) for f in listdir(output)]


__all__ = [
    "ParquetOperation",
    "HashColumn",
    "Deduplicate",
    "Drop",
    "Keep",
    "Filter",
    "Exclude",
    "Rename",
    "Shuffle",
    "Cast",
    "Repartition",
    "modify_parquet",
]
