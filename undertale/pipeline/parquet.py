"""Parquet parsing."""

from os import listdir
from os.path import join
from typing import List, Optional

from dask.dataframe import read_parquet as dask_read_parquet
from pandas import read_parquet as pandas_read_parquet

from ..exceptions import SchemaError
from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_directory, hash

logger = get_logger(__name__)


def hash_parquet_column(input: str, output: str, column: str, target: str) -> str:
    """Hashes a dataset column.

    Arguments:
        input: Path to source dataset.
        output: Path where the hashed dataset should be written.
        column: Name of the column to hash.
        target: Name of the hash column to create.

    Returns:
        The path to the generated parquet file.
    """

    input = assert_path_exists(input)

    logger.info(f"hashing column {column} in {input!r} to {output!r}")

    frame = pandas_read_parquet(input)

    if column not in frame.columns:
        raise SchemaError(f"dataset doesn't include the column {column}")

    frame[target] = frame[column].apply(hash)

    frame.to_parquet(output, schema=None)

    logger.info(f"successfully hashed {len(frame)} rows")

    return output


def resize_parquet(
    input: str | List[str],
    output: str,
    chunks: Optional[int] = None,
    size: Optional[int | str] = None,
    deduplicate: Optional[List[str]] = None,
    drop: Optional[List[str]] = None,
) -> List[str]:
    """Resize a parquet dataset.

    This method is memory-efficient and supports larger-than-memory parquet
    datasets using Dask.

    Exactly one of ``chunks`` or ``size`` must be specified.

    Note:
        The number of chunks is guaranteed but the number of rows per chunk may
        not be exactly the same. If ``chunks`` exceeds the number of rows in
        the dataset, ``chunks`` parquet files will still be created, but some
        of them will be empty.

    Arguments:
        input: Path to the parquet dataset directory or a list of paths to each
            chunk of the dataset.
        output: Path to the target directory.
        chunks: Number of chunk files to generate.
        size: The maximum chunk size in bytes or string representation (e.g.,
            "25MB")
        deduplicate: If provided, deduplicate the dataset by the given list of
            column names (unique together).
        drop: If provided, drop the given column names.

    Returns:
        A list of paths to the generated files.

    Raises:
        ValueError: If not exactly one of ``chunks`` or ``size`` is specified.
    """

    if chunks is None and size is None:
        raise ValueError("exactly one of `chunks` or `size` must be specified")
    if chunks is not None and size is not None:
        raise ValueError("only one of `chunks` or `size` may be specified")

    if isinstance(input, str):
        input = assert_path_exists(input)
    else:
        input = [assert_path_exists(i) for i in input]

    output, created = get_or_create_directory(output)

    if created:
        frame = dask_read_parquet(input)

        if deduplicate:
            logger.info(f"deduplicating dataset by column(s): {', '.join(deduplicate)}")

            for column in deduplicate:
                if column not in frame.columns:
                    raise SchemaError(f"dataset does not include the column {column!r}")

            frame = frame.drop_duplicates(subset=deduplicate, keep="first")

        if drop:
            logger.info(f"dropping dataset column(s): {', '.join(drop)}")

            for column in drop:
                if column not in frame.columns:
                    raise SchemaError(f"dataset does not include the column {column!r}")

            frame = frame.drop(columns=drop)

        logger.info(f"resizing dataset to (chunks={chunks}, size={size})")

        frame = frame.repartition(npartitions=chunks, partition_size=size)

        logger.info(f"writing dataset to {output!r}")

        frame.to_parquet(output, write_index=False, schema=None)

    return [join(output, f) for f in listdir(output)]


__all__ = ["hash_parquet_column", "resize_parquet"]
