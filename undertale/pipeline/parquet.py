"""Parquet parsing."""

from os import listdir
from os.path import join
from typing import List, Optional

from dask.dataframe import read_parquet

from ..logging import get_logger
from ..utils import assert_path_does_not_exist, assert_path_exists

logger = get_logger(__name__)


def resize_parquet(
    input: str | List[str],
    output: str,
    chunks: Optional[int] = None,
    size: Optional[int | str] = None,
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

    Returns:
        A list of paths to the generated files.

    Raises:
        ValueError: If not exactly one of ``chunks`` or ``size`` is specified.
    """

    if chunks is None and size is None:
        raise ValueError("exactly one of `chunks` or `size` must be specified")
    if chunks is not None and size is not None:
        raise ValueError("only one of `chunks` or `size` may be specified")

    logger.info(f"resizing {input!r} to {output!r} (chunks={chunks})")

    if isinstance(input, str):
        input = assert_path_exists(input)
    else:
        input = [assert_path_exists(i) for i in input]

    output = assert_path_does_not_exist(output)

    frame = read_parquet(input)
    frame = frame.repartition(npartitions=chunks, partition_size=size)
    frame.to_parquet(output, write_index=False, schema=None)

    return [join(output, f) for f in listdir(output)]


__all__ = ["resize_parquet"]
