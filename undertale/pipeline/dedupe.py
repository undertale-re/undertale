import hashlib
from typing import List, Optional, Union

from dask.dataframe import read_parquet
from dask.distributed import Client as DaskClient
from dask.distributed import Future
from pandera.typing import Series

from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def sha256_series(value: Series[bytes]) -> Series[str]:
    return value.map(lambda row: hashlib.sha256(row).hexdigest()).astype("string")


def dedupe_by_sha256(
    input: Union[str, List[str], List[Future]],
    output: str,
    column: str = "binary",
    client: Optional[DaskClient] = None,
) -> str:
    """Dedupes a dataset globally using the sha256 of a specified bytes column.

    Arguments:
        input:
            * Path to the parquet dataset directory or
            * A list of paths to each chunk of the dataset or
            * A list of Dask Futures that result in the dataset's file paths (Note: `client` will be required).
        output: Path to the target directory.
        column: Name of the bytes column whose sha256 hash is used for deduplication.
        client: The Dask Client used to issue tasks (required then inputs are Futures).

    Returns:
        A list of paths to the generated files.
    """
    if isinstance(input[0], Future):
        if not client:
            raise ValueError(
                "given that `input` is a list of Dask Futures, a Dask `client` must be specified"
            )
        else:
            input = client.gather(input)

    if isinstance(input, str):
        input = assert_path_exists(input)
    else:
        input = [assert_path_exists(i) for i in input]

    output, _ = get_or_create_directory(output)

    frame = read_parquet(input)

    new_column = f"{column}_sha256"
    frame[new_column] = frame[column].map_partitions(
        sha256_series, meta=(new_column, "string")
    )

    frame = frame.drop_duplicates(subset=[new_column], keep="first")

    frame = frame.drop(columns=[new_column])

    frame.to_parquet(output, schema=None)

    return output


__all__ = ["dedupe_by_sha256"]
