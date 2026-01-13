from typing import List

from dask.dataframe import read_parquet

from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def dedupe_by_hash(input: List[str], output: str, column: str) -> str:
    """Dedupes a dataset globally using the sha256 of a specified bytes column.

    Arguments:
        input: A list of paths to each chunk of the dataset.
        output: Path where the deduped dataset should be written.
        column: Name of the bytes column whose hash is used for deduplication.

    Returns:
        A list of paths to the generated files.
    """

    input = [assert_path_exists(i) for i in input]

    output, _ = get_or_create_directory(output)

    frame = read_parquet(input)

    frame = frame.drop_duplicates(subset=[column], keep="first")
    frame = frame.drop(columns=[column])

    frame.to_parquet(output, schema=None)

    return output


__all__ = ["dedupe_by_hash"]
