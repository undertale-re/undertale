from pandas import DataFrame, read_parquet

from ..exceptions import SchemaError
from ..logging import get_logger
from ..utils import assert_path_exists
from ..utils import hash as hutil

logger = get_logger(__name__)


def hash_column(input: str, output: str, column: str) -> str:
    """Hashes a dataset column using our custom hash utility.

    Arguments:
        input: Path to source dataset.
        output: Path where the hashed binary dataset should be written.
        column: Name of the column to hash.

    Returns:
        The path to the generated parquet file.
    """

    input = assert_path_exists(input)

    logger.info(f"hashing column {column} in {input!r} to {output!r}")

    frame: DataFrame = read_parquet(input)

    if column not in frame.columns:
        raise SchemaError(f"dataset doesn't include the column {column}")

    new_column = f"{column}_hash"
    frame[new_column] = frame[column].apply(hutil)

    frame.to_parquet(output, schema=None)

    logger.info("successfully hashed the dataset")

    return output


__all__ = ["hash_column"]
