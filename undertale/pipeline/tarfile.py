"""Tar file parsing."""

import tarfile

from ..exceptions import InvalidFileType
from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def extract_tarfile(input: str, output: str) -> str:
    """Decompress a given tarfile.

    Arguments:
        input: Path to the tar input file.
        output: Path to the output directory to create.

    Returns:
        The output directory created.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_directory(output)

    if not created:
        return output

    logger.info(f"extracting {input!r} to {output!r}")

    try:
        with tarfile.open(input, "r") as f:
            f.extractall(path=output, filter="data")
    except tarfile.ReadError:
        raise InvalidFileType(f"{input}: is not a valid tar archive")

    return output


def compress_tarfile(input: str, output: str) -> str:
    """Compress a given directory into a tarfile.

    Arguments:
        input: Path to the input directory.
        output: Path to the output tarfile to create.

    Returns:
        The output tarfile created.
    """

    raise NotImplementedError()


__all__ = ["extract_tarfile", "compress_tarfile"]
