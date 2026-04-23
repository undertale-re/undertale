"""Zip file parsing."""

import os
import zipfile

from ..exceptions import InvalidFileType
from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_directory

logger = get_logger(__name__)


def unzip_file(input: str, output: str) -> str:
    """Decompress a given Zip file.

    Arguments:
        input: Path to the Zip input file.
        output: Path to the output directory to create.

    Returns:
        The output directory created.
    """
    input = assert_path_exists(input)
    output, created = get_or_create_directory(os.path.splitext(output)[0])

    if not created:
        return output

    logger.info(f"extracting {input!r} to {output!r}")

    try:
        with zipfile.ZipFile(input) as f:
            f.extractall(path=output)
    except zipfile.BadZipFile:
        raise InvalidFileType(f"{input}: is not a valid Zip file")

    return output


def zip_file(input: str, output: str) -> str:
    """Compress a given directory into a Zip file.

    Arguments:
        input: Path to the input directory.
        output: Path to the output Zip file to create.

    Returns:
        The output Zip file created.
    """
    raise NotImplementedError()


__all__ = ["unzip_file", "zip_file"]
