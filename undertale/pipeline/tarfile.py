import tarfile

from ..exceptions import InvalidFileType
from ..logging import get_logger
from ..utils import assert_path_does_not_exist, assert_path_exists

logger = get_logger(__name__)


def extract_tarfile(input: str, output: str) -> str:
    logger.info(f"extracting {input!r} to {output!r}")

    input = assert_path_exists(input)
    output = assert_path_does_not_exist(output, create=True)

    try:
        with tarfile.open(input, "r") as f:
            f.extractall(path=output, filter="data")
    except tarfile.ReadError:
        raise InvalidFileType(f"{input}: is not a valid tar archive")

    return output


def compress_tarfile(input: str, output: str) -> str:
    raise NotImplementedError()


__all__ = ["extract_tarfile", "compress_tarfile"]
