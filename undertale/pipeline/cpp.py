"""C/C++ compilation."""

from os.path import join
from subprocess import PIPE, run
from tempfile import TemporaryDirectory

from pandas import Series, read_parquet
from pandera.errors import SchemaError as PanderaSchemaError

from ..exceptions import SchemaError
from ..logging import get_logger
from ..schema import SourceDataset
from ..utils import assert_path_exists, find, get_or_create_file

logger = get_logger(__name__)


def compile(row: Series) -> bytes:
    working = TemporaryDirectory()

    source_path = join(working.name, "sample.cpp")
    with open(source_path, "w") as f:
        f.write(row["source"])

    binary_path = join(working.name, "sample")

    logger.debug(f"compiling {row['id']}")

    gpp = find("g++")
    process = run(
        f"{gpp} -c {source_path} -o {binary_path}",
        cwd=working.name,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
    )

    if process.returncode == 0:
        logger.info(f"successfully compiled {row['id']}")

        with open(binary_path, "rb") as f:
            binary = f.read()
        return binary
    else:
        message = "failed to compile source:\n"
        message += "=" * 80 + "\n"
        message += row["source"].strip() + "\n"
        message += "-" * 36 + " stdout " + "-" * 36 + "\n"
        message += process.stdout.decode().strip() + "\n"
        message += "-" * 36 + " stderr " + "-" * 36 + "\n"
        message += process.stderr.decode(errors="ignore").strip() + "\n"
        message += "=" * 80

        logger.warning(message)

        return b""


def compile_cpp(
    input: str,
    output: str,
) -> str:
    """Compile a C/C++ source dataset.

    Arguments:
        input: Path to the source dataset.
        output: Path where the binary dataset should be written.

    Returns:
        The path to the generated parquet file.

    Raises:
        SchemaError: If the input dataset does not match
            :py:class:`SourceDataset <undertale.schema.SourceDataset>`.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    logger.info(f"compiling C/C++ source {input!r} to {output!r}")

    frame = read_parquet(input)

    try:
        SourceDataset.validate(frame)
    except PanderaSchemaError as e:
        logger.error("dataset does not match the expected schema")
        raise SchemaError(str(e))

    frame["binary"] = frame.apply(compile, axis=1)
    success = frame[frame["binary"] != b""]

    logger.info(f"successfully compiled ({len(success)}/{len(frame)}) sources")

    success.to_parquet(output)

    return output


__all__ = ["compile_cpp"]
