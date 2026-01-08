"""JSON parsing."""

import json
from typing import List

from ..logging import get_logger
from ..utils import assert_path_does_not_exist, assert_path_exists

logger = get_logger(__name__)


def split_json(input: str, output: str, chunks: int) -> List[str]:
    """Split one JSON into many files.

    Arguments:
        input: Path to the JSON input file.
        output: Path to the target directory.
        chunks: Number of chunk files to generate.

    Returns:
        A list of paths to the generated files.
    """

    raise NotImplementedError()


def merge_json(inputs: List[str], output: str) -> str:
    """Merge several JSON files into one list.

    Input files may contain a single object or a list of objects. Singular
    objects will ``append()`` the final list while lists of objects will
    ``extend()`` it.

    Arguments:
        inputs: Paths to JSON object files.
        output: Merged output path.

    Returns:
        The path to the merged JSON output file.
    """

    logger.info(f"merging {len(inputs)} results to {output!r}")

    output = assert_path_does_not_exist(output)

    merged = []
    for input in inputs:
        input = assert_path_exists(input)

        with open(input, "r") as f:
            loaded = json.load(f)

            if isinstance(loaded, list):
                merged.extend(loaded)
            else:
                merged.append(loaded)

    with open(output, "w") as f:
        json.dump(merged, f)

    return output


__all__ = ["split_json", "merge_json"]
