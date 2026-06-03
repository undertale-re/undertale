"""JSON parsing."""

import json
from typing import List

from ..logging import get_logger
from ..utils import assert_path_exists, get_or_create_file

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

    output, created = get_or_create_file(output)

    if not created:
        return output

    logger.info(f"merging {len(inputs)} results to {output!r}")

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


def average_json(inputs: List[str], output: str) -> str:
    """Merge several JSON files, averaging their fields.

    JSON must be a flat dictionary of keys mapping to scalar values which can
    be averaged. All input files must match the same schema.

    Arguments:
        inputs: Paths to JSON object files.
        output: Merged output path.

    Returns:
        The path to the merged JSON output file.
    """

    output, created = get_or_create_file(output)

    if not created:
        return output

    logger.info(f"averaging {len(inputs)} results to {output!r}")

    totals: dict = {}
    for path in inputs:
        path = assert_path_exists(path)

        with open(path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            totals[key] = totals.get(key, 0.0) + value

    averaged = {key: value / len(inputs) for key, value in totals.items()}

    with open(output, "w") as f:
        json.dump(averaged, f)

    return output


__all__ = ["split_json", "merge_json", "average_json"]
