import logging
import os
from pathlib import Path

import datasets

from . import dataset, schema
from .transforms.disassemble import capstone
from .transforms.segment import ghidra

logger = logging.getLogger(__name__)


class TestWholeBinary(dataset.Dataset):
    """A precursor dataset used for testing purposes.

    This dataset follows the WholeBinary schema.

    Note: Note: The `parse` method expects a directory containing C source files (.c extension) along
    with their corresponding executable files, each having the same name as its source file.
    """

    path = "test-whole-binary"

    @classmethod
    def parse(cls, path: str, processes=None):
        sources = []
        binaries = []

        working = Path(path)

        for filename in os.listdir(working):
            if filename.endswith(".c"):
                path = os.path.join(working, filename)
                with open(path, "r") as f:
                    sources.append(f.read())
                with open(path[:-2], "rb") as f:
                    binaries.append(f.read())
        dataset = datasets.Dataset.from_dict({"source": sources, "binary": binaries})

        dataset.__class__ = cls

        return dataset


class TestWholeBinarySegment(TestWholeBinary):
    path = "test-whole-binary-segment"
    schema = schema.Function

    transforms = [
        ghidra.GhidraFunctionSegment(),
        capstone.CapstoneDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main([TestWholeBinarySegment])
