"""Dataset schema definition and enforcement."""

from pandera.pandas import DataFrameModel
from pandera.typing import Series


class Dataset(DataFrameModel):
    """Base class for all schema."""

    id: Series[str]
    """Row identifier."""


class SourceDataset(Dataset):
    """Source code."""

    source: Series[str]
    """Source code."""


class SummarizedMixin(DataFrameModel):
    """A mixin adding a ``summary`` field."""

    summary: Series[str]
    """Human-readable summary."""


class SummarizedSourceDataset(SummarizedMixin, SourceDataset):
    """Summarized source code."""


class BinaryDataset(Dataset):
    """Compiled binaries."""

    binary: Series[bytes]
    """Compiled binary code."""


class BinaryDatasetWithSource(BinaryDataset, SourceDataset):
    """Binaries with source code."""


class FunctionDataset(Dataset):
    """Individual functions."""

    name: Series[str]
    """The name of the function."""


class FunctionDatasetWithSource(FunctionDataset, SourceDataset):
    """Functions with source code."""


class DisassembledFunctionDataset(FunctionDataset):
    """Disassembled functions."""

    disassembly: Series[str]


class DisassembledFunctionDatasetWithSource(DisassembledFunctionDataset, SourceDataset):
    """Disassembled functions with source code."""


class SummarizedDisassembledFunctionDatasetWithSource(
    SummarizedMixin, DisassembledFunctionDatasetWithSource
):
    """Summarized, disassembled functions with source code."""


__all__ = [
    "Dataset",
    "SummarizedMixin",
    "SourceDataset",
    "SummarizedSourceDataset",
    "BinaryDataset",
    "BinaryDatasetWithSource",
    "FunctionDataset",
    "FunctionDatasetWithSource",
    "DisassembledFunctionDataset",
    "DisassembledFunctionDatasetWithSource",
    "SummarizedDisassembledFunctionDatasetWithSource",
]
