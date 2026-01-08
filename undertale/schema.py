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


class SummarizedBinaryDatasetWithSource(SummarizedMixin, BinaryDatasetWithSource):
    """Summarized binaries with source code."""


__all__ = [
    "Dataset",
    "SourceDataset",
    "SummarizedSourceDataset",
    "BinaryDataset",
    "BinaryDatasetWithSource",
    "SummarizedMixin",
    "SummarizedBinaryDatasetWithSource",
]
