"""Dataset schema definition and enforcement."""

from typing import Type

from pandas import DataFrame
from pandera.errors import SchemaError as PanderaSchemaError
from pandera.pandas import DataFrameModel
from pandera.typing import Series

from .exceptions import SchemaError


class Dataset(DataFrameModel):
    """Base class for all schema."""

    id: Series[str]
    """Row identifier."""


class TokenizedDataset(Dataset):
    """A tokenized dataset."""

    tokens: Series[object]
    """The token IDs for the tokenized row."""

    mask: Series[object]
    """The attention mask for the tokenized row."""


class TokenizedClassificationDataset(TokenizedDataset):
    """A tokenized dataset with integer classification labels."""

    label: Series[int]
    """Integer class label for the sequence."""


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


class VulnerabilityMixin(DataFrameModel):
    """A mixin adding a ``vulnerability`` field."""

    vulnerability: Series[str]


class VulnerabilityDisassembledFunctionDatasetWithSource(
    VulnerabilityMixin, DisassembledFunctionDatasetWithSource
):
    """Disassembled functions with source code and associated vulnerabilities (if one exists)."""


def validate_dataset(frame: DataFrame, schema: Type[Dataset]) -> None:
    """Validate a dataset against a given schema.

    Arguments:
        frame: The dataset to validate.
        schema: The schema class against which to validate.

    Raises:
        SchemaError: If ``frame`` does not obey ``schema``.
    """

    try:
        schema.validate(frame)
    except PanderaSchemaError as e:
        raise SchemaError(str(e))


__all__ = [
    "Dataset",
    "TokenizedDataset",
    "TokenizedClassificationDataset",
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
    "VulnerabilityDisassembledFunctionDatasetWithSource",
]
