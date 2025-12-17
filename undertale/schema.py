from pandera.pandas import DataFrameModel
from pandera.typing import Series


class Dataset(DataFrameModel):
    id: Series[str]


class SourceDataset(Dataset):
    source: Series[str]


class SummarizedMixin(DataFrameModel):
    Summary: Series[str]


class SummarizedSourceDataset(SummarizedMixin, SourceDataset):
    pass


class BinaryDataset(Dataset):
    binary: Series[bytes]


class BinaryDatasetWithSource(BinaryDataset, SourceDataset):
    pass


class SummarizedBinaryDatasetWithSource(SummarizedMixin, BinaryDatasetWithSource):
    pass


__all__ = [
    "Dataset",
    "SourceDataset",
    "SummarizedSourceDataset",
    "BinaryDataset",
    "BinaryDatasetWithSource",
    "SummarizedBinaryDatasetWithSource",
]
