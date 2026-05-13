from os import listdir
from os.path import isfile, join
from typing import Iterator, List, Optional, Type

from pandas import read_parquet
from pyarrow import parquet
from torch import distributed
from torch.utils.data import IterableDataset, get_worker_info

from ..schema import Dataset, validate_dataset


class ParquetDataset(IterableDataset):
    """An iterable dataset backed by one or more parquet files.

    Loads parquet data sequentially, one file at a time, making it suitable for
    datasets larger than memory.

    When used in a distributed training environment, files are distributed
    across training ranks. When used with a multi-worker ``DataLoader``,
    per-rank files are also distributed across workers. The result is that each
    row is yielded by exactly one worker across all ranks.

    Note:
        ``DataLoader`` shuffle is not supported - shuffling must happen prior
        to loading.

    Note:
        Schema validation is performed against the first file only, as a
        representative check. It is assumed that all files in a directory
        share the same schema.

    Arguments:
        source: Path to a single parquet file or a directory of parquet files.
        schema: An optional schema class to validate the dataset against on
            construction.

    Raises:
        SchemaError: If ``schema`` is provided and the dataset does not
            conform to it.
    """

    def __init__(self, source: str, schema: Optional[Type[Dataset]] = None):
        if isfile(source):
            self._files = [source]
        else:
            self._files = sorted(join(source, f) for f in listdir(source))

        if schema is not None and self._files:
            frame = read_parquet(self._files[0])
            validate_dataset(frame, schema)

    def get_rank_files(self) -> List[str]:
        if distributed.is_initialized():
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
            return self._files[rank::world_size]
        return self._files

    def get_loader_files(self) -> List[str]:
        files = self.get_rank_files()

        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        return files

    def __len__(self) -> int:
        return sum(parquet.read_metadata(f).num_rows for f in self.get_rank_files())

    def __iter__(self) -> Iterator[dict]:
        for file in self.get_loader_files():
            yield from read_parquet(file).to_dict(orient="records")
