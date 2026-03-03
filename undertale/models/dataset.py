from os import listdir
from os.path import isfile, join
from typing import Iterator

import pandas
from torch.utils.data import IterableDataset, get_worker_info


class ParquetDataset(IterableDataset):
    """An iterable dataset backed by one or more parquet files.

    Loads parquet data sequentially, one file at a time, making it suitable
    for datasets larger than memory. When used with a multi-worker
    ``DataLoader``, files are distributed across workers so that each row is
    yielded by exactly one worker.

    Note:
        ``DataLoader`` shuffle is not supported - shuffling must happen prior
        to loading.

    Arguments:
        source: Path to a single parquet file or a directory of parquet files.
    """

    def __init__(self, source: str):
        if isfile(source):
            self.files = [source]
        else:
            self.files = sorted(join(source, f) for f in listdir(source))

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()

        if worker is None:
            files = self.files
        else:
            files = self.files[worker.id :: worker.num_workers]

        for file in files:
            yield from pandas.read_parquet(file).to_dict(orient="records")
