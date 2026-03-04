from os import listdir
from os.path import isfile, join
from typing import Iterator, Optional, Type

import pandas
from pandera.errors import SchemaError as PanderaSchemaError
from pyarrow import parquet
from torch.utils.data import IterableDataset, get_worker_info

from ..exceptions import SchemaError
from ..schema import Dataset


class ParquetDataset(IterableDataset):
    """An iterable dataset backed by one or more parquet files.

    Loads parquet data sequentially, one file at a time, making it suitable for
    datasets larger than memory. When used with a multi-worker ``DataLoader``,
    files are distributed across workers so that each row is yielded by exactly
    one worker.

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
            self.files = [source]
        else:
            self.files = sorted(join(source, f) for f in listdir(source))

        if schema is not None and self.files:
            try:
                schema.validate(pandas.read_parquet(self.files[0]))
            except PanderaSchemaError as e:
                raise SchemaError(str(e))

    def __len__(self) -> int:
        return sum(parquet.read_metadata(f).num_rows for f in self.files)

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()

        if worker is None:
            files = self.files
        else:
            files = self.files[worker.id :: worker.num_workers]

        for file in files:
            yield from pandas.read_parquet(file).to_dict(orient="records")
