from bisect import bisect_right
from functools import lru_cache
from os import listdir
from os.path import isfile, join
from typing import Any, Dict, List, Type

from pandas import read_parquet
from psutil import virtual_memory
from pyarrow import parquet
from torch.utils.data import Dataset

from ..schema import Dataset as DatasetSchema
from ..schema import validate_dataset


class ParquetDataset(Dataset):
    """A parquet-backed dataset featuring random access and caching.

    Loads data from a parquet dataset in one or more shards on disk. Caches
    shard reads in memory to optimize for high-locality access. Automatically
    sizes cache based on available memory.

    If the source dataset is smaller than available memory, then this is
    essentially just a lazy-loaded dataset. If the dataset is larger than
    memory, this is a locality-optimized cached dataset reader.

    Arguments:
        source: Path to a single parquet file or a directory of several parquet
            files.
        utilization: Available memory utilization scaling factor - this
            controls approximately how much of available memory the cache will
            attempt to use.
    """

    def build_cache(self):
        # Automatically determine the appropriate cache size.
        #
        # This assumes all dataset shards are roughly the same size.
        table = parquet.read_table(self.files[0])
        size = table.nbytes
        available = virtual_memory().available
        maxsize = int(available * self.utilization / size)
        maxsize = max(1, min(maxsize, len(self.files)))

        # Initialize cache.
        self.cache = lru_cache(maxsize=maxsize)(lambda path: parquet.read_table(path))

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["cache"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.build_cache()

    def __init__(self, source: str, utilization: float = 0.4):
        self.utilization = utilization

        if isfile(source):
            self.files = [source]
        else:
            self.files = sorted(join(source, f) for f in listdir(source))

        if not self.files:
            raise ValueError(f"no data files found in {source!r}")

        self.offsets: List[int] = [0]

        for path in self.files:
            metadata = parquet.read_metadata(path)
            self.offsets.append(self.offsets[-1] + metadata.num_rows)

        self.build_cache()

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file = bisect_right(self.offsets, index) - 1
        local = index - self.offsets[file]

        table = self.cache(self.files[file])
        row = table.slice(local, 1)

        return {c: row.column(c)[0].as_py() for c in table.schema.names}

    def validate(self, schema: Type[DatasetSchema]) -> None:
        """Validate this dataset against a given schema.

        Arguments:
            schema: A schema class to validate the dataset against.

        Raises:
            SchemaError: If the dataset does not conform to ``schema``.
        """

        frame = read_parquet(self.files[0])
        validate_dataset(frame, schema)
