from bisect import bisect_right
from functools import lru_cache
from math import ceil
from os import listdir
from os.path import isfile, join
from typing import Any, Callable, Dict, List, Optional, Type

from lightning import LightningDataModule
from pandas import read_parquet
from psutil import virtual_memory
from pyarrow import parquet
from torch import distributed
from torch.utils.data import DataLoader, Dataset, Sampler

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


class ChunkedSampler(Sampler):
    """A chunked distributed sampler.

    The default ``DistributedSampler`` from ``pytorch`` uses a strided sampling
    approach that does not suit the locality constraints of the caching
    approach in ``ParquetDataset``. This sampler guarantees per rank and per
    worker contiguity.

    If not provided, this will attempt to discover rank and world size from
    ``torch.distributed`` state. If not running in a distributed environment a
    world size of 1 and rank index of 0 will be assumed - i.e., non-distributed
    training.

    Note:
        This makes some assumptions about relatively stable, but undocumented
        pytorch internals - in particular that the worker distribution strategy
        in ``DataLoader`` is round-robin. If this changes in the future this
        code will break.

    Arguments:
        dataset: The dataset from which to sample.
        batch: Batch size.
        workers: Number of parallel dataset workers. By default, this will
            spawn no dataset workers and fetch data in the main process.
        ranks: The number of distributed ranks.
        rank: The index of this distributed rank.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch: int,
        workers: int,
        ranks: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if ranks is None:
            if distributed.is_initialized():
                ranks = distributed.get_world_size()
            else:
                ranks = 1
                rank = 0

        if rank is None:
            rank = distributed.get_rank()

        self.dataset = dataset
        self.batch = batch
        self.workers = max(workers, 1)
        self.ranks = ranks
        self.rank = rank

        length = len(self.dataset)

        # If samples cannot be distributed evenly to ranks, drop the last few.
        if length % self.ranks != 0:
            self.rank_size = ceil((length - self.ranks) / self.ranks)
        else:
            self.rank_size = length // self.ranks

        # If samples cannot be divided evenly by workers, drop the last few.
        if self.rank_size % self.workers != 0:
            self.worker_size = ceil((self.rank_size - self.workers) / self.workers)
            self.rank_size = self.worker_size * self.workers
        else:
            self.worker_size = self.rank_size // self.workers

        self.total_size = self.rank_size * self.ranks

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # Drop extra samples.
        #
        # Work needs to be divided exactly evenly between ranks and workers.
        indices = indices[: self.total_size]

        # Slice contiguous samples for this rank.
        start = self.rank * self.rank_size
        indices = indices[start : start + self.rank_size]

        if self.workers <= 1:
            return iter(indices)

        # Interleave batched blocks.
        #
        # Divide samples into per-worker batched blocks and reshuffle. This
        # forces the internal round-robin worker scheduling in DataLoader to
        # yield a contiguous region of indices per worker.
        worker_blocks = [
            indices[w * self.worker_size : (w + 1) * self.worker_size]
            for w in range(self.workers)
        ]
        batched_blocks = [
            [block[i : i + self.batch] for i in range(0, len(block), self.batch)]
            for block in worker_blocks
        ]

        interleaved = []
        max_batches = max(len(b) for b in batched_blocks)
        for i in range(max_batches):
            for worker_batches in batched_blocks:
                if i < len(worker_batches):
                    interleaved.extend(worker_batches[i])

        return iter(interleaved)

    def __len__(self) -> int:
        return self.rank_size


def load(
    path: str,
    schema: Type[DatasetSchema],
    collator: Callable,
    batch: int,
    workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the given dataset.

    Arguments:
        path: Path to the dataset to process.
        schema: Expected dataset schema.
        collator: Dataset collator.
        batch: Batch size.
        workers: Number of parallel dataset workers. By default, this will
            spawn no dataset workers and fetch data in the main process.

    Returns:
        A DataLoader for the given dataset at ``path`` with the given
        parameters.
    """

    dataset = ParquetDataset(path)
    dataset.validate(schema)

    sampler = ChunkedSampler(
        dataset,
        batch=batch,
        workers=workers,
    )

    return DataLoader(
        dataset,
        batch_size=batch,
        collate_fn=collator,
        num_workers=workers,
        sampler=sampler,
    )


class DataModule(LightningDataModule):
    """A DataModule that wraps ParquetDatasets.

    Arguments:
        training: Path to the training dataset.
        validation: Path to the validation dataset. If ``None`` validation will
            be skipped.
        schema: Expected dataset schema.
        collator: Dataset collator.
        batch: Batch size.
        workers: Number of parallel dataset workers. By default, this will
            spawn no dataset workers and fetch data in the main process.
    """

    def __init__(
        self,
        training: str,
        validation: Optional[str],
        schema: Type[DatasetSchema],
        collator: Callable,
        batch: int,
        workers: int = 0,
    ):
        super().__init__()

        self.training = training
        self.validation = validation
        self.schema = schema
        self.collator = collator
        self.batch = batch
        self.workers = workers

    def _load(self, dataset: str) -> DataLoader:
        return load(
            dataset,
            schema=self.schema,
            collator=self.collator,
            batch=self.batch,
            workers=self.workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._load(self.training)

    def val_dataloader(self):
        if self.validation is None:
            return []

        return self._load(self.validation)
