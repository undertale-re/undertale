from typing import Any, Callable, Literal

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.writers.disk_base import DiskWriter


class ParquetResumeWriter(ParquetWriter):
    name = "PRW - ParquetResumeWriter"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: (
            Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
        ) = "snappy",
        adapter: Callable = None,
        batch_size: int = 1000,
        expand_metadata: bool = False,
        max_file_size: int = 5 * 2**30,  # 5GB
        schema: Any = None,
    ):
        super().__init__(
            output_folder,
            output_filename,
            compression=compression,
            adapter=adapter,
            batch_size=batch_size,
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
        )

        self.handled = False

    def write(self, document, rank, **kwargs):
        if not self.handled:
            import signal

            from datatrove.utils.logging import logger

            def handler(signum, frame):
                import sys

                from datatrove.utils.logging import logger

                logger.warning("caught signal {}, handling", str(signum))
                self.close()
                sys.exit(1)

            logger.info("SIGTERM Handler registered")
            signal.signal(signal.SIGTERM, handler)
            self.handled = True

            # first time running, skip past existing files
            ofname = self._get_output_filename(document, rank, **kwargs)
            fname = DiskWriter._get_filename_with_file_id(self, ofname)
            while self.output_folder.exists(fname):
                logger.info("file {} already exists, incrementing for resume", fname)
                self.file_id_counter[ofname] += 1
                fname = DiskWriter._get_filename_with_file_id(self, ofname)
        DiskWriter.write(self, document, rank, **kwargs)


class ParquetResumeFilter(PipelineStep):
    type = "🔻 - FILTER"
    name = "🕵 ParquetResumeFilter"

    _requires_dependencies = [
        "pyarrow",
    ]

    def __init__(self, data_folder, columns):
        from datatrove.io import get_datafolder

        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.columns = columns

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import pyarrow.lib
        import pyarrow.parquet as pq
        from datatrove.utils.logging import logger

        self.seen = set()

        glob_pat = f"*_{str(rank).zfill(5)}.parquet"
        logger.info(
            "checking previous files in {} like ", self.data_folder.path, glob_pat
        )
        for myfile in self.data_folder.list_files(glob_pattern=glob_pat):
            try:
                logger.info("updating from {}", str(myfile))
                with self.data_folder.open(myfile, "rb") as f:
                    with pq.ParquetFile(f) as pqf:
                        for batch in pqf.iter_batches(
                            columns=self.columns, batch_size=1000
                        ):
                            for item in batch.to_pylist():
                                id = "|".join([item[f] for f in self.columns])
                                self.seen.add(id)
            except pyarrow.lib.ArrowInvalid:
                logger.warning("invalid parquet file, removing: {}", myfile)
                try:
                    self.data_folder.delete(myfile)
                except Exception as e:
                    logger.error("failed to remove parquet file: {}", str(e))
            except Exception as e:
                logger.error("failed parsing file {}: {} ", str(myfile), str(e))
        logger.info("added {} entries", len(self.seen))

        for document in data:
            try:
                id = "|".join([document.metadata[f] for f in self.columns])
                if id not in self.seen:
                    yield document
                    self.stat_update("unseen")
                else:
                    self.stat_update("seen")
            except Exception as e:
                logger.error(
                    "error processing document {}: {}", str(document.id), str(e)
                )
                self.stat_update("failed")
