import argparse
import json
import logging
import os
import shutil
import tarfile
from datetime import datetime
from logging import WARNING
from os import listdir, makedirs
from os.path import basename, exists, isdir, isfile, join
from tempfile import TemporaryDirectory
from time import sleep
from types import SimpleNamespace
from typing import Dict
from unittest import SkipTest, TestCase
from unittest.mock import patch

from dask.dataframe import from_pandas
from pandas import DataFrame, read_parquet
from pyarrow.parquet import read_metadata as pyarrow_read_metadata
from torch import rand, randint, set_grad_enabled, tensor
from utils import load_resource, main

from undertale.exceptions import EnvironmentError as LocalEnvironmentError
from undertale.exceptions import PathError, SchemaError
from undertale.models.custom import InstructionTracePositionEmbedding
from undertale.models.dataset import ParquetDataset
from undertale.models.maskedlm import MaskedLMCollator
from undertale.models.tokenizer import (
    TOKEN_UNKNOWN,
)
from undertale.models.tokenizer import load as load_tokenizer
from undertale.models.tokenizer import (
    merge_preprocessed_tokens,
    preprocess_tokens,
    tokenize,
    train_tokenizer,
)
from undertale.models.transformer import (
    Attention,
    FeedForward,
    MultiHeadAttention,
    PositionEmbedding,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.pipeline.cpp import compile_cpp
from undertale.pipeline.json import merge_json
from undertale.pipeline.parquet import (
    Deduplicate,
    Drop,
    HashColumn,
    Keep,
    Rename,
    Repartition,
    modify_parquet,
)
from undertale.pipeline.tarfile import extract_tarfile
from undertale.schema import Dataset
from undertale.utils import (
    RemoteException,
    assert_path_exists,
    cache_path,
    enforce_extension,
    find,
    get_or_create_directory,
    get_or_create_file,
    hash,
    subprocess,
    timestamp,
    write_parquet,
)
from undertale.utils.datasets.split import parse_split
from undertale.utils.datasets.split import split as split_dataset


class TestUtilitiesHash(TestCase):
    def test_hash_produces_identical_values(self):
        def check(value: bytes) -> None:
            self.assertEqual(hash(value), hash(value))

        check(b"foo")
        check(b"bar")
        check(b"baz")

    def test_hash_no_collisions(self):
        self.assertNotEqual(hash(b"foo"), hash(b"bar"))
        self.assertNotEqual(hash(b"bar"), hash(b"baz"))
        self.assertNotEqual(hash(b"foo"), hash(b"baz"))


class TestUtilitiesTimestamp(TestCase):
    def test_timestamp_now(self):
        stamp = timestamp()

        self.assertIn("-", stamp)
        self.assertNotIn(" ", stamp)
        self.assertNotIn(":", stamp)

    def test_timestamp_custom_time(self):
        time = datetime(2000, 1, 1, 0, 0)
        stamp = timestamp(time)

        self.assertEqual(stamp, "20000101-000000")


class TestUtilitiesPaths(TestCase):
    def test_enforce_extension_matching(self):
        result = enforce_extension("foo/bar.baz", ".baz")

        self.assertTrue(result.endswith(".baz"))

    def test_enforce_extension_nonmatching(self):
        result = enforce_extension("foo/bar.buzz", ".baz")

        self.assertTrue(result.endswith(".baz"))

    def test_enforce_extension_nonexistent(self):
        result = enforce_extension("foo/bar", ".baz")

        self.assertTrue(result.endswith(".baz"))

    def test_assert_path_exists_existing_directory(self):
        working = TemporaryDirectory()

        assert_path_exists(working.name)

    def test_assert_path_exists_existing_file(self):
        working = TemporaryDirectory()

        target = join(working.name, "test.txt")

        with open(target, "w"):
            pass

        assert_path_exists(target)

    def test_assert_path_exists_nonexistent(self):
        with self.assertRaises(PathError):
            assert_path_exists("foo/bar/baz")

    def test_get_or_create_file_nonexistent(self):
        working = TemporaryDirectory()

        target = join(working.name, "test.txt")

        target, created = get_or_create_file(target)

        self.assertTrue(created)
        self.assertTrue(exists(target))
        self.assertTrue(isfile(target))

    def test_get_or_create_file_existing(self):
        working = TemporaryDirectory()

        target = join(working.name, "test.txt")

        with open(target, "w"):
            pass

        target, created = get_or_create_file(target)

        self.assertFalse(created)
        self.assertTrue(exists(target))

    def test_get_or_create_directory_nonexistent(self):
        working = TemporaryDirectory()

        target = join(working.name, "test")

        target, created = get_or_create_directory(target)

        self.assertTrue(created)
        self.assertTrue(exists(target))
        self.assertTrue(isdir(target))

    def test_get_or_create_directory_existing(self):
        working = TemporaryDirectory()

        target = join(working.name, "test")

        makedirs(target)

        target, created = get_or_create_directory(target)

        self.assertFalse(created)
        self.assertTrue(exists(target))


class TestUtilitiesCachePath(TestCase):
    def test_no_env_var(self):
        source = TemporaryDirectory()
        target = join(source.name, "artifact.txt")

        with open(target, "w") as f:
            f.write("data")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertLogs(level="WARNING") as logs:
                result = cache_path(target)

        self.assertEqual(result, target)
        self.assertTrue(any("UNDERTALE_CACHE" in line for line in logs.output))

    def test_file_is_copied(self):
        source = TemporaryDirectory()
        cache = TemporaryDirectory()

        target = join(source.name, "artifact.txt")
        with open(target, "w") as f:
            f.write("data")

        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertLogs(level="INFO") as logs:
                result = cache_path(target)

        expected = join(cache.name, "artifact.txt")
        self.assertEqual(result, expected)
        self.assertTrue(exists(expected))
        self.assertTrue(any("cached" in line for line in logs.output))

    def test_directory_is_copied_recursively(self):
        source = TemporaryDirectory()
        cache = TemporaryDirectory()

        subdir = join(source.name, "mydir", "sub")
        makedirs(subdir)
        for name in ("a.txt", "b.txt"):
            with open(join(source.name, "mydir", name), "w") as f:
                f.write(name)
        with open(join(subdir, "c.txt"), "w") as f:
            f.write("c")

        source_dir = join(source.name, "mydir")

        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertLogs(level="INFO"):
                result = cache_path(source_dir)

        self.assertTrue(exists(join(result, "a.txt")))
        self.assertTrue(exists(join(result, "b.txt")))
        self.assertTrue(exists(join(result, "sub", "c.txt")))

    def test_file_already_exists_up_to_date(self):
        source = TemporaryDirectory()
        cache = TemporaryDirectory()

        target = join(source.name, "artifact.txt")
        with open(target, "w") as f:
            f.write("data")

        cached = join(cache.name, "artifact.txt")
        shutil.copy2(target, cached)

        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertLogs(level="INFO") as logs:
                cache_path(target)

        self.assertTrue(any("already exists" in line for line in logs.output))
        self.assertFalse(any("cached" in line for line in logs.output))

    def test_file_already_exists_stale(self):
        source = TemporaryDirectory()
        cache = TemporaryDirectory()

        target = join(source.name, "artifact.txt")
        with open(target, "w") as f:
            f.write("updated data")

        cached = join(cache.name, "artifact.txt")
        with open(cached, "w") as f:
            f.write("old")

        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertLogs(level="INFO") as logs:
                cache_path(target)

        self.assertTrue(any("cached" in line for line in logs.output))

        with open(join(cache.name, "artifact.txt")) as f:
            self.assertEqual(f.read(), "updated data")

    def test_nonexistent_path(self):
        cache = TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertRaises(FileNotFoundError):
                cache_path("/nonexistent/path/to/file.txt")

    def test_directory_already_exists_up_to_date(self):
        source = TemporaryDirectory()
        cache = TemporaryDirectory()

        source_dir = join(source.name, "mydir")
        makedirs(source_dir)
        for name in ("x.txt", "y.txt"):
            with open(join(source_dir, name), "w") as f:
                f.write(name)

        dest_dir = join(cache.name, "mydir")
        makedirs(dest_dir)
        for name in ("x.txt", "y.txt"):
            shutil.copy2(join(source_dir, name), join(dest_dir, name))

        with patch.dict(os.environ, {"UNDERTALE_CACHE": cache.name}):
            with self.assertLogs(level="INFO") as logs:
                cache_path(source_dir)

        self.assertTrue(any("already exists" in line for line in logs.output))
        self.assertFalse(any("cached" in line for line in logs.output))


class TestUtilitiesFind(TestCase):
    @staticmethod
    def mock_executable(working: TemporaryDirectory, name: str) -> str:
        path = join(working.name, name)

        with open(path, "w"):
            pass

        return path

    def test_find_simple_path(self):
        path = find("python")

        self.assertTrue(path.endswith("python"))

    def test_find_environment(self):
        working = TemporaryDirectory()
        baz = self.mock_executable(working, "baz")

        try:
            os.environ["BAZ_PATH"] = baz
            path = find("baz", environment="BAZ_PATH")
        finally:
            del os.environ["BAZ_PATH"]

        self.assertTrue(path.endswith("baz"))

    def test_find_guesses(self):
        working = TemporaryDirectory()
        baz = self.mock_executable(working, "baz")

        path = find("baz", guesses=(baz,))

        self.assertTrue(path.endswith("baz"))

    def test_find_nonexistent(self):
        with self.assertRaises(LocalEnvironmentError):
            find("baz")


class TestUtilitiesParquet(TestCase):
    def test_write_parquet_simple(self):
        working = TemporaryDirectory()

        frame = DataFrame([{"foo": "bar"}])

        path = join(working.name, "dataset.parquet")

        write_parquet(frame, path)

        read_parquet(path)

    def test_write_parquet_unsupported(self):
        with self.assertRaises(ValueError):
            write_parquet({}, "foo")

    def test_write_parquet_compression_default_disabled(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]
        frame = DataFrame(dataset)

        path = join(working.name, "dataset")

        write_parquet(frame, path)

        metadata = pyarrow_read_metadata(path)

        self.assertEqual(metadata.row_group(0).column(0).compression, "UNCOMPRESSED")

    def test_write_parquet_compression_specified(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]
        frame = DataFrame(dataset)

        path = join(working.name, "dataset")

        write_parquet(frame, path, compression="snappy")

        metadata = pyarrow_read_metadata(path)

        self.assertEqual(metadata.row_group(0).column(0).compression, "SNAPPY")


class TestUtilitiesSubprocess(TestCase):
    def test_subprocess_simple(self):
        @subprocess
        def test(x: int) -> int:
            return x**2

        self.assertEqual(test(2), 4)
        self.assertEqual(test(4), 16)

    def test_subprocess_timeout(self):
        @subprocess(timeout=0.1)
        def test():
            sleep(1)

        with self.assertRaises(TimeoutError):
            test()

    def test_subprocess_exception(self):
        class CustomException(Exception):
            pass

        @subprocess
        def test():
            raise CustomException("oops")

        with self.assertRaises(RemoteException):
            test()


class TestPipelineDask(TestCase):
    def test_cluster_unsupported_type(self):
        with self.assertRaises(ValueError):
            Cluster("foo")

    def test_fanout_error_no_side_effects(self):
        working = TemporaryDirectory()

        def silence_logging():
            logging.getLogger("distributed").setLevel(logging.CRITICAL)

        class Expected(Exception):
            pass

        def generate():
            raise Expected()

        output = join(working.name, "output")

        try:
            with Cluster(type="local") as cluster, Client(cluster) as client:
                client.run(silence_logging)

                generated = client.submit(generate)
                result = fanout(client, lambda x: x, generated, output)

                result.result()
                flush(client)
        except Expected:
            pass

        self.assertFalse(exists(output))


class TestPipelineJSON(TestCase):
    def test_json_merge_objects_simple(self):
        working = TemporaryDirectory()

        paths = [
            join(working.name, "1.json"),
            join(working.name, "2.json"),
            join(working.name, "3.json"),
        ]

        for i, path in enumerate(paths):
            with open(path, "w") as f:
                json.dump(i + 1, f)

        output = join(working.name, "merged.json")

        result = merge_json(paths, output)

        with open(result, "r") as f:
            loaded = json.load(f)

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded, [1, 2, 3])

    def test_json_merge_lists_simple(self):
        working = TemporaryDirectory()

        paths = [
            join(working.name, "1.json"),
            join(working.name, "2.json"),
            join(working.name, "3.json"),
        ]

        for i, path in enumerate(paths):
            with open(path, "w") as f:
                json.dump([1, 2], f)

        output = join(working.name, "merged.json")

        result = merge_json(paths, output)

        with open(result, "r") as f:
            loaded = json.load(f)

        self.assertEqual(len(loaded), 6)
        self.assertEqual(loaded, [1, 2, 1, 2, 1, 2])

    def test_json_merge_mixed(self):
        working = TemporaryDirectory()

        paths = [
            join(working.name, "1.json"),
            join(working.name, "2.json"),
            join(working.name, "3.json"),
        ]

        with open(paths[0], "w") as f:
            json.dump([1, 2], f)
        with open(paths[1], "w") as f:
            json.dump(1, f)
        with open(paths[2], "w") as f:
            json.dump([1, 2], f)

        output = join(working.name, "merged.json")

        result = merge_json(paths, output)

        with open(result, "r") as f:
            loaded = json.load(f)

        self.assertEqual(len(loaded), 5)
        self.assertEqual(loaded, [1, 2, 1, 1, 2])


class TestPipelineTarfile(TestCase):
    @staticmethod
    def mock_tarfile(
        working: TemporaryDirectory,
        name: str,
        compression: str = "",
    ) -> str:
        paths = []

        path = join(working.name, "test.txt")
        with open(path, "w") as f:
            f.write("hello world")
        paths.append(path)

        path = join(working.name, "data.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        paths.append(path)

        output = join(working.name, name)
        with tarfile.open(output, f"w:{compression}") as f:  # type: ignore[call-overload]
            for path in paths:
                f.add(path, arcname=basename(path))

        return output

    def test_extract_simple(self):
        working = TemporaryDirectory()
        archive = self.mock_tarfile(working, "test.tar")

        extracted = join(working.name, "extracted")
        extract_tarfile(archive, extracted)

        with open(join(extracted, "test.txt"), "r") as f:
            data = f.read()

        self.assertEqual(data, "hello world")

    def test_extract_compressed(self):
        working = TemporaryDirectory()
        archive = self.mock_tarfile(working, "test.tgz", compression="gz")

        extracted = join(working.name, "extracted")
        extract_tarfile(archive, extracted)

        with open(join(extracted, "test.txt"), "r") as f:
            data = f.read()

        self.assertEqual(data, "hello world")

    def test_extract_non_tarfile(self):
        working = TemporaryDirectory()
        nonarchive = join(working.name, "test.txt")

        with open(nonarchive, "w") as f:
            f.write("hello world")

        extracted = join(working.name, "extracted")

        with self.assertRaises(PathError):
            extract_tarfile(nonarchive, extracted)


class TestPipelineParquet(TestCase):
    @staticmethod
    def mock_dataset(
        working: TemporaryDirectory, name: str, size: int, chunks: int = 1
    ) -> str:
        data = []
        for i in range(size):
            data.append(
                {
                    "id": i,
                }
            )

        path = join(working.name, name)

        frame = DataFrame(data)
        write_parquet(from_pandas(frame, npartitions=chunks), path)

        return path

    def test_parquet_hash_column_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            modify_parquet(
                dataset, join(working.name, "hashed"), [HashColumn("mordor", "hash")]
            )

    def test_parquet_hash_column_simple(self):
        working = TemporaryDirectory()

        data = b"\xde\xad\xbe\xef"
        dataset = [{"id": i, "data": data} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "hashed")
        modify_parquet(path, output, [HashColumn("data", "hash")])

        loaded = read_parquet(output)

        self.assertIn("hash", loaded.columns)
        self.assertEqual(loaded["hash"][0], hash(data))

    def test_parquet_repartition_chunks_and_size(self):
        with self.assertRaises(ValueError):
            Repartition(chunks=10, size=10)

    def test_parquet_repartition_neither_chunks_nor_size(self):
        with self.assertRaises(ValueError):
            Repartition()

    def test_parquet_rename_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            modify_parquet(dataset, path, [Rename({"nonexistent": "other"})])

    def test_parquet_rename_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Rename({"foo": "baz"})])

        loaded = read_parquet(output)

        self.assertNotIn("foo", loaded.columns)
        self.assertIn("baz", loaded.columns)
        self.assertEqual(list(loaded["baz"]), ["bar"] * 10)

    def test_parquet_rename_multiple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar", "alpha": "beta"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Rename({"foo": "baz", "alpha": "gamma"})])

        loaded = read_parquet(output)

        self.assertNotIn("foo", loaded.columns)
        self.assertNotIn("alpha", loaded.columns)
        self.assertIn("baz", loaded.columns)
        self.assertIn("gamma", loaded.columns)

    def test_parquet_rename_after_keep(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar", "extra": "drop"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Keep(["foo"]), Rename({"foo": "baz"})])

        loaded = read_parquet(output)

        self.assertNotIn("foo", loaded.columns)
        self.assertNotIn("extra", loaded.columns)
        self.assertIn("baz", loaded.columns)

    def test_parquet_drop_preserves_chunks(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(80)]
        path = join(working.name, "dataset")
        write_parquet(from_pandas(DataFrame(dataset), npartitions=8), path)

        output = join(working.name, "dropped")
        created = modify_parquet(path, output, [Drop(["foo"])])

        # Dask does not guarantee exact partition count after a column drop on
        # small data, so we only assert that chunking was not collapsed to one.
        self.assertGreater(len(created), 1)

        loaded = read_parquet(output)
        self.assertNotIn("foo", loaded.columns)

    def test_parquet_rename_preserves_chunks(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(30)]
        path = join(working.name, "dataset")
        write_parquet(from_pandas(DataFrame(dataset), npartitions=3), path)

        output = join(working.name, "renamed")
        created = modify_parquet(path, output, [Rename({"foo": "baz"})])

        self.assertEqual(len(created), 3)

        loaded = read_parquet(output)
        self.assertIn("baz", loaded.columns)
        self.assertNotIn("foo", loaded.columns)

    def test_parquet_repartition_one_to_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100)

        path = join(working.name, "resized")
        created = modify_parquet(dataset, path, [Repartition(chunks=20)])

        self.assertEqual(len(created), 20)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_repartition_many_to_one_chunk(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        path = join(working.name, "resized")
        created = modify_parquet(dataset, path, [Repartition(chunks=1)])

        self.assertEqual(len(created), 1)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_repartition_many_to_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        path = join(working.name, "resized")
        created = modify_parquet(dataset, path, [Repartition(chunks=25)])

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_repartition_too_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=20)

        path = join(working.name, "resized")
        created = modify_parquet(dataset, path, [Repartition(chunks=25)])

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 20)

    def test_parquet_repartition_size(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100)

        path = join(working.name, "resized")
        created = modify_parquet(dataset, path, [Repartition(size=64)])

        self.assertEqual(len(created), 26)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_repartition_chunk_list_input(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        chunks = [join(dataset, f) for f in listdir(dataset)]

        path = join(working.name, "resized")
        created = modify_parquet(chunks, path, [Repartition(chunks=25)])

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_deduplicate_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            modify_parquet(dataset, path, [Deduplicate(["foo"]), Repartition(chunks=1)])

    def test_parquet_deduplicate_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i} for i in range(10)]
        dataset += [
            {"id": 1337},
            {"id": 1337},
            {"id": 1337},
            {"id": 1338},
            {"id": 301},
            {"id": 302},
            {"id": 303},
            {"id": 1338},
        ]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Deduplicate(["id"]), Repartition(chunks=1)])

        loaded = read_parquet(output)

        self.assertEqual(len(loaded), 15)
        self.assertEqual(len(set(loaded["id"])), 15)

    def test_parquet_drop_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            modify_parquet(dataset, path, [Drop(["foo"]), Repartition(chunks=1)])

    def test_parquet_drop_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Drop(["foo"]), Repartition(chunks=1)])

        loaded = read_parquet(output)

        self.assertNotIn("foo", loaded.columns)

    def test_parquet_keep_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            modify_parquet(dataset, path, [Keep(["foo"]), Repartition(chunks=1)])

    def test_parquet_keep_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar", "baz": "zaa"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Keep(["foo"]), Repartition(chunks=1)])

        loaded = read_parquet(output)

        self.assertIn("foo", loaded.columns)
        self.assertNotIn("id", loaded.columns)
        self.assertNotIn("baz", loaded.columns)

    def test_parquet_repartition_compression(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        modify_parquet(path, output, [Repartition(chunks=1)], compression="snappy")

        files = listdir(output)

        metadata = pyarrow_read_metadata(join(output, files[0]))

        self.assertEqual(metadata.row_group(0).column(0).compression, "SNAPPY")


class TestPipelineCpp(TestCase):
    @staticmethod
    def mock_dataset(frame: DataFrame, working: TemporaryDirectory, name: str) -> str:
        path = join(working.name, name)
        write_parquet(frame, path)

        return path

    @classmethod
    def setUpClass(cls):
        cls.simple_source = load_resource("source/42/42.c").decode()

    def test_cpp_compile_invalid_schema(self):
        working = TemporaryDirectory()

        sources = DataFrame([{"foo": "bar"}])
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        with self.assertRaises(SchemaError):
            path = join(working.name, "compiled.parquet")
            compile_cpp(dataset, path)

    def test_cpp_compile_simple(self):
        working = TemporaryDirectory()

        sources = DataFrame([{"id": "1", "source": self.simple_source}])
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "compiled.parquet")
        compile_cpp(dataset, path)

        loaded = read_parquet(path)

        self.assertIn("binary", loaded)
        self.assertIn("source", loaded)
        self.assertIn("id", loaded)

        self.assertNotEqual(loaded["binary"][0], b"")

    def test_cpp_compile_compilation_failure(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {"id": "1", "source": "foo"},
                {"id": "2", "source": self.simple_source},
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "compiled.parquet")

        with self.assertLogs(level=WARNING):
            compile_cpp(dataset, path)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 1)


class TestPipelineBinary(TestCase):
    def mock_dataset(
        self, frame: DataFrame, working: TemporaryDirectory, name: str
    ) -> str:
        path = join(working.name, name)
        write_parquet(frame, path)

        return path

    @classmethod
    def setUpClass(cls):
        try:
            import binaryninja  # noqa:  F401
        except ImportError:
            raise SkipTest("BinaryNinja is not installed - skipping binary tests")

        cls.simple_source = load_resource("source/42/42.c").decode()
        cls.simple_binary_x86_64_elf = load_resource("binaries/42.x86_64.elf")
        cls.simple_binary_arm_macho = load_resource("binaries/42.arm.macho")
        cls.canary_source = load_resource("source/canary/canary.c").decode()
        cls.canary_binary_x86_64_elf = load_resource("binaries/canary.x86_64.elf")
        cls.tword_source = load_resource("source/tword/tword.c").decode()
        cls.tword_binary_x86_64_elf = load_resource("binaries/tword.x86_64.elf")
        cls.data_source = load_resource("source/data/data.c").decode()
        cls.data_binary_x86_64_elf = load_resource("binaries/data.x86_64.elf")
        cls.relative_source = load_resource("source/relative/relative.c").decode()
        cls.relative_binary_x86_64_elf = load_resource("binaries/relative.x86_64.elf")
        cls.invalid_source = load_resource("source/invalid/invalid.c").decode()
        cls.invalid_binary_x86_64_elf = load_resource("binaries/invalid.x86_64.elf")

    def test_binary_segment_and_disassemble_simple_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.simple_source,
                    "binary": self.simple_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertIn("42", disassembly)

    def test_binary_segment_and_disassemble_fields_preserved(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.simple_source,
                    "binary": self.simple_binary_x86_64_elf,
                    "foo": "bar",
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        self.assertIn("foo", loaded.columns)

    def test_binary_segment_and_disassemble_simple_arm_macho(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.simple_source,
                    "binary": self.simple_binary_arm_macho,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "_main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertIn("42", disassembly)

    def test_binary_segment_and_disassemble_canary_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.canary_source,
                    "binary": self.canary_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertIn("fs + ", disassembly)

    def test_binary_segment_and_disassemble_tword_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.tword_source,
                    "binary": self.tword_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertIn("tword", disassembly)

    def test_binary_segment_and_disassemble_data_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.data_source,
                    "binary": self.data_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertNotIn(disassembly, "data")
        self.assertNotIn(disassembly, "value")

    def test_binary_segment_and_disassemble_relative_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.relative_source,
                    "binary": self.relative_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")
        segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 1)

        disassembly = filtered.get("disassembly").values[0]

        self.assertIn("call rel 5", disassembly)

    def test_binary_segment_and_disassemble_invalid_x86_64_elf(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "source": self.invalid_source,
                    "binary": self.invalid_binary_x86_64_elf,
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "disassembled.parquet")

        with self.assertLogs(level=WARNING):
            segment_and_disassemble_binary(dataset, path)

        loaded = read_parquet(path)

        filtered = loaded[loaded["name"] == "main"]

        self.assertEqual(len(filtered), 0)


class TestModelTokenizer(TestCase):
    def mock_dataset(
        self, frame: DataFrame, working: TemporaryDirectory, name: str
    ) -> str:
        path = join(working.name, name)
        write_parquet(frame, path)

        return path

    def mock_preprocessed(
        self, preprocessed: Dict, working: TemporaryDirectory, name: str
    ) -> str:
        path = join(working.name, name)

        with open(path, "w") as f:
            json.dump(preprocessed, f)

        return path

    def test_tokenizer_preprocess_simple(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "name": "test",
                    "disassembly": "xor eax eax [NEXT] add eax ebx [NEXT] sub eax 42",
                }
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "preprocessed.json")
        preprocess_tokens(dataset, path)

        with open(path, "r") as f:
            loaded = json.load(f)

        self.assertIn("xor", loaded["tokens"])
        self.assertIn("add", loaded["tokens"])
        self.assertIn("sub", loaded["tokens"])
        self.assertIn("eax", loaded["tokens"])
        self.assertIn("ebx", loaded["tokens"])

        self.assertEqual(loaded["tokens"]["eax"], 4)

        self.assertIn("42", loaded["immediates"])

    def test_tokenizer_merge_preprocessed_simple(self):
        working = TemporaryDirectory()

        first = {"tokens": {"xor": 1, "eax": 2}, "immediates": {"42": 1}}
        second = {"tokens": {"add": 1, "eax": 1, "ebx": 3}, "immediates": {"1337": 1}}

        first = self.mock_preprocessed(first, working, "first.json")
        second = self.mock_preprocessed(second, working, "second.json")

        path = join(working.name, "merged.json")
        merge_preprocessed_tokens([first, second], path)

        with open(path, "r") as f:
            loaded = json.load(f)

        self.assertIn("xor", loaded["tokens"])
        self.assertIn("add", loaded["tokens"])
        self.assertIn("eax", loaded["tokens"])
        self.assertIn("ebx", loaded["tokens"])

        self.assertEqual(loaded["tokens"]["xor"], 1)
        self.assertEqual(loaded["tokens"]["eax"], 3)
        self.assertEqual(loaded["tokens"]["ebx"], 3)

        self.assertIn("42", loaded["immediates"])
        self.assertIn("1337", loaded["immediates"])

        self.assertEqual(loaded["immediates"]["42"], 1)
        self.assertEqual(loaded["immediates"]["1337"], 1)

    def test_tokenizer_train_simple(self):
        working = TemporaryDirectory()

        preprocessed = {
            "tokens": {
                "xor": 1,
                "add": 1,
                "eax": 3,
            },
            "immediates": {"42": 1},
        }

        preprocessed = self.mock_preprocessed(
            preprocessed, working, "preprocessed.json"
        )

        path = join(working.name, "tokenizer.json")
        train_tokenizer(preprocessed, path, silent=True)

        tokenizer = load_tokenizer(path)

        encoding = tokenizer.encode("xor eax eax")

        self.assertEqual(encoding.tokens[0], "xor")
        self.assertEqual(encoding.tokens[1], "eax")
        self.assertEqual(encoding.tokens[2], "eax")

        encoding = tokenizer.encode("add eax 42")

        self.assertEqual(encoding.tokens[0], "add")
        self.assertEqual(encoding.tokens[1], "eax")
        self.assertEqual(encoding.tokens[2], "42")

        encoding = tokenizer.encode("add eax 300")

        self.assertEqual(encoding.tokens[2], TOKEN_UNKNOWN)

    def test_tokenizer_tokenize_simple(self):
        working = TemporaryDirectory()

        sources = DataFrame(
            [
                {
                    "id": "1",
                    "name": "test",
                    "disassembly": "xor eax eax add eax eax xor eax 42",
                }
            ]
        )

        # Taken from the above `test_tokenizer_train_simple()`.
        serialized_tokenizer = '{"version":"1.0","truncation":{"direction":"Right","max_length":512,"strategy":"LongestFirst","stride":0},"padding":{"strategy":{"Fixed":512},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[PAD]"},"added_tokens":[{"id":0,"content":"[PAD]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"[SEP]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":4,"content":"[MASK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":5,"content":"[NEXT]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":10,"content":"xor","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},{"id":11,"content":"add","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},{"id":12,"content":"eax","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":"[UNK]","continuing_subword_prefix":"__","end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"[PAD]":0,"[UNK]":1,"[SEP]":2,"[CLS]":3,"[MASK]":4,"[NEXT]":5,"2":6,"4":7,"__2":8,"42":9},"merges":[["4","__2"]]}}'

        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        tokenizer = join(working.name, "tokenizer.json")
        with open(tokenizer, "w") as f:
            f.write(serialized_tokenizer)

        path = join(working.name, "tokenized.parquet")
        tokenize(dataset, path, tokenizer)

        loaded = read_parquet(path)

        tokens = loaded["tokens"][0]
        unpadded = tokens[tokens != tokens[-1]]

        self.assertEqual(len(unpadded), len(sources["disassembly"][0].split()))


class TestModelTransformer(TestCase):
    HIDDEN_DIMENSIONS = 768
    VOCAB_SIZE = 1024
    SEQUENCE_LENGTH = 512
    HEADS = 12
    INTERMEDIATE_DIMENSIONS = 3072
    DROPOUT = 0.1
    EPS = 1e-12

    def setUp(self):
        set_grad_enabled(False)

    def tearDown(self):
        set_grad_enabled(True)

    def test_attention_simple(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        result = layer(state)

        self.assertEqual(result.shape, state.shape)

    def test_attention_masked(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        mask = rand(1, self.SEQUENCE_LENGTH) <= 0.2
        result = layer(state, mask)

        self.assertEqual(result.shape, state.shape)

    def test_attention_unbatched(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected tensor of shape", str(c.exception))

    def test_attention_mismatched_shape(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(1, self.SEQUENCE_LENGTH, 720)

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected tensor with hidden size", str(c.exception))

    def test_attention_masked_mismatched_shape(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        mask = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS) <= 0.2

        with self.assertRaises(ValueError) as c:
            layer(state, mask)

        self.assertIn("expected mask tensor of shape", str(c.exception))

    def test_attention_masked_mismatched_sequence_length(self):
        layer = Attention(self.HIDDEN_DIMENSIONS, self.HIDDEN_DIMENSIONS)
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        mask = rand(1, 256) <= 0.2

        with self.assertRaises(ValueError) as c:
            layer(state, mask)

        self.assertIn("mismatched sequence length", str(c.exception))

    def test_multi_head_attention_simple(self):
        layer = MultiHeadAttention(self.HIDDEN_DIMENSIONS, self.HEADS)
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        result = layer(state)

        self.assertEqual(result.shape, state.shape)

    def test_multi_head_attention_invalid_head_count(self):
        with self.assertRaises(ValueError) as c:
            MultiHeadAttention(self.HIDDEN_DIMENSIONS, 13)

        self.assertIn("invalid number of heads", str(c.exception))

    def test_feed_forward_simple(self):
        layer = FeedForward(
            self.HIDDEN_DIMENSIONS, self.INTERMEDIATE_DIMENSIONS, self.DROPOUT
        )
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        result = layer(state)

        self.assertEqual(result.shape, state.shape)

    def test_feed_forward_mismatched_shape(self):
        layer = FeedForward(
            self.HIDDEN_DIMENSIONS, self.INTERMEDIATE_DIMENSIONS, self.DROPOUT
        )
        state = rand(1, self.SEQUENCE_LENGTH, 720)

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected tensor with hidden size", str(c.exception))

    def test_transformer_encoder_layer_simple(self):
        layer = TransformerEncoderLayer(
            self.HIDDEN_DIMENSIONS,
            self.HEADS,
            self.INTERMEDIATE_DIMENSIONS,
            self.DROPOUT,
        )
        state = rand(1, self.SEQUENCE_LENGTH, self.HIDDEN_DIMENSIONS)
        result = layer(state)

        self.assertEqual(result.shape, state.shape)

    def test_position_embedding_simple(self):
        layer = PositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(1, self.SEQUENCE_LENGTH))
        result = layer(state)

        self.assertEqual(result.ndim, 3)

        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.SEQUENCE_LENGTH)
        self.assertEqual(result.shape[2], self.HIDDEN_DIMENSIONS)

    def test_position_embedding_mismatched_shape(self):
        layer = PositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(self.SEQUENCE_LENGTH,))

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected tensor of shape", str(c.exception))

    def test_position_embedding_mismatched_sequence_length(self):
        layer = PositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(1, 256))

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected sequence length", str(c.exception))

    def test_transformer_encoder_simple(self):
        layer = TransformerEncoder(
            2,
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            2,
            self.INTERMEDIATE_DIMENSIONS,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(1, self.SEQUENCE_LENGTH))
        result = layer(state)

        self.assertEqual(result.ndim, 3)

        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.SEQUENCE_LENGTH)
        self.assertEqual(result.shape[2], self.HIDDEN_DIMENSIONS)


class TestModelCustom(TestCase):
    HIDDEN_DIMENSIONS = 768
    VOCAB_SIZE = 1024
    SEQUENCE_LENGTH = 512
    NEXT_TOKEN_ID = 0
    DROPOUT = 0.1
    EPS = 1e-12

    def setUp(self):
        set_grad_enabled(False)

    def tearDown(self):
        set_grad_enabled(True)

    def test_compute_instruction_index_basic(self):
        state = tensor([[1, 2, 0, 3, 0, 4, 5]])
        result = InstructionTracePositionEmbedding.compute_instruction_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 0, 0, 1, 1, 2, 2]])
        self.assertTrue(result.equal(expected))

    def test_compute_instruction_index_no_next_tokens(self):
        state = tensor([[1, 2, 3, 4]])
        result = InstructionTracePositionEmbedding.compute_instruction_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 0, 0, 0]])
        self.assertTrue(result.equal(expected))

    def test_compute_instruction_index_leading_next(self):
        state = tensor([[0, 1, 2]])
        result = InstructionTracePositionEmbedding.compute_instruction_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 1, 1]])
        self.assertTrue(result.equal(expected))

    def test_compute_instruction_index_batch(self):
        state = tensor([[1, 0, 2], [3, 4, 0]])
        result = InstructionTracePositionEmbedding.compute_instruction_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 0, 1], [0, 0, 0]])
        self.assertTrue(result.equal(expected))

    def test_compute_argument_index_basic(self):
        state = tensor([[1, 2, 0, 3, 0, 4, 5]])
        result = InstructionTracePositionEmbedding.compute_argument_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 1, 2, 0, 1, 0, 1]])
        self.assertTrue(result.equal(expected))

    def test_compute_argument_index_no_next_tokens(self):
        state = tensor([[1, 2, 3, 4]])
        result = InstructionTracePositionEmbedding.compute_argument_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 1, 2, 3]])
        self.assertTrue(result.equal(expected))

    def test_compute_argument_index_leading_next(self):
        state = tensor([[0, 1, 2]])
        result = InstructionTracePositionEmbedding.compute_argument_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 0, 1]])
        self.assertTrue(result.equal(expected))

    def test_compute_argument_index_batch(self):
        state = tensor([[1, 0, 2], [3, 4, 0]])
        result = InstructionTracePositionEmbedding.compute_argument_index(
            state, self.NEXT_TOKEN_ID
        )

        expected = tensor([[0, 1, 0], [0, 1, 2]])
        self.assertTrue(result.equal(expected))

    def test_instruction_argument_position_embedding_simple(self):
        layer = InstructionTracePositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.NEXT_TOKEN_ID,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(1, self.SEQUENCE_LENGTH))
        result = layer(state)

        self.assertEqual(result.ndim, 3)

        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.SEQUENCE_LENGTH)
        self.assertEqual(result.shape[2], self.HIDDEN_DIMENSIONS)

    def test_instruction_argument_position_embedding_mismatched_shape(self):
        layer = InstructionTracePositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.NEXT_TOKEN_ID,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(self.SEQUENCE_LENGTH,))

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected tensor of shape", str(c.exception))

    def test_instruction_argument_position_embedding_mismatched_sequence_length(self):
        layer = InstructionTracePositionEmbedding(
            self.HIDDEN_DIMENSIONS,
            self.VOCAB_SIZE,
            self.SEQUENCE_LENGTH,
            self.NEXT_TOKEN_ID,
            self.DROPOUT,
            self.EPS,
        )
        state = randint(0, self.VOCAB_SIZE, size=(1, 256))

        with self.assertRaises(ValueError) as c:
            layer(state)

        self.assertIn("expected sequence length", str(c.exception))


class TestModelDataset(TestCase):
    def write_chunk(self, directory: str, name: str, rows: list) -> str:
        path = join(directory, name)
        write_parquet(DataFrame(rows), path)
        return path

    def test_dataset_single_file(self):
        working = TemporaryDirectory()
        rows = [{"value": i} for i in range(10)]
        path = self.write_chunk(working.name, "chunk.parquet", rows)

        dataset = ParquetDataset(path)

        self.assertEqual(list(dataset), rows)

    def test_dataset_directory(self):
        working = TemporaryDirectory()
        self.write_chunk(working.name, "a.parquet", [{"value": i} for i in range(5)])
        self.write_chunk(
            working.name, "b.parquet", [{"value": i} for i in range(5, 10)]
        )

        dataset = ParquetDataset(working.name)

        self.assertEqual(len(list(dataset)), 10)

    def test_dataset_directory_sorted(self):
        working = TemporaryDirectory()
        self.write_chunk(working.name, "c.parquet", [{"chunk": "c"}])
        self.write_chunk(working.name, "a.parquet", [{"chunk": "a"}])
        self.write_chunk(working.name, "b.parquet", [{"chunk": "b"}])

        dataset = ParquetDataset(working.name)

        self.assertEqual([r["chunk"] for r in dataset], ["a", "b", "c"])

    def test_dataset_multi_worker(self):
        working = TemporaryDirectory()
        for i in range(4):
            self.write_chunk(working.name, f"chunk_{i}.parquet", [{"chunk": i}])

        dataset = ParquetDataset(working.name)

        with patch("undertale.models.dataset.get_worker_info") as mock:
            mock.return_value = SimpleNamespace(id=0, num_workers=2)
            worker_0 = list(dataset)

            mock.return_value = SimpleNamespace(id=1, num_workers=2)
            worker_1 = list(dataset)

        self.assertEqual(len(worker_0) + len(worker_1), 4)
        self.assertFalse(
            set(r["chunk"] for r in worker_0) & set(r["chunk"] for r in worker_1)
        )

    def test_dataset_empty_directory(self):
        working = TemporaryDirectory()

        dataset = ParquetDataset(working.name)

        self.assertEqual(list(dataset), [])

    def test_dataset_schema_valid(self):
        working = TemporaryDirectory()
        path = self.write_chunk(
            working.name, "chunk.parquet", [{"id": "1", "value": 1}]
        )

        ParquetDataset(path, schema=Dataset)

    def test_dataset_schema_invalid(self):
        working = TemporaryDirectory()
        path = self.write_chunk(working.name, "chunk.parquet", [{"value": 1}])

        with self.assertRaises(SchemaError):
            ParquetDataset(path, schema=Dataset)

    def test_dataset_schema_empty_directory(self):
        working = TemporaryDirectory()

        ParquetDataset(working.name, schema=Dataset)


class TestModelMaskedLMCollator(TestCase):
    SEQUENCE_LENGTH = 16
    VOCAB_SIZE = 100
    MASK_TOKEN_ID = 4

    def make_batch(self, size: int) -> list:
        return [
            {
                "tokens": list(range(1, self.SEQUENCE_LENGTH + 1)),
                "mask": [1] * self.SEQUENCE_LENGTH,
            }
            for _ in range(size)
        ]

    def make_padded_batch(self, size: int, padding: int) -> list:
        tokens = list(range(1, self.SEQUENCE_LENGTH - padding + 1)) + [0] * padding
        mask = [1] * (self.SEQUENCE_LENGTH - padding) + [0] * padding
        return [{"tokens": tokens, "mask": mask} for _ in range(size)]

    def test_collator_returns_dict(self):
        collator = MaskedLMCollator(self.MASK_TOKEN_ID, self.VOCAB_SIZE)
        result = collator(self.make_batch(4))

        self.assertIsInstance(result, dict)
        self.assertIn("tokens", result)
        self.assertIn("mask", result)
        self.assertIn("labels", result)

    def test_collator_output_shapes(self):
        collator = MaskedLMCollator(self.MASK_TOKEN_ID, self.VOCAB_SIZE)
        result = collator(self.make_batch(4))

        self.assertEqual(result["tokens"].shape, (4, self.SEQUENCE_LENGTH))
        self.assertEqual(result["mask"].shape, (4, self.SEQUENCE_LENGTH))
        self.assertEqual(result["labels"].shape, (4, self.SEQUENCE_LENGTH))

    def test_collator_labels_ignore_unmasked(self):
        collator = MaskedLMCollator(
            self.MASK_TOKEN_ID, self.VOCAB_SIZE, probability=1.0
        )
        result = collator(self.make_batch(4))

        self.assertTrue((result["labels"] != -100).all())

    def test_collator_labels_minus_100_at_unmasked(self):
        collator = MaskedLMCollator(
            self.MASK_TOKEN_ID, self.VOCAB_SIZE, probability=0.0
        )
        result = collator(self.make_batch(4))

        self.assertTrue((result["labels"] == -100).all())

    def test_collator_padding_not_masked(self):
        collator = MaskedLMCollator(
            self.MASK_TOKEN_ID, self.VOCAB_SIZE, probability=1.0
        )
        result = collator(self.make_padded_batch(4, padding=4))

        # Labels at padding positions must remain -100.
        self.assertTrue((result["labels"][:, -4:] == -100).all())

    def test_collator_tokens_unchanged_where_not_candidate(self):
        collator = MaskedLMCollator(
            self.MASK_TOKEN_ID, self.VOCAB_SIZE, probability=0.0
        )
        batch = self.make_batch(2)
        result = collator(batch)

        expected = tensor([item["tokens"] for item in batch])
        self.assertTrue(result["tokens"].equal(expected))

    def test_collator_mask_preserved(self):
        collator = MaskedLMCollator(self.MASK_TOKEN_ID, self.VOCAB_SIZE)
        batch = self.make_padded_batch(4, padding=4)
        result = collator(batch)

        expected = tensor([item["mask"] for item in batch])
        self.assertTrue(result["mask"].equal(expected))

    def test_collator_default_probability(self):
        self.assertEqual(MaskedLMCollator.PROBABILITY, 0.15)
        collator = MaskedLMCollator(self.MASK_TOKEN_ID, self.VOCAB_SIZE)
        self.assertEqual(collator.probability, 0.15)


class TestUtilitiesDatasetSplit(TestCase):
    def write_dataset(self, directory: str, rows: list) -> str:
        path = join(directory, "data.parquet")
        write_parquet(DataFrame(rows), path)
        return path

    def test_two_way_split(self):
        working = TemporaryDirectory()
        rows = [{"value": i} for i in range(1000)]
        source = self.write_dataset(working.name, rows)
        output = join(working.name, "out")

        with Cluster(type="local") as cluster, Client(cluster) as client:
            split_dataset(source, output, [("training", 90.0), ("validation", 10.0)])
            flush(client)

        training = read_parquet(f"{output}-training")
        validation = read_parquet(f"{output}-validation")

        self.assertEqual(len(training) + len(validation), 1000)
        self.assertAlmostEqual(len(training) / 1000, 0.9, delta=0.05)

    def test_three_way_split(self):
        working = TemporaryDirectory()
        rows = [{"value": i} for i in range(1000)]
        source = self.write_dataset(working.name, rows)
        output = join(working.name, "out")

        with Cluster(type="local") as cluster, Client(cluster) as client:
            split_dataset(
                source,
                output,
                [("training", 80.0), ("validation", 10.0), ("test", 10.0)],
            )
            flush(client)

        training = read_parquet(f"{output}-training")
        validation = read_parquet(f"{output}-validation")
        test = read_parquet(f"{output}-test")

        self.assertEqual(len(training) + len(validation) + len(test), 1000)
        self.assertAlmostEqual(len(training) / 1000, 0.8, delta=0.05)

    def test_percentages_must_sum_to_100(self):
        working = TemporaryDirectory()
        rows = [{"value": i} for i in range(10)]
        source = self.write_dataset(working.name, rows)
        output = join(working.name, "out")

        with self.assertRaises(ValueError):
            split_dataset(source, output, [("training", 80.0), ("validation", 10.0)])

    def test_invalid_split_format_missing_colon(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_split("training")

    def test_invalid_split_format_non_numeric_percentage(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_split("training:abc")


if __name__ == "__main__":
    main("unit")
