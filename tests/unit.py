import json
import logging
import os
import tarfile
from datetime import datetime
from logging import WARNING
from os import listdir, makedirs
from os.path import basename, exists, isdir, isfile, join
from tempfile import TemporaryDirectory
from time import sleep
from typing import Dict
from unittest import SkipTest, TestCase

from dask.dataframe import from_pandas
from pandas import DataFrame, read_parquet
from pyarrow.parquet import read_metadata as pyarrow_read_metadata
from utils import load_resource, main

from undertale.exceptions import EnvironmentError as LocalEnvironmentError
from undertale.exceptions import PathError, SchemaError
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
from undertale.pipeline import Client, Cluster, fanout, flush
from undertale.pipeline.binary import segment_and_disassemble_binary
from undertale.pipeline.cpp import compile_cpp
from undertale.pipeline.json import merge_json
from undertale.pipeline.parquet import hash_parquet_column, resize_parquet
from undertale.pipeline.tarfile import extract_tarfile
from undertale.utils import (
    RemoteException,
    assert_path_exists,
    enforce_extension,
    find,
    get_or_create_directory,
    get_or_create_file,
    hash,
    subprocess,
    timestamp,
    write_parquet,
)


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

    def test_parquet_resize_compression_specified(self):
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
            hash_parquet_column(dataset, join(working.name, "hashed"), "mordor", "hash")

    def test_parquet_hash_column_simple(self):
        working = TemporaryDirectory()

        data = b"\xde\xad\xbe\xef"
        dataset = [{"id": i, "data": data} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        hashed = hash_parquet_column(path, join(working.name, "hashed"), "data", "hash")

        loaded = read_parquet(hashed)

        self.assertIn("hash", loaded.columns)
        self.assertEqual(loaded["hash"][0], hash(data))

    def test_parquet_resize_chunks_and_size(self):
        with self.assertRaises(ValueError):
            resize_parquet("", "", chunks=10, size=10)

    def test_parquet_resize_neither_chunks_nor_size(self):
        with self.assertRaises(ValueError):
            resize_parquet("", "")

    def test_parquet_resize_one_to_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100)

        path = join(working.name, "resized")
        created = resize_parquet(dataset, path, chunks=20)

        self.assertEqual(len(created), 20)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_resize_many_to_one_chunk(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        path = join(working.name, "resized")
        created = resize_parquet(dataset, path, chunks=1)

        self.assertEqual(len(created), 1)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_resize_many_to_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        path = join(working.name, "resized")
        created = resize_parquet(dataset, path, chunks=25)

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_resize_too_many_chunks(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=20)

        path = join(working.name, "resized")
        created = resize_parquet(dataset, path, chunks=25)

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 20)

    def test_parquet_resize_size(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100)

        path = join(working.name, "resized")
        created = resize_parquet(dataset, path, size=64)

        self.assertEqual(len(created), 26)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_resize_chunk_list_input(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=100, chunks=20)

        chunks = [join(dataset, f) for f in listdir(dataset)]

        path = join(working.name, "resized")
        created = resize_parquet(chunks, path, chunks=25)

        self.assertEqual(len(created), 25)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 100)

    def test_parquet_resize_deduplicate_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            resize_parquet(dataset, path, chunks=1, deduplicate=["foo"])

    def test_parquet_resize_deduplicate_simple(self):
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
        resize_parquet(path, output, chunks=1, deduplicate=["id"])

        loaded = read_parquet(output)

        self.assertEqual(len(loaded), 15)
        self.assertEqual(len(set(loaded["id"])), 15)

    def test_parquet_resize_drop_and_keep(self):
        with self.assertRaises(ValueError):
            resize_parquet("", "", drop=["foo"], keep=["bar"])

    def test_parquet_resize_drop_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            resize_parquet(dataset, path, chunks=1, drop=["foo"])

    def test_parquet_resize_drop_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        resize_parquet(path, output, chunks=1, drop=["foo"])

        loaded = read_parquet(output)

        self.assertNotIn("foo", loaded.columns)

    def test_parquet_resize_keep_invalid_schema(self):
        working = TemporaryDirectory()
        dataset = self.mock_dataset(working, "dataset", size=10)

        with self.assertRaises(SchemaError):
            path = join(working.name, "resized")
            resize_parquet(dataset, path, chunks=1, keep=["foo"])

    def test_parquet_resize_keep_simple(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar", "baz": "zaa"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        resize_parquet(path, output, chunks=1, keep=["foo"])

        loaded = read_parquet(output)

        self.assertIn("foo", loaded.columns)
        self.assertNotIn("id", loaded.columns)
        self.assertNotIn("baz", loaded.columns)

    def test_parquet_resize_compression(self):
        working = TemporaryDirectory()

        dataset = [{"id": i, "foo": "bar"} for i in range(10)]

        frame = DataFrame(dataset)
        path = join(working.name, "dataset")
        write_parquet(frame, path)

        output = join(working.name, "resized")
        resize_parquet(path, output, chunks=1, compression="snappy")

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

        tokens = loaded["input_ids"][0]
        unpadded = tokens[tokens != tokens[-1]]

        self.assertEqual(len(unpadded), len(sources["disassembly"][0].split()))


if __name__ == "__main__":
    main("unit")
