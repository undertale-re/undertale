import json
import os
import tarfile
from datetime import datetime
from os import listdir
from os.path import basename, exists, isdir, join
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

from dask.dataframe import from_pandas
from pandas import DataFrame, read_parquet
from utils import main

from undertale.exceptions import EnvironmentError as LocalEnvironmentError
from undertale.exceptions import PathError, SchemaError
from undertale.pipeline import Cluster
from undertale.pipeline.cpp import compile_cpp
from undertale.pipeline.dedupe import dedupe_by_sha256
from undertale.pipeline.json import merge_json
from undertale.pipeline.parquet import resize_parquet
from undertale.pipeline.tarfile import extract_tarfile
from undertale.utils import (
    assert_path_does_not_exist,
    assert_path_exists,
    find,
    hash,
    timestamp,
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


class TestUtilitiesAsserts(TestCase):
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

    def test_assert_path_does_not_exist_existing_directory(self):
        working = TemporaryDirectory()

        with self.assertRaises(PathError):
            assert_path_does_not_exist(working.name)

    def test_assert_path_does_not_exist_existing_file(self):
        working = TemporaryDirectory()

        target = join(working.name, "test.txt")

        with open(target, "w"):
            pass

        with self.assertRaises(PathError):
            assert_path_does_not_exist(target)

    def test_assert_path_does_not_exist_nonexistent(self):
        assert_path_does_not_exist("foo/bar/baz")

    def test_assert_path_does_not_exist_nonexistent_create(self):
        working = TemporaryDirectory()

        target = join(working.name, "test")

        assert_path_does_not_exist(target, create=True)

        self.assertTrue(exists(target))
        self.assertTrue(isdir(target))


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


class TestPipelineDask(TestCase):
    def test_cluster_unsupported_type(self):
        with self.assertRaises(ValueError):
            Cluster("foo")


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
        from_pandas(frame, npartitions=chunks).to_parquet(path)

        return path

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


class TestPipelineCpp(TestCase):
    @staticmethod
    def mock_dataset(frame: DataFrame, working: TemporaryDirectory, name: str) -> str:
        path = join(working.name, name)
        frame.to_parquet(path)

        return path

    def test_cpp_compile_invalid_schema(self):
        working = TemporaryDirectory()

        sources = DataFrame([{"foo": "bar"}])
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        with self.assertRaises(SchemaError):
            path = join(working.name, "compiled.parquet")
            compile_cpp(dataset, path)

    def test_cpp_compile_simple(self):
        working = TemporaryDirectory()

        sources = DataFrame([{"id": "1", "source": "int main() { return 42; }"}])
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
                {"id": "2", "source": "int main() { return 42; }"},
            ]
        )
        dataset = self.mock_dataset(sources, working, "dataset.parquet")

        path = join(working.name, "compiled.parquet")
        compile_cpp(dataset, path)

        loaded = read_parquet(path)

        self.assertEqual(len(loaded), 1)


class TestPipelineDedupe(TestCase):
    @staticmethod
    def mock_dataset(frames: List[DataFrame], working: str, name: str) -> List[str]:
        paths = []
        for i, frame in enumerate(frames):
            path = join(working, f"{name}-{str(i)}")
            frame.to_parquet(path)
            paths.append(path)

        return paths

    def test_dedupe(self):
        with TemporaryDirectory() as working:
            data = [
                DataFrame({"value": [b"a", b"b", b"a", b"c", b"b"]}),
                DataFrame({"value": [b"a", b"b", b"a", b"c", b"d"]}),
            ]
            dataset = self.mock_dataset(data, working, "dataset.parquet")

            path = join(working, "deduped.parquet")
            dedupe_by_sha256(dataset, path, "value")

            deduped = read_parquet(path)

            self.assertEqual(len(deduped), 4)


if __name__ == "__main__":
    main("unit")
