import json
import logging
import os
import typing
from pathlib import Path

import datasets
import gdown

from . import dataset

logger = logging.getLogger(__name__)


class CoDesc(dataset.Dataset):
    url = "https://github.com/csebuetnlp/CoDesc/tree/master"
    raw_url = "https://drive.google.com/uc?id=14t7fYsW0a09mfBmFJhjsZv-3obnLnNmS"
    description = """A large dataset of 4.2m Java source code and parallel data of their description from code search,
    and code summarization studies."""
    path = "codesc"

    # raw datasets dir
    raw_dir = (Path(dataset.DEFAULT_DATASETS_DIRECTORY) / "raw").expanduser().absolute()
    dataset_name = "CoDesc"
    raw_data_dir = Path(raw_dir) / dataset_name
    zipped_data = raw_data_dir / f"{dataset_name}.7z"
    zipped_size = 647537127
    json_data = raw_data_dir / dataset_name / f"{dataset_name}.json"
    json_size = 4866435476

    @classmethod
    def download(cls, processes: typing.Optional[int] = None):
        # Check if raw zipped data exists and is correct size
        if (not cls.zipped_data.exists()) or (
            cls.zipped_data.stat().st_size != cls.zipped_size
        ):
            cls.raw_data_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading compressed file {cls.dataset_name}.7z")
            gdown.download(cls.raw_url, str(cls.zipped_data))
            assert cls.zipped_data.stat().st_size == cls.zipped_size
            logger.info(f"{cls.dataset_name}.7z downloaded")

    @classmethod
    def parse(cls, path: str, processes: typing.Optional[int] = None):
        if path == "download":
            logger.info(f"Downloading {cls.__name__}")
            cls.download(processes)

        if not cls.json_data.exists():
            cwd = Path().absolute()  # current dir
            os.chdir(cls.raw_data_dir)  # change dir to raw datasets
            logger.info(f"Extracting {cls.dataset_name}.7z")
            os.system(f"7zz x {cls.dataset_name}.7z")
            assert cls.json_data.stat().st_size == cls.json_size

            logger.info(f"{cls.dataset_name}.7z extracted")
            os.chdir(cwd)  # change dir back

        logger.info(f"Parsing {cls.__name__}")

        # Read dataset json
        data = json.load(open(cls.json_data, "r"))

        def gen_dataset():
            for item in data:
                yield {
                    "id": item["id"],
                    "source": item["original_code"],
                    "summary": item["nl"],
                }

        dataset = datasets.Dataset.from_generator(gen_dataset)
        logger.info(f"Parsed {cls.__name__}")

        dataset.__class__ = cls

        return dataset


if __name__ == "__main__":
    dataset.main([CoDesc])
