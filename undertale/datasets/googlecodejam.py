import io
import logging
import os
import tarfile

import datasets
import pandas as pd

from . import dataset
from .transforms import compile
from .transforms.disassemble import capstone

logger = logging.getLogger(__name__)


class GoogleCodeJam(dataset.Dataset):
    url = "https://codingcompetitions.withgoogle.com/codejam/archive"
    description = "Googe Code Jam competition from 2008-2020"
    path = "google-code-jam"

    @classmethod
    def list_bz2_file(cls, dir_path):
        """List all files with .bz2 extension in 'dir_path' and sort."""

        bz2_files = [file for file in os.listdir(dir_path) if file.endswith(".bz2")]
        bz2_files.sort()

        return bz2_files

    @classmethod
    def parse_bz2_file(cls, file_path):
        """Parse .csv.tar.bz2 file to dataframe."""

        with tarfile.open(file_path, "r:bz2") as tar:
            # There should be only one file in the archive
            csv_file_name = tar.getmembers()[0]

            # Extract file as a file-like object
            csv_file = tar.extractfile(csv_file_name)

            if csv_file is None:
                raise ValueError("Failed to extract CSV file from the archive.")

            # Convert csv to dataframe
            df = pd.read_csv(
                io.BytesIO(csv_file.read()),
                dtype={
                    "Unnamed: 0": "int64",
                    "file": "str",
                    "flines": "str",
                    "full_path": "str",
                    "round": "str",
                    "solution": "str",
                    "task": "str",
                    "username": "str",
                    "year": "int64",
                },
            )

            # Select and rename columns
            df = df.drop(
                columns=[
                    "round",
                    "username",
                    "solution",
                    "full_path",
                    "Unnamed: 0",
                    "year",
                    "file",
                ]
            )
            df = df.rename(columns={"flines": "source", "task": "summary"})

        return df

    @classmethod
    def parse(cls, path: str, processes=None):
        raw_dataset_path = os.path.abspath(os.path.expanduser(path))

        # get sorted file list of .bz2 files
        bz2_files = cls.list_bz2_file(raw_dataset_path)
        tot_rows = 0

        # process .bz2 files in raw_dataset directory
        for index, item in enumerate(bz2_files):
            logger.info(f"processing file: {item}")

            file_path = os.path.join(raw_dataset_path, item)

            df = cls.parse_bz2_file(file_path)

            if index == 0:
                DF = df
            else:
                DF = pd.concat([DF, df], ignore_index=True)

            tot_rows += len(df)
            logger.info(f"collected {len(df)} samples from {item}")

        dataset = datasets.Dataset.from_pandas(DF)
        dataset.__class__ = cls

        logger.info(f"collected {tot_rows} from {len(bz2_files)} files")

        return dataset


class GoogleCodeJamCompiled(GoogleCodeJam):
    path = "google-code-jam-compiled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
    ]


class GoogleCodeJamCompiledDisassembled(GoogleCodeJam):
    path = "google-code-jam-compiled-disassembled"

    transforms = [
        compile.Compile(),
        compile.CompileErrorsFilter(),
        capstone.CapstoneDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main(
        [GoogleCodeJamCompiledDisassembled, GoogleCodeJamCompiled, GoogleCodeJam]
    )
