import logging
import os
import sqlite3 as sqlite
import time
import typing

import datasets
import pefile

from . import dataset

logger = logging.getLogger(__name__)

last_time = time.time()
first_time = last_time


def tick():
    global last_time
    global first_time

    t = time.time()
    delta = t - last_time
    f_delta = t - first_time
    last_time = t
    return (delta, f_delta)


class AssemblageWindowsBinariesDataset(dataset.Dataset):
    url = "https://assemblage-dataset.net/"
    description = "assemblage project 62k Windows PE Binaries obtained by compiling source for entire programs or libraries from Github repositories"
    path = "assemblage-windows-binaries"

    @classmethod
    def parse(cls, path: str, processes: typing.Optional[int] = None):
        """Download the dataset from assemblage webserver, un-tar it, and ...

        This is a big dataset. In order to create it, you will have to arrange
        for enough disk space.  Also, note that the code for parsing was
        developed and run on a machine with 512GB RAM.  Takes about 1.5 hours
        to parse on that machine, FYI.

        Disk space requirements.

        1. About 20GB for the two zip files that come from the URL above:
           winpe_licensed.zip and winpe_licensed.sqlite.zip. These should be
           in the directory pointed to by `path` argument.

        2. These uncompress into about 102 GB in same directory.

        3. Hugging face needs to do mysterious big things in your ~.cache
           directory. If that's very big (I have no idea how much space it
           needs) then fine.  Otherwise, set the HF_HOME env variable to
           somewhere very large and HF will use that space instead of ~.cache.

        4. The default output dir (unless you set the
           UNDERTALE_DATASETS_DIRECTORY environment variable) is
           ~/undertale_shared.  The dataset generated there, in the form of
           arrow files, will be about 102GB.  So make sure there is enough
           space there.

        Arguments:
            path: The path to a directory containing the raw data from the
                assemblage folks.
            processes: The number of parallel processes to use.
        """

        logger.warning("DANGER this dataset may contain malware so be careful with it")

        bins_zip = "winpe_licensed.zip"
        sql_zip = "winpe_licensed.sqlite.zip"

        sqlfile = f"{path}/winpe.sqlite"
        bins_dir = f"{path}/winpe"

        os.system(f"/usr/bin/unzip -d {path} {path}/{sql_zip}")
        os.system(f"/usr/bin/unzip -d {path} {path}/{bins_zip}")

        (t, e) = tick()
        logger.info(f"tick={t} el={e}. Unzipping complete")

        with sqlite.connect(sqlfile) as db:
            cur = db.cursor()

            current_binary = None
            current_binary_id = None
            i = 0
            num_with_source = 0
            num_failed_rvas = 0
            functions = {}

            for row in cur.execute(
                "SELECT f.id, b.id, b.file_name, b.platform, b.build_mode, b.optimization, b.path, f.name, f.source_codes, r.start, r.end  FROM functions f INNER JOIN binaries b ON f.binary_id = b.id INNER JOIN rvas r ON f.id = r.function_id ORDER BY b.id;"
            ):
                (
                    f_id,
                    b_id,
                    b_file_name,
                    b_platform,
                    b_build_mode,
                    b_optimization,
                    b_path,
                    f_name,
                    f_source,
                    r_start,
                    r_end,
                ) = row

                if i > 0 and (i % 100000) == 0:
                    (t, e) = tick()
                    logger.info(
                        f"{i} functions so far, tick={t} el={e}, {float(num_with_source)/i:.2f} have source, {num_failed_rvas} with bad rvas"
                    )

                if (current_binary_id is None) or (current_binary_id != b_id):
                    current_binary_id = b_id
                    current_binary = pefile.PE(os.path.join(bins_dir, b_path))

                try:
                    raw_data = current_binary.get_data(
                        rva=r_start, length=r_end - r_start + 1
                    )
                except:
                    logger.warn(
                        f"discarding function {f_id} as rvas for it seem not to make sense with binary: {num_failed_rvas} count"
                    )
                    num_failed_rvas += 1
                    continue

                if f_source == "":
                    f_source = None
                else:
                    num_with_source += 1

                tup = ((r_start, r_end), raw_data, f_source)
                if f_id not in functions:
                    functions[f_id] = []
                functions[f_id].append(tup)
                i += 1

        (t, e) = tick()
        logger.info(f"tick={t} el={e}, done with sql jiggery-pokery")

        # this generator will yield the rows in the dataset one by one
        def gen():
            for f_id, arr in functions.items():
                if len(arr) > 1:
                    arr.sort(key=lambda tup: tup[0][0])
                    code = b""
                    for i in range(len(arr) - 1):
                        x = arr[i + 1][0][0] - arr[i][0][1] - 1
                        code = code + arr[i][1] + (b"\x90") * x + arr[i + 1][1]
                    source = ""
                    for _, _, s in arr:
                        if s is not None:
                            source = source + s
                else:
                    code = arr[0][1]
                    source = arr[0][2]
                if source == "":
                    source = None
                yield {"id": f_id, "code": code, "source": source}

        ds = datasets.Dataset.from_generator(gen)

        (t, e) = tick()
        logger.info(
            f"tick={t} el={e}, completed construction of final dataset via from_generator"
        )

        ds.__class__ = cls

        return ds


if __name__ == "__main__":
    dataset.main([AssemblageWindowsBinariesDataset])
