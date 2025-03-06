import logging
import os
import sqlite3
import time
import typing

import datasets
import pefile

from . import dataset
from .transforms.disassemble import ghidra

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


class AssemblageVcpkg(dataset.Dataset):
    url = "https://huggingface.co/datasets/changliu8541/Assemblage_vcpkgDLL"
    description = "Assemblage DLLs"
    path = "assemblage-vcpkg"

    @classmethod
    def parse(cls, path: str, processes: typing.Optional[int] = None):
        if "assemblage-vcpkg" in path:
            ds = dataset.Dataset.load_from_disk(path)
        else:
            sqlfile = f"{path}/vcpkg.sqlite"
            bins_dir = f"{path}/vcpkg"
            with sqlite3.connect(sqlfile) as db:
                cur = db.cursor()
                current_binary = None
                current_binary_id = None
                i = 0

                functions = {}

                num_failed_rvas = 0
                for row in cur.execute(
                    "SELECT f.id, b.id, b.file_name, b.platform, b.toolset_version, b.optimization, b.path, f.name, r.start, r.end  FROM functions f INNER JOIN binaries b ON f.binary_id = b.id INNER JOIN rvas r ON f.id = r.function_id ORDER BY b.id;"
                ):
                    (
                        f_id,
                        b_id,
                        b_file_name,
                        b_platform,
                        b_toolset_version,
                        b_optimization,
                        b_path,
                        f_name,
                        r_start,
                        r_end,
                    ) = row

                    if i > 0 and (i % 10000) == 0:
                        (t, e) = tick()
                        print(
                            f"{i} functions so far. tick={t} el={e}. {num_failed_rvas} with bad rva.s"
                        )

                    if (current_binary_id is None) or (current_binary_id != b_id):
                        current_binary_id = b_id
                        current_binary = pefile.PE(os.path.join(bins_dir, b_path))

                    try:
                        raw_data = current_binary.get_data(
                            rva=r_start, length=r_end - r_start
                        )
                    except:
                        logger.warn(
                            f"Discarding function {f_id} as rvas for it seem not to make sense with binary: {num_failed_rvas} count"
                        )
                        num_failed_rvas += 1
                        continue

                    tup = (
                        (r_start, r_end),
                        raw_data,
                        b_platform,
                        b_toolset_version,
                        b_optimization,
                        f_name,
                    )
                    if f_id not in functions:
                        functions[f_id] = []
                    functions[f_id].append(tup)
                    i += 1

            def gen():
                for f_id, arr in functions.items():
                    if len(arr) > 1:
                        arr.sort(key=lambda tup: tup[0][0])
                        code = b""
                        for i in range(len(arr) - 1):
                            x = arr[i + 1][0][0] - arr[i][0][1] - 1
                            code = code + arr[i][1] + (b"\x90") * x + arr[i + 1][1]
                    else:
                        code = arr[0][1]
                    platform = arr[0][2]
                    toolset_version = arr[0][3]
                    optimization = arr[0][4]
                    yield {
                        "id": f_id,
                        "code": code,
                        "compiler": toolset_version,
                        "optimization level": optimization,
                        "architecture": platform,
                    }

            (t, e) = tick()
            logger.info(
                f"tick={t} el={e}. Beginning construction of final dataset via from_generator."
            )
            ds = datasets.Dataset.from_generator(gen)
        ds.__class__ = cls
        return ds


class AssemblageVcpkgDisassembled(AssemblageVcpkg):
    path = "assemblage-vcpkg-disassembled"

    transforms = [ghidra.GhidraDisassemble()]


if __name__ == "__main__":
    dataset.main([AssemblageVcpkg, AssemblageVcpkgDisassembled])
