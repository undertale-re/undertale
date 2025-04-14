import logging
import os
import sqlite3
import time

# import typing
from pathlib import Path

import pefile
from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.readers.base import BaseReader
from datatrove.pipeline.writers import ParquetWriter

from .base import DEFAULT_DATASETS_DIRECTORY, Dataset, main
from .pipeline.disassemblers import RadareDisassembler

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


def create_directory(*args, **kwargs):
    import os
    import shutil

    USER = os.environ.get("USER")
    dst = f"/state/partition1/user/{USER}"
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(
        f"/home/gridsan/{USER}/undertale", f"{dst}/undertale", dirs_exist_ok=True
    )
    os.chdir(f"{dst}/undertale")


class AssemblageVcpkgReader(BaseReader):
    type = "📖 - READER"
    name = "A - AssemblageVcpkg"

    def __init__(self):
        self.raw_data_dir = "~/undertale_shared/datasets/raw/assemblage"
        logger.debug("In constructor")

    def run(self, data, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        sqlfile = "../vcpkg.sqlite"
        bins_dir = f"{self.raw_data_dir}/vcpkg"

        functions = {}
        with sqlite3.connect(sqlfile) as db:
            logger.info("Connected to SQL database")
            cur = db.cursor()

            i = 0
            current_binary = None
            current_binary_id = None
            num_failed_rvas = 0

            for row in cur.execute(
                "SELECT f.id, b.id, b.file_name, b.repo_last_update, b.github_url, b.platform, b.build_mode, b.optimization, b.toolset_version, b.path, f.name, r.start, r.end  FROM functions f INNER JOIN binaries b ON f.binary_id = b.id INNER JOIN rvas r ON f.id = r.function_id ORDER BY b.id;"
            ):
                (
                    f_id,
                    b_id,
                    b_file_name,
                    b_repo_last_update,
                    b_github_url,
                    b_platform,
                    b_build_mode,
                    b_optimization,
                    b_toolset_version,
                    b_path,
                    f_name,
                    r_start,
                    r_end,
                ) = row

                if i > 0 and (i % 100000) == 0:
                    logger.info(
                        f"Progress... {i} functions so far, {float(num_failed_rvas)/i:.7f} with bad rvas"
                    )

                if i > 1000000:
                    break

                if (current_binary_id is None) or (current_binary_id != b_id):
                    current_binary_id = b_id
                    current_binary = pefile.PE(os.path.join(bins_dir, b_path))

                try:
                    raw_data = current_binary.get_data(
                        rva=r_start, length=r_end - r_start
                    )
                except:
                    logger.warn(
                        f"discarding function {f_id} as rvas for it seem not to make sense with binary: {num_failed_rvas} count"
                    )
                    num_failed_rvas += 1
                    continue

                tup = (
                    (r_start, r_end),
                    raw_data,
                    b_platform,
                    b_build_mode,
                    b_optimization,
                    b_toolset_version,
                    f_name,
                    b_file_name,
                    b_path,
                    b_repo_last_update,
                    b_github_url,
                    b_id,
                )
                if f_id not in functions:
                    functions[f_id] = []
                functions[f_id].append(tup)
                i += 1

        logger.info("Done with sql shenanigans")
        tick()

        i = 0
        last_build_mode = None
        last_optimization = None
        last_compiler = None
        last_fun_name = None
        for f_id, arr in functions.items():
            i += 1
            if (i % 10000) == 0:
                logger.info(f"{i} of {len(functions)} items processed")
                tick()

            arr.sort(key=lambda tup: tup[0][0])
            (min_a, max_a) = (0xFFFFFFFFFFFFFFFF, 0)
            for (
                rng,
                code,
                platform,
                build_mode,
                optimization,
                compiler,
                fun_name,
                bin_filename,
                bin_path,
                bin_repo_last_update,
                bin_github_url,
                b_id,
            ) in arr:
                min_a = min(rng[0], min_a)
                max_a = max(rng[1], max_a)
            all_code = bytearray(b"\x90" * (max_a - min_a + 1))
            last_platform = None

            for (
                rng,
                code,
                platform,
                build_mode,
                optimization,
                compiler,
                fun_name,
                bin_filename,
                bin_path,
                bin_repo_last_update,
                bin_github_url,
                b_id,
            ) in arr:
                if not (last_platform is None):
                    assert platform == last_platform
                    assert build_mode == last_build_mode
                    assert optimization == last_optimization
                    assert compiler == last_compiler
                    assert fun_name == last_fun_name
                all_code[rng[0] - min_a : rng[1] - min_a + 1] = code

                (
                    _,
                    last_platform,
                    last_build_mode,
                    last_optimization,
                    last_compiler,
                    last_fun_name,
                ) = (rng, platform, build_mode, optimization, compiler, fun_name)

            equiv_class = (
                f"{bin_github_url}-{bin_repo_last_update}-{bin_filename}-{fun_name}"
            )

            yield Document(
                id=f"fid={f_id}",
                text=all_code,
                metadata={
                    "binary": all_code,
                    "architecture": platform,
                    "function_name": fun_name,
                    "equiv_class": equiv_class,
                    "optimization": optimization,
                    "compiler": compiler,
                },
            )

        tick()


class AssemblageVcpkg(Dataset):
    name = "assemblage-vcpkg-dt"

    def get_pipeline(self, input, writer, parallelism):

        if input == "binaries":
            executor = self.get_executor(0, input)
            executor.pipeline.append(writer)
            return executor
        elif input == "r2":
            executor = self.get_executor(0, input)
            executor.depends.pipeline.append(
                ParquetWriter(
                    output_folder=f"{DEFAULT_DATASETS_DIRECTORY}assemblage-vcpkg-dt-disassembled",
                    adapter=lambda self, doc: doc.metadata,
                    max_file_size=100 * 1024 * 1024,
                )
            )
            executor.pipeline.append(writer)
            return executor

        return None

    def get_executor(self, steps, input):
        # Stage 0: Parse function bytes and metadata
        slurm_parse = SlurmPipelineExecutor(
            pipeline=[
                create_directory,
                AssemblageVcpkgReader(),
            ],
            venv_path=os.path.join(f"{Path.home()}/.conda/envs", "ut"),
            logging_dir="~/undertale/logs",
            time="12:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=4,
            tasks=1,
            job_name="parse_vcpkg",
            partition="xeon-p8",
            sbatch_args={"distribution": "cyclic:cyclic"},
        )

        # Stage 1: Disassemble binaries in parallel
        slurm_disassemble = SlurmPipelineExecutor(
            depends=slurm_parse,
            pipeline=[
                create_directory,
                ParquetReader(
                    data_folder=f"{DEFAULT_DATASETS_DIRECTORY}assemblage-vcpkg-dt",
                    adapter=lambda self, data, path, id_in_file: {
                        "id": id_in_file,
                        "text": data["binary"],
                        "metadata": data,
                    },
                ),
                RadareDisassembler(),
            ],
            venv_path=os.path.join(f"{Path.home()}/.conda/envs", "ut"),
            logging_dir="~/undertale/logs",
            time="12:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=4,
            tasks=64,
            job_name="vcpkg_disassemble_r2",
            partition="xeon-p8",
            sbatch_args={"distribution": "cyclic:cyclic"},
        )

        if input == "binaries":
            return slurm_parse
        elif input == "r2":
            return slurm_disassemble
        return None


if __name__ == "__main__":
    main(AssemblageVcpkg)
