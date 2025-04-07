"""

This is a big dataset. In order to create it, you will have to arrange
for enough disk space.  Also, note that the code for parsing was
developed and run on a machine with 512GB RAM.  Takes about 1.5 hours
to parse on that machine, FYI.

Disk space requirements.

1. About 20GB for the two zip files that come from the URL above:
   winpe_licensed.zip and winpe_licensed.sqlite.zip. These should be
   in the directory pointed to by `path` argument.

2. These uncompress into about 102 GB in same directory.

3. Hugging face needs to do mysterious big things in your ~/.cache
   directory. If that's very big (I have no idea how much space it
   needs) then fine.  Otherwise, set the HF_HOME env variable to
   somewhere very large and HF will use that space instead of ~/.cache.

4. The default output dir (unless you set the
   UNDERTALE_DATASETS_DIRECTORY environment variable) is
   ~/undertale_shared.  The dataset generated there, in the form of
   arrow files, will be about 102GB.  So make sure there is enough
   space there.


"""
from base64 import b64encode
import os
import pefile
import sqlite3 as sqlite
import time

from datatrove.pipeline.readers.base import BaseReader
from datatrove.data import Document


from .base import Dataset, main
from .pipeline.compilers import CppCompiler
from .pipeline.disassemblers import RadareDisassembler, GhidraDisassembler


import logging
#import os
#import sqlite3 as sqlite
#import time
#import typing

from typing import Callable
#import datasets

#from . import dataset

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
    logger.info(f"tick={delta} el={f_delta}")


class AssemblageWindowsReader(BaseReader):

    type = "📖 - READER"
    name = "A - Assemblage"
    def __init__(self):
        self.raw_data_dir = "/data/tleek/undertale_raw_data/assemblage"
        
    def run(self, data, rank: int = 0, world_size: int = 1):
        
        logger.warning("DANGER this dataset may contain malware so be careful with it")

        bins_zip = "winpe_licensed.zip"
        sql_zip = "winpe_licensed.sqlite.zip"

        sqlfile = f"{self.raw_data_dir}/winpe.sqlite"
        bins_dir = f"{self.raw_data_dir}/winpe"

        if (not os.path.exists(sqlfile)) or (os.path.getsize(sqlfile) == 0):
            os.system(f"/usr/bin/unzip -o -q -d {self.raw_data_dir} {self.raw_data_dir}/{sql_zip}")
        if not os.path.exists(bins_dir):
            os.system(f"/usr/bin/unzip -o -q -d {self.raw_data_dir} {self.raw_data_dir}/{bins_zip}")

        logger.info("Unzipping complete")
        tick()

        functions = {}
        with sqlite.connect(sqlfile) as db:
            cur = db.cursor()

            i = 0
            current_binary = None
            current_binary_id = None
            num_with_source = 0
            num_failed_rvas = 0

            for row in cur.execute(
                "SELECT f.id, b.id, b.file_name, b.repo_last_update, b.github_url, b.platform, b.build_mode, b.optimization, b.toolset_version, b.path, f.name, f.source_codes, r.start, r.end  FROM functions f INNER JOIN binaries b ON f.binary_id = b.id INNER JOIN rvas r ON f.id = r.function_id ORDER BY b.id;"
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
                    f_source,
                    r_start,
                    r_end,
                ) = row

                if not (b_platform == "x64"):
                    continue
                
                if i > 0 and (i % 100000) == 0:
                    logger.info(
                        f"Progress... {i} functions so far, {float(num_with_source)/i:.4f} have source, {float(num_failed_rvas)/i:.7f} with bad rvas"
                    )

                if i > 1000000:
                    break
 
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

                tup = ((r_start, r_end), raw_data, f_source, b_platform, b_build_mode, b_optimization, b_toolset_version, f_name, b_file_name, b_path, b_repo_last_update, b_github_url, b_id)
                if f_id not in functions:
                    functions[f_id] = []
                # this could be just one of several binary chunks that
                # correspond to this function so we accumlate them in
                # a list here, associated with the function id
                functions[f_id].append(tup)
                i += 1

        logger.info("Done with sql shenanigans")
        tick()
            
        i = 0
        l = len(functions)
        for f_id, arr in functions.items():
            i +=1
            if (i % 10000) == 0:
                logger.info(f"{i} of {l} items processed")
                tick()

            all_source = ""
            arr.sort(key=lambda tup: tup[0][0])                            
            (min_a, max_a) = (0xffffffffffffffff, 0)
            for (rng, code, source, platform, build_mode, optimization, compiler, fun_name, bin_filename, bin_path, bin_repo_last_update, bin_github_url, b_id) in arr:
                min_a = min(rng[0], min_a)
                max_a = max(rng[1], max_a)
            all_code = bytearray(b'\x090' * (max_a - min_a + 1))
            last_platform = None
            all_source = ""
            for (rng, code, source, platform, build_mode, optimization, compiler, fun_name, bin_filename, bin_path, bin_repo_last_update, bin_github_url, b_id) in arr:
                if not (last_platform is None):
                    assert(platform == last_platform)
                    assert(build_mode == last_build_mode)
                    assert(optimization == last_optimization)
                    assert(compiler == last_compiler)
                    assert(fun_name == last_fun_name)
                all_code[rng[0]-min_a:rng[1]-min_a+1] = code                    
                if source is not None:
                    all_source += source
                (last_rng, last_platform, last_build_mode, last_optimization, last_compiler, last_fun_name) = (rng, platform, build_mode, optimization, compiler, fun_name)
            if  all_source == "":
                all_source = None

            # if two functions have same value for this then they differ only by 
            # compiler and optimization, which means they are equivalent; this
            # is used by the transform PairwiseContrastive
            equiv_class = f"{bin_github_url}-{bin_repo_last_update}-{bin_filename}-{fun_name}"

            # this is the binary code for a single function
            yield Document(
                id = f"fid={f_id}", 
                text = all_code, #b64encode(all_code).decode("utf-8"),
                metadata = {
                    "binary": all_code,
                    "architecture": platform,
                    "function_name": fun_name,
                    "equiv_class": equiv_class, 
                    "optimization": optimization,
                    "compiler": compiler
                }
            )

        tick()

        

class AssemblageWindowsPublicDataset(Dataset):
    name = "assemblage-windows-public-dataset"

    def get_pipeline(self, input, writer, parallelism):

        steps = [
            AssemblageWindowsReader(),
            RadareDisassembler(),
        ]
        steps.extend(writer)

        import copy

        # Note: fails here with `TypeError: cannot pickle '_thread.lock' object`
        sc = copy.deepcopy(steps)
        
        return self.get_executor(steps, tasks=parallelism)

if __name__ == "__main__":

    #import pdb
    #pdb.set_trace()
    main(AssemblageWindowsPublicDataset)


