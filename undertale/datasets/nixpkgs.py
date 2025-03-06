import logging
import os
import typing

import datasets

from . import dataset, schema

logger = logging.getLogger(__name__)


class NixPkgsBinaries(dataset.Dataset):
    description = "repository of compiled open source projects with debugging symbols from the Nix packages project"
    notes = "currently just processing already-provided files"
    schema = schema.WholeBinary

    @classmethod
    def parse(cls, path: str, processes: typing.Optional[int] = None):
        base_path = (
            f"/home/gridsan/groups/undertale_shared/binaries/{cls.path}/nix/store"
        )

        def gen_from_dir():
            for root, dirs, files in os.walk(base_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(file_path, start=base_path)
                    package = rel_path.split(os.path.sep)[0]
                    contents = ""
                    logging.info(f"processing {rel_path}...")
                    try:
                        with open(file_path, "rb") as file_obj:
                            contents = file_obj.read()
                    except FileNotFoundError:
                        logger.error(f"cannot open {file_path}")
                        continue
                    yield {
                        "filename": file_name,
                        "binary": contents,
                        "package": package,
                        "path": rel_path,
                    }

        ds = datasets.Dataset.from_generator(gen_from_dir)
        ds.__class__ = cls
        return ds


class NixPkgs2405Binaries(NixPkgsBinaries):
    version = "24.05"
    url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.05.tar.gz"
    path = "nixpkgs-2405"


if __name__ == "__main__":
    dataset.main([NixPkgs2405Binaries])
