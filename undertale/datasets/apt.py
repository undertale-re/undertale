import logging
import os

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from undertale.datasets.base import DEFAULT_DATASETS_DIRECTORY, Dataset, main
from undertale.datasets.pipeline.disassemblers.ghidra import GhidraDisassembler

logger = logging.getLogger(__name__)


BASE_URL = "http://cybersecmirrors.llan.ll.mit.edu/mirrors/ubuntu/pool/universe/"


def adapt_apt_from_dict(data: dict) -> dict:
    return {
        "id": data["filename"],
        "text": data["code"][0],
        "metadata": {"metadata": data["metadata"], "package": data["package"]},
    }


class LoadAPTPackages(PipelineStep):
    _requires_dependencies = [
        "os",
        "shutil",
        "tempfile",
        "requests",
        "bs4",
        "datasets",
        "datatrove",
    ]

    name = "(Down)load Packages"
    type = "📖 - READER"

    def __init__(
        self,
        build_options: list[str] = ["debug"],
        wrapper_args: list[str] | None = None,
        base_url: str = BASE_URL,
        url_list_loc: str = "./apt_url_list.txt",
        downloaded_data_dir: str = "./apt_downloaded",
    ):
        from datatrove.utils.logging import logger

        super().__init__()
        self.wrapper_args = wrapper_args
        self.build_options = build_options
        self.base_url = base_url
        self.adapter = adapt_apt_from_dict
        self.url_list_loc = url_list_loc
        self.downloaded_data_dir = downloaded_data_dir
        logger.debug("In constructor")

    def run(self, data=None, rank=0, world_size=0):
        import os
        import shutil
        import tempfile

        import requests
        from bs4 import BeautifulSoup
        from datatrove.data import Document

        def check_executable(file_path):
            HEADERS = [b"\x7fELF", b"MZ\x90\x00"]  # ELF (Linux) or MZ (Windows)
            if os.access(file_path, os.X_OK):
                try:
                    with open(file_path, "rb") as f:
                        # Check for the presence of common binary file headers
                        header = f.read(4)
                    if header in HEADERS:
                        return True
                except:
                    pass
            return False

        def unpack_deb(orig_file, dest):
            with tempfile.TemporaryDirectory() as tmp_loc:
                command = f"dpkg-deb -R {orig_file} {tmp_loc}"
                status = os.system(command)
                orig_filename = "X"
                if status == 0:
                    success = False
                    for root, dirs, files in os.walk(tmp_loc):
                        for file in files:
                            path = os.path.join(root, file)
                            if check_executable(path):
                                try:
                                    orig_filename = ".".join(
                                        os.path.split(orig_file)[-1].split(".")[:-1]
                                    )
                                    if not os.path.isdir(f"{dest}/{orig_filename}"):
                                        os.mkdir(f"{dest}/{orig_filename}")
                                    shutil.copy(path, f"{dest}/{orig_filename}/{file}")
                                    success = True
                                except PermissionError:
                                    print(
                                        f"cant copy {path} to {dest}/{orig_filename}/{file}"
                                    )
                    os.remove(orig_file)
                    if success:
                        metadata_loc = f"{tmp_loc}/DEBIAN/control"
                        if not os.path.isfile(metadata_loc):
                            raise ValueError
                        with open(f"{tmp_loc}/DEBIAN/control") as f:
                            metadata = f.read()
                        if metadata == "":
                            metadata = "~"
                        return metadata, orig_filename
                    else:
                        return "", orig_filename
            return "", orig_filename

        def generate_url_list(list_loc: str, base_url: str):
            with open(list_loc, "w+") as f:
                f.write("\n")
            all_download_links = []
            response = requests.get(base_url)
            soup = BeautifulSoup(response.content, features="html.parser")
            links = [
                link["href"] for link in soup.find_all("a") if len(link["href"]) == 2
            ]
            for letter in links:
                print(f"Now processing packages starting with: {letter}")
                letter_url = base_url + letter
                response = requests.get(letter_url)
                letter_soup = BeautifulSoup(response.content, features="html.parser")
                letter_links = [link["href"] for link in letter_soup.find_all("a")]
                check = 6
                for letter_link in letter_links[check:]:
                    final_url = letter_url + letter_link
                    response = requests.get(final_url)
                    final_soup = BeautifulSoup(response.content, features="html.parser")
                    final_links = [link["href"] for link in final_soup.find_all("a")]
                    if len(final_links) > check:
                        download_link = final_url + final_links[check]
                        all_download_links.append(download_link)
                        with open(list_loc, "a") as f:
                            f.write(download_link + "\n")

        def create_dataset(
            url_path: str, downloaded_data_path: str, max_file_size: int = int(1e6)
        ):
            if not os.path.isdir(downloaded_data_path):
                os.mkdir(downloaded_data_path)
                download = True
            else:
                download = False
            # downloading the raw packages
            data = {"code": [], "filename": [], "metadata": [], "package": []}
            # data = []
            with open(url_path) as f:
                all_download_links = [
                    link.strip() for link in f.readlines()[1:] if link[-5:-1] == ".deb"
                ]
            unpackaged_path = os.path.join(downloaded_data_path, "unpackaged")
            raw_data_path = os.path.join(downloaded_data_path, "raw")
            if download:
                os.mkdir(unpackaged_path)
                os.mkdir(raw_data_path)
                for download_link in all_download_links:
                    print(f"now downloading {download_link}")
                    floc = os.path.join(raw_data_path, download_link.split("/")[-1])
                    response = requests.get(download_link)
                    with open(floc, "wb+") as f:
                        f.write(response.content)
                    # unpackaging and finding binary executables
                    metadata, pname = unpack_deb(floc, unpackaged_path)
                    project = os.path.join(unpackaged_path, pname)
                    if metadata != "":
                        with open(os.path.join(project, "metadata.txt"), "w+") as f:
                            f.write(metadata)
                        with open(os.path.join(project, "project_name.txt"), "w+") as f:
                            f.write(pname)

            files = os.listdir(unpackaged_path)
            data = []
            for i, pname in enumerate(files):
                project = os.path.join(unpackaged_path, pname)
                with open(os.path.join(project, "metadata.txt")) as f:
                    metadata = f.read()
                with open(os.path.join(project, "project_name.txt")) as f:
                    pname = f.read()
                for fname in os.listdir(project):
                    floc = os.path.join(project, fname)
                    if check_executable(floc):
                        document = {}
                        with open(floc, "rb") as f:
                            code = f.read()
                        if len(code) < 1e6:
                            document["code"] = code
                            document["filename"] = fname
                            document["metadata"] = metadata
                            document["package"] = project
                            data.append(document)
            return data

        if not os.path.isfile(self.url_list_loc):
            generate_url_list(self.url_list_loc, self.base_url)
        ds = create_dataset(self.url_list_loc, self.downloaded_data_dir)

        for row in ds:
            f_id = row["filename"]
            yield Document(
                id=f"fid={f_id}",
                text=row["code"],
                metadata={
                    "binary": row["code"],
                    "text": row["code"],
                    "metadata": {"value": row["metadata"]},
                    "package": row["package"],
                },
            )


class APTpkg(Dataset):
    name = "apt-pkg"
    DEFAULT_DATASETS_DIRECTORY = DEFAULT_DATASETS_DIRECTORY

    def get_pipeline(self, input, writer, parallelism):
        from datatrove.utils.logging import logger

        # def adapt_apt_from_dict(data: dict) -> dict:
        #     return {
        #         "id": data["filename"],
        #         "text": data["code"][0],
        #         "metadata": {
        #             "metadata": {"value": data["metadata"]},
        #             "package": data["package"],
        #         },
        #     }

        if input == "binaries":
            executor = self.get_my_executor(input)
            executor.pipeline.append(
                ParquetWriter(
                    output_folder=f"{self.DEFAULT_DATASETS_DIRECTORY}apt-pkg",
                    adapter=lambda self, doc: doc.metadata,
                    max_file_size=50 * 1024 * 1024,
                )
            )
            logger.info("get_pipeline binaries")
            return executor
        elif input == "r2":
            executor = self.get_my_executor(input)
            executor.pipeline.append(
                ParquetWriter(
                    output_folder=f"{self.DEFAULT_DATASETS_DIRECTORY}apt-pkg-disassembled",
                    adapter=lambda self, doc: doc.metadata,
                    max_file_size=50 * 1024 * 1024,
                )
            )
            return executor

        return None

    def get_my_executor(self, input, ghidra_install_dir, venv_path, partition="RTX-24"):
        # Stage 0: Parse function bytes and metadata
        from datatrove.utils.logging import logger

        os.environ["GHIDRA_INSTALL_DIR"] = ghidra_install_dir

        logger.info("Hello world get_my_executor")
        # parse = LocalPipelineExecutor(
        #     pipeline=[
        #         LoadAPTPackages(),
        #     ],
        # )
        # disassemble = LocalPipelineExecutor(
        #     depends=parse,
        #     pipeline=[
        #         LoadAPTPackages(),
        #         GhidraDisassembler(),
        #     ],
        # )
        slurm_parse = SlurmPipelineExecutor(
            pipeline=[
                LoadAPTPackages(),
            ],
            venv_path=venv_path,
            logging_dir="~/undertale/logs",
            time="48:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=40,
            tasks=10,
            job_name="parse_aptpkg",
            partition=partition,
            sbatch_args={
                "distribution": "cyclic:cyclic",
                "chdir": "~/",
            },
        )

        # Stage 1: Disassemble binaries in parallel
        slurm_disassemble = SlurmPipelineExecutor(
            depends=slurm_parse,
            pipeline=[
                ParquetReader(f"{self.DEFAULT_DATASETS_DIRECTORY}apt-pkg"),
                # LoadAPTPackages(),
                GhidraDisassembler(),
            ],
            venv_path=venv_path,
            logging_dir="~/undertale/logs",
            time="48:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=40,
            tasks=10,
            job_name="disassemble_aptpkg",
            partition=partition,
            sbatch_args={
                "distribution": "cyclic:cyclic",
                "chdir": "~/",
            },
        )

        if input == "binaries":
            # return parse
            return slurm_parse
        elif input == "r2":
            # return disassemble
            return slurm_disassemble
        return None


if __name__ == "__main__":
    os.environ["GHIDRA_INSTALL_DIR"] = ""  # fill in with ghidra install directory
    main(APTpkg)
