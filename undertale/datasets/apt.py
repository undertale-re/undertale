import logging
import os
import shutil
import tempfile

import datasets
import requests
from bs4 import BeautifulSoup

from . import dataset
from .transforms.disassemble import ghidra as ghidra_disassemble
from .transforms.segment import ghidra as ghidra_segment

logger = logging.getLogger(__name__)

HEADERS = [b"\x7fELF", b"MZ\x90\x00"]  # ELF (Linux) or MZ (Windows)


def check_executable(file_path):
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
                            print(f"cant copy {path} to {dest}/{orig_filename}/{file}")
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


class APT(dataset.Dataset):
    description = "A collection of binaries for Debian based linux distributions downloaded by the APT tool "
    path = "apt_dataset_v3"
    base_url = "http://cybersecmirrors.llan.ll.mit.edu/mirrors/ubuntu/pool/universe/"

    @classmethod
    def generate_url_list(cls, list_loc: str):
        with open(list_loc, "w+") as f:
            f.write("\n")
        all_download_links = []
        response = requests.get(cls.base_url)
        soup = BeautifulSoup(response.content, features="html.parser")
        links = [link["href"] for link in soup.find_all("a") if len(link["href"]) == 2]
        for letter in links:
            print(f"Now processing packages starting with: {letter}")
            letter_url = cls.base_url + letter
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

    @classmethod
    def create_dataset(cls, url_path: str, downloaded_data_path: str):
        if not os.path.isdir(downloaded_data_path):
            os.mkdir(downloaded_data_path)
            download = True
        else:
            download = False
        # downloading the raw packages
        data = {"code": [], "filename": [], "metadata": [], "package": []}
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

        for i, pname in enumerate(files):
            project = os.path.join(unpackaged_path, pname)
            with open(os.path.join(project, "metadata.txt")) as f:
                metadata = f.read()
            with open(os.path.join(project, "project_name.txt")) as f:
                pname = f.read()
            for fname in os.listdir(project):
                floc = os.path.join(project, fname)
                if check_executable(floc):
                    with open(floc, "rb") as f:
                        data["code"].append(f.read())
                    data["filename"].append(fname)
                    data["metadata"].append(metadata)
                    data["package"].append(project)

        ds = datasets.Dataset.from_dict(data)
        return ds

    @classmethod
    def parse(cls, path: str, processes=None):
        if path == "download":
            logger.info(f"downloading {cls.__name__} from APT")
            list_loc = "../datasets/apt_url_list.txt"
            if not os.path.isfile(list_loc):
                cls.generate_url_list(list_loc)
            downloaded_data_path = "../datasets/apt_downloaded/"
            dataset = cls.create_dataset(list_loc, downloaded_data_path)
        else:
            dataset = datasets.load_from_disk(path)

        dataset.__class__ = cls

        return dataset


class APTSegmented(APT):
    path = "apt-segmented"

    transforms = [
        ghidra_segment.GhidraFunctionSegment(),
    ]


class APTSegmentedDisassembled(APT):
    path = "apt-segmented-disassembled"

    transforms = [
        ghidra_segment.GhidraFunctionSegment(),
        ghidra_disassemble.GhidraDisassemble(),
    ]


if __name__ == "__main__":
    dataset.main([APTSegmentedDisassembled, APTSegmented, APT])
