import os

from datatrove.data import DocumentsPipeline
from datatrove.executor import SlurmPipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter

from undertale.datasets.transforms.segment.lief import SegmentLief


class FindNixpkgs(PipelineStep):
    name = "🔍 Find Packages"
    type = "⚙️ - PROCESS"

    def __init__(
        self,
        flakes: list[str],
        build_options: list[str] = ["debug"],
        broken_pkgs: list[str] = [],
        wrapper_args: list[str] | None = None,
    ):
        super().__init__()
        self.flakes = flakes
        self.wrapper_args = wrapper_args
        self.build_options = build_options
        pkgs_string = " ".join(f'"{x}"' for x in broken_pkgs)
        self.nix_script = f"""
      let
        pkgs = import <nixpkgs> {{
          system = "x86_64-linux";
          config.allowBroken = false;
          config.allowUnsupportedSystem = false;
          config.allowUnfree = true;
          config.allowInsecurePredicate = (x: true);
        }};
        blacklist = [ {pkgs_string} ];
        lib = pkgs.lib;

        # Our heuristic:
        isC = drv:
          lib.isDerivation drv && (
            (
              builtins.elem pkgs.gcc ((drv.buildInputs or [])
                  ++ (drv.nativeBuildInputs or [])
                  ++ (drv.depsBuildBuild or []))
            ) || (
              builtins.elem pkgs.stdenv.cc ((drv.buildInputs or [])
                  ++ (drv.nativeBuildInputs or [])
                  ++ (drv.depsBuildBuild or []))
            )
          );

        topLevelNames = lib.lists.filter
            (x: !(builtins.elem x blacklist))
            (builtins.attrNames pkgs);

        mapFunc = name: value: {{
            name = "${{name}}";
            urls = let src=value.src or {{}};
            urls=src.urls or []; in urls; }};

        validNames =
          builtins.map (name:
            let
              tname = builtins.trace "Evaluating: ${{name}}" name;
              attempt = builtins.tryEval (pkgs.${{tname}});
            in
              if attempt.success then
                let attemptIsC = builtins.tryEval (mapFunc name attempt.value);
                in if attemptIsC.success then attemptIsC.value else {{}}
              else
                lib.warn "tryEval failed on ${{name}}" false
          ) topLevelNames;

      in
        validNames
    """

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import json
        import os
        import subprocess
        import tempfile
        import traceback

        from datatrove.data import Document
        from datatrove.utils.logging import logger

        for flake_id, flake in enumerate(self.flakes):
            with self.track_time():
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", prefix="findpkgs-", delete=False
                    ) as nixfile:
                        nixfile.write(self.nix_script)
                        nixfile.close()
                        original_args = [
                            "nix",
                            "eval",
                            "--impure",
                            "--json",
                            "--show-trace",
                            "--override-flake",
                            "nixpkgs",
                            flake,
                            "--file",
                            nixfile.name,
                        ]
                        if self.wrapper_args:
                            args = self.wrapper_args.copy()
                            args.append(" ".join(original_args))
                        else:
                            args = original_args
                        result = subprocess.run(
                            args=args, capture_output=True, text=True
                        )
                        os.unlink(nixfile.name)
                    if result.returncode != 0:
                        logger.error(
                            """failed return code: {result.returncode}
                                {result.stdout}
                                {result.stderr}""",
                            result=result,
                        )
                        packages = []
                    else:
                        packages = json.loads(result.stdout)
                        # logger.debug("stderr:\n{}",result.stderr)
                except json.decoder.JSONDecodeError as err:
                    logger.error(
                        """error parsing nix findpkgs results: {err}
                            {result.returncode}
                            {result.stdout}
                            {result.stderr}""",
                        result=result,
                        err=err,
                    )
                    packages = []
                except IOError:
                    logger.error(
                        "IO error processiong nix findpkgs results: {err}\n",
                        err=traceback.format_exc(),
                    )
                    packages = []
                for i, pkg in enumerate(packages):
                    if pkg:
                        yield Document(
                            id=f"pkg-{flake_id}-{i}",
                            text=pkg["name"],
                            metadata={
                                "flake": flake,
                                "urls": pkg["urls"],
                                "build_options": self.build_options,
                            },
                        )


class EnrichGithubPackages(PipelineStep):
    name = "🐙 Enrich Github Packages"
    type = "⚙️ - PROCESS"

    def __init__(
        self,
        github_token: str,
        github_cache_dir: DataFolderLike | None = None,
        proxies: dict[str, str] | None = None,
        wrapper_args: list[str] | None = None,
    ):
        import json

        from datatrove.io import get_datafolder
        from datatrove.utils.logging import logger

        super().__init__()
        self.wrapper_args = wrapper_args
        self.proxies = proxies
        self.cache = {}
        self.cache_folder = None
        if github_cache_dir:
            self.cache_folder = get_datafolder(github_cache_dir)
            for f in self.cache_folder.list_files(glob_pattern="*.json"):
                try:
                    self.cache.update(json.load(self.cache_folder.open(f, "r")))
                except Exception as e:
                    logger.warning("error loading github cache: {}", str(e))

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import json
        import time

        import requests
        from datatrove.utils.logging import get_random_str, logger

        if data:
            newcache = {}
            for doc in data:
                language = None
                urls = doc.metadata.get("urls", [])
                for url in urls:
                    if "github.com" in url and "/repos/" not in url:
                        repo_owner, repo_name = url.split("/")[3:5]
                        api_url = (
                            f"https://api.github.com/repos/{repo_owner}/{repo_name}"
                        )
                        if api_url in self.cache:
                            language = self.cache[api_url]
                        else:
                            try:
                                logger.info("github cache miss: {}", api_url)
                                headers = {"Authorization": f"token {github_token}"}
                                time.sleep(0.8)
                                response = requests.get(
                                    api_url, headers=headers, proxies=self.proxies
                                )
                                if response.status_code == 200:
                                    repo_data = response.json()
                                    language = repo_data.get("language")
                                    newcache[api_url] = language
                                    self.cache[api_url] = language
                                    self.stat_update("github_cache_miss")
                                    break
                                else:
                                    logger.info(
                                        "github error code {} on {}",
                                        response.status_code,
                                        api_url,
                                    )
                                    newcache[api_url] = None
                                    self.cache[api_url] = None
                                    self.stat_update("github_errors")
                            except Exception as e:
                                logger.warning(
                                    "error getting github entry for {}: {}",
                                    api_url,
                                    str(e),
                                )
                                self.stat_update("github_errors")
                doc.metadata["language"] = language
                self.stat_update(f"language.{language}")
                yield doc
            if len(newcache) > 0 and self.cache_folder:
                randomstr = get_random_str()
                fname = f"github_cache-{randomstr}.json"
                logger.info(
                    "saving {} new github cache entries to {}", len(newcache), fname
                )
                with self.cache_folder.open(fname, mode="w") as of:
                    json.dump(newcache, of)


class BuildNixpkgs(PipelineStep):
    name = "🛠  Build Nix Packages"
    type = "⚙️ - PROCESS"

    def __init__(
        self,
        binaries_dir: DataFolderLike,
        working_dir: DataFolderLike | None = None,
        wrapper_args: list[str] | None = None,
        build_timeout: int = 2 * 60 * 60,
    ):
        from datatrove.io import get_datafolder

        super().__init__()
        self.wrapper_args = wrapper_args
        self.working_folder = get_datafolder(working_dir) if working_dir else None
        self.binaries_folder = get_datafolder(binaries_dir)
        self.build_timeout = build_timeout

    def is_binary(self, fpath):
        try:
            with self.working_folder.open(fpath, mode="rb") as fobj:
                header = fobj.read(16)
        except Exception:
            return False
        return header.startswith(b"\x7fELF") or header.startswith(b"!<arch>\n")

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import subprocess
        import tarfile
        import tempfile

        from datatrove.io import DataFolder
        from datatrove.utils.logging import logger

        if not self.working_folder:
            self.working_folder = DataFolder(
                tempfile.TemporaryDirectory(
                    prefix="build-nixpkgs-", delete=False, ignore_cleanup_errors=True
                )
            )
        self.working_folder.makedirs("flake", exist_ok=True)
        flake_folder = DataFolder(self.working_folder.resolve_paths("flake"))
        flake_nix = flake_folder.open("flake.nix", mode="w")
        flake_nix.write(
            """{
        description = "nixpkgs with debugging overlay";
        inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        outputs = { self, nixpkgs }:
          let
            pkgs = import nixpkgs {
              overlays = [ (final: prev: { enableDebugging = true; }) ];
              system = "x86_64-linux";
              config.allowBroken = false;
              config.allowUnsupportedSystem = false;
              config.allowUnfree = true;
              config.allowInsecurePredicate = (x: true);
            };
          in {
            packages.x86_64-linux = pkgs;
          };
      }"""
        )
        flake_nix.close()
        # logger.debug("flake.nix created at {}", flake_nix.name)

        self.binaries_folder.makedirs("", exist_ok=True)
        binary_file_path = self.binaries_folder.resolve_paths(f"binaries-{rank}.tgz")
        tar_file = tarfile.open(name=binary_file_path, mode="w:gz")
        if data:
            for doc in data:
                with self.track_time():
                    # logger.debug("building package {} (flake:{})", doc.text, doc.metadata['flake'])
                    doc.metadata["build_rank"] = rank
                    doc.metadata["build_archive"] = binary_file_path

                    try:
                        original_args = [
                            "nix",
                            "build",
                            "--log-format",
                            "raw",
                            "--print-out-paths",
                            "--show-trace",
                            "--no-link",
                            "--no-write-lock-file",
                            "--store",
                            self.working_folder.path,
                            "--override-input",
                            "nixpkgs",
                            doc.metadata["flake"],
                            f".#{doc.text}",
                        ]
                        if self.wrapper_args:
                            args = self.wrapper_args.copy()
                            args.append(" ".join(original_args))
                        else:
                            args = original_args
                        # logger.debug("running: {}", args)
                        result = subprocess.run(
                            args=args,
                            cwd=flake_folder.path,
                            timeout=self.build_timeout,
                            capture_output=True,
                            text=True,
                        )
                        doc.metadata["build_retval"] = result.returncode
                        doc.metadata["build_stdout"] = result.stdout
                        doc.metadata["build_stderr"] = result.stderr
                        if result.returncode != 0:
                            self.stat_update("build_failure")
                            logger.error(
                                "failed return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
                                result=result,
                            )
                            doc.metadata["build_success"] = False
                            # TODO diagnose failure class from stderr and emit stats
                            # TODO capture nix log references and add to doc
                            yield doc
                            continue
                        else:
                            build_name = result.stdout.removeprefix(
                                "/nix/store/"
                            ).strip()
                            doc.metadata["build_name"] = build_name
                            if build_name == "":
                                self.stat_update("build_failure")
                                logger.warning("missing build name: {}", result.stderr)
                                doc.metadata["build_success"] = False
                                # TODO diagnose failure class from stderr and emit stats
                                # TODO capture nix log references and add to doc
                                yield doc
                                continue
                            # copy binary artifacts out
                            doc.metadata["build_success"] = True
                            self.stat_update("build_success")
                            binaries = []
                            for src_path in self.working_folder.find(
                                f"nix/store/{build_name}"
                            ):
                                if self.is_binary(src_path):
                                    short_path = src_path.removeprefix("nix/store/")
                                    binaries.append(short_path)
                                    try:
                                        tar_file.add(
                                            self.working_folder.resolve_paths(src_path),
                                            arcname=short_path,
                                        )
                                        self.stat_update("binaries")
                                        self.stat_update(
                                            "binaries_size",
                                            value=self.working_folder.stat(src_path)[
                                                "size"
                                            ],
                                        )
                                    except Exception as e:
                                        logger.error(
                                            "tar add failure: {} {}", src_path, str(e)
                                        )
                                        doc.metadata["build_success"] = False
                                        self.stat_update("build_failure")
                                        yield doc
                            doc.metadata["binaries"] = binaries
                            # logger.debug("saving binaries: {}", binaries)
                            yield doc
                    except subprocess.TimeoutExpired as te:
                        logger.error("Timed out building {}: {}", doc.text, str(te))
                        doc.metadata["build_timeout"] = str(te)
                        doc.metadata["build_success"] = False
                        self.stat_update("build_failure")
                        self.stat_update("build_timeout")
                        yield doc
                    except Exception as err:
                        logger.error(
                            "Error processing nix findpkgs results: {err}\n", err=err
                        )
                        doc.metadata["build_exception"] = str(err)
                        doc.metadata["build_success"] = False
                        self.stat_update("build_failure")
                        self.stat_update("build_exception")
                        yield doc
        tar_file.close()


class ExtractBinaryDataset(PipelineStep):
    name = "📦 Extract Binary Dataset"
    type = "⚙️ - PROCESS"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import os
        import tarfile

        from datatrove.data import Document
        from datatrove.utils.logging import logger

        if data:
            for doc in data:
                with self.track_time():
                    if not doc.metadata["build_success"]:
                        continue
                    try:
                        with tarfile.open(
                            doc.metadata["build_archive"], mode="r:gz"
                        ) as tar_file:
                            for rel_path in doc.metadata["binaries"]:
                                package = rel_path.split(os.path.sep)[0]
                                file_name = os.path.basename(rel_path)
                                binary = tar_file.extractfile(
                                    os.path.normpath(rel_path)
                                ).read()
                                yield Document(
                                    id=rel_path,
                                    text=rel_path,
                                    metadata={
                                        "binary": binary,
                                        "package": package,
                                        "filename": file_name,
                                        "path": rel_path,
                                        "properties": {
                                            "flake": doc.metadata["flake"],
                                            "language": doc.metadata["language"],
                                            "build_options": doc.metadata[
                                                "build_options"
                                            ],
                                        },
                                    },
                                )
                                self.stat_update("success")
                    except Exception as e:
                        logger.warning(
                            "failed to decompress binary: ({}) {}", str(e), doc.text
                        )
                        self.stat_update("failure")
                        continue


tmp_dir = "/state/partition1/user/ch17997/tmp"
base_dir = "/home/gridsan/CH17997/test"
log_dir = os.path.join(base_dir, "logging")
binaries_dir = os.path.join(base_dir, "binaries")
packages_dir = os.path.join(base_dir, "packages")
builds_dir = os.path.join(base_dir, "builds")
working_dir = os.path.join(tmp_dir, "working-build")
github_cache_dir = "/home/gridsan/CH17997/github_cache"
github_token = "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"
dataset_dir = "/home/gridsan/groups/undertale_shared/datasets/nixpkgs"
funcs_dir = "/home/gridsan/groups/undertale_shared/datasets/nixpkgs-functions"
dataset_working_dir = os.path.join(tmp_dir, "working-dataset")
proxies = {
    "http": "http://llproxy-rr.llgrid.ll.mit.edu:8080",
    "https": "http://llproxy-rr.llgrid.ll.mit.edu:8080",
    "ftp": "http://llproxy-rr.llgrid.ll.mit.edu:8080",
}

flakes = [
    # "github:NixOS/nixpkgs/nixos-19.03",
    # "github:NixOS/nixpkgs/nixos-19.09",
    # "github:NixOS/nixpkgs/nixos-20.03",
    # "github:NixOS/nixpkgs/nixos-20.09",
    # "github:NixOS/nixpkgs/nixos-21.05",
    # "github:NixOS/nixpkgs/nixos-21.11",
    # "github:NixOS/nixpkgs/nixos-22.05",
    # "github:NixOS/nixpkgs/nixos-22.11",
    # "github:NixOS/nixpkgs/nixos-23.05",
    "github:NixOS/nixpkgs/nixos-23.11",
    "github:NixOS/nixpkgs/nixos-24.05",
    "github:NixOS/nixpkgs/nixos-24.11",
]

broken_pkgs = [
    "cplex",
    "dyalog",
    "input-fonts",
    "joypixels",
    "libreoffice-bin",
    "oilrush",
    "raycast",
    "segger-jlink",
    "segger-jlink-headless",
    "xxe-pe",
    "yabai",
]

find_packages = SlurmPipelineExecutor(
    pipeline=[
        FindNixpkgs(
            flakes=flakes,
            broken_pkgs=broken_pkgs,
            wrapper_args=["/home/gridsan/CH17997/bin/llnix", "-n", "--"],
            build_options=["debug"],
        ),
        # forbidden packages
        LambdaFilter(
            lambda doc: doc.text
            not in [
                "mimikatz",
                "pypykatz",
                "nmap",
                "xmrig",
                "xmrig-mo",
                "xmrig-proxy",
                "mlv-app",
                "stepmani",
                "glider",
                "tor",
                "torbrowser",
            ]
        ),
        EnrichGithubPackages(
            github_token=github_token,
            github_cache_dir=github_cache_dir,
            proxies=proxies,
            wrapper_args=["/home/gridsan/CH17997/bin/llnix", "-n", "--"],
        ),
        LambdaFilter(
            lambda doc: (doc.metadata["language"] in ["C", "C++", "Rust", "Go"])
        ),
        # SamplerFilter(0.002),
        JsonlWriter(
            output_folder=packages_dir,
            max_file_size=2000,
        ),
    ],
    logging_dir=os.path.join(base_dir, "logging", "find"),
    tasks=1,
    time="02:00:00",
    job_name="find_nixpkgs",
    partition="download",
)

build_packages = SlurmPipelineExecutor(
    depends=find_packages,
    pipeline=[
        JsonlReader(data_folder=packages_dir),
        BuildNixpkgs(
            binaries_dir=binaries_dir,
            working_dir=working_dir,
            wrapper_args=["/home/gridsan/CH17997/bin/llnix", "-n", "--"],
            build_timeout=2 * 60 * 60,
        ),
        JsonlWriter(
            output_folder=builds_dir,
            max_file_size=50 * 1024,
        ),
    ],
    logging_dir=f"{log_dir}/build",
    tasks=len(flakes) * 32,
    time="02:00:00",
    job_name="build_nixpkgs",
    mem_per_cpu_gb=1,
    cpus_per_task=4,
    partition="download",
)

extract_dataset = SlurmPipelineExecutor(
    depends=build_packages,
    pipeline=[
        JsonlReader(data_folder=builds_dir),
        ExtractBinaryDataset(),
        ParquetWriter(
            output_folder=dataset_dir,
            adapter=lambda self, doc: doc.metadata,
            max_file_size=100 * 1024 * 1024,
        ),
    ],
    logging_dir=f"{log_dir}/export",
    tasks=len(flakes) * 2400,
    time="02:00:00",
    job_name="extract_nixpkgs",
    mem_per_cpu_gb=4,
    cpus_per_task=2,
    partition="xeon-p8",
    sbatch_args={"distribution": "cyclic:cyclic"},
)

segment_functions = SlurmPipelineExecutor(
    # depends = extract_dataset,
    pipeline=[
        ParquetReader(
            data_folder=dataset_dir,
            adapter=lambda self, data, path, id_in_file: {
                "id": data["filename"],
                "text": data["binary"],
                "metadata": data,
            },
        ),
        SegmentLief(),
        ParquetWriter(
            output_folder=funcs_dir,
            adapter=lambda self, doc: doc.metadata,
            max_file_size=100 * 1024 * 1024,
        ),
    ],
    venv_path="/home/gridsan/CH17997/.venv",
    logging_dir=f"{log_dir}/segment",
    tasks=len(flakes) * 100,
    max_array_size=310,
    time="02:00:00",
    job_name="segment_nixpkgs",
    mem_per_cpu_gb=4,
    cpus_per_task=2,
    partition="xeon-p8",
    sbatch_args={"distribution": "cyclic:cyclic"},
)
if __name__ == "__main__":
    segment_functions.run()
