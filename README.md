# Undertale

[![commit-style-image]][conventional]
[![code-style-image]][black]
[![license-image]][mit]

Large language models (LLMs) for binary program understanding.

## Description

This is all of the code for managing the various datasets, models, and
evaluations involved in the Undertale program.

## Installation

To install Undertale and frozen dependencies from `constraints.txt`, after
cloning this repo run:

```bash
pip install . -c constraints.txt
```

## Usage

This package is primarily a collection of scripts which can be invoked by a
direct module call.

### Datasets

To parse and commit a dataset:

```bash
python -m undertale.datasets.{dataset-module} parse {path-to-raw}
```

Examples:

```bash
# Parse the default HumanEval-X dataset (compiled and disassembled).
python -m undertale.datasets.humanevalx parse download

# Parse the HumanEval-X dataset with 8 parallel processes.
python -m undertale.datasets.humanevalx parse download --processes 8

# Parse the compiled (not disassembled) variant of the HumanEval-X dataset with 8 parallel processes.
python -m undertale.datasets.humanevalx parse download --variant HumanEvalXCompiled --processes 8
```

To load a given dataset and open a shell for exploration:

```bash
python -m undertale.datasets.{dataset-module} shell
```

Example:

```bash
python -m undertale.datasets.humanevalx shell
```

The dataset will be available in a variable called `dataset` in the shell.

To write a script that uses a dataset that has already been parsed and is
available in the cache directory, you can do something like:

```python
from undertale.datasets import humanevalx

dataset = humanevalx.HumanEvalXCompiledDisassembled.fetch()

...
```

## Contributing

Please submit pull requests or issues for any new features and/or bug fixes!

### Development

To set up a development environment, first clone this repo. Next, it is useful
to install Undertale in editable mode with extras for development, using frozen
dependency versions from `constraints.txt`:

```bash
pip install -e .[development] -c constraints.txt
```

#### Nix Support

You can also automatically build and create deployment envrionments and nix
packages for nix with the `flake.nix` in the repository.  

#### Installation of Nix Tooling
To learn how to install nix on your systems [see here](https://nix.dev/install-nix.html)

Once you have nix working on your system make sure to [enable flakes](https://nixos.wiki/wiki/Flakes)

If you'd like to leverage the automatic development environment setup, install [direnv](https://github.com/nix-community/nix-direnv)

If you use `direnv`, once it is installed and hooked into your shell, make sure to allow the
``.envrc` file to turn on automatic shell creation and descruction.

#### Development Environments
Once you've cloned the `undertale` repo, in that directory you can easily enter a
developer shell by running:

```bash
nix develop
```
This may take a while if you've never built anything before.

You will automatically have all of the included development tools in scope as well as all 
of the python dependencies in your python environment.  To exit the shell, run the `exit`
command or hit control-D.

#### Nix Packaging options
If you'd just want to export or package, you can build the package as 

```bash
nix build
```

which will build undertale as a nix package and generate a symlink in the current directory
called `result` pointing to the build entry in `/nix/store`

Additionally, you can also build a package containing all of the environment depenencies for
`undertale` by running

```bash
nix build .#undertale-environment
```

You can also create singularity or docker containers with a full isolated and built
environment by calling the respective commands:

```bash
nix build .#singularity-image
```

or

```bash
nix build .#docker-image
```

which will create a `result` symlink in the current directory pointing to the respective image
binaries.  Since they contain all dependencies, they are quite large tar.gz/sif files (>1GB)
but the docker image is layered, so upon deployment will minimally occupy space as incremental
versions are added.

### Code Style

Pre-commit hooks are available for automatic code formatting, linting, and type
checking via [pre-commit](https://pre-commit.com/). To enable them (after
installing development dependencies), run:

```bash
pre-commit install
```


### Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

© 2025 Massachusetts Institute of Technology

This material is based upon work supported by the Under Secretary of
Defense for Research and Engineering under Air Force Contract
No. FA8702-15-D-0001. Any opinions, findings, conclusions or
recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Under Secretary of
Defense for Research and Engineering.  © 2025 Massachusetts Institute
of Technology.  The software/firmware is provided to you on an As-Is
basis Delivered to the U.S. Government with Unlimited Rights, as
defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding
any copyright notice, U.S. Government rights in this work are defined
by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of
this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.

[commit-style-image]: https://img.shields.io/badge/commits-conventional-fe5196.svg
[conventional]: https://www.conventionalcommits.org/en/v1.0.0/
[code-style-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[black]: https://github.com/psf/black
[license-image]: https://img.shields.io/badge/license-MIT-green.svg
[mit]: ./LICENSE.txt
