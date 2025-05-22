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
python -m undertale.datasets.{dataset-module} {input} {output}
```

Examples:

```bash
# Parse the HumanEval-X dataset.
python -m undertale.datasets.humanevalx _ humanevalx/

# Parse the HumanEval-X dataset with 8 parallel processes.
python -m undertale.datasets.humanevalx _ humanevalx/ --parallelism 8
```

> [!NOTE]
> The HumanEval-X dataset pulls data from the HuggingFace Hub so the `input`
> parameter is ignored. The examples above use `_` for brevity.

To load a given dataset and open a shell for exploration:

```bash
python -m undertale.datasets.scripts.shell {input}
```

Example:

```bash
python -m undertale.datasets.scripts.shell humanevalx/
```

The dataset will be available in a variable called `dataset` in the shell.

To write a script that uses a dataset that has already been parsed and is
available in the cache directory, you can do something like:

```python
from undertale.datasets import Dataset

dataset = Dataset.load(path)

...
```

### Models

#### Pretoken Processing

Before the tokenizer can be trained on a dataset, disassembly must be processed
into pretokens that the tokenizer can consume. To pretokenize e.g., the
HumanEvalX dataset (generated above), run:

```python
python -m undertale.datasets.scripts.pretokenize humanevalx/ humanevalx-pretokenized/
```

#### Tokenizer Training

Next, you can train a tokenizer on the pretokenized dataset:

```bash
python -m undertale.models.item.tokenizer \
    humanevalx-pretokenized/ \
    -o item.tokenizer.json
```

#### Masked Language Modeling Pre-Training

```bash
python -m undertale.models.item.pretrain-maskedlm \
    undertale.datasets.humanevalx:HumanEvalX \
    -t item.tokenizer.json \
    -o pretrain-maskedlm
```

#### Contrastive Embedding Fine-Tuning

```bash
python -m undertale.models.item.finetune-embedding \
    <dataset-tbd> \
    -t item.tokenizer.json \
    -m pretrain-maskedlm/9 \
    -o finetune-embedding
```

#### Masked Language Modeling Inference

> [!WARNING]
> This output is pretty bad right now with only the small dataset - it should
> get better once we can start training with larger datasets.

```bash
python -m undertale.models.item.infer-maskedlm \
    -t item.tokenizer.json \
    -m pretrain-maskedlm/9 \
    "add eax, [MASK]"
```

#### Summarization Inference

> [!WARNING]
> This still uses an untrained code-language connector, the output will be
> gibberish, but it proves that everything is wired up correctly.

```bash
python -m undertale.models.item.infer-summarization \
    -t item.tokenizer.json \
    -m finetune-embedding/9 \
    "add eax, ebx\nxor ecx, ecx"
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
