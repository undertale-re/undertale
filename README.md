# Undertale

[![commit-style-image]][conventional]
[![code-style-image]][black]
[![documentation-image]][pages]
[![license-image]][mitll]

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

Undertale is essentially a collection of scripts invoked via module call, like:

```bash
python -m undertale.{module} ...
```

For more details on available scripts and usage information, see the [full
documentation][pages].

## Contributing

Pull requests and issues are more than welcome. Please review the
``Development`` section of the documentation prior to contributing.

## Publications

| Date | Venue | Publication |
| --- | --- | --- |
| 04/30/2024 | [arXiv][arxiv] | [On Training a Neural Network To Explain Binaries][20240430-arxiv] |
| 09/15/2025 | [IEEE HPEC][HPEC] | [Scaling Performance of Large Language Model Pretraining][20250915-hpec] |

[arxiv]: https://arxiv.org/
[HPEC]: https://ieee-hpec.org/
[20240430-arxiv]: https://arxiv.org/abs/2404.19631
[20250915-hpec]: https://arxiv.org/abs/2509.05258

### Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

© 2023 Massachusetts Institute of Technology

This material is based upon work supported by the Under Secretary of
Defense for Research and Engineering under Air Force Contract
No. FA8702-15-D-0001. Any opinions, findings, conclusions or
recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Under Secretary of
Defense for Research and Engineering.  © 2023 Massachusetts Institute
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
[documentation-image]: https://img.shields.io/badge/docs-latest-green.svg
[pages]: https://undertale-re.github.io/undertale/
[license-image]: https://img.shields.io/badge/license-MIT-green.svg
[mitll]: ./LICENSE.txt
