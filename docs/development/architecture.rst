.. _development-architecture:

Architecture
------------

Undertale is made up of three primary focus areas:

**Datasets**
    Custom-built datasets of binary code for training our models. Our build
    pipelines use `dask <https://www.dask.org/>`_ for scaling workflows and
    we've implemented a number of custom functions for things like compilation,
    disassembly, segmentation, etc. All of our dataset build pipelines are
    fully reproducible and open-source.

**Modeling**
    Our custom model architectures and training schemes for modeling binary
    code.

**Evaluation & Integration**
    Comparisons with current state of the art (natural language LLMs) and
    integrations with popular tools like `Ghidra
    <https://github.com/NationalSecurityAgency/ghidra>`_

Architecturally, these are broken into a few different software components:

**The Undertale Package** (``undertale/``)
    Tested, mature implementations of dataset pipeline steps, model
    implementation code, utilities and helpers.

**Pipeline Scripts** (``pipelines/``)
    Pipeline scripts for datasets, model training, and evaluation.

**Documentation** (``docs/``)
    User and developer documentation and Undertale package auto-documentation.

**Tests** (``tests/``)
    Unit, performance, and integration tests for the Undertale package.
