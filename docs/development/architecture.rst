.. _development-architecture:

Architecture
------------

Undertale is made up of three primary focus areas:

**Datasets**
    Custom-built datasets of binary code for training our models. Our dataset
    build pipelines are based on `datatrove
    <https://github.com/huggingface/datatrove>`_ and we've implemented a number
    of custom ``PipelineSteps`` for things like compilation, disassembly,
    segmentation, etc. All of our dataset build pipelines are fully
    reproducible and open-source.

**Modeling**
    Our custom model architectures and training schemes for modeling binary
    code.

**Evaluation & Integration**
    Comparisons with current state of the art (natural language LLMs) and
    integrations with popular tools like `Ghidra
    <https://github.com/NationalSecurityAgency/ghidra>`_
