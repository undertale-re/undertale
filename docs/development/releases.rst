Releases
--------

Constraints
^^^^^^^^^^^

Both the core package and the inference server pin their dependencies in a
``constraints.txt`` file. The two files are generated from a single, joint
dependency resolution over both projects so their pins can never conflict.
To regenerate them (e.g., after changing dependencies in either
``pyproject.toml``), run:

.. code-block:: bash

    ./scripts/constraints.sh

Then run the test suite against the updated pins and commit both files
together.

Offline Release Bundles
^^^^^^^^^^^^^^^^^^^^^^^

Two scripts build self-contained release bundles for offline (air-gapped)
installation:

- ``scripts/release.sh`` - the core ``undertale`` package
- ``extras/inference-server/scripts/release.sh`` - the inference server
  (includes the ``undertale`` wheel and a pre-seeded HuggingFace cache)

Each bundle contains the project wheel(s) plus complete dependency
wheelhouses for macOS arm64, Linux x86-64 (glibc 2.28+), and Windows amd64,
all targeting Python 3.12. The Linux and Windows wheelhouses use CPU-only
PyTorch builds from the PyTorch package index to keep bundles small (the
default PyPI wheels on those platforms bundle CUDA).

Run either script from a connected machine with a Python 3.12 environment;
archives are written to the respective ``dist/`` directory. Each bundle
includes an ``INSTALL.md`` (from the respective ``scripts/release/``
directory) with the offline installation instructions.

Every wheelhouse is validated during the build by dry-run resolving the
top-level packages against the wheelhouse alone, for each target platform.
Model checkpoints are too large to bundle and are distributed separately -
see the inference server's ``INSTALL.md``.
