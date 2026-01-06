.. _usage-setup:

Setup
-----

Prerequisites
^^^^^^^^^^^^^

- **Ubuntu** >= 24.04 *or* **macOS** >= 15 (with `Homebrew <https://brew.sh/>`_).
- **Python** >= 3.12 (see `pyenv <https://github.com/pyenv/pyenv>`_).
- **Conda** (see `conda documentation <https://docs.conda.io/>`_).
- A copy of the source code (see :ref:`development-workflows-clone`).

.. note::
    
    All of the commands below should be run from the root of the repository.

Setup
^^^^^

To setup a Conda environment with Undertale and all of its dependencies:

.. code-block:: bash

    conda env create -f environment.yml

You can then activate the environment:

.. code-block:: bash

    conda activate undertale
