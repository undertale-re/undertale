.. _development-setup:

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

To setup a development Conda environment with Undertale and all of its
dependencies:

.. code-block:: bash

    conda env create -f environment.yml
    conda env update -f environment.development.yml

Youc an then activate the environment:

.. code-block:: bash

    conda activate undertale

Code Style
^^^^^^^^^^

`Pre-commit <https://pre-commit.com/>`_ hooks are available for automatic code
formatting, linting, and type checking. To enable them (after installing
development dependencies) run:

.. code-block:: bash

    pre-commit install

Documentation
^^^^^^^^^^^^^

To build the documentation (after :ref:`development-setup`), from the root of
the repository run:

.. code-block:: bash

    sphinx-build -b html docs build/documentation/

Or other `supported sphinx output formats
<https://www.sphinx-doc.org/en/master/usage/builders/index.html>`_.

Extras
^^^^^^

Binary Ninja Setup
""""""""""""""""""

Coming soon...
