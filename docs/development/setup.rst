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

Testing
^^^^^^^

To run tests, run:

.. code-block:: bash

    python tests/unit.py --verbose

Extras
^^^^^^

.. _development-binary-ninja-setup:

Binary Ninja Setup
""""""""""""""""""

To work with binary data, you'll need to have a `Binary Ninja
<https://binary.ninja/>`_ license (Commercial or Ultimate). In general, you
should follow `their installation instructions
<https://docs.binary.ninja/getting-started.html#installing-binary-ninja>`_.

Once Binary Ninja is installed, to set up the API bindings, locate
``install_api.py`` in the ``scripts`` directory in your Binary Ninja
installation path. For example:

.. code-block:: bash

    # Ubuntu
    <install-path>/scripts/install_api.py

    # MacOS
    /Applications/Binary\ Ninja.app/Contents/Resources/scripts/install_api.py

Run this script with the same Python environment where Undertale is installed
to set up Binary Ninja's Python API bindings.

You'll also need to ensure your Binary Ninja license key is set up. Typically
this is done by starting Binary Ninja for the first time and using the license
file you were given when you purchased the product. In a headless environment
(where you cannot start the GUI application), you can simply place this license
file in the following location and the API will work as expected (this is not
well documented):

.. code-block:: bash

    ~/.binaryninja/license.dat

Finally, to verify that everything is working correctly, run the binary unit
tests:

.. code-block:: bash

    python tests/unit.py --verbose TestPipelineBinary

Verify that the binary analysis tests pass and are not skipped.
