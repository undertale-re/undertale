.. _development-setup:

Setup
-----

Prerequisites
^^^^^^^^^^^^^

- **Ubuntu** >= 24.04 *or* **macOS** >= 15 (with `Homebrew <https://brew.sh/>`_).
- **Python** >= 3.12 (see `pyenv <https://github.com/pyenv/pyenv>`_).
- A copy of the source code (see :ref:`development-workflows-clone`).

.. note::
    
    All of the commands below should be run from the root of the repository.

Setup
^^^^^

To setup a development environment, first install the necessary development
dependencies:

.. code-block:: bash

    sudo bash dependencies/development.{operating-system}.sh

Where ``{operating-system}`` is either ``ubuntu`` or ``macos``.

.. warning::

    There are some dependencies that the scripts cannot install automatically
    for you - please check the output for warnings and more information.

.. note::

    Python packages should be installed in a `Virtual Environment
    <https://docs.python.org/3/library/venv.html>`_ for development purposes.
    If you're not familiar with using Virtual Environments, go brush up on them
    now.

Then install the Python package with ``pip`` in editable mode, with extras for
development, and with frozen dependencies:

.. code-block:: bash

    pip install -e .[development] -c constraints.txt

Code Style
^^^^^^^^^^

`Pre-commit <https://pre-commit.com/>`_ hooks are available for automatic code
formatting, linting, and type checking. To enable them (after installing
development dependencies) run:

.. code-block:: bash

    pre-commit install

Documentation
^^^^^^^^^^^^^

To build the documentation (after :ref:`development-setup`), from the ``docs/``
directory run:

.. code-block:: bash

    make html

Or other `supported sphinx output formats
<https://www.sphinx-doc.org/en/master/usage/builders/index.html>`_.

Extras
^^^^^^

Installing Ghidra on MacOS
""""""""""""""""""""""""""

In general, you should follow `Ghidra's installation instructions
<https://github.com/NationalSecurityAgency/ghidra?tab=readme-ov-file#install>`_,
but here are a few tips.

#. Install OpenJDK with Homebrew

    .. code-block:: bash

        brew install openjdk

#. Download and extract a Ghidra release directly from GitHub. The directory
   scheme ``/opt/ghidra/{version}/`` works well.

#. Disable MacOS code verification for Ghidra, otherwise you will have to auth
   every time you want to run it.

    .. code-block:: bash

        xattr -dr com.apple.quarantine /opt/ghidra/

#. Make sure you set the ``GHIDRA_INSTALL_DIR`` environment variable when
   running scripts that use Ghidra.

    .. code-block:: bash

        GHIDRA_INSTALL_DIR=/opt/ghidra/11.4.1/ \
            python -m undertale ...
