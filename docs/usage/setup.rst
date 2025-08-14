.. _usage-setup:

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

To setup Undertale, first install the necessary dependencies:

.. code-block:: bash

    sudo bash dependencies/production.{operating-system}.sh

Where ``{operating-system}`` is either ``ubuntu`` or ``macos``.

.. warning::

    There are some dependencies that the scripts cannot install automatically
    for you - please check the output for warnings and more information.

Then install the Python package with ``pip``  with frozen dependencies:

.. code-block:: bash

    pip install . -c constraints.txt
