Datasets
--------

Build a Dataset
^^^^^^^^^^^^^^^

To build a dataset, call the dataset module directly:

.. code:: bash

    python -m undertale.datasets.{dataset} {input} {output}

For example, to build the HumanEval-X dataset from the raw dataset at
``humanevalx-raw/`` and save it to a directory called ``humanevalx/``, run:

.. code:: bash

    # Parse the HumanEvalX dataset.
    python -m undertale.datasets.humanevalx humanevalx-raw/ humanevalx/

    # Parse the HumanEvalX dataset with 8 parallel processes.
    python -m undertale.datasets.humanevalx humanevalx-raw/ humanevalx/ --parallelism 8


Explore a Dataset with a Shell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a dataset into a Python shell, run:

.. code:: bash

    python -m undertale.datasets.scripts.shell {path}

    # Load the HumanEval-X dataset into a shell.
    python -m undertale.dataset.scripts.shell humanevalx/


Use a Dataset in a Script
^^^^^^^^^^^^^^^^^^^^^^^^^

To write a scirpt that uses a dataset that has already been parsed, you can do
something like:

.. code:: python

    from undertale.datasets.base import Dataset

    dataset = Dataset.load(path)

    ...

Where ``path`` is the path to the saved dataset directory.
