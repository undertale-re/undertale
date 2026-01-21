Datasets
--------

Build a Dataset
^^^^^^^^^^^^^^^

To build a dataset, call the dataset pipeline script from the ``pipelines``
directory:

.. code-block:: bash

    python pipelines/dataset-{dataset}.py {input} {output}

For example, to build the HumanEval-X dataset from the raw dataset snapshot at
``humaneval-x-raw/20251114-100300.tgz`` and save it to a directory called
``humaneval-x/``, run:

.. code-block:: bash

    # Parse the HumanEval-X dataset.
    python pipelines/datasets/humaneval-x.py \
        humaneval-x-raw/20251114-100300.tgz \
        humaneval-x

    # Parse the HumanEval-X dataset with 8 parallel processes.
    python pipelines/datasets/humaneval-x.py \
        humaneval-x-raw/20251114-100300.tgz \
        humaneval-x \
        --parallelism 8

Explore a Dataset with a Shell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a dataset into a Python shell, run:

.. code-block:: bash

    python -m undertale.utils.datasets.shell {path}

    # Load the HumanEval-X dataset into a shell.
    python -m undertale.utils.datasets.shell humaneval-x/

Large Datasets
""""""""""""""

The ``undertale.utils.dataset.shell`` utility uses ``pandas`` to load the
dataset - this requires the entire dataset to be loaded into memory. If you're
working with a larger-than-memory dataset, you can use the ``polars`` shell
instead to get a ``LazyFrame`` (see the `Polars Documentation
<https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html>`_ for
more details):

.. code-block:: bash

    python -m undertale.utils.datasets.shell.polars {path}

    # Load the HumanEval-X dataset into a polars.LazyFrame.
    python -m undertale.utils.datasets.shell.polars humaneval-x/

Use a Dataset in a Script
^^^^^^^^^^^^^^^^^^^^^^^^^

Final datasets are simply large directories of parquet. Datasets can be loaded
in Python in all the usual ways you would load parquet - for example,  with
``pandas``:

.. code-block:: python

    import pandas

    dataset = pandas.read_parquet(path)

    ...

Where ``path`` is the path to the saved dataset directory.
