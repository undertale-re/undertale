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


.. _dataset-splitting:

Split a Dataset
^^^^^^^^^^^^^^^

Splitting a dataset ahead of training into training, validation, and test sets
can be efficient and ensure a deterministic split. There is a helper utility
available for that.

.. code-block:: bash

    # Two-way split: 90% training, 10% validation (default).
    #
    # Writes to humaneval-x-training/ and humaneval-x-validation/.
    python -m undertale.utils.datasets.split \
        humaneval-x/ \
        humaneval-x

    # Three-way split: 80% training, 10% validation, 10% test.
    #
    # Writes to humaneval-x-training/, humaneval-x-validation/, and
    # humaneval-x-testing/.
    python -m undertale.utils.datasets.split \
        humaneval-x/ \
        humaneval-x
        --splits training:80 validation:10 testing:10

    # With custom parallelism.
    python -m undertale.utils.datasets.split \
        humaneval-x/ \
        humaneval-x
        --splits training:80 validation:10 testing:10 \
        --parallelism 8

Percentages must sum to 100. See the ``--seed`` option to control split
randomization.

Resize a Dataset
^^^^^^^^^^^^^^^^

To resize a dataset into a fixed number of chunks or by target chunk size, use
the resize utility.

.. code-block:: bash

    # Resize to exactly 32 chunk files.
    python -m undertale.utils.datasets.resize \
        humaneval-x/ \
        humaneval-x-resized \
        --chunks 32

    # Resize by target chunk size.
    python -m undertale.utils.datasets.resize \
        humaneval-x/ \
        humaneval-x-resized \
        --size 25MB

    # Drop columns, deduplicate, and apply compression.
    python -m undertale.utils.datasets.resize \
        humaneval-x/ \
        humaneval-x-resized \
        --chunks 32 \
        --drop metadata source \
        --deduplicate id \
        --compression snappy

    # With custom parallelism.
    python -m undertale.utils.datasets.resize \
        humaneval-x/ \
        humaneval-x-resized \
        --chunks 32 \
        --parallelism 8

Exactly one of ``--chunks`` or ``--size`` must be specified. ``--drop`` and
``--keep`` are mutually exclusive.
