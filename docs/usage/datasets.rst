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

    # Process the HumanEval-X dataset.
    python pipelines/datasets/humaneval-x.py \
        humaneval-x-raw/20251114-100300.tgz \
        humaneval-x

.. _environments:

Environments
^^^^^^^^^^^^

It can be useful to customize your environment to make controling Undertale
parameters for your specific setup a bit easier. There are a couple of example
environment files included in the ``environments`` directory which can be
activated as follows:

.. code-block:: bash

    source environments/example.env

The ``environments`` directory includes a simple example for local development
as well as a more complex example representing a SLURM cluster for distributed
pipelines and training.

.. _parallelism:

Parallelism
^^^^^^^^^^^

All dataset commands support being run in parallel.

.. code-block:: bash

    # Process HumanEval-X with custom local parallelism.
    python pipelines/datasets/humaneval-x.py \
        humaneval-x-raw/20251114-100300.tgz \
        humaneval-x \
        --parallelism 8

    # On a SLURM cluster.
    srun python pipelines/datasets/humaneval-x.py \
        humaneval-x-raw/20251114-100300.tgz \
        humaneval-x \
        --cluster slurm \
        --parallelism 16

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

Percentages must sum to 100. See the ``--seed`` option to control split
randomization.

Repartition a Dataset
^^^^^^^^^^^^^^^^^^^^^

To repartition a dataset into a fixed number of chunks or by target chunk size,
use the ``repartition`` utility.

.. code-block:: bash

    # Repartition to exactly 32 chunk files.
    python -m undertale.utils.datasets.repartition \
        humaneval-x/ \
        humaneval-x-repartitioned \
        --chunks 32

    # Repartition by target chunk size.
    python -m undertale.utils.datasets.repartition \
        humaneval-x/ \
        humaneval-x-repartitioned \
        --size 25MB

Exactly one of ``--chunks`` or ``--size`` must be specified.

Drop or Keep Columns
^^^^^^^^^^^^^^^^^^^^

To drop or keep specific columns from a dataset, use the drop utility.

.. code-block:: bash

    # Drop specific columns.
    python -m undertale.utils.datasets.drop \
        humaneval-x/ \
        humaneval-x-filtered \
        --drop metadata source

    # Keep only specific columns.
    python -m undertale.utils.datasets.drop \
        humaneval-x/ \
        humaneval-x-filtered \
        --keep id solution

Exactly one of ``--drop`` or ``--keep`` must be specified.

Rename Columns
^^^^^^^^^^^^^^

To rename one or more columns in a dataset, use the rename utility.

.. code-block:: bash

    # Rename a single column.
    python -m undertale.utils.datasets.rename \
        humaneval-x/ \
        humaneval-x-renamed \
        --rename source:origin

    # Rename multiple columns at once.
    python -m undertale.utils.datasets.rename \
        humaneval-x/ \
        humaneval-x-renamed \
        --rename source:origin metadata:info

The output dataset preserves the same chunk structure as the input.
