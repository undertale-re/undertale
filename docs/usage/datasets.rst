Datasets
--------

Build a Dataset
^^^^^^^^^^^^^^^

To build a dataset, call the dataset module directly:

.. code-block:: bash

    python -m undertale.datasets.{dataset} {input} {output}

For example, to build the HumanEval-X dataset from the raw dataset at
``humanevalx-raw/`` and save it to a directory called ``humanevalx/``, run:

.. code-block:: bash

    # Parse the HumanEvalX dataset.
    python -m undertale.datasets.humanevalx humanevalx-raw/ humanevalx/

    # Parse the HumanEvalX dataset with 8 parallel processes.
    python -m undertale.datasets.humanevalx humanevalx-raw/ humanevalx/ --parallelism 8


Explore a Dataset with a Shell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a dataset into a Python shell, run:

.. code-block:: bash

    python -m undertale.datasets.scripts.shell {path}

    # Load the HumanEval-X dataset into a shell.
    python -m undertale.dataset.scripts.shell humanevalx/


Use a Dataset in a Script
^^^^^^^^^^^^^^^^^^^^^^^^^

To write a scirpt that uses a dataset that has already been parsed, you can do
something like:

.. code-block:: python

    from undertale.datasets.base import Dataset

    dataset = Dataset.load(path)

    ...

Where ``path`` is the path to the saved dataset directory.

Start a vllm server
^^^^^^^^^^^^^^^^^^^^^^^^^

The run vllm summaries, you need to start a vllm server that the job can query

.. code-block:: shell

    export MODEL="models--Qwen--Qwen3-Coder-30B-A3B-Instruct/"
    export CUDA_VISIBLE_DEVICES=0

    export MASTER_HOST=$(hostname -s)
    export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    export HEAD_NODE_ADDR="$MASTER_HOST:$MASTER_PORT"

    echo "HEAD NODE: $HEAD_NODE_ADDR"

    vllm serve $MODEL \
        --served_model_name qwencoder \
        --max_model_len 32768 \
        --reasoning-parser qwen3

    ...

The ``HEAD_NODE_ADDR`` is part of the VLLM_SERVER_ADDRESS environment variable that you need to set
to run datasets/vllm.py. ``export VLLM_SERVER_ADDRESS=http://$HEAD_NODE_ADDR:8000/v1``
