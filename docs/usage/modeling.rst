Modeling
--------

Tokenizer Training
^^^^^^^^^^^^^^^^^^

The first step in training any of our models is to train a tokenizer. To train
a tokenizer on e.g., the HumanEval-X dataset, run the tokenizer training
pipeline script:

.. code-block:: bash

    # Train a tokenizer on the HumanEval-X dataset.
    python pipelines/models/train-tokenizer.py \
        humaneval-x/ \
        tokenizer

    # Train a tokenizer on the HumanEval-X dataset in parallel.
    python pipelines/models/train-tokenizer.py \
        humaneval-x/ \
        tokenizer \
        --parallelism 8

Tokenization
^^^^^^^^^^^^

With your trained tokenizer you can now tokenize an entire dataset to prepare
for pre-training.

.. code-block:: bash

    # Tokenize the HumanEval-X dataset.
    #
    # Only retain the minimal fields necessary for pre-training.
    python pipelines/models/tokenize-dataset.py \
        humaneval-x/ \
        humaneval-x-pretraining \
        --tokenizer tokenizer.json \
        --minimal

    # Tokenize the HumanEval-X dataset in parallel.
    #
    # Retain all fields this time, as an example.
    python pipelines/models/tokenize-dataset.py \
        humaneval-x/ \
        humaneval-x-tokenized \
        --tokenizer tokenizer.json \
        --parallelism 8

Dataset Split (Optional) 
""""""""""""""""""""""""

Before pre-training, split your tokenized dataset into training and validation
sets.

.. code-block:: bash

    # Split the tokenized HumanEval-X dataset (default: 90% training, 10% validation).
    #
    # Writes to humaneval-x-pretraining-training/ and humaneval-x-pretraining-validation/.
    python -m undertale.utils.datasets.split \
        humaneval-x-pretraining/ \
        humaneval-x-pretraining

    # Split with a custom fraction and parallelism.
    python -m undertale.utils.datasets.split \
        humaneval-x-pretraining/ \
        humaneval-x-pretraining \
        --fraction 0.95 \
        --parallelism 8

The ``--seed`` option controls the random state used for splitting and defaults
to ``42``. Fix this value across runs to ensure a reproducible split.

Pre-Training (Maked Language Modeling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With your tokenized training dataset (and optional validation split) you are
now ready to begin pretraining a model.

.. code-block:: bash

    # Start a pretraining run locally (as an example).
    #
    # Results will be written to maskedlm/.
    
    python pipelines/models/pretrain-maskedlm.py \
        --tokenizer tokenizer.json \
        humaneval-x-pretraining/ \
        maskedlm

    # Include validation data.

    python pipelines/models/pretrain-maskedlm.py \
        --tokenizer tokenizer.json \
        humaneval-x-pretraining-training/ \
        --validation humaneval-x-pretraining-validation/ \
        maskedlm

    # Use multiple accelerators on the same host.

    python pipelines/models/pretrain-maskedlm.py \
        --devices 4 \
        --tokenizer tokenizer.json \
        humaneval-x-pretraining-training/ \
        --validation humaneval-x-pretraining-validation/ \
        maskedlm

    # Distribute training across multiple accelerators on multiple hosts.
    #
    # This uses the Distributed Data Parallel distribution strategy.

    python pipelines/models/pretrain-maskedlm.py \
        --strategy ddp \
        --nodes 8 \
        --devices 2 \
        --tokenizer tokenizer.json \
        humaneval-x-pretraining-training/ \
        --validation humaneval-x-pretraining-validation/ \
        maskedlm

There are several other configurable parameters for other training scenarios -
to get a full list, run:

.. code-block:: bash

    python pipelines/models/pretrain-maskedlm.py --help

Saved model checkpoints are available in the output directory.

Tensorboard
"""""""""""

The pretraining pipeline produces `TensorBoard
<https://www.tensorflow.org/tensorboard>`_-compatible logging in the output
directory. To host a TensorBoard server and monitor training progress, run:

.. code-block:: bash

    tensorboard --logdir maskedlm/

Inference
"""""""""

Coming soon...

Fine-Tuning (Contrastive Embeddings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Inference
"""""""""

Coming soon...

Fine-Tuning (Multi-Modal Summarization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Inference
"""""""""

Coming soon...
