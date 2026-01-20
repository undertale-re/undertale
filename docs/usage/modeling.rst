Modeling
--------

Tokenizer Training
^^^^^^^^^^^^^^^^^^

The first step in training any of our models is to train a tokenizer. To train
a tokenizer on e.g., the HumanEval-X dataset, run the tokenizer training
pipeline script:

.. code-block:: bash

    # Train a tokenizer on the HumanEval-X dataset.
    python pipelines/model-tokenizer-train.py \
        humaneval-x/ \
        tokenizer

    # Train a tokenizer on the HumanEval-X dataset in parallel.
    python pipelines/model-tokenizer-train.py \
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
    python pipelines/model-tokenizer-tokenize.py \
        humaneval-x/ \
        humaneval-x-pretraining \
        --tokenizer tokenizer.json \
        --minimal

    # Tokenize the HumanEval-X dataset in parallel.
    #
    # Retain all fields this time, as an example.
    python pipelines/model-tokenizer-tokenize.py \
        humaneval-x/ \
        humaneval-x-tokenized \
        --tokenizer tokenizer.json \
        --parallelism 8

Pre-Training (Maked Language Modeling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

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
