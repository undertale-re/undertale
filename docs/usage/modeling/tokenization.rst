Tokenization
------------

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

See :ref:`parallelism` for controlling parallel workers and cluster backends.

Tokenization
^^^^^^^^^^^^

With your trained tokenizer you can now tokenize an entire dataset to prepare
for pre-training.

.. code-block:: bash

    # Tokenize the HumanEval-X dataset.
    #
    # Only retain the minimal fields necessary for pre-training.
    python pipelines/models/tokenize-disassembly.py \
        humaneval-x/ \
        humaneval-x-pretraining \
        --tokenizer tokenizer.json \
        --minimal

Consider :ref:`splitting <dataset-splitting>` off some (10%) of your dataset
for validation.

See :ref:`parallelism` for controlling parallel workers and cluster backends.
