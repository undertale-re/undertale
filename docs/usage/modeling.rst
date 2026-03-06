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

See :ref:`parallelism` for controlling parallel workers and cluster backends.

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

Consider :ref:`splitting <dataset-splitting>` off some (10%) of your dataset
for validation.

See :ref:`parallelism` for controlling parallel workers and cluster backends.

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

    # Include validation data (pre-split).
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

    # Distributed training.
    python pipelines/models/pretrain-maskedlm.py \
        --strategy ddp \
        --nodes 8 \
        --devices 2 \
        --tokenizer tokenizer.json \
        humaneval-x-pretraining-training/ \
        --validation humaneval-x-pretraining-validation/ \
        maskedlm

There are several other configurable parameters for other training scenarios -
to get a full list, see the ``--help`` output.

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

With a trained model checkpoint, you can predict masked tokens in a piece of
disassembly input.

.. code-block:: bash

    # Predict masked tokens in a piece of disassembly.
    python pipelines/models/infer-maskedlm.py \
        --tokenizer tokenizer.json \
        --checkpoint maskedlm/checkpoint.ckpt \
        "xor rax [MASK]"

Fine-Tuning (Multi-Modal Summarization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Inference
"""""""""

Coming soon...
