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
    python pipelines/models/tokenize-disassembly.py \
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

    # Distributed training on a SLURM Cluster.
    #
    # This SLURM script requires certain environment variables
    # to be configured - see `environments/example-slurm.env` 
    # for more details or customize the SLURM script to your 
    # environment.
    source environments/example-slurm.env
    sbatch pipelines/models/pretrain-maskedlm.slurm

There are several other configurable parameters for other training scenarios -
to get a full list, see the ``--help`` output.

Saved model checkpoints are available in the output directory.

See :ref:`environments` for for details on configuring the local environment -
in particular for distributed SLURM training.

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

Fine-Tuning (Sequence Classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Inference
"""""""""

Coming soon...

Fine-Tuning (Multi-Modal Sequence Summarization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites
"""""""""""""

The multimodal dataset and model pipelines in this section use trained models
and tokenizers from the `HuggingFace Hub
<https://huggingface.co/docs/hub/index>`_. These will be downloaded and cached
automatically as necessary.

If your pipelines do not have access to the internet, you can download a cache
of all required models for offline use with the following utility:

.. code-block:: bash

    python -m undertale.utils.models.cache.build path/to/output

Then, on your offline system, you can load the cache with the following:

.. code-block:: bash

    python -m undertale.utils.models.cache.load path/to/cache

For offline systems, you should also consider setting the following environment
variable(s):

.. code-block:: bash

    export HF_HUB_OFFLINE=1


Dataset Preparation
"""""""""""""""""""

With an existing dataset, you can tokenize the natural language summaries to
prepare for fine-tuning. This step uses a GPT-2 tokenizer rather than your
trained disassembly tokenizer.

.. code-block:: bash

    # Tokenize summaries in the HumanEval-X dataset.
    python pipelines/models/tokenize-summaries.py \
        humaneval-x/ \
        humaneval-x-summaries \
        --tokenizer gpt2/

Consider :ref:`splitting <dataset-splitting>` off some (10%) of your dataset
for validation.

See :ref:`parallelism` for controlling parallel workers and cluster backends.

Training
""""""""

With your tokenized summary dataset (and optional validation split) you are
now ready to fine-tune for summarization.

.. code-block:: bash

    # Start a fine-tuning run locally (as an example).
    #
    # Results will be written to summarization/.
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        --language-config gpt2/ \
        humaneval-x-summaries/ \
        summarization

    # Initialize from a pre-trained masked LM checkpoint.
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        --language-config gpt2/ \
        --pretrained maskedlm/checkpoint.ckpt \
        humaneval-x-summaries/ \
        summarization

    # Include validation data (pre-split).
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        --language-config gpt2/ \
        --pretrained maskedlm/checkpoint.ckpt \
        humaneval-x-summaries-training/ \
        --validation humaneval-x-summaries-validation/ \
        summarization

    # Use multiple accelerators on the same host.
    python pipelines/models/finetune-summarization.py \
        --devices 4 \
        --tokenizer tokenizer.json \
        --language-config gpt2/ \
        --pretrained maskedlm/checkpoint.ckpt \
        humaneval-x-summaries-training/ \
        --validation humaneval-x-summaries-validation/ \
        summarization

    # Distributed training on a SLURM Cluster.
    #
    # This SLURM script requires certain environment variables
    # to be configured - see `environments/example-slurm.env`
    # for more details or customize the SLURM script to your
    # environment.
    source environments/example-slurm.env
    sbatch pipelines/models/finetune-summarization.slurm

There are several other configurable parameters for other training scenarios -
to get a full list, see the ``--help`` output.

Saved model checkpoints are available in the output directory.

See :ref:`environments` for details on configuring the local environment -
in particular for distributed SLURM training.

Inference
"""""""""

With a trained model checkpoint, you can generate a natural-language summary
of a piece of disassembly input.

.. code-block:: bash

    # Generate a summary for a piece of disassembly.
    python pipelines/models/infer-summarization.py \
        --tokenizer tokenizer.json \
        --checkpoint summarization/checkpoint.ckpt \
        "push rbp [NEXT] mov rbp rsp [NEXT] ..."
