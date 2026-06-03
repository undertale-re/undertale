Fine-Tuning (Multi-Modal Sequence Summarization)
------------------------------------------------

Prerequisites
^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

With an existing dataset, you can tokenize the natural language summaries to
prepare for fine-tuning. This step uses a GPT-2 tokenizer rather than your
trained disassembly tokenizer.

.. code-block:: bash

    # Tokenize summaries in the HumanEval-X dataset.
    python pipelines/models/tokenize-summaries.py \
        humaneval-x/ \
        humaneval-x-summaries

Consider :ref:`splitting <dataset-splitting>` off some (10%) of your dataset
for validation.

See :ref:`parallelism` for controlling parallel workers and cluster backends.

Training
^^^^^^^^

With your tokenized summary dataset (and optional validation split) you are
now ready to fine-tune for summarization.

.. code-block:: bash

    # Start a fine-tuning run locally (as an example).
    #
    # Results will be written to summarization/.
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        humaneval-x-summaries/ \
        summarization

    # Initialize from a pre-trained masked LM checkpoint.
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        --pretrained maskedlm/checkpoint.ckpt \
        humaneval-x-summaries/ \
        summarization

    # Include validation data (pre-split).
    python pipelines/models/finetune-summarization.py \
        --tokenizer tokenizer.json \
        --pretrained maskedlm/checkpoint.ckpt \
        humaneval-x-summaries-training/ \
        --validation humaneval-x-summaries-validation/ \
        summarization

    # Use multiple accelerators on the same host.
    python pipelines/models/finetune-summarization.py \
        --devices 4 \
        --tokenizer tokenizer.json \
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
^^^^^^^^^

With a trained model checkpoint, you can generate a natural-language summary
of a piece of disassembly input.

.. code-block:: bash

    # Generate a summary for a piece of disassembly.
    python pipelines/models/infer-summarization.py \
        --tokenizer tokenizer.json \
        --checkpoint summarization/checkpoint.ckpt \
        "push rbp [NEXT] mov rbp rsp [NEXT] ..."

Evaluation
^^^^^^^^^^

After training, you can evaluate a checkpoint against a tokenized summary
dataset using Rouge-L and BERTScore. The pipeline runs inference and scoring
in parallel across shards and writes a single averaged JSON result.

.. code-block:: bash

    # Evaluate a checkpoint locally.
    python pipelines/models/evaluate-summarization.py \
        --tokenizer tokenizer.json \
        --checkpoint summarization/checkpoint.ckpt \
        humaneval-x-summaries/ \
        summarization-eval

    # Scale evaluation across a SLURM cluster.
    #
    # This requires certain environment variables to be configured -
    # see `environments/example-slurm.env` for more details.
    source environments/example-slurm.env
    python pipelines/models/evaluate-summarization.py \
        --cluster slurm \
        --parallelism 8 \
        --tokenizer tokenizer.json \
        --checkpoint summarization/checkpoint.ckpt \
        humaneval-x-summaries/ \
        summarization-eval

Results are written to ``summarization-eval`` as a JSON file with the
following structure:

.. code-block:: json

    {"rouge-l": 0.0, "bertscore": 0.0}

See :ref:`parallelism` for controlling parallel workers and cluster backends.
