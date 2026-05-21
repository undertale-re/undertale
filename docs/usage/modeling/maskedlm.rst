Pre-Training (Masked Language Modeling)
---------------------------------------

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

See :ref:`environments` for details on configuring the local environment -
in particular for distributed SLURM training.

Tensorboard
^^^^^^^^^^^

The pretraining pipeline produces `TensorBoard
<https://www.tensorflow.org/tensorboard>`_-compatible logging in the output
directory. To host a TensorBoard server and monitor training progress, run:

.. code-block:: bash

    tensorboard --logdir maskedlm/

Inference
^^^^^^^^^

With a trained model checkpoint, you can predict masked tokens in a piece of
disassembly input.

.. code-block:: bash

    # Predict masked tokens in a piece of disassembly.
    python pipelines/models/infer-maskedlm.py \
        --tokenizer tokenizer.json \
        --checkpoint maskedlm/checkpoint.ckpt \
        "xor rax [MASK]"
