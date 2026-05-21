Fine-Tuning (Sequence Classification)
--------------------------------------

With a tokenized dataset that includes integer class labels and a pre-trained
masked LM checkpoint, you are ready to fine-tune for sequence classification.

.. code-block:: bash

    # Start a fine-tuning run locally (as an example).
    #
    # Results will be written to classification/.
    python pipelines/models/finetune-classification.py \
        --tokenizer tokenizer.json \
        --pretrained maskedlm/checkpoint.ckpt \
        --classes 10 \
        dataset/ \
        classification

    # Include validation data (pre-split).
    python pipelines/models/finetune-classification.py \
        --tokenizer tokenizer.json \
        --pretrained maskedlm/checkpoint.ckpt \
        --classes 10 \
        dataset-training/ \
        --validation dataset-validation/ \
        classification

    # Use multiple accelerators on the same host.
    python pipelines/models/finetune-classification.py \
        --devices 4 \
        --tokenizer tokenizer.json \
        --pretrained maskedlm/checkpoint.ckpt \
        --classes 10 \
        dataset-training/ \
        --validation dataset-validation/ \
        classification

    # Distributed training on a SLURM Cluster.
    #
    # This SLURM script requires certain environment variables
    # to be configured - see `environments/example-slurm.env`
    # for more details or customize the SLURM script to your
    # environment.
    source environments/example-slurm.env
    sbatch pipelines/models/finetune-classification.slurm

There are several other configurable parameters for other training scenarios -
to get a full list, see the ``--help`` output.

Saved model checkpoints are available in the output directory.

See :ref:`environments` for details on configuring the local environment -
in particular for distributed SLURM training.

Inference
^^^^^^^^^

With a trained model checkpoint, you can predict the class of a piece of
disassembly input.

.. code-block:: bash

    # Predict the class of a piece of disassembly.
    python pipelines/models/infer-classification.py \
        --tokenizer tokenizer.json \
        --checkpoint classification/checkpoint.ckpt \
        "push rbp [NEXT] mov rbp rsp [NEXT] ..."
