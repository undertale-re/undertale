Modeling
--------

Pretoken Processing
^^^^^^^^^^^^^^^^^^^

Before a tokenizer can be trained on a dataset, disassembly must be processed
into pretokens that the tokenizer can consume. To pretokenize e.g., the
HumanEval-X dataset, run:

.. code:: bash

    python -m undertale.datasets.scripts.pretokenize humanevalx/ humanevalx-pretokenized/

Tokenizer Training
^^^^^^^^^^^^^^^^^^

Next, you can train a tokenizer on the pretokenized dataset:

.. code:: bash

    python -m undertale.models.item.tokenizer \
        humanevalx-pretokenized/ \
        item.tokenizer.json

Tokenization
^^^^^^^^^^^^

With your trained tokenizer you can now tokenize an entire dataset:

.. code:: bash

    python -m undertale.datasets.scripts.tokenize \
        -t item.tokenizer.json \
        -w pretraining \
        humanevalx-pretokenized/ \
        humanevalx-tokenized/

Pre-Training (Maked Language Modeling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With a trained tokenizer and a tokenized dataset, you can now proceed with the
first phase of training:

.. code:: bash

    python -m undertale.models.item.pretrain-maskedlm \
        -t item.tokenizer.json \
        humanevalx-tokenized/ \
        pretrain-maskedlm/

Inference
"""""""""

With a pre-trained model you can now do masked language modeling inference (for
a given pretokenized text):

.. code:: bash

    python -m undertale.models.item.infer-maskedlm \
        -t item.tokenizer.json \
        -c pretrain-maskedlm/version_0/checkpoints/model.ckpt \
        "xor rax [MASK]"

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
