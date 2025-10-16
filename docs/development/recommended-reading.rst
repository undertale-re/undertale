Recommended Reading
-------------------

We use a lot of external libraries in Undertale - if you're not already pretty
familiar with the following, it's worth reading through their documentation and
possibly completing their tutorial(s) before contributing.

`datatrove <https://github.com/huggingface/datatrove>`_
    The dataset building pipeline library from the folks at HuggingFace. We use
    this to codify all of our dataset building pipelines and parallelize them
    across compute infrastructure.

`PyTorch <https://pytorch.org/>`_
    The deep learning library. If you're not already deeply familiar with
    PyTorch, the textbook `Deep Learning with Pytorch
    <https://isip.piconepress.com/courses/temple/ece_4822/resources/books/Deep-Learning-with-PyTorch.pdf>`_
    is an excellent resource.

`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_
    All of our models are written in PyTorch and wrapped in Lightning modules.
    We largely let Lightning handle the complexities of multi-node, multi-GPU
    training, validation, and integration with tensorboard for monitoring
    training.

`Tensorboard <https://www.tensorflow.org/tensorboard>`_
    The visualization tool we use for tracking training runs.

`Sphinx <https://www.sphinx-doc.org/en/master/>`_
    A software documentation library. All of our documentation is written in
    `reStructuredText
    <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
    and built with Sphinx. We also use `autodoc
    <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ with
    Google-style Python docstrings for automatically generated reference
    documentation.

`pyinstrument <https://pyinstrument.readthedocs.io/en/latest/>`_
    The statistical profiler for Python. We use this occasionally for
    performance testing.
