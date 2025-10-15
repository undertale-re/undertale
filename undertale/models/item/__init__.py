"""The Instruction Trace Embedding Model (ITEM).

This is a custom BERT-like model built for binary code - it takes advantage of
unique data (structural, dataflow) in the binary code domain to improve
performance over a naive tranformer implementation.
"""

from . import tokenizer  # noqa: F401, F403
from .model import *  # noqa: F401, F403
from .model import __all__ as __model__

__all__ = ["tokenizer"] + __model__
