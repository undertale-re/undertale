"""Tokenization utilities for the ITEM model."""

import logging

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class ITEMPretokenizer(PipelineStep):
    """Preprocesses disassembly for the tokenizer.

    Input:
        Disassembled code with a `disassembly` field that can be pre-tokenized.

    Output:
        Replaces the current `disassembly` field with preprocessed disassembly.
    """

    type = "✂️ - FORMATTER"
    name = "⚙️ ITEM Pretokenizer"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""
        from undertale.models.item.tokenizer import pretokenize

        if not data:
            return

        for document in data:
            with self.track_time():
                try:
                    document.metadata["disassembly"] = pretokenize(
                        document.metadata["disassembly"]
                    )

                    self.stat_update("succeeded")
                except Exception as e:
                    logger.exception(f"failed to process '{document.id}': {e}")
                    self.stat_update("failed")
                    continue

                yield document


class ITEMTokenizer(PipelineStep):
    """Tokenize preprocessed disassembly.

    Arguments:
        tokenizer: The path to the trained tokenizer.

    Input:
        Disassembled code with a `disassembly` field that has already been pre-tokenized.

    Output:
        Adds a `tokens` field with disassembly tokens.
    """

    def __init__(self, tokenizer: str):
        super().__init__()

        self.tokenizer = tokenizer

    type = "✂️ - FORMATTER"
    name = "🎟️ ITEM Tokenizer"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""
        from undertale.models.item import tokenizer

        if not data:
            return

        tok = tokenizer.load(self.tokenizer)

        for document in data:
            with self.track_time():
                encoding = tok.encode(document.metadata["disassembly"])

                document.metadata["tokens"] = {
                    "input_ids": encoding.ids,
                    "attention_mask": encoding.attention_mask,
                }

                yield document
