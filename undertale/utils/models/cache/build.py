"""Prepare and export the HuggingFace cache.

This script downloads all the necessary models, datasets, tokenizers, code,
etc. required by any of our pipelines and exports them to a specified directory
for offline use.
"""

import os
from argparse import ArgumentParser
from shutil import copytree
from tempfile import TemporaryDirectory

from ....logging import get_logger, setup_logging

logger = get_logger(__name__)


def build(path: str) -> None:
    """Prepare and export the HuggingFace cache.

    Arguments:
        path: Path where the cache should be written after download.
    """

    with TemporaryDirectory() as working:
        os.environ["HF_HOME"] = working

        # Imports must come after HF_HOME is set.
        #
        # The libraries read it at import time.
        import evaluate
        from transformers import (
            AutoModel,
            AutoTokenizer,
            GPT2Config,
            GPT2LMHeadModel,
            GPT2Tokenizer,
        )

        from ....models.summarization import (
            InstructionTraceTransformerEncoderForSequenceSummarizationGPT2,
        )

        model = InstructionTraceTransformerEncoderForSequenceSummarizationGPT2.LANGUAGE
        logger.info(f"downloading GPT2 model ({model!r})")
        GPT2Config.from_pretrained(model)
        GPT2LMHeadModel.from_pretrained(model)
        GPT2Tokenizer.from_pretrained(model)

        logger.info("downloading Rouge metric")
        evaluate.load("rouge")

        logger.info("downloading BERTScore metric")
        evaluate.load("bertscore")

        model = InstructionTraceTransformerEncoderForSequenceSummarizationGPT2.BERTSCORE
        logger.info(f"downloading BERTScore model ({model})")
        AutoTokenizer.from_pretrained(model)
        AutoModel.from_pretrained(model)

        copytree(working, path, dirs_exist_ok=True)

    logger.info(f"cache written to {path!r}")


if __name__ == "__main__":
    parser = ArgumentParser(description="download and export the HuggingFace cache")

    parser.add_argument("output", help="output directory where cache should be written")

    arguments = parser.parse_args()

    setup_logging()

    build(arguments.output)
