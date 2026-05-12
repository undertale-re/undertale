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
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

        logger.info("downloading GPT2 configuration")
        GPT2Config.from_pretrained("gpt2")

        logger.info("downloading GPT2 model")
        GPT2LMHeadModel.from_pretrained("gpt2")

        logger.info("downloading GPT2 tokenizer")
        GPT2Tokenizer.from_pretrained("gpt2")

        logger.info("downloading Rouge metric")
        evaluate.load("rouge")

        copytree(working, path, dirs_exist_ok=True)

    logger.info(f"cache written to {path!r}")


if __name__ == "__main__":
    parser = ArgumentParser(description="download and export the HuggingFace cache")

    parser.add_argument("output", help="output directory where cache should be written")

    arguments = parser.parse_args()

    setup_logging()

    build(arguments.output)
