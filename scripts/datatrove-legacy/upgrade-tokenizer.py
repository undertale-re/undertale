"""Upgrade a legacy tokenizer to the current format.

Apply padding and truncation settings to the saved tokenizer.
"""

import argparse

from tokenizers import Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="upgrade a pre-Dask tokenizer")
    parser.add_argument("input", required=True, help="path to legacy tokenizer file")
    parser.add_argument(
        "output", required=True, help="path for upgraded tokenizer file"
    )
    parser.add_argument(
        "-s",
        "--sequence-length",
        type=int,
        default=512,
        help="sequence length for padding and truncation",
    )

    arguments = parser.parse_args()

    tokenizer = Tokenizer.from_file(arguments.input)

    tokenizer.enable_padding(length=arguments.sequence_length)
    tokenizer.enable_truncation(max_length=arguments.sequence_length)

    tokenizer.save(arguments.output)
