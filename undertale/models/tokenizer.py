"""Tokenizer implementation and utilities."""

import json
from collections import defaultdict
from typing import Dict, List

from pandas import Series
from pandas import read_parquet as pandas_read_parquet
from pandera.errors import SchemaError as PanderaSchemaError
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from ..exceptions import SchemaError
from ..logging import get_logger
from ..schema import DisassembledFunctionDataset
from ..utils import (
    assert_path_exists,
    enforce_extension,
    get_or_create_file,
    write_parquet,
)

logger = get_logger(__name__)


TOKEN_PAD = "[PAD]"
TOKEN_UNKNOWN = "[UNK]"
TOKEN_SEPARATOR = "[SEP]"
TOKEN_CLASSIFY = "[CLS]"
TOKEN_MASK = "[MASK]"
TOKEN_NEXT = "[NEXT]"

SPECIAL_TOKENS = [
    TOKEN_PAD,
    TOKEN_UNKNOWN,
    TOKEN_SEPARATOR,
    TOKEN_CLASSIFY,
    TOKEN_MASK,
    TOKEN_NEXT,
]


def preprocess_tokens(input: str, output: str) -> str:
    """Preprocess a dataset of disassembly.

    Arguments:
        input: Path to disassembly dataset.
        output: Path where the preprocessed tokens should be written.

    Returns:
        The path to the preprocessed tokens file.
    """

    output = enforce_extension(output, ".json")

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    frame = pandas_read_parquet(input)

    try:
        DisassembledFunctionDataset.validate(frame)
    except PanderaSchemaError as e:
        logger.error("dataset does not match the expected schema")
        raise SchemaError(str(e))

    logger.info(f"preprocessing tokens from {input!r}")

    tokens: Dict[str, int] = defaultdict(int)
    immediates: Dict[str, int] = defaultdict(int)

    for sample in frame["disassembly"]:
        for token in sample.split():
            try:
                int(token)
                immediates[token] += 1
            except ValueError:
                tokens[token] += 1

    with open(output, "w") as f:
        json.dump({"tokens": tokens, "immediates": immediates}, f)

    return output


def merge_preprocessed_tokens(inputs: List[str], output: str) -> str:
    """Merge preprocessed token files into a single file for training.

    Arguments:
        inputs: Paths to preprocessed token files.
        output: Merged output path.

    Returns:
        The path to the merged preprocessed token file.
    """

    output = enforce_extension(output, ".json")

    for i, input in enumerate(inputs):
        inputs[i] = assert_path_exists(input)

    output, created = get_or_create_file(output)

    if not created:
        return output

    logger.info(f"merging {len(inputs)} preprocessed token files to {output!r}")

    def add(first, second):
        for k in second:
            first[k] = first.get(k, 0) + second[k]

    merged: Dict[str, Dict[str, int]] = {"tokens": {}, "immediates": {}}
    for input in inputs:
        with open(input, "r") as f:
            loaded = json.load(f)

        add(merged["tokens"], loaded["tokens"])
        add(merged["immediates"], loaded["immediates"])

    with open(output, "w") as f:
        json.dump(merged, f)

    return output


def save(tokenizer: Tokenizer, path: str) -> None:
    """Save a trained tokenizer to a file.

    Arguments:
        tokenizer: A trained tokenizer.
        path: The path where trained tokenizer should be saved.
    """

    tokenizer.save(path)

    logger.info(f"tokenizer saved to {path!r}")


def train_tokenizer(
    input: str,
    output: str,
    sequence_length: int = 512,
    vocabulary_size: int = 4096,
    silent: bool = True,
) -> str:
    """Train a tokenizer on a given dataset.

    This tokenizer essentially computes a dictionary of tokens for all
    instruction mnemonics and registers present in the given dataset and then
    trains a byte pair encoding (BPE) model to represent immediate values to
    constrain the size of the dataset.

    Arguments:
        input: The path to the preprocessed token file on which to train.
        output: The path where the trained tokenizer file should be saved.
        sequence_length: The sequence length for padding and truncation.
        vocabulary_size: The vocabulary size for the immediate BPE model. This is a
            hyperparameter that could be tuned to optimize the token
            representation.
        silent: If ``True``, suppress progress bar display.

    Returns:
        The path to the trained tokenizer file.
    """

    output = enforce_extension(output, ".json")

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    tokenizer = Tokenizer(BPE(unk_token=TOKEN_UNKNOWN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        vocab_size=vocabulary_size,
        continuing_subword_prefix="__",
        show_progress=not silent,
    )

    logger.info(f"loading preprocessed tokens from {input!r}")

    with open(input, "r") as f:
        preprocessed = json.load(f)

    def build_tokenizer_trainer(dictionary):
        for token, count in dictionary.items():
            for _ in range(count):
                yield token

    logger.info(
        f"training tokenizer over {len(preprocessed['tokens']) + len(preprocessed['immediates'])} unique tokens"
    )

    tokenizer.train_from_iterator(
        build_tokenizer_trainer(preprocessed["immediates"]), trainer=trainer
    )

    tokenizer.add_tokens(list(preprocessed["tokens"]))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    tokenizer.enable_padding(length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)

    save(tokenizer, output)

    return output


def load(path: str) -> Tokenizer:
    """Load a trained tokenizer from a file.

    Arguments:
        path: The path to a trained tokenizer file to load.

    Returns:
        A trained tokenizer loaded from ``path``.
    """

    logger.info(f"loading tokenizer from {path!r}")

    tokenizer = Tokenizer.from_file(path)

    return tokenizer


def tokenize(input: str, output: str, tokenizer: str) -> str:
    """Tokenize a given dataset with a trained tokenizer.

    Arguments:
        input: Path to disassembly dataset.
        output: Path where the tokenized dataset should be written.
        tokenizer: Path to the trained tokenizer that should be used.

    Returns:
        The path to the tokenized dataset.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    tok = load(tokenizer)

    frame = pandas_read_parquet(input)

    try:
        DisassembledFunctionDataset.validate(frame)
    except PanderaSchemaError as e:
        logger.error("dataset does not match the expected schema")
        raise SchemaError(str(e))

    logger.info(f"tokenizing {input!r} to {output!r}")

    def process(disassembly: str) -> Series:
        encoding = tok.encode(disassembly)

        return Series(
            {"input_ids": encoding.ids, "attention_mask": encoding.attention_mask}
        )

    frame[["input_ids", "attention_mask"]] = frame["disassembly"].apply(process)
    write_parquet(frame, output)

    logger.info(f"successfully tokenized {len(frame)} rows")

    return output


__all__ = [
    "TOKEN_PAD",
    "TOKEN_UNKNOWN",
    "TOKEN_SEPARATOR",
    "TOKEN_CLASSIFY",
    "TOKEN_MASK",
    "TOKEN_NEXT",
    "preprocess_tokens",
    "merge_preprocessed_tokens",
    "train_tokenizer",
    "tokenize",
    "save",
    "load",
]
