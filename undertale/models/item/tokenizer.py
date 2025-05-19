import argparse
import collections
import logging
import re

import tokenizers

from ... import datasets
from ... import logging as undertale_logging

logger = logging.getLogger(__name__)


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


def preprocess(sample):
    """Preprocess a sample into pretokens.

    This takes a given sample's disassembly directly from Ghidra and parses it
    into pretokens - separating out arguments and immediates, etc.

    Arguments:
        sample: A sample from a function dataset. Must contain the
            `disassembly` field, as specified in the schema.

    Returns:
        A dictionary with the key `preprocessed` containing pretokenized
        disassembly.
    """

    pretokens = []

    for instruction in sample["disassembly"].split("\n"):
        split = instruction.split(" ", maxsplit=1)

        # Instruction prefix (e.g., 'lock add ...')
        if split[0] in ["lock"]:
            assert len(split) == 2
            prefix, remainder = split

            pretokens.append(prefix)

            split = remainder.split(" ", maxsplit=1)

        # Instruction without operands (e.g., `ret`).
        if len(split) == 1:
            pretokens.append(split[0])
            continue
        else:
            mnemonic, operands = split

        pretokens.append(mnemonic)

        for operand in operands.split(", "):
            # Immediate value (e.g., `0x1337`).
            if operand.startswith("0x") or operand.startswith("-0x"):
                operand = str(int(operand, 16))
                pretokens.append(operand)
            # Memory address (e.g., `[rax]`).
            elif "[" in operand:
                # Size directive (e.g., `byte btr [rax]`).
                if "ptr" in operand:
                    size, _, operand = operand.split(maxsplit=2)
                    pretokens.append(size)
                # Segment indicator (e.g., `ds:[rax]`).
                if ":" in operand:
                    segment, operand = operand.split(":")
                    pretokens.append(segment)

                assert operand[0] == "["
                assert operand[-1] == "]"
                operand = operand[1:-1]

                pretokens.append("[")

                # Base, offset, multiplier syntax
                split = re.split(r"(\+|-|\*)", operand)
                split = [o.strip() for o in split]

                for op in split:
                    # Immediate value.
                    if op.startswith("0x") or op.startswith("-0x"):
                        op = str(int(op, 16))

                    pretokens.append(op)

                pretokens.append("]")
            # Everything else should be a register.
            else:
                assert " " not in operand

                pretokens.append(operand)

        pretokens.append(TOKEN_NEXT)

    # Remove final NEXT token.
    if pretokens and pretokens[-1] == TOKEN_NEXT:
        pretokens.pop()

    return {"preprocessed": " ".join(pretokens)}


def preprocess_batch(batch):
    result = {"preprocessed": []}
    for sample in batch["disassembly"]:
        result["preprocessed"].append(
            preprocess({"disassembly": sample})["preprocessed"]
        )

    return result


def train(dataset, vocab_size=4096):
    """Train our custom tokenizer on a given dataset.

    This tokenizer essentially computes a dictionary of tokens for all
    instruction mnemonics and registers present in the given dataset and then
    trains a byte pair encoding (BPE) model to represent immediate values to
    constrain the size of the dataset.

    Arguments:
        dataset: The dataset on which to train. This dataset should conform to
            the `Function` schema.
        vocab_size: The vocabulary size for the immediate BPE model. This is a
            hyperparameter that could be tuned to optimize the token
            representation.

    Returns:
        A trained tokenizer.
    """

    logger.info(f"preprocessing {dataset.__class__.__name__}")

    dataset = dataset.map(preprocess, desc="preprocessing")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=TOKEN_UNKNOWN))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        vocab_size=vocab_size,
        continuing_subword_prefix="__",
    )

    logger.info("training tokenizer")

    # Build a token dictionary, separating out immediate values.
    tokens, immediates = collections.defaultdict(int), collections.defaultdict(int)

    def build_dictionary(sample):
        for token in sample["preprocessed"].split():
            try:
                int(token)
                immediates[token] += 1
            except ValueError:
                tokens[token] += 1

    dataset.map(build_dictionary, desc="compute dictionary")

    def build_tokenizer_trainer(dictionary):
        for token, count in dictionary.items():
            for _ in range(count):
                yield token

    tokenizer.train_from_iterator(build_tokenizer_trainer(immediates), trainer=trainer)

    tokenizer.add_tokens(list(tokens))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    return tokenizer


def load(path, sequence_length=512):
    tokenizer = tokenizers.Tokenizer.from_file(path)
    tokenizer.enable_padding(length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)

    return tokenizer


__all__ = ["train", "load"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train a tokenizer on preprocessed binary data"
    )

    parser.add_argument(
        "dataset",
        help="dataset on which to train the tokenizer (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("-o", "--output", required=True, help="output file")

    parser.add_argument(
        "-l",
        "--logging-level",
        choices=undertale_logging.LEVELS,
        default="info",
        help="logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--logging-file", default=None, help="logging file (default: %(default)s)"
    )

    arguments = parser.parse_args()

    undertale_logging.setup_logging(
        level=undertale_logging.LEVELS[arguments.logging_level],
        file=arguments.logging_file,
    )

    try:
        dataset = datasets.from_specifier(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    tokenizer = train(dataset)
    tokenizer.save(arguments.output)

    logger.info(f"saved tokenizer to: {arguments.output}")
