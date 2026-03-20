"""Upgrade a legacy MaskedLM checkpoint to the current format."""

import argparse

import torch

from undertale.models.maskedlm import InstructionTraceTransformerEncoderForMaskedLM
from undertale.models.tokenizer import TOKEN_MASK, TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="upgrade a legacy MaskedLM checkpoint to the current format"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument("input", required=True, help="path to the old checkpoint")
    parser.add_argument(
        "output", required=True, help="path for the upgraded checkpoint"
    )

    arguments = parser.parse_args()

    tok = load_tokenizer(arguments.tokenizer)
    next_token_id = tok.token_to_id(TOKEN_NEXT)
    mask_token_id = tok.token_to_id(TOKEN_MASK)

    if next_token_id is None:
        raise ValueError(f"token '{TOKEN_NEXT}' not found in tokenizer")
    if mask_token_id is None:
        raise ValueError(f"token '{TOKEN_MASK}' not found in tokenizer")

    checkpoint = torch.load(arguments.input, map_location="cpu", weights_only=False)

    hparams = checkpoint["hyper_parameters"]
    hparams["sequence_length"] = hparams.pop("input_size")
    hparams["next_token_id"] = next_token_id
    hparams["mask_token_id"] = mask_token_id

    torch.save(checkpoint, arguments.output)

    print(f"saved upgraded checkpoint to {arguments.output}")

    model = InstructionTraceTransformerEncoderForMaskedLM.load_from_checkpoint(
        arguments.output
    )

    print(f"validation passed — model loaded successfully ({type(model).__name__})")
