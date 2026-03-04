"""Predict masked tokens given a pretrained model."""

import argparse

from torch import argmax, tensor, where

from undertale.models import tokenizer
from undertale.models.maskedlm import InstructionTraceTransformerEncoderForMaskedLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict masked tokens in a piece of disassembly"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )

    parser.add_argument(
        "input", help="masked disassembly input to fill in (in pretokenized form)"
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = InstructionTraceTransformerEncoderForMaskedLM.load_from_checkpoint(
        arguments.checkpoint
    )

    encoded = tok.encode(arguments.input)
    tokens = tensor(encoded.ids).to(model.device)
    mask = tensor(encoded.attention_mask).to(model.device)

    output = model(tokens.unsqueeze(0), mask.unsqueeze(0)).squeeze()

    # TODO explore Top-K prediction with confidence scores from softmax logits.
    # top = topk(softmax(output, dim=-1), k=5)

    filled = where(
        tokens == tok.token_to_id(tokenizer.TOKEN_MASK), argmax(output, dim=-1), tokens
    )

    predicted = tok.decode(filled.tolist(), skip_special_tokens=False)
    predicted = predicted.replace(tokenizer.TOKEN_PAD, "").strip()

    print(predicted)
