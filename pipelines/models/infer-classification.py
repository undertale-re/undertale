"""Classify a piece of disassembly using a trained model."""

import argparse

from torch import argmax, no_grad, tensor

from undertale.models import tokenizer
from undertale.models.classification import (
    InstructionTraceTransformerEncoderForSequenceClassification,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="classify a piece of disassembly using a trained model"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )

    parser.add_argument(
        "input", help="disassembly input to classify (in pretokenized form)"
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = InstructionTraceTransformerEncoderForSequenceClassification.load_from_checkpoint(
        arguments.checkpoint
    )
    model.eval()

    encoded = tok.encode(arguments.input)
    tokens = tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with no_grad():
        logits = model(tokens, mask)

    prediction = argmax(logits, dim=-1).item()

    print(prediction)
