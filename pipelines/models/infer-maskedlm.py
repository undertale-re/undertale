"""Predict masked tokens given a pretrained model."""

import argparse

from torch import no_grad, tensor

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
    tokens = tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with no_grad():
        filled = model.infer(tokens, mask)

    predicted = tok.decode(filled.tolist(), skip_special_tokens=False)
    predicted = predicted.replace(tokenizer.TOKEN_PAD, "").strip()

    print(predicted)
