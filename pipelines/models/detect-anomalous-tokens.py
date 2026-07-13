"""Detect anomalies in a piece of disassembly using a trained density model."""

import argparse

from torch import no_grad, tensor

from undertale.models import tokenizer
from undertale.models.density import InstructionTraceTransformerEncoderForDensity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="detect anomalies in a piece of disassembly using a trained density model"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )
    parser.add_argument(
        "-r",
        "--threshold",
        type=float,
        default=0.0,
        help="anomaly score threshold above which a token is flagged",
    )
    parser.add_argument(
        "input", help="disassembly input to inspect (in pretokenized form)"
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = InstructionTraceTransformerEncoderForDensity.load_from_checkpoint(
        arguments.checkpoint
    )
    model.eval()

    encoded = tok.encode(arguments.input)
    tokens = tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with no_grad():
        scores = model.score(tokens, mask)
        detected = model.detect(tokens, mask, threshold=arguments.threshold)

    for token, score, anomalous in zip(
        encoded.tokens, scores.squeeze(0).tolist(), detected.squeeze(0).tolist()
    ):
        flag = "*" if anomalous else " "
        print(f"{flag} {score:8.4f}  {token}")
