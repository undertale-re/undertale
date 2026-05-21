"""Generate a summary using a trained model."""

import argparse

from torch import no_grad, tensor
from transformers import GPT2Tokenizer

from undertale.models import tokenizer
from undertale.models.summarization import (
    InstructionTraceTransformerEncoderForSequenceSummarizationGPT2,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="summarize a piece of disassembly using a trained model"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )

    parser.add_argument(
        "input", help="disassembly input to summarize (in pretokenized form)"
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = InstructionTraceTransformerEncoderForSequenceSummarizationGPT2.load_from_checkpoint(
        arguments.checkpoint
    )
    model.eval()

    language_tokenizer = GPT2Tokenizer.from_pretrained(model.LANGUAGE)

    encoded = tok.encode(arguments.input)
    tokens = tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with no_grad():
        generated = model.generate(tokens, mask)

    summary = language_tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    print(summary)
