import argparse

import torch

from undertale.models import item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict masked tokens in a piece of disassembly"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument("-m", "--model", required=True, help="trained model file(s)")

    parser.add_argument("input", help="masked disassembly input to fill in")

    arguments = parser.parse_args()

    tokenizer = item.tokenizer.load(arguments.tokenizer)
    config = item.InstructionTraceConfig.from_tokenizer(tokenizer)
    model = item.InstructionTraceEncoderTransformerForMaskedLM(config)

    model = model.from_pretrained(arguments.model, local_files_only=True)
    model.eval()

    pretokens = item.tokenizer.preprocess({"disassembly": arguments.input})[
        "preprocessed"
    ]
    encoded = tokenizer.encode(pretokens)
    input_ids = torch.tensor((encoded.ids,), dtype=torch.long)
    attention_mask = torch.tensor((encoded.attention_mask,), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    tokens = torch.argmax(output.logits[0], dim=-1)

    decoded = tokenizer.decode(list(tokens))

    print(decoded)
