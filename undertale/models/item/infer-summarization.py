import argparse

import torch
from transformers import AutoTokenizer

from undertale.models import item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="summarize a given piece of assembly")

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument("-m", "--model", required=True, help="trained model file(s)")

    parser.add_argument("input", help="disassembly input to summarize")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="generation temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--maximum-tokens",
        type=int,
        default=20,
        help="maximum new tokens to generate (default: %(default)s)",
    )

    arguments = parser.parse_args()

    tokenizer = item.tokenizer.load(arguments.tokenizer)
    detokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = item.InstructionTraceConfig.from_tokenizer(tokenizer)

    model = item.InstructionTraceEncoderTransformerForSequenceSummarizationGPT2(config)

    # FIXME This is temporary to allow loading an embedding checkpoint
    model.item = model.item.from_pretrained(arguments.model, local_files_only=True)

    pretokens = item.tokenizer.preprocess({"disassembly": arguments.input})[
        "preprocessed"
    ]
    encoded = tokenizer.encode(pretokens)
    input_ids = torch.tensor((encoded.ids,), dtype=torch.long)
    attention_mask = torch.tensor((encoded.attention_mask,), dtype=torch.long)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=arguments.temperature,
        max_new_tokens=arguments.maximum_tokens,
    )
    response = detokenizer.batch_decode(generated)[0]

    print("=" * 80)
    print(arguments.input)
    print("-" * 80)
    print(response)
    print("=" * 80)
