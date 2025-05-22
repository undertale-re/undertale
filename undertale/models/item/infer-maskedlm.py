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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tokenizer = item.tokenizer.load(arguments.tokenizer)
    config = item.InstructionTraceConfig.from_tokenizer(tokenizer)
    model = item.InstructionTraceEncoderTransformerForMaskedLM(config)

    model = model.from_pretrained(arguments.model, local_files_only=True)
    model = model.to(device)
    model.eval()

    pretokens = item.tokenizer.pretokenize(arguments.input)
    encoded = tokenizer.encode(pretokens)
    
    input_ids = torch.tensor((encoded.ids,), dtype=torch.long)
    input_ids = input_ids.to(device)
    
    attention_mask = torch.tensor((encoded.attention_mask,), dtype=torch.long)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    tokens = torch.argmax(output.logits[0], dim=-1)

    decoded = tokenizer.decode(list(tokens))

    print(decoded)
