import argparse
from types import SimpleNamespace

import torch
from torch import argmax, tensor, where

from . import tokenizer
from .finetune_summarization import SummarizeModel
from .model import TransformerEncoderForSequenceSummarization

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

    connector_config = SimpleNamespace(
        **{
            "model_type": "transformer",
            "prefix_size": 768,
            "prefix_length_const": 40,
            "prefix_length_assembly": 40,
            "num_layers": 8,
        }
    )

    inner = TransformerEncoderForSequenceSummarization(
        connector_config=connector_config, vocab_size=tok.get_vocab_size()
    )
    model = SummarizeModel(
        inner,
        prefix_length=connector_config.prefix_length_const,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    ckpt = torch.load(arguments.checkpoint, map_location=device)

    model.load_state_dict(ckpt["state_dict"], strict=True)

    encoded = tok.encode(arguments.input)
    tokens, mask = tensor(encoded.ids), tensor(encoded.attention_mask)

    with torch.no_grad():
        encoder_embedding = model.model.embed_assembly(
            tokens.unsqueeze(0), mask.unsqueeze(0)
        )
        encoder_embedding = encoder_embedding.mean(dim=1)

        prefixes = model.model.connector(encoder_embedding).view(
            -1,
            model.model.prefix_length_const,
            model.model.llm_embedding_size,
        )

        predicted = model.model.generate(
            embed=prefixes,
            entry_length=150,
            do_sample=False,
            temperature=1.0,
            beam=True,
            num_beams=5,
        )

    print(predicted)
