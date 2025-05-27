import argparse

from transformers import FillMaskPipeline, PreTrainedTokenizerFast

from undertale.models import item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict masked tokens in a piece of disassembly"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument("-m", "--model", required=True, help="trained model file(s)")

    parser.add_argument(
        "input", help="masked disassembly input to fill in (in pretokenized form)"
    )

    arguments = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=arguments.tokenizer,
        mask_token=item.tokenizer.TOKEN_MASK,
        unk_token=item.tokenizer.TOKEN_UNKNOWN,
        pad_token=item.tokenizer.TOKEN_PAD,
    )

    model = item.InstructionTraceEncoderTransformerForMaskedLM.from_pretrained(
        arguments.model, local_files_only=True
    )
    model.eval()

    pipeline = FillMaskPipeline(model, tokenizer)

    for result in pipeline(arguments.input):
        print("=" * 80)
        print(f"token: {result['token_str']!r}")
        print(f"score: {result['score']:.3f}")
        print("-" * 80)
        print(f"{result['sequence']}")
    print("=" * 80)
