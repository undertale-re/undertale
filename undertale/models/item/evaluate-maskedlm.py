import argparse

import transformers
from torch import argmax, where
from torch.utils.data import DataLoader

from undertale.datasets.base import Dataset
from undertale.models.item import tokenizer
from undertale.models.item.model import TransformerEncoderForMaskedLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a model on a masked language modeling dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="trained model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        help="dataset on which to evaluate the model",
    )

    parser.add_argument(
        "-s",
        "--samples",
        default=10,
        help="the number of samples to run",
    )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = TransformerEncoderForMaskedLM.load_from_checkpoint(arguments.checkpoint)

    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=transformers.PreTrainedTokenizerFast(
            tokenizer_file=arguments.tokenizer,
            mask_token=tokenizer.TOKEN_MASK,
            unk_token=tokenizer.TOKEN_UNKNOWN,
            pad_token=tokenizer.TOKEN_PAD,
        ),
        mlm_probability=0.15,
    )
    dataset = Dataset.load(arguments.dataset)
    dataset = dataset.train_test_split(test_size=arguments.samples)
    validation = DataLoader(
        dataset["test"],
        shuffle=False,
        batch_size=1,
        collate_fn=collator,
    )
    for batch in validation:
        input_ids = batch.input_ids.to(model.device)
        attention_mask = batch.attention_mask.to(model.device)
        output = model(input_ids, attention_mask)
        filled = where(
            input_ids == tok.token_to_id(tokenizer.TOKEN_MASK),
            argmax(output, dim=-1),
            input_ids,
        )
        input_seq = (
            tok.decode(input_ids[0].tolist(), skip_special_tokens=False)
            .replace("[NEXT]", "\n")
            .replace("[PAD]", "")
            .strip()
        )
        predicted = tok.decode(filled[0].tolist(), skip_special_tokens=False)
        predicted = (
            predicted.replace(tokenizer.TOKEN_PAD, "").replace("[NEXT]", "\n").strip()
        )
        print(f"input:\n {input_seq}\n\noutput:\n {predicted}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
