import argparse

import transformers
from torch import argmax, where
from torch.utils.data import DataLoader

from undertale.datasets.base import Dataset
from undertale.models.item import tokenizer
from undertale.models.item.model import TransformerEncoderForMaskedLM

parser = argparse.ArgumentParser(
    description="pretrain the model on a masked language modeling dataset",
)

parser.add_argument(
    "-m",
    "--model",
    help="model to evaluate",
)
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument(
    "--dataset",
    help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
)
parser.add_argument(
    "-n",
    "--num_examples",
    default=10,
    help="The number of examples to run",
)
parser.add_argument(
    "--mode",
    default="MLM",
    help="What kind of model to test",
)
args = parser.parse_args()
if args.mode not in ["MLM"]:
    raise NotImplementedError(f"{args.mode} is not a currently supported mode.")
tokenizer_loc = args.tokenizer
checkpoint = args.model

tok = tokenizer.load(tokenizer_loc)
model = TransformerEncoderForMaskedLM.load_from_checkpoint(checkpoint)
print("running model on ", model.device)

dataset_loc = args.dataset
batch_size = 1

if args.mode == "MLM":
    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=transformers.PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_loc,
            mask_token=tokenizer.TOKEN_MASK,
            unk_token=tokenizer.TOKEN_UNKNOWN,
            pad_token=tokenizer.TOKEN_PAD,
        ),
        mlm_probability=0.15,
    )
    dataset = Dataset.load(dataset_loc)
    dataset = dataset.train_test_split(test_size=10)
    validation = DataLoader(
        dataset["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=8,
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
        # TODO explore Top-K prediction with confidence scores from softmax logits.
        # top = topk(softmax(output, dim=-1), k=5)
        print(f"input:\n {input_seq}\n\noutput:\n {predicted}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
