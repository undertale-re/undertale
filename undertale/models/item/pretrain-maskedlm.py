import argparse
import logging

import torch
import transformers
from lightning import Trainer
from torch.utils.data import DataLoader

from ... import logging as undertale_logging
from ...datasets.base import Dataset
from . import tokenizer
from .model import Defaults, TransformerEncoderForMaskedLM

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain the model on a masked language modeling dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "dataset",
        help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("output", help="output model directory")

    parser.add_argument(
        "-c",
        "--checkpoint",
        help="trained model checkpoint from which to resume training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="number of epochs for which to train",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="starting epoch number"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")

    arguments = parser.parse_args()

    undertale_logging.setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    tok = tokenizer.load(arguments.tokenizer, sequence_length=512)

    model = TransformerEncoderForMaskedLM(
        depth=Defaults.depth,
        hidden_dimensions=Defaults.hidden_dimensions,
        vocab_size=tok.get_vocab_size(),
        input_size=Defaults.input_size,
        heads=Defaults.heads,
        intermediate_dimensions=Defaults.intermediate_dimensions,
        dropout=Defaults.dropout,
    )

    if arguments.checkpoint:
        # TODO
        # model = model.from_pretrained(arguments.checkpoint, local_files_only=True)
        pass

    try:
        dataset = Dataset.load(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    dataset = dataset.train_test_split(test_size=0.1)

    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=transformers.PreTrainedTokenizerFast(
            tokenizer_file=arguments.tokenizer,
            mask_token=tokenizer.TOKEN_MASK,
            unk_token=tokenizer.TOKEN_UNKNOWN,
            pad_token=tokenizer.TOKEN_PAD,
        ),
        mlm_probability=0.15,
    )

    batch_size = arguments.batch_size
    training = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=8,
    )
    validation = DataLoader(
        dataset["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=8,
    )

    trainer = Trainer(
        # limit_train_batches=2,
        # limit_val_batches=2,
        # max_epochs=1,
    )
    trainer.fit(model, train_dataloaders=training, val_dataloaders=validation)
