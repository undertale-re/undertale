import argparse
import logging
import os

import torch
import transformers
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ... import logging as undertale_logging
from ...datasets.base import Dataset
from . import tokenizer
from .model import Defaults, TransformerEncoderForSequenceSimilarity

logger = logging.getLogger(__name__)


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="finetune the model on a pairwise embedding dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "dataset",
        help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("-o", "--output", required=True, help="output model directory")

    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument(
        "-m", "--model", help="pretrained model from which to begin training"
    )

    start.add_argument(
        "-c",
        "--checkpoint",
        help="trained model checkpoint from which to resume training",
    )
    
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")

    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )

    parser.add_argument(
        "-d",
        "--devices",
        default=1,
        type=int,
        help="number of accelerator devices to use (per node)",
    )
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, help="number of nodes to use"
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

    
    model = TransformerEncoderForSequenceSimilarity(
        depth=Defaults.depth,
        hidden_dimensions=Defaults.hidden_dimensions,
        vocab_size=tok.get_vocab_size(),
        input_size=Defaults.input_size,
        heads=Defaults.heads,
        intermediate_dimensions=Defaults.intermediate_dimensions,
        dropout=Defaults.dropout,
        eps=Defaults.eps,
        lr=Defaults.lr,
        warmup=Defaults.warmup,
        embedding_size=Defaults.embedding_size,
        embedding_dropout_prob=Defaults.dropout
    )

    if (arguments.model):
        # Seems like I want to load the previously trained
        # TransformerEncoderForMaskedLM and then pull out its
        # model1.encoder and hand it to this model we'll be using for seq
        # similarity training.
        #
        # Presumably, arguments.model is meant to point to that pre-trained mlm.
        #
        mlm = TransformerEncoderForMaskedLM.load_from_checkpoint(arguments.model)
        model.encoder = mlm.encoder
    elif arguments.checkpoint:
        model = model.from_pretrained(arguments.checkpoint, local_files_only=True)
            
    try:
        dataset = Dataset.load(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    dataset = dataset.train_test_split(test_size=0.1)
    
    def tokenize(batch):
        preprocessed = tokenizer.preprocess_batch(batch)
        encoded = tok.encode_batch(preprocessed["preprocessed"])

        batch["input_ids"] = [s.ids for s in encoded]
        batch["attention_mask"] = [s.attention_mask for s in encoded]

        return batch

    def tokenize_pair(batch):
        first = tokenize(batch["first"])
        second = tokenize(batch["second"])

        batch = {}
        batch["input_ids1"] = first["input_ids"]
        batch["attention_mask1"] = first["attention_mask"]
        batch["input_ids2"] = second["input_ids"]
        batch["attention_mask2"] = second["attention_mask"]
        batch["labels"] = batch["similarity"]

        return batch

    dataset = dataset.map(
        tokenize_pair,
        batched=True,
        remove_columns=dataset.column_names,
        desc="tokenizing",
    )

    dataset = dataset.train_test_split(test_size=0.1)

    collator = transformers.DefaultDataCollator()

    batch_size = arguments.batch_size
    training = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator
    )
    validation = DataLoader(
        dataset["test"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator
    )

    output = os.path.abspath(os.path.expanduser(arguments.output))
    
    progress = ProgressBar(leave=True)
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}-{valid_f1:.2f}",
        save_top_k=-1,
    )
    stop = EarlyStopping(monitor="valid_f1", mode="max", patience=5, min_delta=0.001)
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=arguments.version,
    )

    trainer = Trainer(
        callbacks=[progress, checkpoint, stop],
        logger=logger,
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        num_nodes=arguments.nodes,
        strategy="ddp",
        max_epochs=96,
    )
    trainer.fit(
        model,
        train_dataloaders=training,
        val_dataloaders=validation,
        ckpt_path=arguments.checkpoint,
    )


