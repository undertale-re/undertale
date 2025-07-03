import argparse
import logging
import os

import torch
import transformers
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from ... import logging as undertale_logging
from ...datasets.base import Dataset
from . import tokenizer
from .model import Defaults, TransformerEncoderForMaskedLM

logger = logging.getLogger(__name__)


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ValidationCallback(Callback):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_validation_end(self, trainer, pl_module):
        for batch in self.dataloader:
            input_ids = batch.input_ids.to(pl_module.device)
            attention_mask = batch.attention_mask.to(pl_module.device)
            output = pl_module(input_ids, attention_mask)
            filled = torch.where(
                input_ids == pl_module.tok.token_to_id(tokenizer.TOKEN_MASK),
                torch.argmax(output, dim=-1),
                input_ids,
            )
            input_seq = (
                pl_module.tok.decode(input_ids[0].tolist(), skip_special_tokens=False)
                .replace("[NEXT]", "\n")
                .replace("[PAD]", "")
                .strip()
            )
            predicted = pl_module.tok.decode(
                filled[0].tolist(), skip_special_tokens=False
            )
            predicted = (
                predicted.replace(tokenizer.TOKEN_PAD, "")
                .replace("[NEXT]", "\n")
                .strip()
            )
            if isinstance(pl_module.logger.experiment, SummaryWriter):
                pl_module.logger.experiment.add_text(
                    "mask prediction", f"input: {input_seq}\n\noutput:{predicted}"
                )


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
    parser.add_argument("-v", "--version", help="training run version name")
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to output validation examples",
    )

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
        eps=Defaults.eps,
        lr=Defaults.lr,
        warmup=Defaults.warmup,
        tokenizer_loc=arguments.tokenizer,
    )

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

    if arguments.validation:
        random_sampler = RandomSampler(dataset["test"], num_samples=5)
        random_validation = DataLoader(
            dataset["test"],
            sampler=random_sampler,
            batch_size=1,
            collate_fn=collator,
            num_workers=8,
        )
        validation_check = ValidationCallback(random_validation)
        callbacks = [progress, checkpoint, stop, validation_check]
    else:
        callbacks = [progress, checkpoint, stop]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        num_nodes=arguments.nodes,
        strategy="ddp",
        max_epochs=96,
        # Testing
        # log_every_n_steps=1,
        # limit_train_batches=2,
        # limit_val_batches=2,
    )
    trainer.fit(
        model,
        train_dataloaders=training,
        val_dataloaders=validation,
        ckpt_path=arguments.checkpoint,
    )
