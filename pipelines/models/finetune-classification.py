from os.path import basename, dirname
from typing import Callable, Optional

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from undertale.models.classification import (
    ClassificationCollator,
    InstructionTraceTransformerEncoderForSequenceClassification,
)
from undertale.models.configuration import (
    InstructionTraceTransformerEncoderConfiguration,
)
from undertale.models.dataset import ParquetDataset
from undertale.models.tokenizer import TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import TokenizedClassificationDataset
from undertale.utils import cache_path


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def load_dataset(
    path: str, batch: int, collator: Optional[Callable] = None, workers: int = 0
) -> DataLoader:
    cached = cache_path(path)

    return DataLoader(
        ParquetDataset(cached, schema=TokenizedClassificationDataset),
        batch_size=batch,
        collate_fn=collator,
        num_workers=workers,
    )


if __name__ == "__main__":
    parser = ModelArgumentParser(description="sequence classification fine-tuning")

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="path to a trained tokenizer"
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        required=True,
        help="path to a pretrained masked LM checkpoint",
    )
    parser.add_argument(
        "-k", "--classes", type=int, required=True, help="number of output classes"
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    tokenizer = load_tokenizer(cache_path(arguments.tokenizer))

    vocab_size = tokenizer.get_vocab_size()
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    model = InstructionTraceTransformerEncoderForSequenceClassification(
        vocab_size=vocab_size,
        next_token_id=next_token_id,
        classes=arguments.classes,
        lr=arguments.learning_rate,
        warmup=arguments.warmup,
        **InstructionTraceTransformerEncoderConfiguration.medium,
    )

    pretrained = torch.load(cache_path(arguments.pretrained), map_location="cpu")
    model.load_state_dict(pretrained["state_dict"], strict=False)

    collator = ClassificationCollator()

    training = load_dataset(
        arguments.dataset,
        arguments.batch_size,
        collator=collator,
        workers=arguments.dataloaders,
    )

    if arguments.validation is not None:
        validation = load_dataset(
            arguments.validation,
            arguments.batch_size,
            collator=collator,
            workers=arguments.dataloaders,
        )
        stop = EarlyStopping(
            monitor="valid_f1", mode="max", patience=5, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(
            filename="{epoch}-{train_loss:.2f}-{valid_f1:.2f}", save_top_k=-1
        )
    else:
        validation = None
        stop = EarlyStopping(
            monitor="train_loss", mode="min", patience=5, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(filename="{epoch}-{train_loss:.2f}", save_top_k=-1)

    progress = ProgressBar(leave=True)

    logger = TensorBoardLogger(
        save_dir=dirname(arguments.output),
        name=basename(arguments.output),
        version=arguments.version,
    )

    trainer = Trainer(
        callbacks=[progress, checkpoint, stop],
        logger=logger,
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        num_nodes=arguments.nodes,
        strategy=arguments.strategy,
        max_epochs=arguments.epochs,
    )
    trainer.fit(
        model,
        train_dataloaders=training,
        val_dataloaders=validation,
        ckpt_path=arguments.checkpoint,
    )
