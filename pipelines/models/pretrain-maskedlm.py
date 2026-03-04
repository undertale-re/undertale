from os.path import basename, dirname
from typing import Callable, Optional

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from undertale.models.dataset import ParquetDataset
from undertale.models.maskedlm import (
    InstructionTraceTransformerEncoderForMaskedLM,
    InstructionTraceTransformerEncoderForMaskedLMConfiguration,
    MaskedLMCollator,
)
from undertale.models.tokenizer import TOKEN_MASK, TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import TokenizedDataset


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def load_dataset(
    path: str, batch: int, collator: Optional[Callable] = None, workers: int = 0
) -> DataLoader:
    return DataLoader(
        ParquetDataset(path, schema=TokenizedDataset),
        batch_size=batch,
        collate_fn=collator,
        num_workers=workers,
    )


if __name__ == "__main__":
    parser = ModelArgumentParser(description="masked language modeling pretraining")

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="path to a trained tokenizer"
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    tokenizer = load_tokenizer(arguments.tokenizer)
    model = InstructionTraceTransformerEncoderForMaskedLM(
        vocab_size=tokenizer.get_vocab_size(),
        next_token_id=tokenizer.token_to_id(TOKEN_NEXT),
        lr=arguments.learning_rate,
        warmup=arguments.warmup,
        **InstructionTraceTransformerEncoderForMaskedLMConfiguration.medium,
    )

    collator = MaskedLMCollator(
        mask_token_id=tokenizer.token_to_id(TOKEN_MASK),
        vocab_size=tokenizer.get_vocab_size(),
    )

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
