from os.path import basename, dirname

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from undertale.models.configuration import (
    InstructionTraceTransformerEncoderConfiguration,
)
from undertale.models.dataset import DataModule
from undertale.models.density import (
    DensityCollator,
    InstructionTraceTransformerEncoderForDensity,
)
from undertale.models.tokenizer import TOKEN_MASK, TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import TokenizedDataset
from undertale.utils import cache_path


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


if __name__ == "__main__":
    parser = ModelArgumentParser(description="density model fine-tuning")

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
        "--freeze-encoder",
        action="store_true",
        help="freeze the pretrained encoder and only train the density head",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    tokenizer = load_tokenizer(cache_path(arguments.tokenizer))

    vocab_size = tokenizer.get_vocab_size()
    mask_token_id = tokenizer.token_to_id(TOKEN_MASK)
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    collator = DensityCollator(
        mask_token_id=mask_token_id, vocab_size=vocab_size, next_token_id=next_token_id
    )

    dataset = cache_path(arguments.dataset)
    validation = arguments.validation
    if validation is not None:
        validation = cache_path(arguments.validation)

    datamodule = DataModule(
        dataset,
        validation,
        schema=TokenizedDataset,
        collator=collator,
        batch=arguments.batch_size,
        workers=arguments.dataloaders,
        memory=arguments.dataloader_memory,
    )

    model = InstructionTraceTransformerEncoderForDensity(
        vocab_size=vocab_size,
        next_token_id=next_token_id,
        lr=arguments.learning_rate,
        warmup=arguments.warmup,
        freeze_encoder=arguments.freeze_encoder,
        **InstructionTraceTransformerEncoderConfiguration.medium,
    )

    pretrained = torch.load(cache_path(arguments.pretrained), map_location="cpu")
    model.load_state_dict(pretrained["state_dict"], strict=False)

    if arguments.validation is not None:
        stop = EarlyStopping(
            monitor="valid_loss", mode="min", patience=5, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(
            filename="{epoch}-{train_loss:.2f}-{valid_loss:.2f}", save_top_k=-1
        )
    else:
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
        use_distributed_sampler=False,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=arguments.checkpoint,
    )
