from os.path import basename, dirname
from typing import Callable, Optional

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from undertale.models.dataset import ParquetDataset
from undertale.models.maskedlm import (
    InstructionTraceTransformerEncoderForMaskedLM,
    InstructionTraceTransformerEncoderForMaskedLMConfiguration,
    MaskedLMCollator,
)
from undertale.models.tokenizer import TOKEN_MASK, TOKEN_NEXT, TOKEN_PAD
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import TokenizedDataset


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class InferenceLogger(Callback):
    """Runs inference on probe strings and logs results to TensorBoard.

    At the end of each training epoch the model is switched to evaluation
    mode, each probe is encoded and filled, and the result is written as
    Markdown text to TensorBoard.

    Arguments:
        tokenizer: The trained tokenizer used to encode and decode probes.
    """

    PROBES = [
        # Simple XOR (self).
        ("simple-xor-self", "xor rax [MASK]"),
        # Simple ADD (immediate).
        ("simple-add-immediate", "add rax [MASK]"),
        # Control flow (addresses).
        ("control-flow-address", "jmp [MASK]"),
        # Register width alignment.
        ("register-width-16", "mov [MASK] bl"),
        ("register-width-32", "mov [MASK] eax"),
        ("register-width-64", "mov [MASK] rax"),
        ("memory-dereference-32", "mov eax [MASK] [ esi ]"),
        ("memory-dereference-64", "mov rax [MASK] [ rsi ]"),
        ("memory-dereference-extended", "movzx eax [MASK] [ esi ]"),
        # Control flow.
        ("control-flow-next-instruction", "test eax eax [NEXT] [MASK] [MASK]"),
        # Calling conventions.
        (
            "calling-conventions-register-preserved",
            "push [MASK] [NEXT] mov rbx 42 [NEXT] add rax rbx [NEXT] pop [MASK] [NEXT] ret",
        ),
        # Generation.
        ("primitive-generation", "[MASK] [MASK] [MASK] [MASK] [MASK]"),
        (
            "conditioned-primitive-generateion",
            "add rax [MASK] [NEXT] mov rax [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
        ),
    ]

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @staticmethod
    def format_tokens(tokens: str):
        formatted = tokens.replace(f"{TOKEN_NEXT} ", f"{TOKEN_NEXT}\n")
        formatted = formatted.replace(TOKEN_MASK, f"**{TOKEN_MASK}**")
        return formatted

    def on_train_epoch_end(self, trainer, model) -> None:
        model.eval()
        with torch.no_grad():
            for index, (name, probe) in enumerate(self.PROBES):
                encoded = self.tokenizer.encode(probe)
                tokens = torch.tensor(encoded.ids).unsqueeze(0).to(model.device)
                mask = (
                    torch.tensor(encoded.attention_mask).unsqueeze(0).to(model.device)
                )

                filled = model.infer(tokens, mask)

                decoded = self.tokenizer.decode(
                    filled.tolist(), skip_special_tokens=False
                )
                decoded = decoded.replace(TOKEN_PAD, "").strip()

                probe = self.format_tokens(probe)
                decoded = self.format_tokens(decoded)

                trainer.logger.experiment.add_text(
                    f"inference/{index}-{name}",
                    f"####Input\n{probe}\n####Output\n{decoded}",
                    global_step=trainer.current_epoch,
                )
        model.train()


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

    vocab_size = tokenizer.get_vocab_size()
    mask_token_id = tokenizer.token_to_id(TOKEN_MASK)
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    model = InstructionTraceTransformerEncoderForMaskedLM(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        next_token_id=next_token_id,
        lr=arguments.learning_rate,
        warmup=arguments.warmup,
        **InstructionTraceTransformerEncoderForMaskedLMConfiguration.medium,
    )

    collator = MaskedLMCollator(
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
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
    probe = InferenceLogger(tokenizer)

    logger = TensorBoardLogger(
        save_dir=dirname(arguments.output),
        name=basename(arguments.output),
        version=arguments.version,
    )

    trainer = Trainer(
        callbacks=[progress, checkpoint, probe, stop],
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
