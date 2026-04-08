import argparse
import os
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss

# from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from undertale.logging import setup_logging
from undertale.models.classification_dataset import CustomCollator
from undertale.models.classification_model import (
    TransformerEncoderForSequenceClassification,
)
from undertale.models.maskedlm import (
    InstructionTraceTransformerEncoderForMaskedLMConfiguration,
)
from undertale.models.tokenizer import TOKEN_MASK, TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser


def dataset_size_type(x):
    x = int(x)
    if x == 0:
        raise argparse.ArgumentTypeError(
            "dataset_size cannot be 0. Use -1 for full dataset or a positive integer."
        )
    return x


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ValidationCallback(Callback):
    def __init__(
        self,
        dataloader,
        run_on_val_end=True,
        run_on_fit_end=False,
        tag=None,
        end_to_end=False,
        args=None,
    ):
        super().__init__()

        # self.bertscore_model_path=args.bertscore_model_path
        self.dataloader = dataloader
        self.save_dir = args.generated_output_paths
        self.end_to_end = end_to_end
        self.run_on_val_end = run_on_val_end
        self.run_on_fit_end = run_on_fit_end
        self.tag = tag

        self.do_sample = False

        os.makedirs(self.save_dir, exist_ok=True)

    def _run_validation(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device

        loss_function = CrossEntropyLoss()

        score_sum = 0
        n_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):

                dis_tokens = batch["disassembly_tokens"].to(device)
                dis_mask = batch["disassembly_mask"].to(device)
                labels = batch["labels"].to(device)

                # ---------------- encoder path (UNCHANGED) ----------------
                encoder_embedding = model.assembly_encoder(dis_tokens, dis_mask)
                encoder_embedding = encoder_embedding.mean(dim=1)

                # ---------------- classification ----------------
                classifiction = pl_module.model.head(encoder_embedding)
                score_sum += loss_function(classifiction, labels)

        if was_training:
            pl_module.train()

        score = score_sum / max(1, n_samples)

        trainer.print(f"[val metrics] Cross Entropy Loss={score:.4f}")

        if self.tag is None:
            path = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch}.txt")
        else:
            path = os.path.join(
                self.save_dir, f"{self.tag}_epoch_{trainer.current_epoch}.txt"
            )
        self.log("score", score, sync_dist=True)
        with open(path, "w") as f:

            f.write(
                f"Cross Entropy Score: {score:.6f}\n"
                f"NUM_SAMPLES: {n_samples}\n"
                "=================\n\n"
            )

        trainer.print(f"Saved validation outputs to {path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_on_val_end:
            self._run_validation(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.run_on_fit_end:
            self._run_validation(trainer, pl_module)


if __name__ == "__main__":

    parser = ModelArgumentParser()
    parser.add_argument("--dataset_size", default=-1)
    parser.add_argument("--test_size", default=0.1)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--model_type", default="mlp", help="the type of the connector")
    parser.add_argument(
        "--num_layers", default=2, help="the number of layers in the connector"
    )
    # parser.add_argument("--output", default="./vuln_class_output/")
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--warmup_steps", default=1)
    parser.add_argument("--generated_output_paths")
    parser.add_argument("--tokenizer_size", default=512)
    parser.add_argument("--tokenizer")
    parser.add_argument("--val_dataset")

    args = parser.parse_args()
    parser.setup(args)
    setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # set up dataloaders

    if os.path.exists(args.dataset):
        dataset = load_dataset(args.dataset)
    else:
        raise FileNotFoundError(f"File not found: {args.dataset}")
    if os.path.exists(args.val_dataset):
        val_dataset = load_dataset(args.val_dataset)
    else:
        raise FileNotFoundError(f"File not found: {args.dataset}")

    def create_label(row):
        row["label"] = row["vulnerability"] != ""
        return row

    cols = dataset["train"].column_names
    if "vulnerability" in cols and "label" not in cols:
        dataset["train"] = dataset["train"].map(create_label)

    dataset = dataset["train"].select_columns(["tokens", "label", "mask"])

    cols = val_dataset["train"].column_names
    if "vulnerability" in cols and "label" not in cols:
        val_dataset["train"] = val_dataset["train"].map(create_label)

    val_dataset = val_dataset["train"].select_columns(["tokens", "label", "mask"])

    if not args.dataset_size == -1:
        dataset = dataset.select(range(args.dataset_size))

    train_dataset = dataset["train"]
    val_dataset = val_dataset["train"]

    tokenizer = load_tokenizer(args.tokenizer)

    vocab_size = tokenizer.get_vocab_size()
    mask_token_id = tokenizer.token_to_id(TOKEN_MASK)
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    collator = CustomCollator(mask_token_id=mask_token_id, vocab_size=vocab_size)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=8,
    )

    # computing weights to account for imbalanced data
    counts = Counter(train_dataset["label"])
    weights = 1.0 / np.array([counts[False], counts[True]])
    weights = 2 * weights / weights.sum()

    model = TransformerEncoderForSequenceClassification(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        next_token_id=next_token_id,
        lr=args.learning_rate,
        warmup=args.warmup,
        num_classes=2,
        head_hidden_size=64,
        balance_weights=list(weights),
        **InstructionTraceTransformerEncoderForMaskedLMConfiguration.medium,
    )

    output = os.path.abspath(os.path.expanduser(args.output))

    progress = ProgressBar(leave=True)
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=-1,
    )
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001)
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=args.version,
    )

    callbacks = [progress, checkpoint, stop]

    sample = val_dataloader.dataset[0]

    trainer = Trainer(
        strategy="ddp",
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
    )

    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    loaded_state_dict = checkpoint["state_dict"]
    current_model_dict = model.state_dict()
    new_state_dict = {
        k: v for k, v in loaded_state_dict.items() if k in current_model_dict
    }
    model.load_state_dict(new_state_dict, strict=False)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
