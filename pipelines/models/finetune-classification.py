import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss

# from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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


class ClassifyModel(LightningModule, torch.nn.Module):
    def __init__(self, model, lr=2e-5, warmup_steps=5000, end_to_end=True):
        super().__init__()

        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.end_to_end = end_to_end
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def forward(self, inp, mask=None):
        encoder_embedding = self.model.encoder(inp)
        return self.model(encoder_embedding, mask)

    def training_step(self, batch, batch_idx):

        labels, dissassembly_tokens, dissassembly_mask = (
            batch["labels"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],
        )

        if self.end_to_end:
            with torch.no_grad():
                embeddings = self.model.embed_assembly(
                    dissassembly_tokens, dissassembly_mask
                )
        else:
            embeddings = dissassembly_tokens

        outputs = self(embeddings)

        loss = F.cross_entropy(outputs, labels, ignore_index=0)

        self.log("train_loss", loss, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        labels, dissassembly_tokens, dissassembly_mask = (
            batch["labels"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],
        )

        if self.end_to_end:
            with torch.no_grad():
                embeddings = self.model.embed_assembly(
                    dissassembly_tokens, dissassembly_mask
                )
        else:
            embeddings = dissassembly_tokens

        outputs = self(embeddings)
        loss = F.cross_entropy(outputs, labels, ignore_index=0)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):

        total_steps = self.trainer.estimated_stepping_batches

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        config_optim = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config_optim


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

    def create_label(row):
        row["label"] = row["vulnerability"] != ""
        return row

    cols = dataset["train"].column_names
    if "vulnerability" in cols and "label" not in cols:
        dataset["train"] = dataset["train"].map(create_label)

    dataset = dataset["train"].select_columns(["disassembly", "label"])

    if not args.dataset_size == -1:
        dataset = dataset.select(range(args.dataset_size))
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    train_dataset = split_dataset["train"]

    val_dataset = split_dataset["test"]

    collator = CustomCollator(args, device)

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

    # set up model
    connector_config = {
        "model_type": args.model_type,
        "num_layers": args.num_layers,
    }

    connector_config_namespace = SimpleNamespace(**connector_config)

    tokenizer = load_tokenizer(args.tokenizer)

    vocab_size = tokenizer.get_vocab_size()
    mask_token_id = tokenizer.token_to_id(TOKEN_MASK)
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    model = TransformerEncoderForSequenceClassification(
        vocab_size=vocab_size,
        lr=args.learning_rate,
        warmup=args.warmup,
        num_classes=2,
        head_hidden_size=64,
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

    # if args.validation:

    #     random_sampler = RandomSampler(val_dataset, num_samples=5)
    #     random_validation = DataLoader(
    #         val_dataset,
    #         sampler=random_sampler,
    #         batch_size=1,
    #         collate_fn=collator,
    #         num_workers=8,
    #     )

    #     final_validation = DataLoader(
    #         val_dataset, batch_size=1, collate_fn=collator, num_workers=8
    #     )

    #     final_validation_check = ValidationCallback(
    #         final_validation,
    #         end_to_end=True,
    #         tag="final_full_val",
    #         run_on_val_end=False,
    #         run_on_fit_end=True,
    #         args=args,
    #     )

    #     validation_check = ValidationCallback(
    #         random_validation,
    #         end_to_end=True,
    #         tag=None,
    #         run_on_val_end=True,
    #         run_on_fit_end=False,
    #         args=args,
    #     )

    #     callbacks = [
    #         progress,
    #         checkpoint,
    #         stop,
    #         validation_check,
    #         final_validation_check,
    #     ]

    # else:
    callbacks = [progress, checkpoint, stop]

    classify_model = ClassifyModel(
        model,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        end_to_end=True,
    )

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

    trainer.fit(
        classify_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.classifier_checkpoint,
    )
