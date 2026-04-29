"""Sequence classification implementation."""

from typing import List, Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import Tensor, argmax, stack, tensor
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .custom import InstructionTraceTransformerEncoder


class ClassificationCollator:
    """Collation function for sequence classification.

    Stacks ``tokens`` and ``mask`` tensors and gathers integer ``label``
    values into a 1-D tensor.
    """

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of dataset rows.

        Arguments:
            batch: A list of dataset rows, each containing ``tokens``,
                ``mask``, and ``label`` fields.

        Returns:
            A batch ready for input to the classification model.
        """

        tokens = stack([tensor(item["tokens"]) for item in batch])
        mask = stack([tensor(item["mask"]) for item in batch])
        labels = tensor([item["label"] for item in batch]).to(int)

        return {"tokens": tokens, "mask": mask, "labels": labels}


class ClassificationHead(Module):
    """Sequence classification head.

    A single linear projection from hidden state space to class logits.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        classes: The number of output classes.
    """

    def __init__(self, hidden_dimensions: int, classes: int):
        super().__init__()

        self.projection = Linear(hidden_dimensions, classes)

    def forward(self, state: Tensor) -> Tensor:
        """Project hidden state to class logits.

        Arguments:
            state: The input state tensor.

        Returns:
            A tensor of class logits.
        """

        return self.projection(state)


class InstructionTraceTransformerEncoderForSequenceClassification(
    LightningModule, Module
):
    """A transformer encoder with a sequence classification head.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        next_token_id: The ID of the special ``NEXT`` token.
        classes: The number of output classes.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        lr: Peak learning rate reached after warmup.
        warmup: Fraction of total steps used for linear warmup.
    """

    LR = 1e-4
    WARMUP = 0.025

    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        heads: int,
        intermediate_dimensions: int,
        next_token_id: int,
        classes: int,
        dropout: float,
        eps: float,
        lr: float = LR,
        warmup: float = WARMUP,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = InstructionTraceTransformerEncoder(
            depth,
            hidden_dimensions,
            vocab_size,
            sequence_length,
            heads,
            intermediate_dimensions,
            next_token_id,
            dropout,
            eps,
        )
        self.head = ClassificationHead(hidden_dimensions, classes)

        self.lr = lr or self.LR
        self.warmup = warmup or self.WARMUP

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode and classify the input sequence.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.

        Returns:
            A tensor of class logits derived from masked, mean-pooled hidden
            state.
        """

        hidden = self.encoder(state, mask)

        if mask is not None:
            expanded = mask.unsqueeze(-1).float()
            pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        return self.head(pooled)

    def configure_optimizers(self):
        """"""
        optimizer = AdamW(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup * total_steps)
        decay_steps = total_steps - warmup_steps

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])
        loss = cross_entropy(output, batch["labels"])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])

        references = batch["labels"]
        predictions = argmax(output, dim=-1)

        f1 = f1_score(references.tolist(), predictions.tolist(), average="micro")

        self.log("valid_f1", f1, prog_bar=True, sync_dist=True)
