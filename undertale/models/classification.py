"""Sequence classification implementation."""

from re import sub
from typing import List, Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import Tensor, argmax, float32, stack, tensor, zeros
from torch.nn import Linear, Module
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
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


class MultiLabelClassificationCollator:
    """Collation function for multi-label sequence classification.

    Stacks ``tokens`` and ``mask`` tensors and turns each row's ``labels`` list
    of class names into a multi-hot vector.

    Arguments:
        classes: Class names, in the order of the model's outputs.
    """

    def __init__(self, classes: List[str]):
        self.index = {name: position for position, name in enumerate(classes)}

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of dataset rows.

        Arguments:
            batch: A list of dataset rows, each containing ``tokens``,
                ``mask``, and ``labels`` fields.

        Returns:
            A batch ready for input to the classification model.

        Raises:
            ValueError: If a row has no labels or a label is not a known class.
        """

        tokens = stack([tensor(item["tokens"]) for item in batch])
        mask = stack([tensor(item["mask"]) for item in batch])

        labels = zeros(len(batch), len(self.index), dtype=float32)
        for row, item in enumerate(batch):
            if not item["labels"]:
                raise ValueError(
                    "row has no labels; drop unlabeled rows before training"
                )
            for name in item["labels"]:
                if name not in self.index:
                    raise ValueError(f"unknown class {name!r}")
                labels[row, self.index[name]] = 1.0

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
            state: Pooled encoder hidden state.

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
        class_weights: Optional[List[float]] = None,
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

        if class_weights is not None:
            self.class_weights = Tensor(class_weights)
        else:
            self.class_weights = None

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

    def on_train_start(self):
        """"""
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

    def training_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])
        loss = cross_entropy(output, batch["labels"], weight=self.class_weights)

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


class InstructionTraceTransformerEncoderForMultiLabelSequenceClassification(
    InstructionTraceTransformerEncoderForSequenceClassification
):
    """A transformer encoder with a multi-label sequence classification head.

    Each class is scored independently with a sigmoid, so a sequence may
    belong to any number of classes. Validation reports micro and macro F1
    over the whole validation set, plus per-class F1.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        next_token_id: The ID of the special ``NEXT`` token.
        classes: Class names, in output order.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        lr: Peak learning rate reached after warmup.
        warmup: Fraction of total steps used for linear warmup.
        positive_weights: Optional per-class weight on positive examples,
            typically the ratio of negatives to positives.
        threshold: Probability above which a class is predicted.
    """

    THRESHOLD = 0.5

    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        heads: int,
        intermediate_dimensions: int,
        next_token_id: int,
        classes: List[str],
        dropout: float,
        eps: float,
        lr: float = InstructionTraceTransformerEncoderForSequenceClassification.LR,
        warmup: float = InstructionTraceTransformerEncoderForSequenceClassification.WARMUP,
        positive_weights: Optional[List[float]] = None,
        threshold: float = THRESHOLD,
    ):
        super().__init__(
            depth=depth,
            hidden_dimensions=hidden_dimensions,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            heads=heads,
            intermediate_dimensions=intermediate_dimensions,
            next_token_id=next_token_id,
            classes=len(classes),
            dropout=dropout,
            eps=eps,
            lr=lr,
            warmup=warmup,
        )

        # Record this class's arguments rather than the parent's, so that
        # ``load_from_checkpoint`` gets class names and not a count.
        self.save_hyperparameters()

        self.classes = classes
        self.threshold = threshold

        # A buffer follows the model across devices; it is not persisted since
        # the weights are already recorded among the hyperparameters.
        weights = tensor(positive_weights) if positive_weights is not None else None
        self.register_buffer("positive_weights", weights, persistent=False)

    def predict(self, logits: Tensor) -> Tensor:
        """Decide which classes each sequence belongs to.

        Every class whose probability exceeds ``threshold`` is predicted. Every
        labeled sequence has at least one class, so when none clears the
        threshold the most probable class is predicted alone.

        Arguments:
            logits: Class logits, as returned by ``forward``.

        Returns:
            A boolean tensor the same shape as ``logits``.
        """

        predictions = logits.sigmoid() > self.threshold

        empty = ~predictions.any(dim=-1)
        predictions[empty, argmax(logits[empty], dim=-1)] = True

        return predictions

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Binary cross entropy over every class."""

        return binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.positive_weights
        )

    def training_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])
        loss = self.loss(output, batch["labels"])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        """"""
        # Rows are true positives, false positives, false negatives per class.
        self.counts = zeros(3, len(self.classes), device=self.device)

    def validation_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])
        references = batch["labels"].bool()
        predictions = self.predict(output)

        self.counts[0] += (predictions & references).sum(dim=0)
        self.counts[1] += (predictions & ~references).sum(dim=0)
        self.counts[2] += (~predictions & references).sum(dim=0)

        self.log("valid_loss", self.loss(output, batch["labels"]), sync_dist=True)

    def on_validation_epoch_end(self):
        """"""
        # F1 is not an average of per-batch F1, so tally counts across every
        # batch and rank before computing it.
        counts = self.trainer.strategy.reduce(self.counts, reduce_op="sum")
        positives, false_positives, false_negatives = counts

        micro = (
            2
            * positives.sum()
            / (
                2 * positives.sum() + false_positives.sum() + false_negatives.sum()
            ).clamp(min=1)
        )

        denominator = 2 * positives + false_positives + false_negatives
        scores = 2 * positives / denominator.clamp(min=1)

        # Classes absent from both labels and predictions say nothing about
        # the model, so they are left out of the macro average.
        present = denominator > 0
        macro = scores[present].mean() if present.any() else scores.new_zeros(())

        self.log("valid_f1", macro, prog_bar=True)
        self.log("valid_micro_f1", micro)
        for name, score in zip(self.classes, scores):
            tag = sub(r"\W+", "_", name).strip("_")
            self.log(f"valid_f1_class/{tag}", score)
