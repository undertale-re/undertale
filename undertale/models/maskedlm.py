"""Masked language modeling implementation."""

from typing import Any, Dict, List, Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import (
    Tensor,
    argmax,
    full_like,
    rand,
    randint,
    stack,
    tensor,
)
from torch.nn import GELU, LayerNorm, Linear, Module
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .custom import InstructionTraceTransformerEncoder


class MaskedLMCollator:
    """Collation function for masked language modeling.

    Masking follows the BERT convention: of the candidate positions selected
    at the given ``probability``, 80% are replaced with ``[MASK]``, 10% with
    a random token from the vocabulary, and 10% are left unchanged. Only
    non-padding positions are eligible for masking.

    Arguments:
        mask_token_id: The token ID of the ``[MASK]`` special token.
        vocab_size: The vocabulary size, used for random token replacement.
        probability: The fraction of non-padding tokens selected as masking
            candidates per sequence.
    """

    PROBABILITY = 0.15

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        probability: float = PROBABILITY,
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.probability = probability

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of dataset rows.

        Arguments:
            batch: A list of dataset rows, each containing ``tokens`` and
                ``mask`` fields.

        Returns:
            A masked batch ready for input to the model.
        """

        tokens = stack([tensor(item["tokens"]) for item in batch])
        mask = stack([tensor(item["mask"]) for item in batch])

        # Select masking candidates from non-padding positions.
        candidates = (rand(tokens.shape) < self.probability) & (mask == 1)

        # Labels are the original token IDs at masked positions, -100 elsewhere.
        labels = full_like(tokens, -100)
        labels[candidates] = tokens[candidates]

        # Apply BERT masking strategy: 80% [MASK], 10% random, 10% unchanged.
        decision = rand(tokens.shape)
        replace_with_mask = candidates & (decision < 0.8)
        replace_with_random = candidates & (decision >= 0.8) & (decision < 0.9)

        tokens[replace_with_mask] = self.mask_token_id
        tokens[replace_with_random] = randint(
            0, self.vocab_size, (replace_with_random.sum().item(),)
        )

        return {"tokens": tokens, "mask": mask, "labels": labels}


class MaskedLMHead(Module):
    """Masked language modeling head.

    A simple linear transform and decode - the standard masked language
    modeling head.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        eps: Layer normalization stabalization parameter.
    """

    def __init__(self, hidden_dimensions: int, vocab_size: int, eps: float):
        super().__init__()

        self.transform = Linear(hidden_dimensions, hidden_dimensions)
        self.activation = GELU()
        self.norm = LayerNorm(hidden_dimensions, eps=eps)

        self.decode = Linear(hidden_dimensions, vocab_size)

    def forward(self, state: Tensor) -> Tensor:
        """Decode to vocabulary tokens.

        Arguments:
            state: The input state tensor from the hidden state of a
                transformer.

        Returns:
            A decoded state tensor in vocabulary token space.
        """

        hidden = self.activation(self.transform(state))
        hidden = self.norm(hidden)

        output = self.decode(hidden)

        return output


class InstructionTraceTransformerEncoderForMaskedLM(LightningModule, Module):
    """A transformer encoder with a masked language modeling head.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        next_token_id: The ID of the special ``NEXT`` token.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        lr: Peak learning rate reached after warmup.
        warmup: Number of linear warmup steps before cosine decay begins.
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
        self.head = MaskedLMHead(hidden_dimensions, vocab_size, eps)

        self.lr = lr or self.LR
        self.warmup = warmup or self.WARMUP

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode and decode with the language modeling head.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.

        Returns:
            The computed output tensor in output token space.
        """

        hidden = self.encoder(state, mask)
        output = self.head(hidden)

        return output

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
        loss = cross_entropy(output.view(-1, output.size(-1)), batch["labels"].view(-1))

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        """"""
        output = self(batch["tokens"], batch["mask"])

        references = batch["labels"]
        predictions = argmax(output, dim=-1)

        predictions = predictions[references != -100]
        references = references[references != -100]

        f1 = f1_score(references.tolist(), predictions.tolist(), average="micro")

        self.log("valid_f1", f1, prog_bar=True, sync_dist=True)


class InstructionTraceTransformerEncoderForMaskedLMConfiguration:
    """Model size configurations with associated parameters.

    To make use of this class, simply pass the model size dictionary to model
    initialization as kwargs.
    """

    regularization: Dict[str, Any] = {
        "dropout": 0.1,
        "eps": 1e-12,
    }

    medium: Dict[str, Any] = {
        "sequence_length": 512,
        "depth": 12,
        "heads": 12,
        "hidden_dimensions": 768,
        "intermediate_dimensions": 3072,
        **regularization,
    }

    options = {"medium": medium}
