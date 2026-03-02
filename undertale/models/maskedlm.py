"""Masked language modeling implementation."""

from typing import Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import (
    Tensor,
    argmax,
)
from torch.nn import GELU, LayerNorm, Linear, Module
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .custom import InstructionTraceTransformerEncoder
from .tokenizer import SPECIAL_TOKENS, TOKEN_NEXT


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
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        lr: Learning rate.
        warmup: Learning rate warmup steps.
    """

    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        heads: int,
        intermediate_dimensions: int,
        dropout: float,
        eps: float,
        lr: float,
        warmup: float,
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
            SPECIAL_TOKENS.index(TOKEN_NEXT),
            dropout,
            eps,
        )
        self.head = MaskedLMHead(hidden_dimensions, vocab_size, eps)

        self.lr = lr
        self.warmup = warmup
        self.steps_per_epoch = None

    def on_fit_start(self):
        """"""
        self.steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

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

        def constant_with_linear_warmup(step):
            if self.steps_per_epoch is None:
                return 1
            return min(step / self.warmup * self.steps_per_epoch, 1)

        scheduler = LambdaLR(optimizer, constant_with_linear_warmup)

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
        output = self(batch.input_ids, batch.attention_mask)
        loss = cross_entropy(output.view(-1, output.size(-1)), batch.labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        """"""
        output = self(batch.input_ids, batch.attention_mask)

        references = batch.labels
        predictions = argmax(output, dim=-1)

        predictions = predictions[references != -100]
        references = references[references != -100]

        f1 = f1_score(references.tolist(), predictions.tolist(), average="micro")

        self.log("valid_f1", f1, prog_bar=True, sync_dist=True)
