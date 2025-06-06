from typing import Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import (
    Tensor,
    arange,
    argmax,
    bincount,
    cat,
    cumsum,
    long,
    roll,
    stack,
    zeros,
)
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn import TransformerEncoder as TorchTransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .tokenizer import SPECIAL_TOKENS, TOKEN_NEXT


class Defaults:
    depth = 12
    hidden_dimensions = 768
    input_size = 512
    heads = 12
    intermediate_dimensions = 3072
    dropout = 0.1
    lr = 5e-5
    warmup = 4000


class PositionEmbedding(Module):
    def __init__(
        self, hidden_dimensions: int, vocab_size: int, input_size: int, dropout: float
    ):
        super().__init__()

        self.token = Embedding(vocab_size, hidden_dimensions)
        self.instruction = Embedding(input_size, hidden_dimensions)
        self.argument = Embedding(input_size, hidden_dimensions)
        self.norm = LayerNorm(hidden_dimensions, eps=1e-12)
        self.dropout = Dropout()

        self.next_token_id = SPECIAL_TOKENS.index(TOKEN_NEXT)

    def forward(self, state: Tensor) -> Tensor:
        # FIXME this could probably be optimized
        starts = roll(state == self.next_token_id, 1)
        starts[:, 0] = False
        instructions = cumsum(starts, dim=-1)

        arguments = zeros(instructions.shape, dtype=long, device=state.device)
        for i, batch in enumerate(instructions):
            arguments[i] = cat([arange(v) for v in bincount(batch)])

        tokens = self.token(state)
        instructions = self.instruction(instructions)
        arguments = self.argument(arguments)

        embedded = tokens + instructions + arguments

        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class TransformerEncoder(Module):
    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        input_size: int,
        heads: int,
        intermediate_dimensions: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = PositionEmbedding(
            hidden_dimensions, vocab_size, input_size, dropout
        )

        layer = TransformerEncoderLayer(
            d_model=hidden_dimensions,
            nhead=heads,
            dim_feedforward=intermediate_dimensions,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = TorchTransformerEncoder(layer, num_layers=depth)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        embedded = self.embedding(state)
        output = self.encoder(embedded, src_key_padding_mask=mask == 1)

        return output


class MaskedLMHead(Module):
    def __init__(self, hidden_dimensions, vocab_size):
        super().__init__()

        self.linear = Linear(hidden_dimensions, vocab_size)

    def forward(self, state: Tensor) -> Tensor:
        return self.linear(state)


class TransformerEncoderForMaskedLM(LightningModule, Module):
    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        input_size: int,
        heads: int,
        intermediate_dimensions: int,
        dropout: float,
        lr: float,
        warmup: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TransformerEncoder(
            depth,
            hidden_dimensions,
            vocab_size,
            input_size,
            heads,
            intermediate_dimensions,
            dropout,
        )
        self.head = MaskedLMHead(hidden_dimensions, vocab_size)

        self.lr = lr
        self.warmup = warmup

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        hidden = self.encoder(state, mask)
        output = self.head(hidden)

        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        def constant_with_linear_warmup(step):
            return min(step / self.warmup, 1)

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
        output = self(batch.input_ids, batch.attention_mask)
        loss = cross_entropy(output.view(-1, output.size(-1)), batch.labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        output = self(batch.input_ids, batch.attention_mask)

        references = batch.labels
        predictions = argmax(output, dim=-1)

        predictions = predictions[references != -100]
        references = references[references != -100]

        f1 = f1_score(references.tolist(), predictions.tolist(), average="micro")

        self.log("valid_f1", f1, prog_bar=True, sync_dist=True)


class TransformerEncoderForSequenceSimilarity(Module):
    pass


class TransformerEncoderForSequenceClassification(Module):
    def __init__(
        self,
        classes: int,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        input_size: int,
        heads: int,
        intermediate_dimensions: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            depth,
            hidden_dimensions,
            vocab_size,
            input_size,
            heads,
            intermediate_dimensions,
            dropout,
        )
        self.dropout = Dropout(dropout)
        self.head = Linear(hidden_dimensions, classes)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Select only the final state to classify.
        hidden = self.encoder(state, mask)[:, 0, :]
        hidden = self.dropout(hidden)
        output = self.head(hidden)

        return output


class LanguageConnector(Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()

        self.connectors = ModuleList(
            [Linear(input_size, output_size) for _ in range(hidden_size)]
        )

    def forward(self, state: Tensor) -> Tensor:
        return stack([c(state) for c in self.connectors], dim=1)


class TransformerEncoderForSequenceSummarizationGPT2(Module):
    pass


__all__ = [
    "Defaults",
    "TransformerEncoder",
    "TransformerEncoderForMaskedLM",
    "TransformerEncoderForSequenceSimilarity",
    "TransformerEncoderForSequenceClassification",
    "TransformerEncoderForSequenceSummarizationGPT2",
]
