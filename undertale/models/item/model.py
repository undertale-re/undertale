from math import sqrt
from typing import Optional

from lightning import LightningModule
from sklearn.metrics import f1_score
from torch import (
    Tensor,
    arange,
    argmax,
    bincount,
    bmm,
    cat,
    cumsum,
    long,
    roll,
    softmax,
    stack,
    where,
    zeros,
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from . import tokenizer
from .tokenizer import SPECIAL_TOKENS, TOKEN_NEXT


class Defaults:
    input_size = 512
    depth = 12
    heads = 12
    hidden_dimensions = 768
    intermediate_dimensions = 3072
    dropout = 0.1
    eps = 1e-12
    lr = 1e-4
    warmup = 0.5


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    scores = bmm(query, key.transpose(-2, -1)) / sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2) == 0, -1e9)

    weights = softmax(scores, dim=-1)
    return bmm(weights, value)


class Attention(Module):
    def __init__(self, hidden_dimensions: int, head_dimensions: int):
        super().__init__()

        self.q = Linear(hidden_dimensions, head_dimensions)
        self.k = Linear(hidden_dimensions, head_dimensions)
        self.v = Linear(hidden_dimensions, head_dimensions)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return scaled_dot_product_attention(
            self.q(state), self.k(state), self.v(state), mask=mask
        )


class MultiHeadAttention(Module):
    def __init__(self, hidden_dimensions: int, heads: int):
        super().__init__()

        head_dimensions = hidden_dimensions // heads

        self.heads = ModuleList(
            [Attention(hidden_dimensions, head_dimensions) for _ in range(heads)]
        )

        self.output = Linear(hidden_dimensions, hidden_dimensions)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        attended = cat([h(state, mask) for h in self.heads], dim=-1)
        output = self.output(attended)

        return output


class FeedForward(Module):
    def __init__(
        self, hidden_dimensions: int, intermediate_dimensions: int, dropout: float
    ):
        super().__init__()

        self.linear1 = Linear(hidden_dimensions, intermediate_dimensions)
        self.linear2 = Linear(intermediate_dimensions, hidden_dimensions)
        self.activation = GELU()
        self.dropout = Dropout(dropout)

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.activation(self.linear1(state))
        output = self.dropout(self.linear2(hidden))

        return output


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        hidden_dimensions: int,
        heads: int,
        intermediate_dimensions: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = LayerNorm(hidden_dimensions)
        self.attention = MultiHeadAttention(hidden_dimensions, heads)
        self.norm2 = LayerNorm(hidden_dimensions)
        self.ff = FeedForward(hidden_dimensions, intermediate_dimensions, dropout)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Vanila.
        # attended = self.attention(state)
        # output = self.ff(attended)

        # Add layer normalization.
        # attended = self.attention(self.norm1(state))
        # output = self.ff(self.norm2(attended))

        # Add skip connections.
        hidden = self.norm1(state)
        output = state + self.attention(hidden, mask)
        hidden = self.norm2(output)
        output = output + self.ff(hidden)

        return output


class PositionEmbedding(Module):
    def __init__(
        self,
        hidden_dimensions: int,
        vocab_size: int,
        input_size: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()

        self.token = Embedding(vocab_size, hidden_dimensions)
        self.instruction = Embedding(input_size, hidden_dimensions)
        self.argument = Embedding(input_size, hidden_dimensions)
        self.norm = LayerNorm(hidden_dimensions, eps=eps)
        self.dropout = Dropout(dropout)

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
        eps: float,
    ):
        super().__init__()

        self.embedding = PositionEmbedding(
            hidden_dimensions, vocab_size, input_size, dropout, eps
        )
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_dimensions, heads, intermediate_dimensions, dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = self.embedding(state)

        for layer in self.layers:
            output = layer(output, mask)

        return output


class MaskedLMHead(Module):
    def __init__(self, hidden_dimensions: int, vocab_size: int, eps: float):
        super().__init__()

        self.transform = Linear(hidden_dimensions, hidden_dimensions)
        self.activation = GELU()
        self.norm = LayerNorm(hidden_dimensions, eps=eps)

        self.decode = Linear(hidden_dimensions, vocab_size)

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.activation(self.transform(state))
        hidden = self.norm(hidden)

        output = self.decode(hidden)

        return output


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
        eps: float,
        lr: float,
        warmup: float,
        tokenizer_loc: str = "",
        val_example_count: int = 5
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
            eps,
        )
        self.head = MaskedLMHead(hidden_dimensions, vocab_size, eps)
        if tokenizer_loc != "":
            self.tok = tokenizer.load(tokenizer_loc)
        else:
            self.tok = None
        self.val_example_count = val_example_count

        self.lr = lr
        self.warmup = warmup
        self.steps_per_epoch = None

    def on_fit_start(self):
        self.steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        hidden = self.encoder(state, mask)
        output = self.head(hidden)

        return output

    def configure_optimizers(self):
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
        if self.tok is not None:
            if int(index) < self.val_example_count:
                filled = where(
                batch.input_ids == self.tok.token_to_id(tokenizer.TOKEN_MASK), argmax(output, dim=-1), batch.input_ids
                )
                input_seq = self.tok.decode(batch.input_ids[0].tolist(), skip_special_tokens=False).replace("[NEXT]", "\n").replace("[PAD]", "").strip()
                predicted = self.tok.decode(filled[0].tolist(), skip_special_tokens=False)
                predicted = predicted.replace(tokenizer.TOKEN_PAD, "").replace("[NEXT]", "\n").strip()
                if isinstance(self.logger.experiment, SummaryWriter):
                    self.logger.experiment.add_text("mask prediction", f"index: {index}\ninput: {input_seq}\n\noutput:{predicted}")



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
