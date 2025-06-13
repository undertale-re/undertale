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
    zeros,
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .tokenizer import SPECIAL_TOKENS, TOKEN_NEXT


class Defaults:
    input_size = 512
    depth = 12
    heads = 12
    hidden_dimensions = 768
    intermediate_dimensions = 3072
    embedding_size = 128
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




class CosineInstructionTraceDifference(Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        difference = cos(first, second)

        return self.activation(difference)


class EuclidianInstructionTraceDifference(Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        difference = torch.sqrt(torch.sum((first - second) ** 2, dim=-1))

        return self.activation(difference)


class LearnedInstructionTraceDifference(Module):
    def __init__(self, embedding_size:int):
        super().__init__()

        self.linear = nn.Linear(embedding_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        difference = torch.abs(first - second)
        outputs = self.linear(difference).squeeze()
        outputs = self.activation(outputs)

        return outputs


@dataclass
class EmbeddingModelOutput(ModelOutput):
    embedded: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def encode(self):
        data = self.embedded.squeeze().tolist()
        data = struct.pack(f"{len(data)}f", *data)
        data = base64.b64encode(data).decode("utf-8")

        return data

    @classmethod
    def decode(cls, data):
        data = base64.b64decode(data)
        data = struct.unpack(f"{len(data) // 4}f", data)
        data = torch.tensor(data).unsqueeze(0).to(device)

        return cls(embedded=data)


class InstructionTraceEmbeddingHead(Module):
    def __init__(self, hidden_size:int, embedding_size:int, embedding_dropout_prob:int):
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = nn.Linear(hidden_size // 2, embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout(outputs)

        return outputs


class TransformerEncoderForSequenceEmbedding(Module):

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
            embedding_size: int,
            embedding_dropout_prob: float):
        ):

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
        
        self.embedding = InstructionTraceEmbeddingHead(hidden_dimensions, embedding_size, embedding_dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool sequence model by taking the embedding of the [CLS] token - this
        # is how HuggingFace handles pooling, we just don't want the extra
        # dense layer that they add
        pooled = outputs.last_hidden_state[:, 0]

        # Mean pooling - Trex's approach
        # pooled = output.last_hidden_state.mean(dim=-2)

        embedded = self.embedding(pooled)

        return EmbeddingModelOutput(
            embedded=embedded,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
class TransformerEncoderForSequenceSimilarity(Module):
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
            embedding_size: int,
            embedding_dropout_prob: float):
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.warmup = warmup
        self.steps_per_epoch = None

        self.embedding = TransformerEncoderForSequenceEmbedding(
            # these are for TransformerEncoder
            depth,
            hidden_dimensions,
            vocab_size,
            input_size,
            heads,
            intermediate_dimensions,
            dropout,
            eps,
            # these are for the embedding head
            hidden_dimension, 
            128, # embedding_size
            0.1 # embedding_dropout_prob            
        )
        
        self.difference = CosineInstructionTraceDifference()

        
    def on_fit_start(self):
        self.steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )


    def forward(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels=None
    ):
        embedded1 = self.embedding(input_ids=input_ids1, attention_mask=attention_mask1)
        embedded2 = self.embedding(input_ids=input_ids2, attention_mask=attention_mask2)

        logits = self.difference(embedded1.embedded, embedded2.embedded)

        if labels is not None:
            function = nn.BCELoss()
            loss = function(logits, labels.float())
        else:
            loss = None

        return SimilarityModelOutput(
            loss=loss,
            logits=logits,
            embedded1=embedded1.embedded,
            embedded2=embedded2.embedded,
            hidden_states1=embedded1.hidden_states,
            hidden_states2=embedded2.hidden_states,
            attentions1=embedded1.attentions,
            attentions2=embedded2.attentions,
        )

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
        # does this work?
        outputs = self(**batch)
        return outputs.loss

    def validation_step(self, batch, index):
        outputs = self(**batch)
        references = batch["labels"]
        predictions = outputs.logits.squeeze()
        f1 = f1_score(references.tolist(), predictions.tolist(), average="micro")
        self.log("valid_f1", f1, prog_bar=True, sync_dist=True)



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
