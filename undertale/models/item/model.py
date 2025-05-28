from math import sqrt

from torch import Tensor, arange, bmm, cat, long, stack
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import softmax


class Defaults:
    depth = 12
    hidden_dimensions = 768
    input_size = 512
    heads = 12
    intermediate_dimension = 3072
    dropout = 0.1


class PositionEmbedding(Module):
    def __init__(self, hidden_dimensions: int, vocab_size: int, input_size: int):
        super().__init__()

        self.token = Embedding(vocab_size, hidden_dimensions)
        self.position = Embedding(input_size, hidden_dimensions)
        self.norm = LayerNorm(hidden_dimensions, eps=1e-12)
        self.dropout = Dropout()

    def forward(self, embedding):
        length = embedding.size(1)
        positions = arange(length, dtype=long).unsqueeze(0)

        tokens = self.token(embedding)
        positions = self.position(positions)

        embedded = tokens + positions

        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    scores = bmm(query, key.transpose(1, 2)) / sqrt(query.size(-1))
    weights = softmax(scores, dim=-1)
    return bmm(weights, value)


class Attention(Module):
    def __init__(self, embedded_dimensions: int, head_dimensions: int):
        super().__init__()

        self.q = Linear(embedded_dimensions, head_dimensions)
        self.k = Linear(embedded_dimensions, head_dimensions)
        self.v = Linear(embedded_dimensions, head_dimensions)

    def forward(self, state: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(state), self.k(state), self.v(state))


class MultiHeadAttention(Module):
    def __init__(self, embedded_dimensions: int, heads: int):
        super().__init__()

        head_dimensions = embedded_dimensions // heads

        self.heads = ModuleList(
            [Attention(embedded_dimensions, head_dimensions) for _ in range(heads)]
        )

        self.output = Linear(embedded_dimensions, embedded_dimensions)

    def forward(self, state):
        attended = cat([h(state) for h in self.heads], dim=-1)
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

    def forward(self, state: Tensor) -> Tensor:
        # Vanila.
        # attended = self.attention(state)
        # output = self.ff(attended)

        # Add layer normalization.
        # attended = self.attention(self.norm1(state))
        # output = self.ff(self.norm2(attended))

        # Add skip connections.
        hidden = self.norm1(state)
        output = state + self.attention(hidden)
        hidden = self.norm2(output)
        output = output + self.ff(hidden)

        return output


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

        self.embedding = PositionEmbedding(hidden_dimensions, vocab_size, input_size)
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_dimensions, heads, intermediate_dimensions, dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(self, state: Tensor) -> Tensor:
        output = self.embedding(state)

        for layer in self.layers:
            output = layer(output)

        return output


class MaskedLMHead(Module):
    def __init__(self, hidden_dimensions, vocab_size):
        super().__init__()

        self.linear = Linear(hidden_dimensions, vocab_size)

    def forward(self, state: Tensor) -> Tensor:
        return softmax(self.linear(state), dim=-1)


class TransformerEncoderForMaskedLM(Module):
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

    def forward(self, state: Tensor) -> Tensor:
        hidden = self.encoder(state)
        output = self.head(hidden)

        return output


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

    def forward(self, state: Tensor) -> Tensor:
        # Select only the final state to classify.
        hidden = self.encoder(state)[:, 0, :]
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
