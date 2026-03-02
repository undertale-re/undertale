"""Basic transformer implementation."""

from typing import Optional

from torch import (
    Tensor,
    arange,
    bincount,
    cat,
    cumsum,
    long,
    roll,
    zeros,
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import scaled_dot_product_attention


class Attention(Module):
    """Single-headed attention module.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        head_dimensions: The size of the output space.
    """

    def __init__(self, hidden_dimensions: int, head_dimensions: int):
        super().__init__()

        self.hidden_dimensions = hidden_dimensions

        self.q = Linear(hidden_dimensions, head_dimensions)
        self.k = Linear(hidden_dimensions, head_dimensions)
        self.v = Linear(hidden_dimensions, head_dimensions)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention.

        Arguments:
            state: The input state tensor.
            mask: Optional attention mask.

        Returns:
            A tensor in attended state space.
        """

        if state.ndim != 3:
            raise ValueError(
                f"expected tensor of shape (batch_size, sequence_length, hidden_size) but got {tuple(state.shape)}"
            )

        if state.shape[-1] != self.hidden_dimensions:
            raise ValueError(
                f"expected tensor with hidden size of {self.hidden_dimensions} but got tensor of shape {tuple(state.shape)}"
            )

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(
                    f"expected mask tensor of shape (batch_size, sequence_length) but got {tuple(mask.shape)}"
                )

            if mask.shape[-1] != state.shape[-2]:
                raise ValueError(
                    f"mismatched sequence length - got tensor of shape {tuple(state.shape)} and mask of shape {tuple(mask.shape)}"
                )

        return scaled_dot_product_attention(
            self.q(state),
            self.k(state),
            self.v(state),
            attn_mask=mask.unsqueeze(-2).bool() if mask is not None else None,
        )


class MultiHeadAttention(Module):
    """Multi-headed attention.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        heads: The number of attention heads.

    The number of dimensions for each head is computed as::

        head_dimensions = hidden_dimensions // heads
    """

    def __init__(self, hidden_dimensions: int, heads: int):
        if hidden_dimensions % heads != 0:
            raise ValueError(
                f"invalid number of heads, {heads} is not an even multiple of {hidden_dimensions}"
            )

        super().__init__()

        head_dimensions = hidden_dimensions // heads

        self.heads = ModuleList(
            [Attention(hidden_dimensions, head_dimensions) for _ in range(heads)]
        )

        self.output = Linear(hidden_dimensions, hidden_dimensions)

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention.

        Arguments:
            state: The input state tensor.
            mask: Optional attention mask.

        Returns:
            A tensor in attended state space.
        """

        attended = cat([h(state, mask) for h in self.heads], dim=-1)
        output = self.output(attended)

        return output


class FeedForward(Module):
    """A simple feed-forward network module.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        intermediate_dimensions: The size of the intermediate state space.
        dropout: Dropout probability.

    This computes a non-linear feature transformation over the inputs in
    ``intermediate_dimensions``.
    """

    def __init__(
        self, hidden_dimensions: int, intermediate_dimensions: int, dropout: float
    ):
        super().__init__()

        self.hidden_dimensions = hidden_dimensions

        self.linear1 = Linear(hidden_dimensions, intermediate_dimensions)
        self.linear2 = Linear(intermediate_dimensions, hidden_dimensions)
        self.activation = GELU()
        self.dropout = Dropout(dropout)

    def forward(self, state: Tensor) -> Tensor:
        """Compute forward pass.

        Arguments:
            state: The input state tensor.

        Returns:
            Computed state.
        """

        if state.shape[-1] != self.hidden_dimensions:
            raise ValueError(
                f"expected tensor with hidden size of {self.hidden_dimensions} but got tensor of shape {tuple(state.shape)}"
            )

        hidden = self.activation(self.linear1(state))
        output = self.dropout(self.linear2(hidden))

        return output


class TransformerEncoderLayer(Module):
    """An individual layer of an encoder with attention.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        dropout: Dropout probability.
    """

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
        """Compute attention plus non-linear transform.

        Includes regularization (layer normalization, dropout) and skip
        connections.

        Arguments:
            state: The input state tensor.
            mask: Optional attention mask.

        Returns:
            Transformed state.
        """

        hidden = self.norm1(state)
        output = state + self.attention(hidden, mask)
        hidden = self.norm2(output)
        output = output + self.ff(hidden)

        return output


class PositionEmbedding(Module):
    """Standard transformer positional encoding.

    Injects positional information by embedding and adding token index.

    Requires a fixed-size input vector (padding and truncation).

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
    """

    def __init__(
        self,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.token = Embedding(vocab_size, hidden_dimensions)
        self.position = Embedding(sequence_length, hidden_dimensions)
        self.norm = LayerNorm(hidden_dimensions, eps=eps)
        self.dropout = Dropout(dropout)

    def forward(self, state: Tensor) -> Tensor:
        """Inject positional information.

        Arguments:
            state: The input state tensor.

        Returns:
            Modified state with positional information.
        """

        if state.ndim != 2:
            raise ValueError(
                f"expected tensor of shape (batch_size, sequence_length) but got {tuple(state.shape)}"
            )

        if state.shape[-1] != self.sequence_length:
            raise ValueError(
                f"expected sequence length of {self.sequence_length} but got tensor of shape {tuple(state.shape)}"
            )

        length = state.size(1)
        positions = arange(length, dtype=long).unsqueeze(0)

        tokens = self.token(state)
        positions = self.position(positions)

        embedded = tokens + positions

        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class InstructionArgumentPositionEmbedding(Module):
    """Custom instruction and argument positional encoding.

    Injects positional information by embedding and adding two features to each
    token:

    1. Instruction number (e.g., ``foo bar baz NEXT blah`` -> ``0 0 0 - 1``).
    2. Argument number (e.g., ``foo bar baz NEXT blah`` -> ``0 1 2 - 0``).

    Requires a special ``NEXT`` token that is used to identify instruction
    boundaries.

    Requires a fixed-size input vector (padding and truncation).

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        next_token_id: The ID of the special ``NEXT`` token.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
    """

    def __init__(
        self,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        next_token_id: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.next_token_id = next_token_id

        self.token = Embedding(vocab_size, hidden_dimensions)
        self.instruction = Embedding(sequence_length, hidden_dimensions)
        self.argument = Embedding(sequence_length, hidden_dimensions)
        self.norm = LayerNorm(hidden_dimensions, eps=eps)
        self.dropout = Dropout(dropout)

    def forward(self, state: Tensor) -> Tensor:
        """Inject positional information.

        Arguments:
            state: The input state tensor.

        Returns:
            Modified state with positional information.
        """

        if state.ndim != 2:
            raise ValueError(
                f"expected tensor of shape (batch_size, sequence_length) but got {tuple(state.shape)}"
            )

        if state.shape[-1] != self.sequence_length:
            raise ValueError(
                f"expected sequence length of {self.sequence_length} but got tensor of shape {tuple(state.shape)}"
            )

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
    """A full transformer encoder with configurable positional encoding.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        embedding: Optional custom position embedding module.
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
        embedding: Optional[Module] = None,
    ):
        super().__init__()

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = PositionEmbedding(
                hidden_dimensions, vocab_size, sequence_length, dropout, eps
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
        """Encode the given state.

        Arguments:
            state: The input state tensor.
            mask: Optional attention mask.

        Returns:
            Encoded state.
        """

        output = self.embedding(state)

        for layer in self.layers:
            output = layer(output, mask)

        return output
