"""Transformer customizations for binary code."""

from typing import Optional

from torch import (
    Tensor,
    arange,
    cummax,
    cumsum,
    roll,
)
from torch.nn import Dropout, Embedding, LayerNorm, Module, ModuleList

from .transformer import TransformerEncoderLayer


class InstructionTracePositionEmbedding(Module):
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

    @staticmethod
    def compute_instruction_index(state: Tensor, next_token_id: int) -> Tensor:
        """Compute the instruction index for each token.

        Each token is assigned the index of the instruction it belongs to. The
        first instruction is index 0, incrementing after each ``NEXT`` token.

        Arguments:
            state: The input state tensor.
            next_token_id: The ID of the special ``NEXT`` token.

        Returns:
            A tensor of instruction indices.
        """

        starts = roll(state == next_token_id, 1)
        starts[:, 0] = False

        return cumsum(starts, dim=-1)

    @staticmethod
    def compute_argument_index(state: Tensor, next_token_id: int) -> Tensor:
        """Compute the argument index for each token within its instruction.

        Each token is assigned its position within the current instruction,
        resetting to 0 after each ``NEXT`` token.

        Arguments:
            state: The input state tensor.
            next_token_id: The ID of the special ``NEXT`` token.

        Returns:
            A tensor of argument indices.
        """

        starts = roll(state == next_token_id, 1)
        starts[:, 0] = False
        position = (
            arange(state.size(-1), device=state.device).unsqueeze(0).expand_as(state)
        )
        origin, _ = cummax(position * starts, dim=-1)
        return position - origin

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

        instructions = self.compute_instruction_index(state, self.next_token_id)
        arguments = self.compute_argument_index(state, self.next_token_id)

        tokens = self.token(state)
        instructions = self.instruction(instructions)
        arguments = self.argument(arguments)

        embedded = tokens + instructions + arguments

        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class InstructionTraceTransformerEncoder(Module):
    """A transformer encoder for instruction traces.

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
    """

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
    ):
        super().__init__()

        self.embedding = InstructionTracePositionEmbedding(
            hidden_dimensions, vocab_size, sequence_length, next_token_id, dropout, eps
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
