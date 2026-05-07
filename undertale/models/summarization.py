"""Sequence summarization implementation."""

from typing import Dict, List, Optional

from pytorch_lightning import LightningModule
from torch import Tensor, stack, tensor
from torch.nn import GELU, Linear, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import GPT2Config, GPT2LMHeadModel

from .custom import InstructionTraceTransformerEncoder


class SummarizationCollator:
    """Collation function for sequence summarization."""

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of dataset rows.

        Arguments:
            batch: A list of dataset rows.

        Returns:
            A batch ready for input to the summarization model.
        """

        tokens = stack([tensor(item["tokens"]) for item in batch])
        mask = stack([tensor(item["mask"]) for item in batch])
        summary_tokens = stack([tensor(item["summary_tokens"]) for item in batch])
        summary_mask = stack([tensor(item["summary_mask"]) for item in batch])

        return {
            "tokens": tokens,
            "mask": mask,
            "summary_tokens": summary_tokens,
            "summary_mask": summary_mask,
        }


class MLPConnector(Module):
    """A simple multi-layer perceptron language connector.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        connector_dimensions: The size of the intermediate state space.
        language_dimensions: The size of the language state space.
        language_tokens: The number of language tokens to produce.
    """

    def __init__(
        self,
        hidden_dimensions: int,
        connector_dimensions: int,
        language_dimensions: int,
        language_tokens: int,
    ):
        super().__init__()

        self.language_tokens = language_tokens
        self.language_dimensions = language_dimensions

        self.linear1 = Linear(hidden_dimensions, connector_dimensions)
        self.linear2 = Linear(
            connector_dimensions, language_dimensions * language_tokens
        )
        self.activation = GELU()

    def forward(self, state: Tensor) -> Tensor:
        """Project the input tensor into language token space.

        Arguments:
            state: Pooled encoder hidden state.

        Returns:
            A batched projection into a tensor of shape ``(language_tokens,
            language_dimensions)``.
        """

        hidden = self.activation(self.linear1(state))
        values = self.linear2(hidden)
        output = values.view(-1, self.language_tokens, self.language_dimensions)

        return output


class InstructionTraceTransformerEncoderForSequenceSummarizationGPT2(
    LightningModule, Module
):
    """A transformer encoder for multi-modal sequence summarization.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        next_token_id: The ID of the special ``NEXT`` token.
        connector_dimensions: The size of the intermediate state space.
        language_dimensions: The size of the language state space.
        language_tokens: The number of language tokens to produce.
        language_config: The GPT2 language model configuration as a dict.
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
        connector_dimensions: int,
        language_tokens: int,
        language_config: Dict,
        dropout: float,
        eps: float,
        lr: float = LR,
        warmup: float = WARMUP,
    ):
        super().__init__()

        self.save_hyperparameters()

        gpt2_config = GPT2Config.from_dict(language_config)
        language_dimensions = gpt2_config.n_embd

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
        self.connector = MLPConnector(
            hidden_dimensions,
            connector_dimensions,
            language_dimensions,
            language_tokens,
        )
        self.language = GPT2LMHeadModel(gpt2_config)

        self.lr = lr or self.LR
        self.warmup = warmup or self.WARMUP

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode and summarize the input sequence.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.

        Returns:
            Language model summary logits.
        """

        hidden = self.encoder(state, mask)

        if mask is not None:
            expanded = mask.unsqueeze(-1).float()
            pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        language_embedded = self.connector(pooled)

        output = self.language(inputs_embeds=language_embedded).logits

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

        raise NotImplementedError()

    def validation_step(self, batch, index):
        """"""

        raise NotImplementedError()
