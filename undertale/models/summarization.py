"""Sequence summarization implementation."""

from typing import List, Optional, Tuple

import evaluate
from pandas import Series
from pandas import read_parquet as pandas_read_parquet
from pytorch_lightning import LightningModule
from torch import Tensor, cat, exp, full, long, ones, randn, stack, tensor
from torch.nn import GELU, Linear, Module, ModuleList, Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from ..logging import get_logger
from ..schema import SummarizedDataset, validate_dataset
from ..utils import assert_path_exists, get_or_create_file, write_parquet
from .custom import InstructionTraceTransformerEncoder
from .transformer import TransformerEncoderLayer

logger = get_logger(__name__)


def tokenize_summaries_gpt2(input: str, output: str) -> str:
    """Tokenize a summarized dataset for GPT2.

    Arguments:
        input: Path to the summarized dataset.
        output: Path where the tokenized summarized dataset should be written.

    Returns:
        The path to the tokenized dataset.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    tok = GPT2Tokenizer.from_pretrained(
        InstructionTraceTransformerEncoderForSequenceSummarizationGPT2.LANGUAGE
    )

    frame = pandas_read_parquet(input)

    validate_dataset(frame, SummarizedDataset)

    logger.info(f"tokenizing summaries {input!r} to {output!r}")

    def process(summary: str) -> Series:
        encoding = tok(summary)

        return Series(
            {
                "summary_tokens": encoding["input_ids"],
                "summary_mask": encoding["attention_mask"],
            }
        )

    frame[["summary_tokens", "summary_mask"]] = frame["summary"].apply(process)
    write_parquet(frame, output)

    logger.info(f"successfully tokenized {len(frame)} rows")

    return output


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
        connector_dimensions: Scaling factor; intermediate MLP size is
            ``language_dimensions × connector_dimensions``.
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

        intermediate_dimensions = language_dimensions * connector_dimensions / 8

        self.linear1 = Linear(hidden_dimensions, intermediate_dimensions)
        self.linear2 = Linear(
            intermediate_dimensions,
            language_dimensions * language_tokens,
        )
        self.activation = GELU()

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Project the input tensor into language token space.

        Arguments:
            state: Encoder hidden states of shape
                ``(batch, seq_len, hidden_dimensions)``.
            mask: Optional attention mask used for pooling.

        Returns:
            A batched projection into a tensor of shape ``(batch,
            language_tokens, language_dimensions)``.
        """

        if mask is not None:
            expanded = mask.unsqueeze(-1).float()
            pooled = (state * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = state.mean(dim=1)

        hidden = self.activation(self.linear1(pooled))
        values = self.linear2(hidden)
        output = values.view(-1, self.language_tokens, self.language_dimensions)

        return output


class TransformerConnector(Module):
    """A transformer language connector.

    Design inspiration taken from the paper "ClipCap: CLIP Prefix for Image
    Captioning."

    Uses learnable prefix queries that attend over the full encoder sequence
    via transformer self-attention to produce richer prefix representations
    than the MLP connector.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        connector_dimensions: Number of attention heads and transformer layers.
            Must evenly divide ``language_dimensions``.
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

        self.projection = Linear(hidden_dimensions, language_dimensions)
        self.prefix = Parameter(randn(language_tokens, language_dimensions))
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    language_dimensions,
                    connector_dimensions,
                    4 * language_dimensions,
                    dropout=0.0,
                )
                for _ in range(connector_dimensions)
            ]
        )

    def forward(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Project the input tensor into language token space.

        Arguments:
            state: Encoder hidden states of shape
                ``(batch, seq_len, hidden_dimensions)``.
            mask: Optional attention mask over the encoder sequence.

        Returns:
            A batched projection into a tensor of shape ``(batch,
            language_tokens, language_dimensions)``.
        """

        batch_size = state.size(0)
        projected = self.projection(state)
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
        combined = cat([prefix, projected], dim=1)

        if mask is not None:
            prefix_mask = ones(batch_size, self.language_tokens, device=state.device)
            combined_mask = cat([prefix_mask, mask.float()], dim=1)
        else:
            combined_mask = None

        for layer in self.layers:
            combined = layer(combined, combined_mask)

        return combined[:, : self.language_tokens, :]


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
        dropout: Dropout probability.
        eps: Layer normalization stabalization parameter.
        lr: Peak learning rate reached after warmup.
        warmup: Fraction of total steps used for linear warmup.
    """

    LANGUAGE = "gpt2"
    LR = 1e-4
    WARMUP = 0.025

    BERTSCORE = "distilbert-base-uncased"

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
        connector_dimensions: int = 8,
        language_tokens: int = 40,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.language_tokens = language_tokens

        gpt2_config = GPT2Config.from_pretrained(self.LANGUAGE)
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

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.lr = lr or self.LR
        self.warmup = warmup or self.WARMUP

        self.language_tokenizer = None
        self.rouge = None
        self.bertscore = None

    def encode(self, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode and compute language tokens.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.

        Returns:
            A tensor in language token space encoding the given input.
        """

        hidden = self.encoder(state, mask)
        return self.connector(hidden, mask)

    def forward(
        self,
        state: Tensor,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Encode and summarize the input sequence.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.
            labels: Optional summary ground truth tokens.

        Returns:
            Language model summary logits. If ``labels`` are provided a tuple
            of summary logits and computed loss are returned.
        """

        language_embedded = self.encode(state, mask)

        # Teacher forcing optimization, if labels are provided.
        #
        # Append target labels to input vector during training - allows single
        # step loss computation instead of iterative loss during generation.
        if labels is not None:
            labels_embedded = self.language.transformer.wte(labels)
            language_embedded = cat([language_embedded, labels_embedded], dim=1)

            # Ignore prefix tokens in loss computation (-100).
            prefix_labels = full(
                (state.size(0), self.language_tokens),
                -100,
                device=state.device,
                dtype=long,
            )
            labels = cat([prefix_labels, labels], dim=1)

        output = self.language(inputs_embeds=language_embedded, labels=labels)

        if labels is not None:
            return output.logits, output.loss
        else:
            return output.logits

    def generate(
        self, state: Tensor, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        """Generate a summary from a given input.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.
            **kwargs: Forwarded to ``GPT2LMHeadModel.generate`` (e.g.
                ``max_new_tokens``, ``do_sample``, ``temperature``).

        Returns:
            A tensor of generated token IDs.
        """

        language_embedded = self.encode(state, mask)

        return self.language.generate(inputs_embeds=language_embedded, **kwargs)

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
        labels = batch["summary_tokens"].masked_fill(
            ~batch["summary_mask"].bool(), -100
        )
        _, loss = self(batch["tokens"], batch["mask"], labels=labels)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        """"""
        if self.language_tokenizer is None:
            self.language_tokenizer = GPT2Tokenizer.from_pretrained(self.LANGUAGE)

        if self.rouge is None:
            self.rouge = evaluate.load("rouge")

        if self.bertscore is None:
            self.bertscore = evaluate.load("bertscore")

        # Compute Perplexity.
        labels = batch["summary_tokens"].masked_fill(
            ~batch["summary_mask"].bool(), -100
        )
        logits, loss = self(batch["tokens"], batch["mask"], labels=labels)
        perplexity = exp(loss)

        # Decode summary tokens for metrics.
        N = batch["summary_tokens"].size(1)
        pred_ids = logits[
            :, self.language_tokens - 1 : self.language_tokens + N - 1, :
        ].argmax(dim=-1)

        predictions = []
        references = []
        for i in range(batch["summary_tokens"].size(0)):
            mask = batch["summary_mask"][i].bool()
            predictions.append(
                self.language_tokenizer.decode(pred_ids[i][mask].tolist())
            )
            references.append(
                self.language_tokenizer.decode(
                    batch["summary_tokens"][i][mask].tolist()
                )
            )

        # Compute Rouge-L.
        rouge_l = self.rouge.compute(predictions=predictions, references=references)[
            "rougeL"
        ]

        # Compute BERTScore.
        result = self.bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=self.BERTSCORE,
        )
        bert_score = sum(result["f1"]) / len(result["f1"])

        self.log("valid_perplexity", perplexity, prog_bar=True, sync_dist=True)
        self.log("valid_rougeL", rouge_l, prog_bar=True, sync_dist=True)
        self.log("valid_bertscore", bert_score, prog_bar=True, sync_dist=True)
