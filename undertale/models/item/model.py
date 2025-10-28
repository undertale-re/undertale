from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as nnf
from lightning import LightningModule
from numpy import np
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
    zeros,
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel

from . import tokenizer
from .model_connector import MLP, TransformerConnector
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


class TransformerEncoderForSequenceSummarization(Module):
    """
    code based on https://github.com/rmokady/CLIP_prefix_caption/blob/main/train.py

    """

    def __init__(
        self,
        assembly_checkpoint,
        connector_config,
        llm_checkpoint,
        end2end=True,
        tune_llm=True,
    ):
        super().__init__()

        self.tune_llm = tune_llm

        self.assembly_checkpoint = assembly_checkpoint
        self.assembly_model = None
        if end2end:
            self.assembly_encoder = TransformerEncoderForMaskedLM.load_from_checkpoint(
                assembly_checkpoint
            )

        try:
            self.llm = GPT2LMHeadModel.from_pretrained(llm_checkpoint)
        except:
            print("downloading gpt2 from huggingface...")
            self.llm = GPT2LMHeadModel.from_pretrained("gpt2")
            self.llm.save_pretrained(llm_checkpoint)

        if not self.tune_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        else:
            self.llm.train()

        self.llm_embedding_size = self.llm.transformer.wte.weight.shape[1]
        self.prefix_length_const = connector_config.prefix_length_const
        prefix_length_assembly = connector_config.prefix_length_assembly

        output_size = self.llm_embedding_size * self.prefix_length_const
        hidden_size = output_size // 2

        if connector_config.model_type.lower() == "mlp":
            self.connector = MLP(
                (connector_config.prefix_size_const, hidden_size, output_size)
            )
        else:
            self.connector = TransformerConnector(
                connector_config.prefix_size,
                self.llm_embedding_size,
                self.prefix_length_const,
                prefix_length_assembly,
                connector_config.num_layers,
            )

    def forward(self, text_tokens, encoder_embedding, mask=None, labels=None):
        embedding_text = self.llm.transformer.wte(text_tokens)
        encoder_embedding = encoder_embedding.mean(dim=1)
        prefixes = self.connector(encoder_embedding).view(
            -1, self.prefix_length_const, self.llm_embedding_size
        )

        embedding_cat = torch.cat((prefixes, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(text_tokens.shape[0], text_tokens.device)
            labels = torch.cat((dummy_token, text_tokens), dim=1)

        out = self.llm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out

    def embed_assembly(self, assembly_tokens, assembly_mask=None):

        if self.assembly_model is None:
            self.assembly_encoder = TransformerEncoderForMaskedLM.load_from_checkpoint(
                self.assembly_checkpoint
            )
        if self.assembly_encoder.training:
            self.assembly_encoder.eval()
        return self.assembly_encoder.encoder(assembly_tokens, assembly_mask)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length_const, dtype=torch.int64, device=device
        )

    def generate_beam(
        self,
        beam_size: int = 5,
        prefix=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
    ):

        self.llm.eval()
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(self.llm.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prefix))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = self.llm.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = self.llm(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.llm.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            self.tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts

    def generate2(
        self,
        tokens=None,
        prefix=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
        stop_token: str = ".",
    ):

        self.llm.eval()
        generated_list = []
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(self.llm.parameters()).device

        with torch.no_grad():

            for entry_idx in range(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(self.tokenizer.encode(prefix))
                        tokens = tokens.unsqueeze(0).to(device)

                    generated = self.llm.transformer.wte(tokens)

                for i in range(entry_length):

                    outputs = self.llm(inputs_embeds=generated)
                    logits = outputs.logits
                    logits = logits[:, -1, :] / (
                        temperature if temperature > 0 else 1.0
                    )
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        nnf.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = self.llm.transformer.wte(next_token)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = self.tokenizer.decode(output_list)
                generated_list.append(output_text)

        return generated_list[0]


__all__ = [
    "Defaults",
    "TransformerEncoder",
    "TransformerEncoderForMaskedLM",
    "TransformerEncoderForSequenceSimilarity",
    "TransformerEncoderForSequenceClassification",
    "TransformerEncoderForSequenceSummarization",
]
