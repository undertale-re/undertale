from math import sqrt
from typing import Optional

import torch
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
    zeros,
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, GPT2LMHeadModel

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


# from typing import Tuple

# eventually move this to its own file
# from torch import nn


# class MLP(nn.Module):

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)

#     def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
#         super(MLP, self).__init__()
#         layers = []
#         for i in range(len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
#             if i < len(sizes) - 2:
#                 layers.append(act())
#         self.model = nn.Sequential(*layers)


class TransformerEncoderForSequenceClassification(Module):
    def __init__(
        self,
        assembly_checkpoint,
        num_classes=2,
        end2end=True,
        tune_llm=True,
        head_hidden_size=64,
    ):
        super().__init__()

        self.tune_llm = tune_llm

        self.assembly_checkpoint = assembly_checkpoint
        self.assembly_encoder = None
        if end2end:
            self.assembly_encoder = TransformerEncoderForMaskedLM.load_from_checkpoint(
                assembly_checkpoint
            )
        else:
            raise ValueError("For now only end2end is supported")
        output_size = self.assembly_encoder.hidden_dimensions

        self.head = MLP(output_size, head_hidden_size, num_classes)

    def embed_assembly(self, assembly_tokens, assembly_mask=None):

        if self.assembly_encoder is None:
            self.assembly_encoder = TransformerEncoderForMaskedLM.load_from_checkpoint(
                self.assembly_checkpoint
            )
        if self.assembly_encoder.training:

            self.assembly_encoder.eval()
        return self.assembly_encoder.encoder(assembly_tokens, assembly_mask)

    def forward(self, encoder_embedding, mask=None, labels=None):
        encoder_embedding = encoder_embedding.mean(dim=1)

        out = self.head(encoder_embedding)

        return out


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
        self.assembly_encoder = None
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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_checkpoint, local_files_only=True
            )
            self.stop_token = self.tokenizer.eos_token_id
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model/tokenizer from {llm_checkpoint}. "
                f"Make sure you download it first.\n{e}"
            )

        if not self.tune_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        else:
            print("TRAINING LLM")
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

        if self.assembly_encoder is None:
            self.assembly_encoder = TransformerEncoderForMaskedLM.load_from_checkpoint(
                self.assembly_checkpoint
            )
        if self.assembly_encoder.training:

            self.assembly_encoder.eval()
        return self.assembly_encoder.encoder(assembly_tokens, assembly_mask)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, self.prefix_length_const),
            -100,
            device=device,
            dtype=torch.long,
        )

    def generate(
        self,
        tokens=None,
        prefix=None,
        embed=None,
        entry_length=67,
        top_p=0.8,
        temperature=1.0,
        do_sample=False,
        beam=False,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=True,
        min_new_tokens=1,
    ):
        self.llm.eval()
        device = next(self.llm.parameters()).device

        stop_token_index = (
            self.stop_token
        )  # your custom stop token id (may or may not equal eos)

        if temperature is None or temperature <= 0:
            temperature = 1.0

        pad_token_id = 0

        with torch.no_grad():
            # 1) prompt embeds
            if embed is not None:
                inputs_embeds = embed.to(device)  # [1, T, n_embd]
            else:
                if tokens is None:
                    tokens = torch.tensor(
                        self.tokenizer.encode(prefix or ""),
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                else:
                    tokens = tokens.to(device)
                inputs_embeds = self.llm.transformer.wte(tokens)  # [1, T, n_embd]

            # 2) attention mask (no padding here => all ones)
            attention_mask = torch.ones(
                inputs_embeds.size()[:2], dtype=torch.long, device=device
            )

            # 3) generation kwargs
            gen_kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=entry_length,
                min_new_tokens=min_new_tokens,
                pad_token_id=pad_token_id,
            )

            gen_kwargs["eos_token_id"] = stop_token_index

            # Beam toggle
            if beam:
                # Standard beam search is deterministic; sampling is typically OFF.
                gen_kwargs.update(
                    do_sample=do_sample,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    length_penalty=length_penalty,
                )
            else:
                # Sampling / nucleus
                gen_kwargs.update(
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                )

            out_ids = self.llm.generate(**gen_kwargs)[0].tolist()
            if self.stop_token in out_ids:
                out_ids = out_ids[: out_ids.index(self.stop_token)]
            # Decode full sequence
            output_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            return output_text


#     def generate2(
#         self,
#         tokens=None,
#         prefix=None,
#         embed=None,         # optional: prompt token embeddings [1, T, n_embd]
#         entry_length=67,    # max new tokens
#         top_p=0.8,
#         temperature=1.0,
#         do_sample=True,
#     ):
#         """
#         Adaptation of your old generate2 to use HF `model.generate(inputs_embeds=...)`
#         with Option A multimodal soft-prefix prepended at the embedding layer.

#         Backward-compatible call pattern:
#           generate2(tokens=..., prefix="text prompt", embed=...)

#         New multimodal prefix:
#           generate2(prefix="text prompt", mm_prefix=z)

#         Requirements:
#           - self.modality_to_prefix(mm_prefix) -> [1, P, n_embd]
#           - self.stop_token is an int token id (or None)
#         """
#         self.llm.eval()
#         device = next(self.llm.parameters()).device
#         stop_token_index = self.stop_token
#         if temperature is None or temperature <= 0:
#             temperature = 1.0

#         with torch.no_grad():
#             # -------------------------
#             # 1) Build prompt tokens + embeds (same as old, but "prefix" stays the text prompt)
#             # -------------------------
#             if embed is not None:
#                 prompt_embeds = embed.to(device)  # [1, T, n_embd]
#                 # For decoding: if tokens not provided, we'll decode from generated ids returned by generate.
#                 prompt_tokens = tokens.to(device) if tokens is not None else None
#             else:
#                 if tokens is None:
#                     tokens = torch.tensor(self.tokenizer.encode(prefix or ""), dtype=torch.long)
#                     tokens = tokens.unsqueeze(0).to(device)  # [1, T]
#                 else:
#                     tokens = tokens.to(device)

#                 prompt_tokens = tokens
#                 prompt_embeds = self.llm.transformer.wte(prompt_tokens)  # [1, T, n_embd]

#             # -------------------------
#             # 2) Build multimodal soft-prefix embeds (Option A) and prepend
#             # -------------------------
#             prefix_len = 0

#             inputs_embeds = prompt_embeds  # [1, T, n_embd]

#             # attention_mask: all ones (no padding)
#             attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)

#             # -------------------------
#             # 3) Generate with HF
#             # -------------------------
#             out_ids = self.llm.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 max_new_tokens=entry_length,
#                 do_sample=do_sample,
#                 top_p=top_p,
#                 temperature=temperature,
#                 eos_token_id=stop_token_index if stop_token_index is not None else self.tokenizer.eos_token_id,
#                 pad_token_id=stop_token_index,
#             )  # [1, (P+T)+new] in "dummy token space"

#             #print(out_ids)
#             # Drop the dummy prefix positions before decoding
#             #out_ids = out_ids[:, prefix_len:]  # [1, T+new] aligned to real tokens

#             # Match your old behavior: decode the full returned sequence (prompt + continuation)
#             output_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=False)
#             return output_text


__all__ = [
    "Defaults",
    "TransformerEncoder",
    "TransformerEncoderForMaskedLM",
    "TransformerEncoderForSequenceSimilarity",
    "TransformerEncoderForSequenceClassification",
    "TransformerEncoderForSequenceSummarization",
]
