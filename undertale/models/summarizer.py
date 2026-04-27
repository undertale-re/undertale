"""Sequence summarization implementation."""

from typing import Optional

import torch
from torch import Tensor, cat
from torch.nn import GELU, Linear, Module, ModuleList, Sequential
from transformers import GPT2Config, GPT2LMHeadModel

from .custom import InstructionTraceTransformerEncoder
from .tokenizer import SPECIAL_TOKENS, TOKEN_NEXT
from .transformer import TransformerEncoderLayer


class MLP(Module):
    """A simple multi-layer perceptron connector."""

    def __init__(self, sizes, bias: bool = True, act=GELU):
        super().__init__()

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())

        self.model = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Project the input tensor through the connector network."""

        return self.model(x)


class TransformerConnector(Module):
    """A transformer-based connector from encoder features to GPT prefixes."""

    def __init__(
        self,
        prefix_size: int,
        dim_embedding: int,
        prefix_length: int,
        assembly_length: int,
        num_layers: int = 8,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.assembly_length = assembly_length
        self.linear = Linear(prefix_size, assembly_length * dim_embedding)
        self.prefix_const = torch.nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    dim_embedding,
                    heads,
                    dim_embedding * 2,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map an encoder embedding into learned GPT prefix embeddings."""

        x = self.linear(x).view(x.shape[0], self.assembly_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        output = cat((x, prefix), dim=1)

        for layer in self.layers:
            output = layer(output)

        return output[:, self.assembly_length :]


class InstructionTraceTransformerEncoderForSequenceSummarization(Module):
    """Legacy-compatible sequence summarizer built from modular components."""

    def __init__(
        self,
        assembly_encoder_config,
        connector_config,
        llm_config,
        end2end: bool = True,
        tune_llm: bool = True,
    ):
        super().__init__()

        self.tune_llm = tune_llm
        self.assembly_encoder = None
        if end2end:
            next_token_id = assembly_encoder_config.get(
                "next_token_id", SPECIAL_TOKENS.index(TOKEN_NEXT)
            )
            sequence_length = assembly_encoder_config.get(
                "sequence_length",
                assembly_encoder_config.get("input_size"),
            )
            if sequence_length is None:
                raise KeyError(
                    "assembly_encoder_config must define either 'sequence_length' or 'input_size'"
                )
            self.assembly_encoder = InstructionTraceTransformerEncoder(
                assembly_encoder_config["depth"],
                assembly_encoder_config["hidden_dimensions"],
                assembly_encoder_config["vocab_size"],
                sequence_length,
                assembly_encoder_config["heads"],
                assembly_encoder_config["intermediate_dimensions"],
                next_token_id,
                assembly_encoder_config["dropout"],
                assembly_encoder_config["eps"],
            )

        self.llm = GPT2LMHeadModel(GPT2Config(**llm_config))
        self.tokenizer = None
        self.stop_token = None

        if not self.tune_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        else:
            self.llm.train()

        self.llm_embedding_size = self.llm.transformer.wte.weight.shape[1]
        self.prefix_length_const = connector_config["prefix_length_const"]
        prefix_length_assembly = connector_config["prefix_length_assembly"]

        output_size = self.llm_embedding_size * self.prefix_length_const
        hidden_size = output_size // 2

        if connector_config["model_type"].lower() == "mlp":
            self.connector = MLP(
                (connector_config["prefix_size"], hidden_size, output_size)
            )
        else:
            self.connector = TransformerConnector(
                connector_config["prefix_size"],
                self.llm_embedding_size,
                self.prefix_length_const,
                prefix_length_assembly,
                connector_config["num_layers"],
            )

    def set_tokenizer(self, tokenizer) -> None:
        """Set the decoder tokenizer used for generation."""

        self.tokenizer = tokenizer
        self.stop_token = tokenizer.eos_token_id

    def masked_mean_pool(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Pool sequence features while ignoring padding."""

        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        text_tokens: Tensor,
        encoder_embedding: Tensor,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ):
        """Run summarization with a learned prefix prepended to decoder tokens."""

        embedding_text = self.llm.transformer.wte(text_tokens)
        if encoder_embedding.dim() == 3:
            if encoder_attention_mask is None:
                raise ValueError(
                    "encoder_attention_mask is required when encoder_embedding is 3D."
                )
            encoder_embedding = self.masked_mean_pool(
                encoder_embedding, encoder_attention_mask
            )
        prefixes = self.connector(encoder_embedding).view(
            -1, self.prefix_length_const, self.llm_embedding_size
        )

        embedding_cat = torch.cat((prefixes, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(text_tokens.shape[0], text_tokens.device)
            labels = torch.cat((dummy_token, text_tokens), dim=1)

        return self.llm(
            inputs_embeds=embedding_cat,
            labels=labels,
            attention_mask=mask,
        )

    def embed_assembly(
        self, assembly_tokens: Tensor, assembly_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode assembly tokens using the optional instruction-trace encoder."""

        if self.assembly_encoder is None:
            raise RuntimeError(
                "assembly_encoder is not initialized for non-end2end mode."
            )
        if self.assembly_encoder.training:
            self.assembly_encoder.eval()
        return self.assembly_encoder(assembly_tokens, assembly_mask)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> Tensor:
        """Create ignored labels for the learned prefix positions."""

        return torch.full(
            (batch_size, self.prefix_length_const),
            -100,
            device=device,
            dtype=torch.long,
        )

    def generate(
        self,
        tokens: Optional[Tensor] = None,
        prefix: Optional[str] = None,
        embed: Optional[Tensor] = None,
        entry_length: int = 67,
        top_p: float = 0.8,
        temperature: float = 1.0,
        do_sample: bool = False,
        beam: bool = False,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        min_new_tokens: int = 1,
    ) -> str:
        """Generate a summary from prompt tokens or prompt embeddings."""

        if self.tokenizer is None or self.stop_token is None:
            raise RuntimeError("Tokenizer must be set before calling generate().")

        self.llm.eval()
        device = next(self.llm.parameters()).device

        stop_token_index = self.stop_token
        if temperature <= 0:
            temperature = 1.0

        with torch.no_grad():
            if embed is not None:
                inputs_embeds = embed.to(device)
            else:
                if tokens is None:
                    tokens = torch.tensor(
                        self.tokenizer.encode(prefix or ""),
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                else:
                    tokens = tokens.to(device)
                inputs_embeds = self.llm.transformer.wte(tokens)

            attention_mask = torch.ones(
                inputs_embeds.size()[:2], dtype=torch.long, device=device
            )

            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": entry_length,
                "min_new_tokens": min_new_tokens,
                "pad_token_id": 0,
                "eos_token_id": stop_token_index,
            }

            if beam:
                gen_kwargs.update(
                    do_sample=do_sample,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    length_penalty=length_penalty,
                )
            else:
                gen_kwargs.update(
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                )

            output_ids = self.llm.generate(**gen_kwargs)[0].tolist()
            if self.stop_token in output_ids:
                output_ids = output_ids[: output_ids.index(self.stop_token)]

            return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class TransformerEncoderForSequenceSummarization(
    InstructionTraceTransformerEncoderForSequenceSummarization
):
    """Legacy class name preserved for low-churn migration."""


__all__ = [
    "TransformerEncoderForSequenceSummarization",
    "InstructionTraceTransformerEncoderForSequenceSummarization",
    "MLP",
    "TransformerConnector",
]
