from math import sqrt
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput
from lightning import LightningModule
from sklearn.metrics import f1_score
import torch
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
    nn
)
from torch.nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, ModuleList, CosineSimilarity,Sigmoid,BCEWithLogitsLoss
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch.nn import CosineEmbeddingLoss

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

class CosineInstructionTraceDifference(Module):
    def __init__(self):
        super().__init__()

        self.activation = Sigmoid()

    def forward(self, first, second):
        cos = CosineSimilarity(dim=-1, eps=1e-6)
        difference = cos(first, second)

        return self.activation(difference)

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

@dataclass
class SimilarityModelOutput(ModelOutput):
    loss:torch.FloatTensor = None
    logits:Optional[Tuple[torch.FloatTensor]] = None
    embedded1: Optional[Tuple[torch.FloatTensor]]  = None
    hidden_states1: Optional[Tuple[torch.FloatTensor]] = None
    attentions1: Optional[Tuple[torch.FloatTensor]] = None
    embedded2: Optional[Tuple[torch.FloatTensor]] =  None
    hidden_states2: Optional[Tuple[torch.FloatTensor]] = None
    attentions2: Optional[Tuple[torch.FloatTensor]] = None

        

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

        self.linear1 = Linear(hidden_size, hidden_size // 2)
        self.linear2 = Linear(hidden_size // 2, embedding_size)
        self.activation = GELU()
        self.dropout = Dropout(embedding_dropout_prob)

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)
        outputs = self.linear1(inputs)

        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout(outputs)
        return outputs

class SimilarityLoss(Module):
    def __init__(self):
        super().__init__()
        self.simloss = CosineSimilarity(dim=2,eps=1e-6)
        self.activation = Sigmoid()
    def forward(self, first, second):
        logits = self.simloss(first, second)
        logits = self.activation(logits)
        return logits  


class TransformerEncoderForSequenceSimilarity(LightningModule, Module):
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

        
        self.lr = lr
        self.warmup = warmup
        self.steps_per_epoch = None
        self.embedloss = SimilarityLoss()
        self.bceloss = BCEWithLogitsLoss()
        self.head = MaskedLMHead(hidden_dimensions, vocab_size, eps)
        self.save_hyperparameters()
        


    def on_fit_start(self):
        self.steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

    def on_save_checkpoint(self, checkpoint):
        model_state_dict = checkpoint['state_dict']
        for param_name, param_tensor in model_state_dict.items():
            print(f"{param_name}\t{param_tensor.size()}")
            
    def forward(
        self, input_ids_d1, attention_mask_d1, input_ids_d2, attention_mask_d2,similarity=None
    ):
        embedded1 = self.encoder(input_ids_d1, attention_mask_d1)
        embedded2 = self.encoder(input_ids_d2, attention_mask_d2)
        diffembed = self.embedloss(embedded1, embedded2)
        diffembed = torch.mean(diffembed, dim=[1]) #Collapse (512,768) to mean value
        #similarity = (similarity +1) /2 # Convert -1 to 1 to 0 to 1 range
        return diffembed#similarity

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
        batch_size = batch["input_ids_d1"].shape[0]
        outputs = torch.empty(batch_size)
        outputs = []
        for i in range(batch_size):
            references = torch.stack([batch["similarity"][i]])
            diffembed = self(torch.stack([batch["input_ids_d1"][i,:]]),torch.stack([batch["attention_mask_d1"][i,:]]),torch.stack([batch["input_ids_d2"][i,:]]),torch.stack([batch["attention_mask_d2"][i,:]]))
            loss = self.bceloss(diffembed, references)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss 
            #outputs.append(torch.stack(loss))    
        #return torch.cat(outputs)

    def xvalidation_step(self, batch, index):
        references = batch["similarity"]
        batch_size = references.shape[0]
        for i in range(batch_size):
            target = torch.stack([batch["similarity"][i]])
            predictions = self(torch.stack([batch["input_ids_d1"][i,:]]),torch.stack([batch["attention_mask_d1"][i,:]]),torch.stack([batch["input_ids_d2"][i,:]]),torch.stack([batch["attention_mask_d2"][i,:]]))
            #threshold = 0.5
            #similarity = torch.where(loss > threshold, 1.0, 0.0) # Convert continous values to 0 - 1 
            f1 = f1_score(target.tolist(), predictions.tolist(), average="micro")
            self.log("valid_f1", f1, prog_bar=True, sync_dist=True)
        #return output

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