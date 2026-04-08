import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from undertale.models.classification_connector import MLP
from undertale.models.maskedlm import InstructionTraceTransformerEncoderForMaskedLM


class TransformerEncoderForSequenceClassification(
    InstructionTraceTransformerEncoderForMaskedLM
):
    LR = 2e-5
    WARMUP = 0.025

    def __init__(
        self,
        # parameters from source model
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        heads: int,
        intermediate_dimensions: int,
        next_token_id: int,
        mask_token_id: int,
        dropout: float,
        eps: float,
        lr: float = LR,
        warmup: float = WARMUP,
        # specific parameters for sequence classification
        num_classes: int = 2,
        head_hidden_size: int = 64,
        balance_weights: list[float] = [1.0, 1.0],
    ):
        super().__init__(
            depth,
            hidden_dimensions,
            vocab_size,
            sequence_length,
            heads,
            intermediate_dimensions,
            next_token_id,
            mask_token_id,
            dropout,
            eps,
            lr=lr,
            warmup=warmup,
        )

        output_size = 768  # self.assembly_encoder.hidden_dimensions

        for param in self.parameters():
            param.requires_grad = False

        self.head = MLP((output_size, head_hidden_size, num_classes))
        self.end_to_end = True
        self.balance_weights = torch.tensor(balance_weights)

    def embed_assembly(self, assembly_tokens, assembly_mask=None):
        # if self.encoder.training:
        #     self.encoder.eval()
        return self.encoder(assembly_tokens, assembly_mask)

    def forward(self, inp, mask=None):
        encoder_embedding = self.encoder(inp, mask).mean(dim=1)
        return self.head(encoder_embedding)

    def training_step(self, batch, batch_idx):

        labels, dissassembly_tokens, dissassembly_mask = (
            batch["labels"],
            batch["tokens"],
            batch["mask"],
        )

        outputs = self.forward(dissassembly_tokens, dissassembly_mask)

        loss = F.cross_entropy(
            outputs, labels, weight=self.balance_weights.to(outputs.device)
        )

        self.log("train_loss", loss, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        labels, dissassembly_tokens, dissassembly_mask = (
            batch["labels"],
            batch["tokens"],
            batch["mask"],
        )

        outputs = self.forward(dissassembly_tokens, dissassembly_mask)

        loss = F.cross_entropy(
            outputs, labels, weight=self.balance_weights.to(outputs.device)
        )

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup * total_steps)
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        config_optim = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return config_optim
