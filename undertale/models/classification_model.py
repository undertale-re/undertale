from classification_connector import MLP
from item.model import TransformerEncoderForMaskedLM
from torch.nn import Module


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
            ).encoder
        else:
            raise ValueError("For now only end2end is supported")
        output_size = 768  # self.assembly_encoder.hidden_dimensions

        self.head = MLP((output_size, head_hidden_size, num_classes))

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
