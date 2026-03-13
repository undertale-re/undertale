from undertale.models.classification_connector import MLP
from undertale.models.item.model import TransformerEncoderForMaskedLM


class TransformerEncoderForSequenceClassification(TransformerEncoderForMaskedLM):
    def __init__(
        self,
        # parameters from source model
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
        # specific parameters for sequence classification
        num_classes: int = 2,
        head_hidden_size: int = 64,
    ):
        super().__init__(
            depth,
            hidden_dimensions,
            vocab_size,
            input_size,
            heads,
            intermediate_dimensions,
            dropout,
            eps,
            lr,
            warmup,
        )

        output_size = 768  # self.assembly_encoder.hidden_dimensions

        self.head = MLP((output_size, head_hidden_size, num_classes))

    def embed_assembly(self, assembly_tokens, assembly_mask=None):

        if self.encoder.training:

            self.encoder.eval()
        return self.encoder(assembly_tokens, assembly_mask)

    def forward(self, encoder_embedding, mask=None, labels=None):
        encoder_embedding = encoder_embedding.mean(dim=1)

        out = self.head(encoder_embedding)

        return out
