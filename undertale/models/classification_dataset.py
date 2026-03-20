# import os
# import time
import torch
from torch import (
    stack,
    tensor,
)


class CustomCollator:
    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def __call__(self, batch):
        labels = torch.tensor([item["label"] for item in batch]).to(int)
        tokens = stack([tensor(item["tokens"]) for item in batch])
        mask = stack([tensor(item["mask"]) for item in batch])

        return {"tokens": tokens, "mask": mask, "labels": labels}
