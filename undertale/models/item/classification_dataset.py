# import os
# import time
import torch
from transformers import (
    PreTrainedTokenizerFast,
)

from undertale.models import tokenizer


class CustomCollator:
    def __init__(self, args, device):

        self.tokenizer = tokenizer.load(args.tokenizer)
        self.max_length = args.tokenizer_size
        self.tokenizer.pad_token = tokenizer.TOKEN_PAD
        self.tok_fast = None
        self.device = device

    def __call__(self, batch):
        if self.tok_fast is None:
            self.tok_fast = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        labels = torch.tensor([row["label"] for row in batch]).to(int)
        disassembly_infos = [row["disassembly"] for row in batch]
        self.tok_fast.pad_token = tokenizer.TOKEN_PAD
        disassembly_batch = self.tok_fast(
            disassembly_infos,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {
            "labels": labels.to(self.device),
            "disassembly_tokens": disassembly_batch["input_ids"].to(self.device),
            "disassembly_mask": disassembly_batch["attention_mask"].to(self.device),
        }
