# import os
# import time

from transformers import (
    PreTrainedTokenizerFast,
)

from . import tokenizer


class CustomCollator:
    def __init__(self, args, max_seq_len, device, pad_id):

        self.tokenizer = tokenizer.load(args.tokenizer)
        self.max_length = args.tokenizer_size

        self.tok_fast = None
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.prefix_length = args.prefix_length_const
        self.device = device

    def __call__(self, batch):
        if self.tok_fast is None:
            self.tok_fast = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        labels, disassembly_infos = zip(*batch)

        disassembly_batch = self.tok_fast(
            disassembly_infos,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {
            "labels": labels,
            "disassembly_tokens": disassembly_batch["input_ids"],
            "disassembly_mask": disassembly_batch["attention_mask"],
        }
