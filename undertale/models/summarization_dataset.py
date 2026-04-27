"""Summarization dataset and collation utilities."""

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from . import tokenizer


class SummarizerDataset(torch.utils.data.Dataset):
    """Dataset wrapper supporting raw, tokenized, and prefix-based assembly inputs."""

    def __init__(
        self,
        dataset,
        prefix_length,
        gpt2path,
        normalize_prefix=False,
        end2end=True,
        token_batchsize=1024,
        summary_tokens_column="summary_tokens",
        summary_text_column="summary",
        assembly_text_column="disassembly",
        assembly_tokens_column="disassembly_tokens",
        assembly_mask_column="disassembly_mask",
        assembly_prefix_column="disassembly_prefixes",
    ):
        self.end2end = end2end
        self.normalize_prefix = normalize_prefix

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                gpt2path, local_files_only=True
            )
            self.stop_token = self.tokenizer.eos_token_id
            config = AutoConfig.from_pretrained(gpt2path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model/tokenizer from {gpt2path}. "
                f"Make sure you download it first.\n{exc}"
            ) from exc

        self.prefix_length = prefix_length
        self.pad_id = 0
        self.dataset = dataset

        self.summary_tokens_column = summary_tokens_column
        self.summary_text_column = summary_text_column
        self.assembly_text_column = assembly_text_column
        self.assembly_tokens_column = assembly_tokens_column
        self.assembly_mask_column = assembly_mask_column
        self.assembly_prefix_column = assembly_prefix_column

        if hasattr(config, "n_positions"):
            max_positions = config.n_positions
        elif hasattr(config, "max_position_embeddings"):
            max_positions = config.max_position_embeddings
        else:
            raise ValueError(
                f"Cannot determine max context length from {gpt2path}"
            )

        self.max_seq_len = max_positions - prefix_length - 1
        if self.max_seq_len < 0:
            raise ValueError(
                f"prefix_length={prefix_length} leaves no room for summary tokens "
                f"and the stop token with max_positions={max_positions}."
            )

        if self.summary_tokens_column not in self.dataset.column_names:
            self.tokenize_dataset(token_batchsize)

    def tokenize_dataset(self, batch_size=1024):
        """Tokenize summary text only when pre-tokenized ids are unavailable."""

        del batch_size

        captions = self.dataset[self.summary_text_column]
        tokenized = []
        for caption in captions:
            encoded = self.tokenizer.encode(
                caption, truncation=True, max_length=self.max_seq_len
            )
            encoded = encoded + [self.stop_token]
            tokenized.append(encoded)

        self.dataset = self.dataset.add_column(self.summary_tokens_column, tokenized)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int):
        row = self.dataset[item]
        tokens = row[self.summary_tokens_column]

        if self.end2end:
            if self.assembly_tokens_column in row and self.assembly_mask_column in row:
                disassembly_info = {
                    "mode": "tokenized_assembly",
                    "tokens": row[self.assembly_tokens_column],
                    "mask": row[self.assembly_mask_column],
                }
            else:
                disassembly_info = {
                    "mode": "raw_assembly",
                    "text": row[self.assembly_text_column],
                }
        else:
            prefix = row[self.assembly_prefix_column]
            if self.normalize_prefix:
                prefix = torch.tensor(prefix, dtype=torch.float32)
                norm = prefix.norm(2, -1, keepdim=True).clamp_min(1e-12)
                prefix = (prefix / norm).tolist()

            disassembly_info = {
                "mode": "assembly_prefix",
                "prefix": prefix,
            }

        return tokens, disassembly_info


class CustomCollator:
    """Collate summary tokens with raw assembly, tokenized assembly, or prefixes."""

    def __init__(self, args, max_seq_len, pad_id):
        self.tokenizer = (
            tokenizer.load(args.tokenizer, sequence_length=args.tokenizer_size)
            if args.tokenizer
            else None
        )
        self.max_length = args.tokenizer_size
        self.tok_fast = None
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.prefix_length = args.prefix_length_const

    def _pad_assembly_sequences(self, sequences, padding_value: int):
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, self.max_length),
            padding_value,
            dtype=torch.long,
        )

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_length)
            padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)

        return padded

    def _pad_summary_tokens(self, tokens):
        token_size = len(tokens)
        padded = torch.full((token_size, self.max_seq_len), -1, dtype=torch.long)

        for i, seq in enumerate(tokens):
            seq_len = min(len(seq), self.max_seq_len)
            padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)

        masks = padded.ge(0)
        padded = padded.masked_fill_(~masks, self.pad_id)
        prefix_mask = torch.ones((token_size, self.prefix_length), dtype=torch.float32)
        masks = torch.cat((prefix_mask, masks.float()), dim=1)
        return padded, masks

    def __call__(self, batch):
        if self.tokenizer is not None and self.tok_fast is None:
            self.tok_fast = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)

        tokens, disassembly_infos = zip(*batch)
        padded, masks = self._pad_summary_tokens(tokens)

        mode = disassembly_infos[0]["mode"]

        if mode == "raw_assembly":
            if self.tok_fast is None:
                raise ValueError(
                    "--tokenizer is required when assembly data is provided as raw text."
                )
            disassembly_batch = self.tok_fast(
                [item["text"] for item in disassembly_infos],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=self.max_length,
            )
            disassembly_tokens = disassembly_batch["input_ids"]
            disassembly_mask = disassembly_batch["attention_mask"]
        elif mode == "tokenized_assembly":
            disassembly_tokens = self._pad_assembly_sequences(
                [item["tokens"] for item in disassembly_infos],
                padding_value=0,
            )
            disassembly_mask = self._pad_assembly_sequences(
                [item["mask"] for item in disassembly_infos],
                padding_value=0,
            )
        elif mode == "assembly_prefix":
            disassembly_tokens = torch.tensor(
                [item["prefix"] for item in disassembly_infos],
                dtype=torch.float32,
            )
            disassembly_mask = torch.ones(
                disassembly_tokens.shape[:-1], dtype=torch.long
            )
        else:
            raise ValueError(f"Unsupported batch mode: {mode}")

        return {
            "tokens": padded,
            "mask": masks,
            "disassembly_tokens": disassembly_tokens,
            "disassembly_mask": disassembly_mask,
        }


__all__ = ["CustomCollator", "SummarizerDataset"]
