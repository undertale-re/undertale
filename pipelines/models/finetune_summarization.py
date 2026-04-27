"""Fine-tune the summarization model."""

import argparse
import json
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from undertale import logging as undertale_logging
from undertale.models.summarization_dataset import CustomCollator, SummarizerDataset
from undertale.models.summarizer import TransformerEncoderForSequenceSummarization
from undertale.models.tokenizer import TOKEN_NEXT
from undertale.models import tokenizer as undertale_tokenizer


def dataset_size_type(x):
    x = int(x)
    if x == 0 or x < -1:
        raise argparse.ArgumentTypeError(
            "dataset_size must be -1 for the full dataset or a positive integer."
        )
    return x


def optional_str(x):
    if x is None:
        return None
    value = str(x).strip()
    return None if value.lower() in {"", "none", "null"} else value


def load_preprocessing_spec(dataset_path):
    path = Path(dataset_path)
    candidate_dirs = [path] if path.is_dir() else [path.parent]
    for directory in candidate_dirs:
        spec_path = directory / "preprocessing_spec.json"
        if spec_path.exists():
            with spec_path.open("r", encoding="utf-8") as handle:
                return spec_path, json.load(handle)
    return None, None


def maybe_override_prefix_lengths_from_spec(args, spec):
    if spec is None:
        return args

    spec_prefix_length = spec.get("prefix_length")
    if spec_prefix_length is not None:
        args.prefix_length_const = int(spec_prefix_length)
        args.prefix_length_assembly = int(spec_prefix_length)
    return args


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def build_assembly_encoder_config_from_checkpoint(checkpoint_path, tokenizer_path=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})

    next_token_id = None
    if tokenizer_path:
        tokenizer = undertale_tokenizer.load(tokenizer_path)
        next_token_id = tokenizer.token_to_id(TOKEN_NEXT)
    if next_token_id is None:
        next_token_id = hparams.get("next_token_id", 5)

    return {
        "depth": hparams["depth"],
        "hidden_dimensions": hparams["hidden_dimensions"],
        "vocab_size": hparams["vocab_size"],
        "sequence_length": hparams.get("sequence_length", hparams.get("input_size")),
        "heads": hparams["heads"],
        "intermediate_dimensions": hparams["intermediate_dimensions"],
        "dropout": hparams["dropout"],
        "eps": hparams["eps"],
        "next_token_id": next_token_id,
    }


def load_assembly_encoder_weights(encoder, checkpoint_path):
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[len("encoder."):]] = value
    return encoder.load_state_dict(encoder_state_dict, strict=False)


def build_llm_config_dict(gpt2path):
    config = AutoConfig.from_pretrained(gpt2path, local_files_only=True)
    return config.to_dict()


def load_llm_weights(llm, gpt2path):
    pretrained_llm = GPT2LMHeadModel.from_pretrained(gpt2path, local_files_only=True)
    try:
        missing, unexpected = llm.load_state_dict(pretrained_llm.state_dict(), strict=False)
    finally:
        del pretrained_llm
    return missing, unexpected


def resolve_parquet_files(dataset_path: str) -> list[str]:
    """Match undertale-main parquet directory handling while returning HF inputs."""

    if isfile(dataset_path):
        return [dataset_path] if dataset_path.endswith(".parquet") else []

    return sorted(
        join(dataset_path, entry)
        for entry in listdir(dataset_path)
        if entry.endswith(".parquet")
    )


def load_training_dataset(dataset_path, end2end):
    del end2end

    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {dataset_path}")

    if path.is_file():
        if path.suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset file type: {path}")

    if (path / "dataset_info.json").exists() and (path / "state.json").exists():
        return load_from_disk(str(path))

    parquet_files = resolve_parquet_files(str(path))
    if parquet_files:
        return load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
        )

    raise ValueError(f"Could not load dataset from {dataset_path}")


def resolve_column_name(column_names, override, candidates):
    columns = set(column_names)
    if override is not None:
        if override not in columns:
            raise ValueError(
                f"Requested column '{override}' was not found in dataset columns: {sorted(columns)}"
            )
        return override

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in columns:
            return candidate
    return None


def infer_columns_from_spec_and_data(dataset, spec, args):
    columns = dataset.column_names
    output_columns = (spec or {}).get("output_columns", {})
    spec_assembly_col = (spec or {}).get("assembly_column")
    spec_summary_col = (spec or {}).get("summary_column")

    summary_text = resolve_column_name(
        columns,
        args.summary_text_column,
        ["summary", spec_summary_col, "function_name", "caption", "summaries"],
    )
    assembly_text = resolve_column_name(
        columns,
        args.assembly_text_column,
        ["disassembly", spec_assembly_col, "assembly", "asm", "assembly_code"],
    )
    summary_tokens = resolve_column_name(
        columns,
        args.summary_tokens_column,
        [
            output_columns.get("summary_tokens_column"),
            f"{summary_text}_tokens" if summary_text else None,
            "summary_tokens",
            "function_name_tokens",
            "target_ids",
        ],
    )
    assembly_tokens = resolve_column_name(
        columns,
        args.assembly_tokens_column,
        [
            output_columns.get("assembly_tokens_column"),
            f"{assembly_text}_tokens" if assembly_text else None,
            "disassembly_tokens",
            "assembly_tokens",
            "tokens",
            "asm_ids",
        ],
    )
    assembly_mask = resolve_column_name(
        columns,
        args.assembly_mask_column,
        [
            output_columns.get("assembly_mask_column"),
            f"{assembly_text}_mask" if assembly_text else None,
            "disassembly_mask",
            "assembly_mask",
            "mask",
            "asm_mask",
            "asm_attention_mask",
        ],
    )
    assembly_prefix = resolve_column_name(
        columns,
        args.assembly_prefix_column,
        [
            output_columns.get("assembly_embedding_column"),
            "disassembly_prefixes",
            "disassembly_embedding",
            f"{assembly_text}_embedding" if assembly_text else None,
            "assembly_embedding",
            "asm_embed",
        ],
    )
    return {
        "summary_text": summary_text,
        "summary_tokens": summary_tokens,
        "assembly_text": assembly_text,
        "assembly_tokens": assembly_tokens,
        "assembly_mask": assembly_mask,
        "assembly_prefix": assembly_prefix,
    }


def resolve_dataset_modes(args, resolved_columns):
    summary_mode = args.summary_input_mode
    assembly_mode = args.assembly_input_mode
    columns = resolved_columns

    if summary_mode == "auto":
        summary_mode = "tokens" if columns["summary_tokens"] else "raw"
    if assembly_mode == "auto":
        if args.end2end:
            assembly_mode = "raw"
        elif columns["assembly_prefix"]:
            assembly_mode = "prefix"
        elif columns["assembly_tokens"] and columns["assembly_mask"]:
            assembly_mode = "tokens"
        else:
            assembly_mode = "raw"

    if summary_mode == "tokens" and not columns["summary_tokens"]:
        raise ValueError("summary_input_mode=tokens requires a summary token column.")
    if summary_mode == "raw" and not columns["summary_text"]:
        raise ValueError("summary_input_mode=raw requires a summary text column.")

    if args.end2end:
        if assembly_mode == "prefix":
            raise ValueError("assembly_input_mode=prefix is only valid when --end2end is disabled.")
        if assembly_mode == "tokens" and not (columns["assembly_tokens"] and columns["assembly_mask"]):
            raise ValueError("assembly_input_mode=tokens requires assembly token and mask columns.")
        if assembly_mode == "raw" and not columns["assembly_text"]:
            raise ValueError("assembly_input_mode=raw requires an assembly text column.")
    else:
        if assembly_mode == "prefix":
            if not columns["assembly_prefix"]:
                raise ValueError("assembly_input_mode=prefix requires a prefix column.")
        elif assembly_mode == "tokens":
            if not (columns["assembly_tokens"] and columns["assembly_mask"]):
                raise ValueError("assembly_input_mode=tokens requires assembly token and mask columns.")
        elif assembly_mode == "raw":
            if not columns["assembly_text"]:
                raise ValueError("assembly_input_mode=raw requires an assembly text column.")
        else:
            raise ValueError(f"Unsupported assembly_input_mode: {assembly_mode}")

    return summary_mode, assembly_mode


def select_dataset_columns(dataset, resolved_columns, summary_mode, assembly_mode):
    required = []
    if summary_mode == "tokens":
        required.append(resolved_columns["summary_tokens"])
    else:
        required.append(resolved_columns["summary_text"])

    if assembly_mode == "prefix":
        required.append(resolved_columns["assembly_prefix"])
    elif assembly_mode == "tokens":
        required.extend(
            [resolved_columns["assembly_tokens"], resolved_columns["assembly_mask"]]
        )
    else:
        required.append(resolved_columns["assembly_text"])

    required = [column for column in required if column]
    if hasattr(dataset, "select_columns"):
        return dataset.select_columns(required)
    return dataset


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ValidationCallback(Callback):
    def __init__(
        self,
        dataloader,
        end2end,
        tag,
        run_on_val_end,
        run_on_fit_end,
        args,
    ):
        super().__init__()
        self.dataloader = dataloader
        self.end2end = end2end
        self.tag = tag
        self.run_on_val_end = run_on_val_end
        self.run_on_fit_end = run_on_fit_end
        self.save_dir = args.generated_output_paths
        self.max_new_tokens = args.max_new_tokens
        self.beam = args.beam
        self.num_beams = args.num_beams
        self.temperature = args.temperature
        self.bertscore_model_path = args.bertscore_model_path

    def _run_validation(self, trainer, pl_module):
        from bert_score import score as bert_score
        from rouge_score import rouge_scorer

        os.makedirs(self.save_dir, exist_ok=True)
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        outputs = []
        references = []
        predictions = []
        n_samples = 0
        model_was_training = pl_module.model.training
        pl_module.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                tokens = batch["tokens"].to(pl_module.device)
                disassembly_tokens = batch["disassembly_tokens"].to(pl_module.device)
                disassembly_mask = batch["disassembly_mask"].to(pl_module.device)

                if self.end2end:
                    prefix = pl_module.model.embed_assembly(disassembly_tokens, disassembly_mask)
                    prefix = pl_module.model.connector(
                        pl_module.model.masked_mean_pool(prefix, disassembly_mask)
                    ).view(
                        -1,
                        pl_module.model.prefix_length_const,
                        pl_module.model.llm_embedding_size,
                    )
                else:
                    prefix = pl_module.model.connector(disassembly_tokens).view(
                        -1,
                        pl_module.model.prefix_length_const,
                        pl_module.model.llm_embedding_size,
                    )

                text = pl_module.model.generate(
                    embed=prefix,
                    entry_length=self.max_new_tokens,
                    beam=self.beam,
                    num_beams=self.num_beams,
                    temperature=self.temperature,
                )

                prefix_len = pl_module.model.prefix_length_const
                target_ids = tokens[0].tolist()
                if 0 in target_ids:
                    target_ids = target_ids[: target_ids.index(0)]
                caption = pl_module.model.tokenizer.decode(target_ids, skip_special_tokens=True)

                outputs.append((caption, text))
                references.append(caption)
                predictions.append(text)
                n_samples += 1

        if model_was_training:
            pl_module.model.train()

        rouge_scores = [rouge.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
        rouge_l_f1 = sum(rouge_scores) / max(1, len(rouge_scores))

        _, _, bert_f1 = bert_score(
            predictions,
            references,
            model_type=self.bertscore_model_path,
            verbose=False,
        )
        bert_f1 = bert_f1.mean().item()

        if self.tag is None:
            path = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch}.txt")
        else:
            path = os.path.join(self.save_dir, f"{self.tag}_epoch_{trainer.current_epoch}.txt")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"ROUGE-L(F1): {rouge_l_f1:.6f}\n"
                f"BERTScore(F1): {bert_f1:.6f}\n"
                f"NUM_SAMPLES: {n_samples}\n"
                "=================\n\n"
            )
            for cap, pred in outputs:
                handle.write(f"GROUND TRUTH CAPTION:\n{cap}\n\n")
                handle.write(f"PREDICTED CAPTION:\n{pred}\n\n")
                handle.write("_________________\n")

        trainer.print(f"Saved validation outputs to {path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_on_val_end:
            self._run_validation(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.run_on_fit_end:
            self._run_validation(trainer, pl_module)


class SummarizeModel(LightningModule, torch.nn.Module):
    def __init__(self, model, prefix_length, lr=2e-5, warmup_steps=5000, end2end=True):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.end2end = end2end

    def forward(self, text, encoder_embedding, mask=None, labels=None, encoder_attention_mask=None):
        return self.model(
            text,
            encoder_embedding,
            mask,
            labels,
            encoder_attention_mask=encoder_attention_mask,
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        tokens, mask, disassembly_tokens, disassembly_mask = (
            batch["tokens"],
            batch["mask"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],
        )

        if self.end2end:
            with torch.no_grad():
                prefix = self.model.embed_assembly(disassembly_tokens, disassembly_mask)
        else:
            prefix = disassembly_tokens

        outputs = self(tokens, prefix, mask, encoder_attention_mask=disassembly_mask)
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
        )

        self.log("train_loss", loss, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        tokens, mask, disassembly_tokens, disassembly_mask = (
            batch["tokens"],
            batch["mask"],
            batch["disassembly_tokens"],
            batch["disassembly_mask"],
        )

        if self.end2end:
            with torch.no_grad():
                prefix = self.model.embed_assembly(disassembly_tokens, disassembly_mask)
        else:
            prefix = disassembly_tokens

        outputs = self(tokens, prefix, mask, encoder_attention_mask=disassembly_mask)
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
        )

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="fine-tune the summarization model")
    parser.add_argument("--gpt2path", type=str)
    parser.add_argument("--assembly_checkpoint", type=str)
    parser.add_argument("-c", "--summarizer_checkpoint")
    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prefix_length_const", type=int, default=40)
    parser.add_argument("--prefix_length_assembly", type=int, default=40)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("-t", "--tokenizer", help="trained assembly tokenizer file")
    parser.add_argument("--tokenizer_size", type=int, default=512)
    parser.add_argument("-e", "--end2end", dest="end2end", action="store_true")
    parser.add_argument("--tune_llm", dest="tune_llm", action="store_true")
    parser.add_argument("--normalize_prefix", dest="normalize_prefix", action="store_true")
    parser.add_argument("--token_batchsize", type=int, default=1024)
    parser.add_argument("--dataset", type=str, help="dataset on which to train the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--dataset_size", type=dataset_size_type, default=-1)
    parser.add_argument(
        "--summary_input_mode",
        type=str,
        choices=["auto", "raw", "tokens"],
        default="auto",
    )
    parser.add_argument(
        "--assembly_input_mode",
        type=str,
        choices=["auto", "raw", "tokens", "prefix"],
        default="auto",
    )
    parser.add_argument("--summary_text_column", type=optional_str, default=None)
    parser.add_argument("--summary_tokens_column", type=optional_str, default=None)
    parser.add_argument("--assembly_text_column", type=optional_str, default=None)
    parser.add_argument("--assembly_tokens_column", type=optional_str, default=None)
    parser.add_argument("--assembly_mask_column", type=optional_str, default=None)
    parser.add_argument("--assembly_prefix_column", type=optional_str, default=None)
    parser.add_argument("--output", help="output model directory")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-warmup", "--warmup_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("-a", "--accelerator", default="auto")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("-d", "--devices", default=1, type=int)
    parser.add_argument("-n", "--nodes", default=1, type=int)
    parser.add_argument("-v", "--version")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--generated_output_paths", default="./validation_outputs", type=str)
    parser.add_argument("--bertscore_model_path", type=str)

    args = parser.parse_args()
    undertale_logging.setup_logging()

    spec_path, preprocessing_spec = load_preprocessing_spec(args.dataset)
    args = maybe_override_prefix_lengths_from_spec(args, preprocessing_spec)
    if spec_path is not None:
        print(f"Loaded preprocessing spec from {spec_path}")
        print(
            "Prefix lengths overridden from preprocessing spec: "
            f"prefix_length_const={args.prefix_length_const}, "
            f"prefix_length_assembly={args.prefix_length_assembly}"
        )

    dataset = load_training_dataset(args.dataset, args.end2end)
    resolved_columns = infer_columns_from_spec_and_data(dataset, preprocessing_spec, args)

    summary_mode, assembly_mode = resolve_dataset_modes(args, resolved_columns)
    print(f"Using summary_input_mode={summary_mode}")
    print(f"Using assembly_input_mode={assembly_mode}")

    if assembly_mode == "raw" and not args.tokenizer:
        raise ValueError("--tokenizer is required when assembly_input_mode=raw.")

    dataset = select_dataset_columns(dataset, resolved_columns, summary_mode, assembly_mode)

    if args.dataset_size != -1:
        actual_size = min(args.dataset_size, len(dataset))
        dataset = dataset.select(range(actual_size))

    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    train_dataset = SummarizerDataset(
        dataset=split_dataset["train"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
        summary_tokens_column=resolved_columns["summary_tokens"] if summary_mode == "tokens" else "summary_tokens",
        summary_text_column=resolved_columns["summary_text"] if summary_mode == "raw" else "summary",
        assembly_text_column=resolved_columns["assembly_text"] if assembly_mode == "raw" else "disassembly",
        assembly_tokens_column=resolved_columns["assembly_tokens"] if assembly_mode == "tokens" else "disassembly_tokens",
        assembly_mask_column=resolved_columns["assembly_mask"] if assembly_mode == "tokens" else "disassembly_mask",
        assembly_prefix_column=resolved_columns["assembly_prefix"] if assembly_mode == "prefix" else "disassembly_prefixes",
    )
    val_dataset = SummarizerDataset(
        dataset=split_dataset["test"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
        summary_tokens_column=resolved_columns["summary_tokens"] if summary_mode == "tokens" else "summary_tokens",
        summary_text_column=resolved_columns["summary_text"] if summary_mode == "raw" else "summary",
        assembly_text_column=resolved_columns["assembly_text"] if assembly_mode == "raw" else "disassembly",
        assembly_tokens_column=resolved_columns["assembly_tokens"] if assembly_mode == "tokens" else "disassembly_tokens",
        assembly_mask_column=resolved_columns["assembly_mask"] if assembly_mode == "tokens" else "disassembly_mask",
        assembly_prefix_column=resolved_columns["assembly_prefix"] if assembly_mode == "prefix" else "disassembly_prefixes",
    )

    collator = CustomCollator(args, train_dataset.max_seq_len, train_dataset.pad_id)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=8,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=8,
        drop_last=True,
    )

    connector_config = {
        "model_type": args.model_type,
        "prefix_size": 768,
        "prefix_length_const": args.prefix_length_const,
        "prefix_length_assembly": args.prefix_length_assembly,
        "num_layers": args.num_layers,
    }
    assembly_encoder_config = (
        build_assembly_encoder_config_from_checkpoint(args.assembly_checkpoint, args.tokenizer)
        if args.end2end
        else None
    )
    if args.end2end and assembly_encoder_config["sequence_length"] != args.tokenizer_size:
        raise ValueError(
            "tokenizer_size must match the assembly encoder sequence length "
            f"for end2end summarization: tokenizer_size={args.tokenizer_size}, "
            f"encoder sequence_length={assembly_encoder_config['sequence_length']}"
        )
    llm_config = build_llm_config_dict(args.gpt2path)

    model = TransformerEncoderForSequenceSummarization(
        assembly_encoder_config,
        connector_config,
        llm_config,
        args.end2end,
        args.tune_llm,
    )

    if args.end2end:
        missing, unexpected = load_assembly_encoder_weights(
            model.assembly_encoder,
            args.assembly_checkpoint,
        )
        if missing:
            print(f"Assembly encoder missing keys after load: {missing}")
        if unexpected:
            print(f"Assembly encoder unexpected keys after load: {unexpected}")

    missing, unexpected = load_llm_weights(model.llm, args.gpt2path)
    if missing:
        print(f"LLM missing keys after load: {missing}")
    if unexpected:
        print(f"LLM unexpected keys after load: {unexpected}")

    model.set_tokenizer(AutoTokenizer.from_pretrained(args.gpt2path, local_files_only=True))

    output = os.path.abspath(os.path.expanduser(args.output))
    progress = ProgressBar(leave=True)
    checkpoint = ModelCheckpoint(filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}", save_top_k=-1)
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001)
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=args.version,
    )

    if args.validation:
        random_sampler = RandomSampler(val_dataset, num_samples=5)
        random_validation = DataLoader(
            val_dataset,
            sampler=random_sampler,
            batch_size=1,
            collate_fn=collator,
            num_workers=8,
        )
        final_validation = DataLoader(
            val_dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=8,
        )
        final_validation_check = ValidationCallback(
            final_validation,
            end2end=args.end2end,
            tag="final_full_val",
            run_on_val_end=False,
            run_on_fit_end=True,
            args=args,
        )
        validation_check = ValidationCallback(
            random_validation,
            end2end=args.end2end,
            tag=None,
            run_on_val_end=True,
            run_on_fit_end=False,
            args=args,
        )
        callbacks = [progress, checkpoint, stop, validation_check, final_validation_check]
    else:
        callbacks = [progress, checkpoint, stop]

    summarize_model = SummarizeModel(
        model,
        args.prefix_length_const,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        end2end=args.end2end,
    )

    num_train_batches = len(train_dataloader)
    if num_train_batches > 2000:
        val_check_interval = 2000
        check_val_every_n_epoch = None
    else:
        val_check_interval = num_train_batches
        check_val_every_n_epoch = 1

    if args.num_epochs is not None:
        trainer_max_steps = args.num_epochs * num_train_batches
    else:
        trainer_max_steps = args.max_steps

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        max_steps=trainer_max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(
        summarize_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.summarizer_checkpoint,
    )


if __name__ == "__main__":
    main()
