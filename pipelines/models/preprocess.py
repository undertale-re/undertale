"""Preprocess summarization parquet datasets for model training."""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from transformers import AutoTokenizer

from undertale import logging as undertale_logging
from undertale.models.custom import InstructionTraceTransformerEncoder
from undertale.models.tokenizer import TOKEN_NEXT
from undertale.models import tokenizer as undertale_tokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_ASSEMBLY_CANDIDATES = [
    "disassembly",
    "assembly",
    "assembly_code",
    "asm",
]

DEFAULT_SUMMARY_CANDIDATES = [
    "summary",
    "summaries",
    "caption",
    "captions",
    "function_name",
]


class ArrowArrayDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for pre-tokenized ids and attention masks."""

    def __init__(
        self,
        input_ids: Sequence[Sequence[int]],
        attention_masks: Sequence[Sequence[int]],
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return {
            "row_idx": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }


class SimpleCollator:
    """Stack variable-length tokenized rows into one batch."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def _pad_sequences(self, sequences, padding_value: int):
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, self.sequence_length),
            padding_value,
            dtype=torch.long,
        )

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.sequence_length)
            padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)

        return padded

    def __call__(self, batch):
        return {
            "row_idx": torch.stack([item["row_idx"] for item in batch]),
            "input_ids": self._pad_sequences(
                [item["input_ids"].tolist() for item in batch],
                padding_value=0,
            ),
            "attention_mask": self._pad_sequences(
                [item["attention_mask"].tolist() for item in batch],
                padding_value=0,
            ),
        }


class DistributedPredictionWriter(BasePredictionWriter):
    """Persist prediction shards per rank so rank 0 can merge them."""

    def __init__(self, output_dir: Optional[Path] = None):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        del pl_module, batch_indices, batch, dataloader_idx

        if self.output_dir is None:
            raise RuntimeError("Prediction writer output_dir was not configured.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"rank{trainer.global_rank:04d}_batch{batch_idx:08d}.pt"
        torch.save(prediction, destination)


class EmbeddingProgressCallback(Callback):
    """Show progress while embedding assembly rows."""

    def __init__(self):
        self.total_batches = None

    def set_total_batches(self, total_batches: int) -> None:
        self.total_batches = total_batches


class AssemblyEmbeddingPredictor(LightningModule):
    """Lightning wrapper that runs the assembly encoder for prediction."""

    def __init__(self, encoder_config: Dict[str, Any]):
        super().__init__()
        self.assembly_encoder = InstructionTraceTransformerEncoder(
            encoder_config["depth"],
            encoder_config["hidden_dimensions"],
            encoder_config["vocab_size"],
            encoder_config["sequence_length"],
            encoder_config["heads"],
            encoder_config["intermediate_dimensions"],
            encoder_config["next_token_id"],
            encoder_config["dropout"],
            encoder_config["eps"],
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        hidden = self.assembly_encoder(batch["input_ids"], batch["attention_mask"])
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {
            "row_idx": batch["row_idx"].detach().cpu(),
            "embeddings": pooled.detach().cpu(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess summarization parquet files with optional assembly tokenization, embeddings, and GPT tokenization."
    )
    parser.add_argument("--dataset", required=True, help="Input parquet dataset directory.")
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output directory for parquet shards and preprocessing_spec.json.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.parquet",
        help="Glob pattern for parquet discovery inside folder.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of datapoints to preprocess across the dataset. Use -1 to process all rows.",
    )
    parser.add_argument("--assembly_column", type=str, default=None)
    parser.add_argument("--summary_column", type=str, default=None)
    parser.add_argument("--assembly_tokenizer", type=str, default=None)
    parser.add_argument("--assembly_checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_size", type=int, default=512)
    parser.add_argument("--gpt2path", type=str, default="gpt2")
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=40,
        help="Number of prefix tokens reserved for summary decoding.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("-a", "--accelerator", default="auto")
    parser.add_argument("-d", "--devices", default=1, type=int)
    parser.add_argument("-n", "--nodes", default=1, type=int)
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp_find_unused_parameters_true",
        help="Lightning distributed strategy for embedding prediction.",
    )
    parser.add_argument("--tokenize_assembly", action="store_true")
    parser.add_argument("--embed_assembly", action="store_true")
    parser.add_argument("--tokenize_summaries", action="store_true")
    parser.add_argument("--assembly_tokens_column", type=str, default=None)
    parser.add_argument("--assembly_mask_column", type=str, default=None)
    parser.add_argument("--assembly_embedding_column", type=str, default=None)
    parser.add_argument("--summary_tokens_column", type=str, default=None)
    parser.add_argument("--summary_mask_column", type=str, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.tokenize_assembly or args.embed_assembly or args.tokenize_summaries):
        raise ValueError(
            "At least one of --tokenize_assembly, --embed_assembly, or --tokenize_summaries must be set."
        )
    if (args.tokenize_assembly or args.embed_assembly) and not args.assembly_tokenizer:
        raise ValueError(
            "--assembly_tokenizer is required when --tokenize_assembly or --embed_assembly is used."
        )
    if args.embed_assembly and not args.assembly_checkpoint:
        raise ValueError("--assembly_checkpoint is required when --embed_assembly is used.")
    if args.prefix_length < 0:
        raise ValueError("--prefix_length must be >= 0.")
    if args.dataset_size == 0 or args.dataset_size < -1:
        raise ValueError("--dataset_size must be -1 for the full dataset or a positive integer.")


def find_parquet_files(folder: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder} with pattern {pattern!r}")
    return files


def infer_column(
    columns: Sequence[str],
    requested: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> str:
    if requested:
        if requested not in columns:
            raise KeyError(f"Requested {kind} column {requested!r} not present. Found columns: {list(columns)}")
        return requested

    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    raise KeyError(
        f"Could not infer {kind} column. Found columns: {list(columns)}. "
        f"Pass --{kind}_column explicitly."
    )


def resolve_output_names(args: argparse.Namespace, assembly_col: str, summary_col: str) -> dict:
    return {
        "assembly_tokens": args.assembly_tokens_column or f"{assembly_col}_tokens",
        "assembly_mask": args.assembly_mask_column or f"{assembly_col}_mask",
        "assembly_embedding": args.assembly_embedding_column or f"{assembly_col}_embedding",
        "summary_tokens": args.summary_tokens_column or f"{summary_col}_tokens",
        "summary_mask": args.summary_mask_column or f"{summary_col}_mask",
    }


def tokenize_assembly_texts(texts: Sequence[str], tokenizer) -> Tuple[List[List[int]], List[List[int]]]:
    token_ids = []
    masks = []
    for text in texts:
        encoding = tokenizer.encode(text)
        token_ids.append(list(encoding.ids))
        masks.append(list(encoding.attention_mask))
    return token_ids, masks


def resolve_summary_tokenization_settings(tokenizer, gpt2path: str, prefix_length: int) -> Tuple[int, int]:
    stop_token = tokenizer.eos_token_id
    if stop_token is None:
        raise ValueError(f"Tokenizer {gpt2path!r} has no eos_token_id; cannot append stop token.")

    max_positions = None
    for attr in ("n_positions", "max_position_embeddings"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            max_positions = value
            break

    if max_positions is None and hasattr(tokenizer, "model_max_length"):
        model_max_length = getattr(tokenizer, "model_max_length")
        if isinstance(model_max_length, int) and model_max_length < 10**9:
            max_positions = model_max_length

    if max_positions is None:
        raise ValueError(f"Cannot determine max context length from {gpt2path}")

    max_seq_len = max_positions - prefix_length - 1
    if max_seq_len < 0:
        raise ValueError(
            f"prefix_length={prefix_length} leaves no room for summary tokens and stop token with max_positions={max_positions}."
        )

    return stop_token, max_seq_len


def tokenize_summary_texts(
    texts: Sequence[str],
    tokenizer,
    max_seq_len: int,
    stop_token: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    token_ids = []
    masks = []
    for text in texts:
        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
        )
        ids = list(ids) + [stop_token]
        token_ids.append(ids)
        masks.append([1] * len(ids))
    return token_ids, masks


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def build_assembly_encoder_config_from_checkpoint(
    checkpoint_path: str,
    tokenizer_path: Optional[str],
) -> Dict[str, Any]:
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


def build_prediction_trainer(
    args: argparse.Namespace,
    output_root: Path,
    prediction_writer: DistributedPredictionWriter,
    embedding_progress: EmbeddingProgressCallback,
) -> Trainer:
    return Trainer(
        strategy=args.strategy,
        callbacks=[prediction_writer, embedding_progress],
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(output_root.resolve()),
    )


def resolve_prediction_shard_root(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}_predict_shards"


def generate_embeddings(
    token_ids: Sequence[Sequence[int]],
    masks: Sequence[Sequence[int]],
    trainer: Trainer,
    predictor: AssemblyEmbeddingPredictor,
    prediction_writer: DistributedPredictionWriter,
    embedding_progress: EmbeddingProgressCallback,
    batch_size: int,
    num_workers: int,
) -> Optional[List[List[float]]]:
    dataset = ArrowArrayDataset(token_ids, masks)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SimpleCollator(predictor.assembly_encoder.embedding.sequence_length),
    )
    embedding_progress.set_total_batches(len(loader))

    temp_root = resolve_prediction_shard_root(Path(trainer.default_root_dir))
    if trainer.is_global_zero and temp_root.exists():
        shutil.rmtree(temp_root)
    trainer.strategy.barrier("cleanup_predict_shards")

    prediction_writer.output_dir = temp_root
    trainer.predict(predictor, dataloaders=loader, return_predictions=False)

    trainer.strategy.barrier("predict_embeddings_done")

    if not trainer.is_global_zero:
        return None

    shard_paths = sorted(temp_root.glob("rank*_batch*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No prediction shards were written to {temp_root}")

    merged: List[Optional[List[float]]] = [None] * len(token_ids)
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        row_indices = shard["row_idx"].tolist()
        embeddings = shard["embeddings"].tolist()
        for row_idx, embedding in zip(row_indices, embeddings):
            merged[row_idx] = embedding

    missing_rows = [idx for idx, embedding in enumerate(merged) if embedding is None]
    if missing_rows:
        raise RuntimeError(
            f"Missing embeddings for {len(missing_rows)} rows; first missing row index: {missing_rows[0]}"
        )

    shutil.rmtree(temp_root)
    return merged


def sanitize_text_column(series: pd.Series) -> List[str]:
    return series.fillna("").astype(str).tolist()


def write_parquet_preserving_name(df: pd.DataFrame, source_path: Path, output_root: Path) -> Path:
    destination = output_root / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, destination)
    return destination


def process_file(
    parquet_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    assembly_tok,
    summary_tok,
    trainer: Optional[Trainer],
    predictor: Optional[AssemblyEmbeddingPredictor],
    prediction_writer: Optional[DistributedPredictionWriter],
    embedding_progress: Optional[EmbeddingProgressCallback],
    row_limit: Optional[int] = None,
) -> Path:
    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    df = dataset.to_pandas()
    if row_limit is not None:
        df = df.iloc[:row_limit].copy()

    assembly_col = infer_column(df.columns, args.assembly_column, DEFAULT_ASSEMBLY_CANDIDATES, "assembly")
    summary_col = infer_column(df.columns, args.summary_column, DEFAULT_SUMMARY_CANDIDATES, "summary")
    output_names = resolve_output_names(args, assembly_col, summary_col)

    assembly_texts = sanitize_text_column(df[assembly_col])
    summary_texts = sanitize_text_column(df[summary_col])

    assembly_ids = None
    assembly_masks = None
    if args.tokenize_assembly or args.embed_assembly:
        assembly_ids, assembly_masks = tokenize_assembly_texts(assembly_texts, assembly_tok)

    if args.tokenize_assembly:
        df[output_names["assembly_tokens"]] = assembly_ids
        df[output_names["assembly_mask"]] = assembly_masks

    if args.embed_assembly:
        embeddings = generate_embeddings(
            assembly_ids,
            assembly_masks,
            trainer,
            predictor,
            prediction_writer,
            embedding_progress,
            args.batch_size,
            args.num_workers,
        )
        if embeddings is not None:
            df[output_names["assembly_embedding"]] = embeddings

    if args.tokenize_summaries:
        summary_stop_token, summary_max_seq_len = resolve_summary_tokenization_settings(
            summary_tok,
            args.gpt2path,
            args.prefix_length,
        )
        summary_ids, summary_masks = tokenize_summary_texts(
            summary_texts,
            summary_tok,
            summary_max_seq_len,
            summary_stop_token,
        )
        df[output_names["summary_tokens"]] = summary_ids
        df[output_names["summary_mask"]] = summary_masks

    if args.embed_assembly and (trainer is None or not trainer.is_global_zero):
        return output_root / parquet_path.name

    return write_parquet_preserving_name(df, parquet_path, output_root)


def resolve_output_root(input_root: Path, args: argparse.Namespace) -> Path:
    if args.output_folder:
        return Path(args.output_folder).expanduser().resolve()
    return input_root.parent / f"{input_root.name}_preprocessed"


def build_preprocessing_manifest(
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    parquet_files: Sequence[Path],
    summary_tok=None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "input_folder": str(input_root),
        "output_folder": str(output_root),
        "glob": args.glob,
        "dataset_size": args.dataset_size,
        "num_input_files": len(parquet_files),
        "input_files": [p.name for p in parquet_files],
        "assembly_column": args.assembly_column,
        "summary_column": args.summary_column,
        "assembly_tokenizer": args.assembly_tokenizer,
        "assembly_checkpoint": args.assembly_checkpoint,
        "tokenizer_size": args.tokenizer_size,
        "gpt2path": args.gpt2path,
        "prefix_length": args.prefix_length,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "runtime": {
            "accelerator": args.accelerator,
            "devices": args.devices,
            "nodes": args.nodes,
            "strategy": args.strategy,
        },
        "tokenize_assembly": args.tokenize_assembly,
        "embed_assembly": args.embed_assembly,
        "tokenize_summaries": args.tokenize_summaries,
        "output_columns": {
            "assembly_tokens_column": args.assembly_tokens_column,
            "assembly_mask_column": args.assembly_mask_column,
            "assembly_embedding_column": args.assembly_embedding_column,
            "summary_tokens_column": args.summary_tokens_column,
            "summary_mask_column": args.summary_mask_column,
        },
    }

    if args.tokenize_summaries and summary_tok is not None:
        stop_token, max_seq_len = resolve_summary_tokenization_settings(
            summary_tok,
            args.gpt2path,
            args.prefix_length,
        )
        manifest["summary_tokenization"] = {
            "stop_token": stop_token,
            "max_summary_tokens_before_stop": max_seq_len,
            "appends_stop_token": True,
        }

    return manifest


def write_preprocessing_manifest(output_root: Path, manifest: Dict[str, Any]) -> Path:
    manifest_path = output_root / "preprocessing_spec.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def teardown_prediction_runtime(trainer: Optional[Trainer]) -> None:
    if trainer is None:
        return

    strategy = getattr(trainer, "strategy", None)
    if strategy is not None:
        try:
            strategy.barrier("preprocess_shutdown")
        except Exception:
            LOGGER.exception("Failed during final distributed barrier.")

        teardown = getattr(strategy, "teardown", None)
        if callable(teardown):
            try:
                teardown()
            except Exception:
                LOGGER.exception("Failed to tear down Lightning strategy.")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            LOGGER.exception("Failed to destroy torch distributed process group.")


def main() -> None:
    args = parse_args()
    undertale_logging.setup_logging()
    validate_args(args)

    input_root = Path(args.dataset).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a directory: {input_root}")

    output_root = resolve_output_root(input_root, args)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_files = find_parquet_files(input_root, args.glob)
    LOGGER.info("Found %d parquet files under %s", len(parquet_files), input_root)

    assembly_tok = None
    if args.tokenize_assembly or args.embed_assembly:
        assembly_tok = undertale_tokenizer.load(
            args.assembly_tokenizer,
            sequence_length=args.tokenizer_size,
        )

    summary_tok = None
    if args.tokenize_summaries:
        summary_tok = AutoTokenizer.from_pretrained(args.gpt2path, local_files_only=True)
        if summary_tok.pad_token is None:
            summary_tok.pad_token = summary_tok.eos_token

    trainer = None
    predictor = None
    prediction_writer = None
    embedding_progress = None
    if args.embed_assembly:
        encoder_config = build_assembly_encoder_config_from_checkpoint(
            args.assembly_checkpoint,
            args.assembly_tokenizer,
        )
        if encoder_config["sequence_length"] != args.tokenizer_size:
            raise ValueError(
                "tokenizer_size must match the assembly encoder sequence length "
                f"for embedding generation: tokenizer_size={args.tokenizer_size}, "
                f"encoder sequence_length={encoder_config['sequence_length']}"
            )
        predictor = AssemblyEmbeddingPredictor(encoder_config)
        missing, unexpected = load_assembly_encoder_weights(
            predictor.assembly_encoder,
            args.assembly_checkpoint,
        )
        if missing:
            LOGGER.warning("Assembly encoder missing keys after load: %s", missing)
        if unexpected:
            LOGGER.warning("Assembly encoder unexpected keys after load: %s", unexpected)
        predictor.assembly_encoder.eval()
        prediction_writer = DistributedPredictionWriter()
        embedding_progress = EmbeddingProgressCallback()
        trainer = build_prediction_trainer(args, output_root, prediction_writer, embedding_progress)

    try:
        manifest = build_preprocessing_manifest(
            args=args,
            input_root=input_root,
            output_root=output_root,
            parquet_files=parquet_files,
            summary_tok=summary_tok,
        )
        if trainer is None or trainer.is_global_zero:
            manifest_path = write_preprocessing_manifest(output_root, manifest)
            LOGGER.info("Wrote preprocessing manifest to %s", manifest_path)
        if trainer is not None:
            trainer.strategy.barrier("manifest_written")

        remaining_rows = None if args.dataset_size == -1 else args.dataset_size
        for parquet_path in parquet_files:
            if remaining_rows == 0:
                break

            row_limit = None
            if remaining_rows is not None:
                dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
                file_rows = len(dataset)
                if file_rows == 0:
                    continue
                row_limit = min(remaining_rows, file_rows)
                remaining_rows -= row_limit

            destination = process_file(
                parquet_path=parquet_path,
                output_root=output_root,
                args=args,
                assembly_tok=assembly_tok,
                summary_tok=summary_tok,
                trainer=trainer,
                predictor=predictor,
                prediction_writer=prediction_writer,
                embedding_progress=embedding_progress,
                row_limit=row_limit,
            )
            LOGGER.info("Wrote %s", destination)
            if trainer is not None:
                trainer.strategy.barrier("file_processed")
    finally:
        teardown_prediction_runtime(trainer)


if __name__ == "__main__":
    main()
