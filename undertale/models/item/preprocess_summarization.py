import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from transformers import AutoTokenizer

from ... import logging as undertale_logging
from . import tokenizer as undertale_tokenizer
from .model import TransformerEncoder


# Module-level logger used to report progress while processing files.
LOGGER = logging.getLogger("preprocess_summarization")


# Common fallback names used to infer the assembly text column when the user
# does not pass --assembly_column explicitly.
DEFAULT_ASSEMBLY_CANDIDATES = [
    "disassembly",
    "assembly",
    "assembly_code",
    "asm",
]

# Common fallback names used to infer the summary text column when the user
# does not pass --summary_column explicitly.
DEFAULT_SUMMARY_CANDIDATES = [
    "summary",
    "summaries",
    "caption",
    "captions",
    "function_name",
]


class ArrowArrayDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for pre-tokenized ids and attention masks."""

    def __init__(self, input_ids: Sequence[Sequence[int]], attention_masks: Sequence[Sequence[int]]):
        # Store already-tokenized sequences so they can be batched by a DataLoader.
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        # Number of examples available in the dataset.
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        # Convert one tokenized example into torch tensors that can be fed to the model.
        return {
            "row_idx": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }


class SimpleCollator:
    """Stacks already-padded or variable-length tensors into a batch.

    This assumes the tokenization step already produced per-row attention masks.
    """

    def __call__(self, batch):
        # Sequences may have different lengths, so pad them to the length of the
        # longest example in the batch before stacking them.
        return {
            "row_idx": torch.stack([item["row_idx"] for item in batch]),
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
        }


class DistributedPredictionWriter(BasePredictionWriter):
    """Persist prediction shards per rank so they can be merged on global rank 0."""

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
            raise RuntimeError("Prediction writer output_dir was not set before prediction started.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"rank{trainer.global_rank:04d}_batch{batch_idx:08d}.pt"
        torch.save(prediction, destination)


class AssemblyEmbeddingPredictor(LightningModule):
    """Lightning wrapper that runs the plain assembly encoder for prediction."""

    def __init__(self, encoder_config: Dict[str, Any]):
        super().__init__()
        self.assembly_encoder = TransformerEncoder(
            encoder_config["depth"],
            encoder_config["hidden_dimensions"],
            encoder_config["vocab_size"],
            encoder_config["input_size"],
            encoder_config["heads"],
            encoder_config["intermediate_dimensions"],
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
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Preprocess summarization parquet files with optional assembly tokenization, assembly embeddings, and GPT-2 summary tokenization."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset folder containing parquet files to preprocess.",
    )

    # Output directory. Outputs are written directly here; run settings are saved in a JSON manifest.
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help=(
            "Output directory where processed parquet files and a preprocessing_spec.json manifest are written. "
            "Defaults to <input_parent>/<input_name>_preprocessed."
        ),
    )

    # File discovery pattern inside the input folder.
    parser.add_argument(
        "--glob",
        type=str,
        default="*.parquet",
        help="Glob pattern for parquet discovery inside folder. Default: *.parquet",
    )

    # Optional explicit column names to avoid relying on fallback inference.
    parser.add_argument(
        "--assembly_column",
        type=str,
        default=None,
        help="Column containing assembly/disassembly text. If omitted, tries common names.",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="Column containing summary text. If omitted, tries common names.",
    )

    # Resources needed for assembly tokenization / embedding.
    parser.add_argument(
        "--assembly_tokenizer",
        type=str,
        default=None,
        help="Path to the Undertale assembly tokenizer JSON. Required if --tokenize_assembly or --embed_assembly is set.",
    )
    parser.add_argument(
        "--assembly_checkpoint",
        type=str,
        default=None,
        help="Checkpoint for the assembly encoder. Required if --embed_assembly is set.",
    )

    # GPT-2 tokenizer path used for summary text tokenization.
    parser.add_argument(
        "--gpt2path",
        type=str,
        default="gpt2",
        help="Path or model name for the GPT-2 tokenizer used on summaries.",
    )
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=40,
        help="Number of prefix tokens reserved for summary decoding. Used to limit summary token length before appending the stop token.",
    )

    # DataLoader settings used during embedding generation.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for assembly embedding.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers for assembly embedding.",
    )
    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )
    parser.add_argument(
        "-d",
        "--devices",
        default=1,
        type=int,
        help="number of accelerator devices to use (per node)",
    )
    parser.add_argument("-n", "--nodes", default=1, type=int, help="number of nodes to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp_find_unused_parameters_true",
        help="Lightning distributed strategy for embedding prediction.",
    )

    # Feature-selection flags that control which derived columns are produced.
    parser.add_argument(
        "--tokenize_assembly",
        action="store_true",
        help="Save tokenized assembly code.",
    )
    parser.add_argument(
        "--embed_assembly",
        action="store_true",
        help="Save assembly embeddings from the bare assembly encoder.",
    )
    parser.add_argument(
        "--tokenize_summaries",
        action="store_true",
        help="Save GPT-2 tokenized summaries.",
    )

    # Optional custom names for each derived output column.
    parser.add_argument(
        "--assembly_tokens_column",
        type=str,
        default=None,
        help="Optional output column name for assembly token ids. Default: <assembly_column>_tokens",
    )
    parser.add_argument(
        "--assembly_mask_column",
        type=str,
        default=None,
        help="Optional output column name for assembly attention mask. Default: <assembly_column>_mask",
    )
    parser.add_argument(
        "--assembly_embedding_column",
        type=str,
        default=None,
        help="Optional output column name for assembly embeddings. Default: <assembly_column>_embedding",
    )
    parser.add_argument(
        "--summary_tokens_column",
        type=str,
        default=None,
        help="Optional output column name for GPT-2 summary token ids. Default: <summary_column>_tokens",
    )
    parser.add_argument(
        "--summary_mask_column",
        type=str,
        default=None,
        help="Optional output column name for GPT-2 summary attention mask. Default: <summary_column>_mask",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate flag combinations and required supporting paths."""

    if not args.dataset:
        raise ValueError("--dataset is required.")

    # Require the user to ask for at least one preprocessing action.
    if not (args.tokenize_assembly or args.embed_assembly or args.tokenize_summaries):
        raise ValueError(
            "At least one of --tokenize_assembly, --embed_assembly, or --tokenize_summaries must be set."
        )

    # Assembly tokenizer is needed both for raw assembly token export and as the
    # first step in embedding generation.
    if (args.tokenize_assembly or args.embed_assembly) and not args.assembly_tokenizer:
        raise ValueError(
            "--assembly_tokenizer is required when --tokenize_assembly or --embed_assembly is used."
        )

    # The encoder checkpoint is only required when generating embeddings.
    if args.embed_assembly and not args.assembly_checkpoint:
        raise ValueError("--assembly_checkpoint is required when --embed_assembly is used.")

    if args.prefix_length < 0:
        raise ValueError("--prefix_length must be >= 0.")


def find_parquet_files(folder: Path, pattern: str) -> List[Path]:
    """Return top-level parquet files from the input folder."""

    # Search the folder for files matching the user's glob pattern.
    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder} with pattern {pattern!r}")
    return files


def infer_column(columns: Sequence[str], requested: Optional[str], candidates: Sequence[str], kind: str) -> str:
    """Resolve a column name either explicitly or from fallback candidates."""

    # If the caller explicitly requested a column, trust it but verify that it exists.
    if requested:
        if requested not in columns:
            raise KeyError(f"Requested {kind} column {requested!r} not present. Found columns: {list(columns)}")
        return requested

    # Otherwise, try to infer the column name by matching common aliases case-insensitively.
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    # If no candidate matches, fail loudly and tell the user how to override it.
    raise KeyError(
        f"Could not infer {kind} column. Found columns: {list(columns)}. "
        f"Pass --{kind}_column explicitly."
    )


def resolve_output_names(args: argparse.Namespace, assembly_col: str, summary_col: str) -> dict:
    """Build the output column names for derived features."""

    # Each output column can be user-defined; otherwise it is derived from the
    # source column name.
    return {
        "assembly_tokens": args.assembly_tokens_column or f"{assembly_col}_tokens",
        "assembly_mask": args.assembly_mask_column or f"{assembly_col}_mask",
        "assembly_embedding": args.assembly_embedding_column or f"{assembly_col}_embedding",
        "summary_tokens": args.summary_tokens_column or f"{summary_col}_tokens",
        "summary_mask": args.summary_mask_column or f"{summary_col}_mask",
    }


def tokenize_assembly_texts(texts: Sequence[str], tokenizer) -> Tuple[List[List[int]], List[List[int]]]:
    """Tokenize assembly strings with the Undertale tokenizer."""

    token_ids: List[List[int]] = []
    masks: List[List[int]] = []
    for text in texts:
        # Encode one assembly string into token ids and its attention mask.
        encoding = tokenizer.encode(text)
        token_ids.append(list(encoding.ids))
        masks.append(list(encoding.attention_mask))
    return token_ids, masks


def resolve_summary_tokenization_settings(tokenizer, gpt2path: str, prefix_length: int) -> Tuple[int, int]:
    """Return stop token id and usable summary length after reserving prefix space."""

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
    prefix_length: int,
    max_seq_len: int,
    stop_token: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Tokenize summaries, reserve prefix space, and append the stop token."""

    del prefix_length  # already accounted for in max_seq_len

    token_ids: List[List[int]] = []
    masks: List[List[int]] = []
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
    """Load a PyTorch/Lightning checkpoint and return its state dict payload."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def build_assembly_encoder_config_from_checkpoint(checkpoint_path):
    """Extract plain-Python encoder hyperparameters from a masked-LM checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})

    return {
        "depth": hparams["depth"],
        "hidden_dimensions": hparams["hidden_dimensions"],
        "vocab_size": hparams["vocab_size"],
        "input_size": hparams["input_size"],
        "heads": hparams["heads"],
        "intermediate_dimensions": hparams["intermediate_dimensions"],
        "dropout": hparams["dropout"],
        "eps": hparams["eps"],
    }


def load_assembly_encoder_weights(encoder, checkpoint_path):
    """Load only encoder weights from a masked-LM Lightning checkpoint."""
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[len("encoder."):]] = value

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    return missing, unexpected


def build_prediction_trainer(
    args: argparse.Namespace,
    output_root: Path,
    prediction_writer: DistributedPredictionWriter,
) -> Trainer:
    """Build the Lightning trainer used for distributed embedding prediction."""

    return Trainer(
        strategy=args.strategy,
        callbacks=[prediction_writer],
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=os.path.abspath(str(output_root)),
    )


def resolve_prediction_shard_root(output_root: Path) -> Path:
    """Store temporary prediction shards outside the user-visible output folder."""

    return output_root.parent / f".{output_root.name}_predict_shards"


def generate_embeddings(
    token_ids: Sequence[Sequence[int]],
    masks: Sequence[Sequence[int]],
    trainer: Trainer,
    predictor: AssemblyEmbeddingPredictor,
    prediction_writer: DistributedPredictionWriter,
    batch_size: int,
    num_workers: int,
) -> Optional[List[List[float]]]:
    """Generate one mean-pooled embedding vector per row."""

    # Wrap tokenized rows in a Dataset, then use a DataLoader to efficiently batch
    # them for inference.
    dataset = ArrowArrayDataset(token_ids, masks)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SimpleCollator(),
    )

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
    """Convert a column to clean strings, replacing nulls with empty text."""

    # Normalize the column so downstream tokenizers always receive strings.
    return series.fillna("").astype(str).tolist()


def write_parquet_preserving_name(df: pd.DataFrame, source_path: Path, output_root: Path) -> Path:
    """Write one output parquet per input parquet, preserving the file name."""

    # Keep the original file name, but place it under the derived output root.
    destination = output_root / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Convert pandas -> Arrow table -> parquet for output.
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
) -> Path:
    """Load one parquet file, add requested derived columns, and write it back out."""

    LOGGER.info("Reading %s", parquet_path)

    # Read the parquet file through the Hugging Face datasets loader, then convert
    # it to a pandas DataFrame for easier column manipulation.
    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    df = dataset.to_pandas()

    # Figure out which source columns contain assembly text and summaries.
    assembly_col = infer_column(df.columns, args.assembly_column, DEFAULT_ASSEMBLY_CANDIDATES, "assembly")
    summary_col = infer_column(df.columns, args.summary_column, DEFAULT_SUMMARY_CANDIDATES, "summary")

    # Build the final names of any columns we may add.
    output_names = resolve_output_names(args, assembly_col, summary_col)

    # Pull source text into clean Python string lists before tokenization.
    assembly_texts = sanitize_text_column(df[assembly_col])
    summary_texts = sanitize_text_column(df[summary_col])

    assembly_ids = None
    assembly_masks = None
    if args.tokenize_assembly or args.embed_assembly:
        # Assembly tokenization is shared by both the "save tokens" and
        # "generate embeddings" paths, so do it once and reuse the results.
        assembly_ids, assembly_masks = tokenize_assembly_texts(assembly_texts, assembly_tok)

    if args.tokenize_assembly:
        # Add raw assembly token ids and attention masks as new dataframe columns.
        df[output_names["assembly_tokens"]] = assembly_ids
        df[output_names["assembly_mask"]] = assembly_masks

    if args.embed_assembly:
        # Feed the tokenized assembly into the encoder and store one embedding vector
        # per row.
        embeddings = generate_embeddings(
            assembly_ids,
            assembly_masks,
            trainer,
            predictor,
            prediction_writer,
            args.batch_size,
            args.num_workers,
        )
        if embeddings is not None:
            df[output_names["assembly_embedding"]] = embeddings

    if args.tokenize_summaries:
        # Independently tokenize summaries using GPT-2's tokenizer, reserving prefix
        # space and appending the stop token.
        summary_stop_token, summary_max_seq_len = resolve_summary_tokenization_settings(
            summary_tok,
            args.gpt2path,
            args.prefix_length,
        )
        summary_ids, summary_masks = tokenize_summary_texts(
            summary_texts,
            summary_tok,
            args.prefix_length,
            summary_max_seq_len,
            summary_stop_token,
        )
        df[output_names["summary_tokens"]] = summary_ids
        df[output_names["summary_mask"]] = summary_masks

    # Write the augmented dataframe back to parquet in the output directory.
    if args.embed_assembly and (trainer is None or not trainer.is_global_zero):
        return output_root / parquet_path.name

    destination = write_parquet_preserving_name(df, parquet_path, output_root)
    LOGGER.info("Wrote %s", destination)
    return destination


def resolve_output_root(input_root: Path, args: argparse.Namespace) -> Path:
    """Resolve the output directory without encoding flags into its name."""

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
    """Build a JSON-serializable manifest describing preprocessing settings."""

    manifest: Dict[str, Any] = {
        "input_folder": str(input_root),
        "output_folder": str(output_root),
        "glob": args.glob,
        "num_input_files": len(parquet_files),
        "input_files": [p.name for p in parquet_files],
        "assembly_column": args.assembly_column,
        "summary_column": args.summary_column,
        "assembly_tokenizer": args.assembly_tokenizer,
        "assembly_checkpoint": args.assembly_checkpoint,
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
    """Write preprocessing settings to JSON next to the output parquet files."""

    manifest_path = output_root / "preprocessing_spec.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def teardown_prediction_runtime(trainer: Optional[Trainer]) -> None:
    """Best-effort cleanup for Lightning and torch distributed state."""

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
    """Run preprocessing over every parquet file in the input folder."""

    # Parse and validate command-line inputs before doing any work.
    args = parse_args()
    undertale_logging.setup_logging()
    validate_args(args)

    # Resolve and validate the input folder path.
    input_root = Path(args.dataset).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a directory: {input_root}")

    # Build the final output directory path and create it if needed.
    output_root = resolve_output_root(input_root, args)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover all input parquet files.
    parquet_files = find_parquet_files(input_root, args.glob)
    LOGGER.info("Found %d parquet files under %s", len(parquet_files), input_root)

    assembly_tok = None
    if args.tokenize_assembly or args.embed_assembly:
        # Load the custom assembly tokenizer only if it is actually needed.
        assembly_tok = undertale_tokenizer.load(args.assembly_tokenizer)

    summary_tok = None
    if args.tokenize_summaries:
        # Load the GPT-2 tokenizer only if summary tokenization is requested.
        summary_tok = AutoTokenizer.from_pretrained(args.gpt2path, local_files_only=True)
        if summary_tok.pad_token is None:
            # GPT-2 commonly has no pad token configured, so reuse EOS when needed.
            summary_tok.pad_token = summary_tok.eos_token

    trainer = None
    predictor = None
    prediction_writer = None
    if args.embed_assembly:
        encoder_config = build_assembly_encoder_config_from_checkpoint(args.assembly_checkpoint)
        predictor = AssemblyEmbeddingPredictor(encoder_config)
        missing, unexpected = load_assembly_encoder_weights(
            predictor.assembly_encoder,
            args.assembly_checkpoint,
        )
        if missing:
            print(f"Assembly encoder missing keys after load: {missing}")
        if unexpected:
            print(f"Assembly encoder unexpected keys after load: {unexpected}")
        predictor.assembly_encoder.eval()
        prediction_writer = DistributedPredictionWriter()
        trainer = build_prediction_trainer(args, output_root, prediction_writer)

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

        # Process each parquet file independently and write a corresponding output file.
        for parquet_path in parquet_files:
            process_file(
                parquet_path=parquet_path,
                output_root=output_root,
                args=args,
                assembly_tok=assembly_tok,
                summary_tok=summary_tok,
                trainer=trainer,
                predictor=predictor,
                prediction_writer=prediction_writer,
            )
            if trainer is not None:
                trainer.strategy.barrier("file_processed")
    finally:
        teardown_prediction_runtime(trainer)


if __name__ == "__main__":
    # Standard Python entry point so the script can be run directly.
    main()
