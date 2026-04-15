"""Preprocess summarization parquet datasets with the datatrove pipeline."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ... import logging as undertale_logging
from ...models.item.model import TransformerEncoder
from ..base import Dataset, EXECUTORS
from ..pipeline.formatters import ITEMTokenizer

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

TEMP_ASSEMBLY_COLUMN_KEY = "__undertale_assembly_column__"
TEMP_SUMMARY_COLUMN_KEY = "__undertale_summary_column__"
TEMP_SYNTHETIC_DISASSEMBLY_KEY = "__undertale_synthetic_disassembly__"
TEMP_ORIGINAL_DISASSEMBLY_KEY = "__undertale_original_disassembly__"
TEMP_HAD_ORIGINAL_DISASSEMBLY_KEY = "__undertale_had_original_disassembly__"


def adapt_dataset_from_parquet(self, data: dict, path: str, id_in_file: int | str) -> dict:
    row = dict(data)
    document_id = row.pop("id", id_in_file)

    return {
        "id": document_id,
        "text": str(row.get("code", "") or ""),
        "metadata": row,
    }


def adapt_document_to_parquet(self, document) -> dict:
    row = {"id": document.id}
    row.update(document.metadata)
    return row


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

    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        resolved = lowered.get(candidate.lower())
        if resolved is not None:
            return resolved

    raise KeyError(
        f"Could not infer {kind} column. Found columns: {list(columns)}. "
        f"Pass --{kind}_column explicitly."
    )


def resolve_output_names(args: argparse.Namespace, assembly_col: str, summary_col: str) -> Dict[str, str]:
    return {
        "assembly_tokens": args.assembly_tokens_column or f"{assembly_col}_tokens",
        "assembly_mask": args.assembly_mask_column or f"{assembly_col}_mask",
        "assembly_embedding": args.assembly_embedding_column or f"{assembly_col}_embedding",
        "summary_tokens": args.summary_tokens_column or f"{summary_col}_tokens",
        "summary_mask": args.summary_mask_column or f"{summary_col}_mask",
    }


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def find_parquet_files(folder: Path) -> list[Path]:
    files = sorted(path for path in folder.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found under {folder}")
    return files


def stage_exact_subset_input(input_root: Path, output_root: Path, dataset_size: int) -> Optional[Path]:
    """Materialize the first N rows into a temporary parquet directory."""

    if dataset_size == -1:
        return None

    subset_root = output_root.parent / f".{output_root.name}_subset_input"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    parquet_files = find_parquet_files(input_root)
    remaining_rows = dataset_size

    for parquet_path in parquet_files:
        if remaining_rows <= 0:
            break

        parquet_file = pq.ParquetFile(parquet_path)
        file_rows = parquet_file.metadata.num_rows
        if file_rows <= 0:
            continue

        take_rows = min(remaining_rows, file_rows)
        destination = subset_root / parquet_path.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if take_rows == file_rows:
            shutil.copy2(parquet_path, destination)
        else:
            table = pq.read_table(parquet_path).slice(0, take_rows)
            pq.write_table(table, destination)

        remaining_rows -= take_rows

    LOGGER.info(
        "Staged exact subset input at %s using %d requested rows",
        subset_root,
        dataset_size,
    )
    return subset_root


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


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def build_assembly_encoder_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
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


def load_assembly_encoder_weights(encoder, checkpoint_path: str) -> None:
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[len("encoder."):]] = value

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    if missing:
        LOGGER.warning("Assembly encoder missing keys after load: %s", missing)
    if unexpected:
        LOGGER.warning("Assembly encoder unexpected keys after load: %s", unexpected)


class PrepareAssemblyForTokenizer(PipelineStep):
    """Expose the chosen assembly column as `disassembly` for ITEMTokenizer."""

    type = "✂️ - FORMATTER"
    name = "🧭 Summarization Assembly Resolver"

    def __init__(self, requested_assembly_column: Optional[str]):
        super().__init__()
        self.requested_assembly_column = requested_assembly_column
        self._assembly_column: Optional[str] = None

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        del rank, world_size
        if not data:
            return

        for document in data:
            with self.track_time():
                metadata = document.metadata
                assembly_column = self._assembly_column
                if assembly_column is None or assembly_column not in metadata:
                    assembly_column = infer_column(
                        list(metadata.keys()),
                        self.requested_assembly_column,
                        DEFAULT_ASSEMBLY_CANDIDATES,
                        "assembly",
                    )
                    self._assembly_column = assembly_column

                metadata[TEMP_ASSEMBLY_COLUMN_KEY] = assembly_column
                if "disassembly" in metadata:
                    metadata[TEMP_HAD_ORIGINAL_DISASSEMBLY_KEY] = True
                    metadata[TEMP_ORIGINAL_DISASSEMBLY_KEY] = metadata["disassembly"]

                if assembly_column != "disassembly":
                    metadata[TEMP_SYNTHETIC_DISASSEMBLY_KEY] = True

                metadata["disassembly"] = sanitize_text(metadata.get(assembly_column))
                yield document


class SummarizationAugmenter(PipelineStep):
    """Add assembly tokens, embeddings, and summary tokens to each row."""

    type = "✂️ - FORMATTER"
    name = "🧠 Summarization Preprocessor"

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self._summary_column: Optional[str] = None
        self._summary_tokenizer = None
        self._summary_stop_token: Optional[int] = None
        self._summary_max_seq_len: Optional[int] = None
        self._encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_summary_tokenizer(self):
        if self._summary_tokenizer is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.args.gpt2path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        stop_token, max_seq_len = resolve_summary_tokenization_settings(
            tokenizer,
            self.args.gpt2path,
            self.args.prefix_length,
        )

        self._summary_tokenizer = tokenizer
        self._summary_stop_token = stop_token
        self._summary_max_seq_len = max_seq_len

    def _load_encoder(self):
        if self._encoder is not None:
            return

        config = build_assembly_encoder_config_from_checkpoint(self.args.assembly_checkpoint)
        encoder = TransformerEncoder(
            config["depth"],
            config["hidden_dimensions"],
            config["vocab_size"],
            config["input_size"],
            config["heads"],
            config["intermediate_dimensions"],
            config["dropout"],
            config["eps"],
        )
        load_assembly_encoder_weights(encoder, self.args.assembly_checkpoint)
        encoder.eval()
        encoder.to(self._device)
        self._encoder = encoder
        LOGGER.info("Summarization encoder running on device: %s", self._device)

    def _resolve_summary_column(self, metadata: Dict[str, Any]) -> str:
        summary_column = metadata.get(TEMP_SUMMARY_COLUMN_KEY)
        if summary_column is not None:
            return summary_column

        if self._summary_column is None or self._summary_column not in metadata:
            self._summary_column = infer_column(
                list(metadata.keys()),
                self.args.summary_column,
                DEFAULT_SUMMARY_CANDIDATES,
                "summary",
            )

        metadata[TEMP_SUMMARY_COLUMN_KEY] = self._summary_column
        return self._summary_column

    def _build_embedding(self, input_ids: Sequence[int], attention_mask: Sequence[int]) -> list[float]:
        self._load_encoder()
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self._device)
        mask = torch.tensor([list(attention_mask)], dtype=torch.long, device=self._device)

        with torch.inference_mode():
            hidden = self._encoder(ids, mask)
            pooled = hidden.mean(dim=1)[0]

        return pooled.detach().cpu().tolist()

    def _tokenize_summary(self, text: str) -> Tuple[list[int], list[int]]:
        self._load_summary_tokenizer()
        ids = self._summary_tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self._summary_max_seq_len,
        )
        ids = list(ids) + [self._summary_stop_token]
        return ids, [1] * len(ids)

    def _cleanup(self, metadata: Dict[str, Any]) -> None:
        metadata.pop("tokens", None)
        metadata.pop(TEMP_ASSEMBLY_COLUMN_KEY, None)
        metadata.pop(TEMP_SUMMARY_COLUMN_KEY, None)

        had_synthetic_disassembly = bool(metadata.pop(TEMP_SYNTHETIC_DISASSEMBLY_KEY, False))
        had_original_disassembly = bool(metadata.pop(TEMP_HAD_ORIGINAL_DISASSEMBLY_KEY, False))
        original_disassembly = metadata.pop(TEMP_ORIGINAL_DISASSEMBLY_KEY, None)

        if had_synthetic_disassembly:
            metadata.pop("disassembly", None)
        elif had_original_disassembly:
            metadata["disassembly"] = original_disassembly

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        del rank, world_size
        if not data:
            return

        for document in data:
            with self.track_time():
                metadata = document.metadata
                assembly_column = metadata.get(TEMP_ASSEMBLY_COLUMN_KEY)
                summary_column = self._resolve_summary_column(metadata)

                if assembly_column is None and (self.args.tokenize_assembly or self.args.embed_assembly):
                    assembly_column = infer_column(
                        list(metadata.keys()),
                        self.args.assembly_column,
                        DEFAULT_ASSEMBLY_CANDIDATES,
                        "assembly",
                    )

                output_names = resolve_output_names(self.args, assembly_column or "assembly", summary_column)

                if self.args.tokenize_assembly or self.args.embed_assembly:
                    tokens = metadata.get("tokens")
                    if not isinstance(tokens, dict):
                        raise ValueError(
                            "Assembly tokenization metadata was not found. "
                            "The pipeline must run ITEMTokenizer before augmentation."
                        )

                    input_ids = list(tokens["input_ids"])
                    attention_mask = list(tokens["attention_mask"])

                    if self.args.tokenize_assembly:
                        metadata[output_names["assembly_tokens"]] = input_ids
                        metadata[output_names["assembly_mask"]] = attention_mask

                    if self.args.embed_assembly:
                        metadata[output_names["assembly_embedding"]] = self._build_embedding(
                            input_ids,
                            attention_mask,
                        )

                if self.args.tokenize_summaries:
                    summary_text = sanitize_text(metadata.get(summary_column))
                    summary_ids, summary_mask = self._tokenize_summary(summary_text)
                    metadata[output_names["summary_tokens"]] = summary_ids
                    metadata[output_names["summary_mask"]] = summary_mask

                self._cleanup(metadata)
                yield document


class SummarizationPreprocessor(Dataset):
    def __init__(self, *args, cli_args: argparse.Namespace, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args

    def build(self, input: str, output: str, parallelism: int = 1) -> None:
        writer = [
            ParquetWriter(
                output,
                adapter=adapt_document_to_parquet,
                max_file_size=100 * 1024 * 1024,
            )
        ]
        executor = self.get_pipeline(input, writer, parallelism)
        executor.run()

    def get_pipeline(self, input, writer, parallelism):
        steps = [
            ParquetReader(
                input,
                adapter=adapt_dataset_from_parquet,
            ),
        ]

        if self.cli_args.tokenize_assembly or self.cli_args.embed_assembly:
            steps.append(PrepareAssemblyForTokenizer(self.cli_args.assembly_column))
            steps.append(ITEMTokenizer(self.cli_args.assembly_tokenizer))

        steps.append(SummarizationAugmenter(self.cli_args))
        steps.extend(writer)

        return self.get_executor(
            steps,
            venv_path=os.path.join(f"{Path.home()}/.conda/envs", "undertale"),
            time="48:00:00",
            cpus_per_task=1,
            mem_per_cpu_gb=8,
            tasks=parallelism,
            job_name="preprocess-summarization",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )


def build_preprocessing_manifest(
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "input_folder": str(input_root),
        "output_folder": str(output_root),
        "assembly_column": args.assembly_column,
        "summary_column": args.summary_column,
        "assembly_tokenizer": args.assembly_tokenizer,
        "assembly_checkpoint": args.assembly_checkpoint,
        "gpt2path": args.gpt2path,
        "prefix_length": args.prefix_length,
        "dataset_size": args.dataset_size,
        "executor": args.executor,
        "parallelism": args.parallelism,
        "logging_directory": args.logging_directory,
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

    if args.tokenize_summaries:
        tokenizer = AutoTokenizer.from_pretrained(args.gpt2path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_token, max_seq_len = resolve_summary_tokenization_settings(
            tokenizer,
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
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "preprocessing_spec.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess summarization parquet files with the dataset pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", required=True, help="Input parquet dataset directory.")
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Output directory for parquet shards and preprocessing_spec.json.",
    )
    parser.add_argument(
        "-e",
        "--executor",
        choices=EXECUTORS,
        default="local",
        help="executor on which to run the given pipeline",
    )
    parser.add_argument(
        "-l",
        "--logging-directory",
        help="override logging directory path",
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        default=1,
        help="degree of parallelism",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of datapoints to preprocess across the dataset. Use -1 to process all rows.",
    )
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
    parser.add_argument(
        "--assembly_tokenizer",
        type=str,
        default=None,
        help="Path to the Undertale assembly tokenizer JSON.",
    )
    parser.add_argument(
        "--assembly_checkpoint",
        type=str,
        default=None,
        help="Checkpoint for the assembly encoder. Required if --embed_assembly is set.",
    )
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
        help="Number of prefix tokens reserved for summary decoding.",
    )
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


def main() -> None:
    undertale_logging.setup_logging()

    args = parse_args()
    validate_args(args)

    input_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output_folder).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a directory: {input_root}")

    staged_input_root = stage_exact_subset_input(input_root, output_root, args.dataset_size)
    effective_input_root = staged_input_root or input_root

    manifest = build_preprocessing_manifest(args, input_root, output_root)
    manifest_path = write_preprocessing_manifest(output_root, manifest)
    LOGGER.info("Wrote preprocessing manifest to %s", manifest_path)

    try:
        dataset = SummarizationPreprocessor(
            writer="parquet",
            executor=args.executor,
            logging_directory=args.logging_directory,
            cli_args=args,
        )
        dataset.build(
            input=str(effective_input_root),
            output=str(output_root),
            parallelism=args.parallelism,
        )
    finally:
        if staged_input_root is not None and staged_input_root.exists():
            shutil.rmtree(staged_input_root)


if __name__ == "__main__":
    main()
