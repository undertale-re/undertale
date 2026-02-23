import argparse
import logging
from pathlib import Path

from undertale import logging as undertale_logging
from undertale.datasets.base import Dataset
from undertale.models.item.summarization_dataset import SummarizerDataset

logger = logging.getLogger(__name__)


def split_dataset(args):
    logger.info("loading dataset")
    dataset = Dataset.load(args.dataset)
    logger.info("...done loading dataset")

    cols = dataset.column_names
    if "function_name" in cols and "summary" not in cols:
        dataset = dataset.rename_column("function_name", "summary")
    dataset = dataset.select_columns(["disassembly", "summary"])

    logger.info("splitting dataset")
    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    logger.info("...done splitting dataset")

    logger.info("processing train dataset")
    train_dataset = SummarizerDataset(
        dataset=split["train"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )
    output = Path(args.output) / "train.dataset"
    train_dataset.dataset.save_to_disk(output)
    logger.info(f"...tokenized train dataset written to: {output}")

    logger.info("processing validation dataset")
    validation_dataset = SummarizerDataset(
        dataset=split["test"],
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )
    output = Path(args.output) / "validation.dataset"
    validation_dataset.dataset.save_to_disk(output)
    logger.info(f"...tokenized validation dataset written to: {output}")


if __name__ == "__main__":
    undertale_logging.setup_logging()

    parser = argparse.ArgumentParser(
        description="generate validation split",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="test split ratio (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed to split dataset")
    parser.add_argument(
        "--prefix_length_const",
        type=int,
        default=40,  # i increased from 10 (without tuning with llm)
        help="length for additional prefix constant tokens",
    )
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--tokenizer_size", type=int, default=512)
    parser.add_argument("--token_batchsize", type=int, default=1024)

    # REQUIRED ----------------------------------------------------------------------------

    parser.add_argument("--dataset", type=str, help="path to the dataset")
    parser.add_argument(
        "--output", type=str, help="output directory for validation split"
    )
    parser.add_argument(
        "-e",
        "--end2end",
        dest="end2end",
        action="store_true",
        help="whether to train from assembly code embeddings",
    )
    # parser.add_argument(
    #     "-t", "--tokenizer", required=True, help="trained assembly tokenizer file"
    # )
    parser.add_argument("--gpt2path", type=str)

    args = parser.parse_args()
    split_dataset(args)
