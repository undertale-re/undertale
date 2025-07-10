import argparse
import logging
import os

import torch
import tqdm
import transformers
from sklearn import metrics
from torch.utils.data import DataLoader

from ... import datasets
from ... import logging as undertale_logging
from . import model, tokenizer

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="finetune the model on a pairwise embedding dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "dataset",
        help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("-o", "--output", required=True, help="output model directory")

    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument(
        "-m", "--model", help="pretrained model from which to begin training"
    )
    start.add_argument(
        "-c",
        "--checkpoint",
        help="trained model checkpoint from which to resume training",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="number of epochs for which to train",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="starting epoch number"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")

    arguments = parser.parse_args()