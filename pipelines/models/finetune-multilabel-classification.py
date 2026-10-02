"""Fine-tune a pretrained encoder to tag functions with taxonomy classes.

Each function may belong to any number of the taxonomy's classes. The training
and validation datasets are tokenized shards carrying the ``labels`` column
written by ``pipelines/datasets/label_parquet.py``: a list of class names, most
specific first. Rows whose labeling failed must be dropped beforehand (for
instance with ``label_parquet.py --drop-unlabeled``).
"""

import json
from os.path import basename, dirname
from pathlib import Path
from typing import List

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pyarrow import parquet

from undertale.models.classification import (
    InstructionTraceTransformerEncoderForMultiLabelSequenceClassification,
    MultiLabelClassificationCollator,
)
from undertale.models.configuration import (
    InstructionTraceTransformerEncoderConfiguration,
)
from undertale.models.dataset import DataModule, ParquetDataset
from undertale.models.tokenizer import TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import TokenizedMultiLabelClassificationDataset
from undertale.utils import cache_path

TAXONOMY = (
    Path(__file__).resolve().parent.parent / "datasets" / "function_taxonomy.json"
)


def load_classes(path: str) -> List[str]:
    """Read class names from the taxonomy, in the order the labeler uses.

    Arguments:
        path: Path to the taxonomy JSON.

    Returns:
        Class names, which become the model's outputs in this order.
    """

    taxonomy = json.loads(Path(path).read_text())
    return [entry["name"] for entry in taxonomy["classes"]]


def compute_positive_weights(dataset: str, classes: List[str]) -> List[float]:
    """Compute per-class positive weights to offset class imbalance.

    Each weight is the ratio of rows without the class to rows with it, so a
    rare class's few positives count for as much as its many negatives. Only
    the ``labels`` column is read, which is much cheaper than iterating the
    dataloader.

    Arguments:
        dataset: Path to the training dataset.
        classes: Class names, in output order.

    Returns:
        One weight per class.
    """

    index = {name: position for position, name in enumerate(classes)}
    counts = [0] * len(classes)
    total = 0

    for path in ParquetDataset(dataset).files:
        for labels in parquet.read_table(path, columns=["labels"])["labels"]:
            total += 1
            for name in labels.as_py() or []:
                counts[index[name]] += 1

    # A class with no positives contributes no positive terms, so its weight
    # is irrelevant; one keeps it finite.
    return [(total - count) / count if count else 1.0 for count in counts]


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


if __name__ == "__main__":
    parser = ModelArgumentParser(
        description="multi-label sequence classification fine-tuning"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="path to a trained tokenizer"
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        required=True,
        help="path to a pretrained masked LM checkpoint",
    )
    parser.add_argument(
        "--taxonomy",
        default=str(TAXONOMY),
        help="taxonomy JSON whose classes the dataset was labeled with",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=InstructionTraceTransformerEncoderForMultiLabelSequenceClassification.THRESHOLD,
        help="probability above which a class is predicted",
    )
    parser.add_argument(
        "--label-balance",
        action="store_true",
        help="weight positive examples per class to offset an unbalanced dataset",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    classes = load_classes(arguments.taxonomy)

    tokenizer = load_tokenizer(cache_path(arguments.tokenizer))

    vocab_size = tokenizer.get_vocab_size()
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    collator = MultiLabelClassificationCollator(classes)

    dataset = cache_path(arguments.dataset)
    validation = arguments.validation
    if validation is not None:
        validation = cache_path(arguments.validation)

    datamodule = DataModule(
        dataset,
        validation,
        schema=TokenizedMultiLabelClassificationDataset,
        collator=collator,
        batch=arguments.batch_size,
        workers=arguments.dataloaders,
        memory=arguments.dataloader_memory,
    )

    weights = None
    if arguments.label_balance:
        weights = compute_positive_weights(dataset, classes)

    model = InstructionTraceTransformerEncoderForMultiLabelSequenceClassification(
        vocab_size=vocab_size,
        next_token_id=next_token_id,
        classes=classes,
        lr=arguments.learning_rate,
        warmup=arguments.warmup,
        positive_weights=weights,
        threshold=arguments.threshold,
        **InstructionTraceTransformerEncoderConfiguration.medium,
    )

    pretrained = torch.load(cache_path(arguments.pretrained), map_location="cpu")
    model.load_state_dict(pretrained["state_dict"], strict=False)

    # ``valid_f1`` is macro F1: with a dominant catch-all class, micro F1 can
    # look good while the rarer classes are never predicted.
    if arguments.validation is not None:
        stop = EarlyStopping(
            monitor="valid_f1", mode="max", patience=5, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(
            filename="{epoch}-{train_loss:.2f}-{valid_f1:.2f}", save_top_k=-1
        )
    else:
        stop = EarlyStopping(
            monitor="train_loss", mode="min", patience=5, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(filename="{epoch}-{train_loss:.2f}", save_top_k=-1)

    progress = ProgressBar(leave=True)

    logger = TensorBoardLogger(
        save_dir=dirname(arguments.output),
        name=basename(arguments.output),
        version=arguments.version,
    )

    trainer = Trainer(
        callbacks=[progress, checkpoint, stop],
        logger=logger,
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        num_nodes=arguments.nodes,
        strategy=arguments.strategy,
        max_epochs=arguments.epochs,
        use_distributed_sampler=False,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=arguments.checkpoint,
    )
