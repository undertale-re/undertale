import json
from collections import Counter
from os.path import basename, dirname
from typing import Callable, Iterable, List, Optional, Type

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from undertale.models.classification import (
    ClassificationCollator,
    InstructionTraceTransformerEncoderForMultiLabelSequenceClassification,
    InstructionTraceTransformerEncoderForSequenceClassification,
    MultiLabelClassificationCollator,
)
from undertale.models.configuration import (
    InstructionTraceTransformerEncoderConfiguration,
)
from undertale.models.dataset import DataModule
from undertale.models.tokenizer import TOKEN_NEXT
from undertale.models.tokenizer import load as load_tokenizer
from undertale.parsers import ModelArgumentParser
from undertale.schema import Dataset as DatasetSchema
from undertale.schema import (
    TokenizedClassificationDataset,
    TokenizedMultiLabelClassificationDataset,
)
from undertale.utils import cache_path


def compute_class_weights(dataset: Iterable) -> List[float]:
    """Computes class weights for a dataset to handle class imbalance.

    This function calculates the weights for each class in the dataset based on
    the inverse frequency of occurrences of each class. The weights are returned
    as a PyTorch tensor, which can be used to adjust the loss function during
    training (e.g., for weighted cross-entropy loss).

    Args:
        dataset: An iterable of batches where each batch is a dictionary
            containing a key ``labels`` that maps to a list of class labels.

    Returns:
        A list of class weights, where the weight for each class is
        proportional to the inverse of its frequency in the dataset.
    """

    # Count occurrences of each class
    targets = []
    for batch in dataset:
        targets += [int(i) for i in batch["labels"]]
    class_counts = Counter(targets)
    total_samples = len(targets)

    # Compute weights (inverse frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    return [class_weights[i] for i in range(len(class_counts))]


def load_classes(path: str) -> List[str]:
    """Read the taxonomy's class names, in order.

    The order must match the one ``label_parquet.py`` labeled against, so the
    multi-hot column a name maps to is the same at training time as it was at
    labeling time. Both read it from the same taxonomy file.

    Args:
        path: Path to the taxonomy JSON.

    Returns:
        The class names, most specific first.
    """

    taxonomy = json.loads(open(path).read())
    return [entry["name"] for entry in taxonomy["classes"]]


def compute_positive_weights(dataset: Iterable, classes: int) -> List[float]:
    """Compute a per-class positive weight for multi-label training.

    Each class is a separate binary decision, so imbalance is handled per class
    rather than across one shared distribution. The weight on a class's positive
    examples is the ratio of its negatives to its positives, which pushes the
    loss back towards rare classes.

    Args:
        dataset: An iterable of batches, each a dict whose ``labels`` key maps
            to a multi-hot float tensor of shape ``(batch, classes)``.
        classes: The number of classes.

    Returns:
        One weight per class, in class order.
    """

    positives = torch.zeros(classes)
    total = 0
    for batch in dataset:
        labels = batch["labels"]
        positives += labels.sum(dim=0)
        total += labels.shape[0]

    negatives = total - positives
    # Guard against a class with no positive examples, whose ratio is undefined.
    return (negatives / positives.clamp(min=1)).tolist()


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


if __name__ == "__main__":
    parser = ModelArgumentParser(description="sequence classification fine-tuning")

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
        "-k",
        "--classes",
        type=int,
        help="number of output classes (single-label only; multi-label derives "
        "the count from --taxonomy)",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="train the multi-label head on the `labels` name-list written by "
        "label_parquet.py, scoring each taxonomy class independently",
    )
    parser.add_argument(
        "--taxonomy",
        help="path to the taxonomy JSON whose class order defines the label "
        "columns; required with --multi-label",
    )
    parser.add_argument(
        "--label-balance",
        action="store_true",
        help="apply class weights during training for an unbalanced dataset",
    )

    arguments = parser.parse_args()
    parser.setup(arguments)

    if arguments.multi_label:
        if not arguments.taxonomy:
            parser.error("--taxonomy is required with --multi-label")
    elif not arguments.classes:
        parser.error("--classes is required without --multi-label")

    tokenizer = load_tokenizer(cache_path(arguments.tokenizer))

    vocab_size = tokenizer.get_vocab_size()
    next_token_id = tokenizer.token_to_id(TOKEN_NEXT)

    collator: Callable
    schema: Type[DatasetSchema]

    if arguments.multi_label:
        classes = load_classes(cache_path(arguments.taxonomy))
        collator = MultiLabelClassificationCollator(classes)
        schema = TokenizedMultiLabelClassificationDataset
    else:
        collator = ClassificationCollator()
        schema = TokenizedClassificationDataset

    dataset = cache_path(arguments.dataset)
    validation = arguments.validation
    if validation is not None:
        validation = cache_path(arguments.validation)

    datamodule = DataModule(
        dataset,
        validation,
        schema=schema,
        collator=collator,
        batch=arguments.batch_size,
        workers=arguments.dataloaders,
        memory=arguments.dataloader_memory,
    )

    if arguments.multi_label:
        positive_weights: Optional[list[float]] = None
        if arguments.label_balance:
            positive_weights = compute_positive_weights(
                datamodule.train_dataloader(), len(classes)
            )

        model = InstructionTraceTransformerEncoderForMultiLabelSequenceClassification(
            vocab_size=vocab_size,
            next_token_id=next_token_id,
            classes=classes,
            lr=arguments.learning_rate,
            warmup=arguments.warmup,
            positive_weights=positive_weights,
            **InstructionTraceTransformerEncoderConfiguration.medium,
        )
    else:
        weights: Optional[list[float]] = None
        if arguments.label_balance:
            weights = compute_class_weights(datamodule.train_dataloader())

        model = InstructionTraceTransformerEncoderForSequenceClassification(
            vocab_size=vocab_size,
            next_token_id=next_token_id,
            classes=arguments.classes,
            lr=arguments.learning_rate,
            warmup=arguments.warmup,
            class_weights=weights,
            **InstructionTraceTransformerEncoderConfiguration.medium,
        )

    pretrained = torch.load(cache_path(arguments.pretrained), map_location="cpu")
    model.load_state_dict(pretrained["state_dict"], strict=False)

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
