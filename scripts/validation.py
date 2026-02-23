import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from datasets import load_from_disk
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch import cuda
from torch.utils.data import DataLoader, RandomSampler

from undertale import logging as undertale_logging
from undertale.models.item import TransformerEncoderForSequenceSummarization
from undertale.models.item.finetune_summarization import (
    ProgressBar,
    SummarizeModel,
    ValidationCallback,
    dataset_size_type,
)
from undertale.models.item.summarization_dataset import (
    CustomCollator,
    SummarizerDataset,
)

logger = logging.getLogger(__name__)


def validate(args):
    logger.info("validation began")
    device = "cuda" if cuda.is_available() else "cpu"
    if cuda.is_available():
        cuda.set_device(0)

    logger.info("loading train dataset")
    input = Path(args.dataset) / "train"
    train = load_from_disk(input)
    train_dataset = SummarizerDataset(
        dataset=train,
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )

    logger.info("loading validation dataset")
    input = Path(args.dataset) / "validation"
    validation = load_from_disk(input)
    validation_dataset = SummarizerDataset(
        dataset=validation,
        prefix_length=args.prefix_length_const,
        gpt2path=args.gpt2path,
        normalize_prefix=args.normalize_prefix,
        end2end=args.end2end,
        token_batchsize=args.token_batchsize,
    )

    collator = CustomCollator(
        args=args,
        max_seq_len=train_dataset.max_seq_len,
        device=device,
        pad_id=train_dataset.pad_id,
    )

    final_validation = DataLoader(
        validation_dataset, batch_size=1, collate_fn=collator, num_workers=8
    )

    final_validation_check = ValidationCallback(
        final_validation,
        end2end=args.end2end,
        tag="final_full_val",
        run_on_val_end=True,
        run_on_fit_end=False,
        args=args,
    )

    connector_config = {
        "model_type": args.model_type,
        "prefix_size": 768,
        "prefix_length_const": args.prefix_length_const,
        "prefix_length_assembly": args.prefix_length_assembly,
        "num_layers": args.num_layers,
    }

    connector_config = SimpleNamespace(**connector_config)

    logger.info("setting up the model")
    model = TransformerEncoderForSequenceSummarization(
        args.assembly_checkpoint,
        connector_config,
        args.gpt2path,
        args.end2end,
        args.tune_llm,
    )

    output = os.path.abspath(os.path.expanduser(args.output))

    progress = ProgressBar(leave=True)

    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=args.version,
    )

    callbacks = [
        progress,
        final_validation_check,
    ]

    summarize_model = SummarizeModel(
        model,
        args.prefix_length_const,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        end2end=args.end2end,
    )

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=tensorboard_logger,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        max_epochs=args.num_epochs,
        # Testing
        # log_every_n_steps=1,
        # limit_train_batches=2,
        # limit_val_batches=2,
    )

    logger.info("runnning .validate()")
    results = trainer.validate(
        summarize_model,
        dataloaders=final_validation,
        ckpt_path=args.summarizer_checkpoint,
    )
    # results = trainer.validate(summarize_model, dataloaders=random_validation, ckpt_path=args.summarizer_checkpoint)

    print(f"🎉RESULTS: {results}")


if __name__ == "__main__":
    undertale_logging.setup_logging()

    parser = argparse.ArgumentParser(
        description="validation step",
    )
    parser.add_argument(
        "--prefix_length_const",
        type=int,
        default=40,  # i increased from 10 (without tuning with llm)
        help="length for additional prefix constant tokens",
    )
    parser.add_argument(
        "--prefix_length_assembly",
        type=int,
        default=40,  # i increased from 10 (without tuning with llm)
        help="break down assembly output into sequence length",
    )
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    parser.add_argument("--token_batchsize", type=int, default=1024)
    parser.add_argument(
        "--model_type", type=str, default="transformer", help="model for connector"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="number layers for transformer"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "-warmup",
        "--warmup_steps",
        type=int,
        default=1,  # used to be 50000
        help="number of warmup steps",
    )
    parser.add_argument(
        "-a", "--accelerator", default="auto", help="accelerator to use"
    )
    parser.add_argument("--tokenizer_size", type=int, default=512)
    parser.add_argument(
        "--dataset_size",
        type=dataset_size_type,
        default=-1,
        help="subsample dataset. -1 means use entire dataset. will throw error is choose 0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="max number tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for generation",
    )

    # MAYBE -------------------------------------------------------------------------------------
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, help="number of nodes to use"
    )
    parser.add_argument(
        "-d",
        "--devices",
        default=1,
        type=int,
        help="number of accelerator devices to use (per node)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="number epochs to train model"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="if beam is toggled, number of beams to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")

    # REQUIRED ----------------------------------------------------------------------------------
    parser.add_argument("--dataset", type=str, help="path to the validation dataset")
    parser.add_argument("--gpt2path", type=str)
    parser.add_argument(
        "--bertscore_model_path",
        type=str,
        help="path for bert score model",
    )
    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained assembly tokenizer file"
    )
    parser.add_argument(
        "-c",
        "--summarizer_checkpoint",
        help="trained model checkpoint from which to resume training",
    )
    parser.add_argument(
        "--assembly_checkpoint",
        type=str,
        help="trained model checkpoint from which to resume training",
    )
    parser.add_argument("--output", help="output model directory")

    parser.add_argument(
        "--generated_output_paths",
        default="./validation_outputs",
        type=str,
        help="where to output validation examples",
    )
    parser.add_argument(
        "--beam",
        action="store_true",
        help="whether to use beam search for generation of text",
    )
    parser.add_argument(
        "-e",
        "--end2end",
        dest="end2end",
        action="store_true",
        help="whether to train from assembly codo embeddings ",
    )
    parser.add_argument("--tune_llm", dest="tune_llm", action="store_true")
    parser.add_argument("-v", "--version", help="training run version name")

    args = parser.parse_args()

    validate(args)
