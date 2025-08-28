import argparse
import logging
import os

import torch
import transformers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.model_summary import ModelSummary
from ... import logging as undertale_logging
from ...datasets.base import Dataset
from . import tokenizer
from .model import Defaults, TransformerEncoderForSequenceSimilarity

logger = logging.getLogger(__name__)


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


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

    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")

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
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, help="number of nodes to use"
    )

    parser.add_argument(
        "-v", "--version", default=1.0, type=int, help="Tensorboard logger"
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

    arguments = parser.parse_args()

    undertale_logging.setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision('high')
        
    tok = tokenizer.load(arguments.tokenizer, sequence_length=512)


    model = TransformerEncoderForSequenceSimilarity(
        depth=Defaults.depth,
        hidden_dimensions=Defaults.hidden_dimensions,
        vocab_size=tok.get_vocab_size(),
        input_size=Defaults.input_size,
        heads=Defaults.heads,
        intermediate_dimensions=Defaults.intermediate_dimensions,
        dropout=Defaults.dropout,
        eps=Defaults.eps,
        lr=Defaults.lr,
        warmup=Defaults.warmup,
        #embedding_size=128, #ASK TODO REVISIT
        #embedding_dropout_prob=Defaults.dropout
    )
    #summary = ModelSummary(model, max_depth=-1) # Use -1 to show all modules
    #print(summary)
    model =  TransformerEncoderForSequenceSimilarity.load_from_checkpoint(arguments.model)
    model_state_dict = model.state_dict()
    for param_name, param_tensor in model_state_dict.items():
        if (param_name =='encoder.embedding.token.weight'):
            print(f"{param_name}\t{param_tensor}")
    #if (arguments.model):
    #    maskedLMModel = TransformerEncoderForMaskedLM.load_from_checkpoint(arguments.model)
    #elif arguments.checkpoint:
    #    maskedLMModel = model.from_pretrained(arguments.checkpoint, local_files_only=True)
    #    print("CHECKPOINT")

    #model.encoder = maskedLMModel.encoder
    #model.head = maskedLMModel.head
    #model.save_hyperparameters()
    #print("************************************KSRTC")
    #ASK
    summary = ModelSummary(model, max_depth=-1) # Use -1 to show all modules
    print(summary)

    try:
        dataset = Dataset.load(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset.column_names)

    collator = transformers.DefaultDataCollator()

    batch_size = arguments.batch_size
    training = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=7
    )
    validation = DataLoader(
        dataset["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=7
    )

    output = os.path.abspath(os.path.expanduser(arguments.output))

    progress = ProgressBar()
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}",
        save_top_k=-1,
    )
    #stop = EarlyStopping(monitor="valid_f1", mode="max", patience=5, min_delta=0.001)  
    stop = EarlyStopping(
        monitor="val_loss",  # Or any other logged metric
        patience=3,
        verbose=False,
        mode="min", 
    )

    logger = TensorBoardLogger(
        save_dir=os.path.dirname(output),
        name=os.path.basename(output),
        version=arguments.version,
    )
    
    trainer = Trainer(
        callbacks=[progress, checkpoint, stop],
        logger=logger,
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        num_nodes=arguments.nodes,
        strategy = DDPStrategy(find_unused_parameters=True),
        max_epochs=96,
    )

    trainer.fit(
        model,
        train_dataloaders=training,
        val_dataloaders=validation,
        ckpt_path=arguments.checkpoint,
    )
