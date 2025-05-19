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

    undertale_logging.setup_logging()

    os.makedirs(arguments.output, exist_ok=True)

    sequence_length = 512

    tok = tokenizer.load(arguments.tokenizer, sequence_length=sequence_length)

    configuration = model.InstructionTraceConfig(
        vocab_size=tok.get_vocab_size(),
        next_token_id=tok.token_to_id("[NEXT]"),
        max_position_embeddings=sequence_length,
        type_vocab_size=1,
    )

    model = model.InstructionTraceEncoderTransformerForSequenceSimilarity(configuration)

    if arguments.model:
        model.embedding = model.embedding.from_pretrained(
            arguments.model, local_files_only=True
        )
    elif arguments.checkpoint:
        model = model.from_pretrained(arguments.checkpoint, local_files_only=True)

    try:
        dataset = datasets.from_specifier(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    def tokenize(batch):
        preprocessed = tokenizer.preprocess_batch(batch)
        encoded = tok.encode_batch(preprocessed["preprocessed"])

        batch["input_ids"] = [s.ids for s in encoded]
        batch["attention_mask"] = [s.attention_mask for s in encoded]

        return batch

    def tokenize_pair(batch):
        first = tokenize(batch["first"])
        second = tokenize(batch["second"])

        batch = {}
        batch["input_ids1"] = first["input_ids"]
        batch["attention_mask1"] = first["attention_mask"]
        batch["input_ids2"] = second["input_ids"]
        batch["attention_mask2"] = second["attention_mask"]
        batch["labels"] = batch["similarity"]

        return batch

    dataset = dataset.map(
        tokenize_pair,
        batched=True,
        remove_columns=dataset.column_names,
        desc="tokenizing",
    )

    dataset = dataset.train_test_split(test_size=0.1)

    collator = transformers.DefaultDataCollator()

    batch_size = arguments.batch_size
    training = DataLoader(
        dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=collator
    )
    validation = DataLoader(
        dataset["test"], shuffle=True, batch_size=batch_size, collate_fn=collator
    )

    learning_rate = 1e-4
    epochs = arguments.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    batches = len(training)
    steps = epochs * batches

    scheduler = transformers.get_scheduler(
        "constant",
        optimizer=optimizer,
        num_training_steps=steps,
    )

    parallel = False
    if torch.cuda.device_count() > 1:
        parallel = True
        model.embedding.bert = torch.nn.DataParallel(model.embedding.bert)

    model.to(model.device)

    for epoch in range(arguments.start_epoch, arguments.start_epoch + epochs):
        print(f"epoch {epoch}")

        model.train()
        losses = []
        loop = tqdm.tqdm(training, desc="training")
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            loop.set_postfix(loss=sum(losses) / len(losses))

        if parallel:
            model.embedding.bert = model.embedding.bert.module
            model.save_pretrained(f"{arguments.output}/{epoch}")
            model.embedding.bert = torch.nn.DataParallel(model.embedding.bert)
        else:
            model.save_pretrained(f"{arguments.output}/{epoch}")

        model.eval()
        values, labels = [], []
        performance = {}
        loop = tqdm.tqdm(validation, desc="validating")
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            references = batch["labels"]
            predictions = outputs.logits.squeeze()

            labels.extend(references.tolist())
            losses.append(predictions.tolist())

            performance["roc-auc"] = metrics.roc_auc_score(labels, values)

            loop.set_postfix(**performance)

        print(f"evaluation performance: {performance}")
