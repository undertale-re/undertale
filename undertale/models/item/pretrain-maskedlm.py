import argparse
import logging
import math
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
        description="pretrain the model on a masked language modeling dataset",
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "dataset",
        help="dataset on which to train the model (format: `{module.path}:{DatasetClass}`)",
    )
    parser.add_argument("-o", "--output", required=True, help="output model directory")

    parser.add_argument(
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
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

    model = model.InstructionTraceEncoderTransformerForMaskedLM(configuration)
    model = model.to(device)
    if arguments.checkpoint:
        model = model.from_pretrained(arguments.checkpoint, local_files_only=True)

    try:
        dataset = datasets.from_specifier(
            arguments.dataset, schema=datasets.schema.Function
        )
    except ValueError as e:
        logger.critical(e)
        exit(1)

    def tokenize(batch):
        preprocessed = tokenizer.preprocess_batch(batch)
        encoded = tok.encode_batch(preprocessed["preprocessed"])

        batch["input_ids"] = [s.ids for s in encoded]
        batch["attention_mask"] = [s.attention_mask for s in encoded]

        return batch

    dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names, desc="tokenizing"
    )

    dataset = dataset.train_test_split(test_size=0.1)

    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=transformers.PreTrainedTokenizerFast(
            tokenizer_file=arguments.tokenizer,
            mask_token=tokenizer.TOKEN_MASK,
            unk_token=tokenizer.TOKEN_UNKNOWN,
            pad_token=tokenizer.TOKEN_PAD,
        ),
        mlm_probability=0.15,
    )

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
    warmup = steps // (2 * epochs)
    if arguments.checkpoint:
        warmup = 0

    scheduler = transformers.get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=steps,
    )

    parallel = False
    if torch.cuda.device_count() > 1:
        parallel = True
        model.bert = torch.nn.DataParallel(model.bert)

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
            model.bert = model.bert.module
            model.save_pretrained(f"{arguments.output}/{epoch}")
            model.bert = torch.nn.DataParallel(model.bert)
        else:
            model.save_pretrained(f"{arguments.output}/{epoch}")

        model.eval()
        values, labels, losses = [], [], []
        performance = {}
        loop = tqdm.tqdm(validation, desc="validating")
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            references = batch["labels"]
            predictions = torch.argmax(outputs.logits, dim=-1)

            predictions = predictions[references != -100]
            references = references[references != -100]

            values.extend(predictions.tolist())
            labels.extend(references.tolist())
            losses.append(outputs.loss.item())

            # micro-averaged f1 score of masked token prediction
            performance["f1"] = metrics.f1_score(labels, values, average="micro")

            # ppl is ill-defined for masked language modeling, however this is how
            # the code from the Trex paper calcualtes it for masked language
            # pretraining
            loss = sum(losses) / len(losses) / math.sqrt(2)
            performance["ppl"] = 2**loss

            loop.set_postfix(**performance)

        print(f"evaluation performance: {performance}")
