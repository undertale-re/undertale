import argparse
import logging
import math
import os

import torch
import torch.distributed as dist
import tqdm
import transformers
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ... import logging as undertale_logging
from ...datasets.base import Dataset
from . import model, tokenizer

logger = logging.getLogger(__name__)


def setup(local_process_rank, global_process_rank, world_size):
    """Initialize the communication backend for each process."""

    node_rank = os.environ["SLURM_NODEID"]
    log = f"[{node_rank}:{local_process_rank}:{global_process_rank}] "
    logger.info(log + "setting up backend")

    # sets the process on the device in the node (local scope)
    torch.cuda.set_device(local_process_rank)

    # sets up communication of processes in the distributed setting (global scope)
    dist.init_process_group(
        backend="nccl",
        rank=global_process_rank,
        world_size=world_size,
    )


def cleanup(log):
    """Cleanup used resources."""

    logger.debug(log + "waiting at barrier")
    dist.barrier()

    dist.destroy_process_group()
    logger.debug(log + "ran destroy_process_group()")


def train(arguments, model):
    node_rank = int(os.environ["SLURM_NODEID"])  # rank of node
    world_size = int(os.environ["WORLD_SIZE"])  # num of total processes
    local_process_rank = int(os.environ["LOCAL_RANK"])  # rank of process in node
    global_process_rank = int(os.environ["RANK"])  # rank of process in world

    log = f"[{node_rank}:{local_process_rank}:{global_process_rank}] "

    # loads the dataset
    try:
        dataset = Dataset.load(arguments.dataset)
    except ValueError as e:
        logger.critical(e)
        exit(1)

    try:
        # initialize communication backend for the process
        setup(local_process_rank, global_process_rank, world_size)

        # splits the dataset in traning and validation sets
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

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=global_process_rank, shuffle=True
        )

        batch_size = arguments.batch_size

        training = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            collate_fn=collator,
            sampler=sampler,
        )
        validation = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            collate_fn=collator,
            sampler=sampler,
        )

        # sets training parameters
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

        model.to(int(local_process_rank))
        model = DDP(model, device_ids=[local_process_rank])

        for epoch in range(arguments.start_epoch, arguments.start_epoch + epochs):
            logger.info(log + f"at epoch {epoch}")

            model.train()
            losses = []
            loop = tqdm.tqdm(training, desc=log + "training")
            for idx, batch in enumerate(loop):
                sampler.set_epoch(epoch)
                batch = {k: v.to(int(local_process_rank)) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                count = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None and count < 5:
                        logger.debug(
                            log
                            + f"epoch {epoch}: batch {idx}: gradient {name}: {param.grad.flatten()[0:5]}"
                        )
                        count += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                loop.set_postfix(loss=sum(losses) / len(losses))
                logger.debug(log + f"at barrier for epoch {epoch}")
                dist.barrier()

            if global_process_rank == 0:
                # torch.save(model.module.state_dict(), f"{arguments.output}/{epoch}")
                model.module.save_pretrained(f"{arguments.output}/{epoch}")

            model.eval()
            values, labels, losses = [], [], []
            performance = {}
            loop = tqdm.tqdm(validation, desc="validating")
            for batch in loop:
                batch = {k: v.to(int(local_process_rank)) for k, v in batch.items()}

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

            logger.info(log + f"evaluation performance: {performance}")
    except:
        logger.critical(log + "an error occurred")
    finally:
        cleanup(log)


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
    parser.add_argument(
        "-l",
        "--logging-level",
        choices=undertale_logging.LEVELS,
        default="info",
        help="logging level (default: %(default)s)",
    )

    # general setup
    arguments = parser.parse_args()
    undertale_logging.setup_logging(
        level=undertale_logging.LEVELS[arguments.logging_level]
    )
    os.makedirs(arguments.output, exist_ok=True)

    # setup tokenizer
    sequence_length = 512
    tok = tokenizer.load(arguments.tokenizer, sequence_length=sequence_length)
    configuration = model.InstructionTraceConfig(
        vocab_size=tok.get_vocab_size(),
        next_token_id=tok.token_to_id("[NEXT]"),
        max_position_embeddings=sequence_length,
        type_vocab_size=1,
    )

    # instantiate model
    model = model.InstructionTraceEncoderTransformerForMaskedLM(configuration)
    model.to(int(os.environ["LOCAL_RANK"]))

    if arguments.checkpoint:
        model = model.from_pretrained(arguments.checkpoint, local_files_only=True)

    # train
    train(arguments, model)
