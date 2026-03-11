"""Utilities for hyperparameter tuning."""

from typing import Callable

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader


def find_optimal_batch_size(
    model: LightningModule,
    load_dataset: Callable[[int], DataLoader],
    utilization: float = 0.95,
) -> int:
    """Find the maximum per-GPU batch size that fits in memory.

    Performs a doubling search followed by binary search to find the largest
    batch size that does not cause an OOM error, then scales it by a utilization
    factor. Uses a single-device Trainer per probe, which is compatible with
    DDP training (each GPU in DDP independently holds ``batch_size`` samples).

    Args:
        model: The LightningModule to probe. Its state is restored after
            probing so the caller receives an unmodified model.
        load_dataset: Callable that accepts a batch size and returns a
            DataLoader configured with that batch size.
        utilization: Fraction of the maximum fitting batch size to return.
            Defaults to 0.95 to leave a small safety margin.

    Returns:
        The recommended per-GPU batch size.

    Raises:
        RuntimeError: If the model does not fit in memory even at batch_size=1.
    """

    state = {k: v.cpu() for k, v in model.state_dict().items()}

    def probe(batch_size: int) -> bool:
        trainer = Trainer(
            accelerator="auto",
            devices=1,
            num_nodes=1,
            strategy="auto",
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        try:
            trainer.fit(model, train_dataloaders=load_dataset(batch_size))
            return True
        except torch.cuda.OutOfMemoryError:
            del trainer
            model.zero_grad()
            model.load_state_dict(state)
            torch.cuda.empty_cache()
            return False

    # Find an upper bound.
    low = 1
    high = 1
    while probe(high):
        low = high
        high *= 2

    if low == high:
        raise RuntimeError("model does not fit in GPU memory")

    # Binary search until the optimal batch size.
    while high - low > 1:
        mid = (low + high) // 2
        if probe(mid):
            low = mid
        else:
            high = mid

    model.load_state_dict(state)

    return max(1, int(low * utilization))
