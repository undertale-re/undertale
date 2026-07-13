"""Density modeling implementation."""

from typing import Optional

from lightning import LightningModule
from torch import Tensor, arange, exp, full_like, gather, log
from torch.nn import GELU, LayerNorm, Linear, Module
from torch.nn.functional import gaussian_nll_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .custom import InstructionTraceTransformerEncoder
from .maskedlm import MaskedLMCollator


class DensityCollator(MaskedLMCollator):
    """Collation function for unsupervised density modeling.

    The task is to model the density of the observed tokens: given a
    sequence with some tokens masked out, predict the original token at each
    masked position. This is identical to ``MaskedLMCollator`` (see that
    class for masking details); it is named separately to distinguish the
    training objective where it's configured in pipelines.
    """


class DensityHead(Module):
    """Density head predicting a per-vocabulary Gaussian per position.

    The head has its own nonlinear adaptation space on top of the shared encoder
    representation, then splits into two projections:
     - the means
     - the log-variances of a diagonal Gaussian over vocabulary elements

    Log-variance is predicted rather than variance directly, for numerical
    stability and to guarantee positivity after exponentiation.

    Arguments:
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        eps: Layer normalization stabilization parameter.
    """

    def __init__(self, hidden_dimensions: int, vocab_size: int, eps: float):
        super().__init__()

        self.transform = Linear(hidden_dimensions, hidden_dimensions)
        self.activation = GELU()
        self.norm = LayerNorm(hidden_dimensions, eps=eps)

        self.mean = Linear(hidden_dimensions, vocab_size)
        self.log_variance = Linear(hidden_dimensions, vocab_size)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Predict per-element means and log-variances.

        Arguments:
            state: The input state tensor from the hidden state of a
                transformer, shaped ``(batch, sequence, hidden)``.

        Returns:
            A tuple ``(mean, log_variance)``, each shaped
            ``(batch, sequence, vocab_size)``.
        """

        hidden = self.activation(self.transform(state))
        hidden = self.norm(hidden)

        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden)

        return mean, log_variance


class InstructionTraceTransformerEncoderForDensity(LightningModule, Module):
    """A transformer encoder with a density head for anomaly detection.

    The model predicts, per position, a Gaussian (mean and variance) for every
    vocabulary element. It is trained unsupervised by minimizing the Gaussian
    negative log-likelihood of the observed token under its own predicted
    Gaussian. The per-token NLL doubles as the anomaly score at inference time.

    Arguments:
        depth: The number of stacked transformer layers.
        hidden_dimensions: The size of the hidden state space.
        vocab_size: The size of the vocabulary.
        sequence_length: The fixed size of the input vector.
        heads: The number of attention heads.
        intermediate_dimensions: The size of the intermediate state space.
        next_token_id: The ID of the special ``NEXT`` token.
        dropout: Dropout probability.
        eps: Layer normalization stabilization parameter.
        lr: Peak learning rate reached after warmup.
        warmup: Fraction of total steps spent in linear warmup.
        freeze_encoder: If true, the encoder weights are frozen and only the
            density head is trained.
        exclude_next: Whether ``NEXT`` positions are excluded from scoring.
        full_variance_nll: Whether to include the additive Gaussian
            normalization constant (``0.5 * log(2 * pi)``) in the loss. The
            variance term itself always contributes to the loss regardless of
            this flag.
    """

    LR = 1e-4
    WARMUP = 0.025
    PAD_LABEL = -100
    NLL_EPS = 1e-6

    def __init__(
        self,
        depth: int,
        hidden_dimensions: int,
        vocab_size: int,
        sequence_length: int,
        heads: int,
        intermediate_dimensions: int,
        next_token_id: int,
        dropout: float,
        eps: float,
        lr: float = LR,
        warmup: float = WARMUP,
        freeze_encoder: bool = False,
        exclude_next: bool = True,
        full_variance_nll: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.next_token_id = next_token_id
        self.exclude_next = exclude_next

        self.encoder = InstructionTraceTransformerEncoder(
            depth,
            hidden_dimensions,
            vocab_size,
            sequence_length,
            heads,
            intermediate_dimensions,
            next_token_id,
            dropout,
            eps,
        )
        self.head = DensityHead(hidden_dimensions, vocab_size, eps)

        self.lr = lr or self.LR
        self.warmup = warmup or self.WARMUP
        self.full_variance_nll = full_variance_nll

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze the encoder so only the density head trains."""

        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self, state: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Encode and predict per-element Gaussian parameters.

        Arguments:
            state: The tokenized input state tensor.
            mask: Optional attention mask.

        Returns:
            A tuple ``(mean, log_variance)`` each shaped
            ``(batch, sequence, vocab_size)``.
        """

        hidden = self.encoder(state, mask)
        mean, log_variance = self.head(hidden)

        return mean, log_variance

    def _score_mask(self, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Boolean mask of positions that count toward scoring/loss.

        True where a position is a real (non-padding) token and, when
        ``exclude_next`` is set, not the structural ``NEXT`` token.

        Arguments:
            tokens: The token ID tensor.
            mask: Optional 0/1 padding mask (1 = real token).

        Returns:
            A boolean tensor shaped like ``tokens``.
        """

        if mask is not None:
            valid = mask == 1
        else:
            # No padding mask provided: treat every position as real.
            valid = full_like(tokens, 1, dtype=bool)

        if self.exclude_next:
            valid = valid & (tokens != self.next_token_id)

        return valid

    def _score(
        self,
        tokens: Tensor,
        mean: Tensor,
        log_variance: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-token anomaly score from precomputed Gaussian parameters.

        Factored out of ``score`` so callers that already ran a forward pass
        (e.g. ``validation_step``) can reuse ``mean``/``log_variance`` instead
        of encoding the batch a second time.

        Arguments:
            tokens: Pre-tokenized input tensor, shaped ``(batch, sequence)``.
            mean: Predicted means, ``(batch, sequence, vocab_size)``.
            log_variance: Predicted log-variances, same shape.
            mask: Optional attention/padding mask tensor.

        Returns:
            A tensor of per-token anomaly scores shaped ``(batch, sequence)``,
            with excluded positions (padding, and optionally ``NEXT``) zeroed.
        """

        # Select the predicted mean/variance for the observed token at each
        # position - i.e. the vocabulary element actually present.
        index = tokens.unsqueeze(-1)
        token_mean = gather(mean, -1, index).squeeze(-1)
        token_log_variance = gather(log_variance, -1, index).squeeze(-1)

        # Floor the variance to match the stability guarantee `gaussian_nll_loss`
        # applies in `_nll`; otherwise a collapsed variance can divide by ~0.
        variance = exp(token_log_variance).clamp(min=self.NLL_EPS)

        # Gaussian NLL of the observed element against its one-hot target (1.0).
        nll = 0.5 * (log(variance) + ((1.0 - token_mean) ** 2) / variance)

        valid = self._score_mask(tokens, mask)
        nll = nll * valid

        return nll

    def score(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute a per-token anomaly score.

        The score for each position is the Gaussian negative log-likelihood of
        the observed token under the predicted Gaussian for the corresponding
        vocabulary element. Higher means more anomalous. Excluded positions
        (padding, and optionally ``NEXT``) are scored as zero.

        This is the convenient one-call entry point for inference: it runs a
        forward pass internally. Callers that already have ``mean`` and
        ``log_variance`` should call ``_score`` directly to avoid a redundant
        forward pass.

        Arguments:
            tokens: Pre-tokenized input tensor, shaped ``(batch, sequence)``.
            mask: Optional attention/padding mask tensor.

        Returns:
            A tensor of per-token anomaly scores shaped ``(batch, sequence)``.
        """

        mean, log_variance = self(tokens, mask)

        return self._score(tokens, mean, log_variance, mask)

    def detect(
        self, tokens: Tensor, mask: Optional[Tensor] = None, threshold: float = 0.0
    ) -> Tensor:
        """Flag anomalous positions by thresholding the per-token score.

        Arguments:
            tokens: Pre-tokenized input tensor, shaped ``(batch, sequence)``.
            mask: Optional attention/padding mask tensor.
            threshold: NLL threshold above which a position is flagged
                anomalous. Calibrate this against a held-out, known-normal
                dataset for the deployed model.

        Returns:
            A boolean tensor shaped ``(batch, sequence)``, True where a
            position's score exceeds ``threshold``. Excluded positions
            (padding, and optionally ``NEXT``) are always ``False``.
        """

        scores = self.score(tokens, mask)
        valid = self._score_mask(tokens, mask)

        return (scores > threshold) & valid

    def _nll(self, mean: Tensor, log_variance: Tensor, labels: Tensor) -> Tensor:
        """Gaussian negative log-likelihood over valid positions.

        The observed token is treated as a one-hot target over the vocabulary:
        the predicted Gaussian for the present element is fit toward 1.0 and
        all absent elements toward 0.0. This shapes a per-element density whose
        likelihood of the realized observation is the training signal.

        Arguments:
            mean: Predicted means, ``(batch, sequence, vocab_size)``.
            log_variance: Predicted log-variances, same shape.
            labels: Observed token IDs with excluded positions as ``PAD_LABEL``.

        Returns:
            A scalar loss tensor. Zero (with a gradient connection to the
            head, so ``backward()`` remains safe to call) if the batch has no
            valid positions.
        """

        valid = labels != self.PAD_LABEL

        if not valid.any():
            return (mean.sum() + log_variance.sum()) * 0.0

        mean = mean[valid]
        log_variance = log_variance[valid]
        target_index = labels[valid]

        # Build one-hot targets over the vocabulary for the valid positions.
        target = mean.new_zeros(mean.shape)
        rows = arange(target.size(0), device=target.device)
        target[rows, target_index] = 1.0

        variance = exp(log_variance)

        return gaussian_nll_loss(
            mean,
            target,
            variance,
            full=self.full_variance_nll,
            eps=self.NLL_EPS,
            reduction="mean",
        )

    def configure_optimizers(self):
        """"""
        # Only optimize parameters that require gradients (handles a frozen
        # encoder cleanly).
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(parameters, lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.warmup * total_steps))
        decay_steps = total_steps - warmup_steps

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, index):
        mean, log_variance = self(batch["tokens"], batch["mask"])
        loss = self._nll(mean, log_variance, batch["labels"])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

        return loss

    def validation_step(self, batch, index):
        mean, log_variance = self(batch["tokens"], batch["mask"])

        loss = self._nll(mean, log_variance, batch["labels"])

        scores = self._score(batch["tokens"], mean, log_variance, batch["mask"])
        valid = self._score_mask(batch["tokens"], batch["mask"])
        mean_score = scores[valid].mean() if valid.any() else scores.new_zeros(())

        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_score", mean_score, prog_bar=True, sync_dist=True)
