import logging
import multiprocessing
import os
import time

from inference import settings
from inference.logging import get_logger, setup_logging
from inference.models import Completion, CompletionState, CompletionType, connect
from sqlalchemy import select, update
from sqlalchemy.orm import Session

logger = get_logger(__name__)


class Worker(multiprocessing.Process):
    """Background process that consumes queued completions and runs inference."""

    def run(self):
        from undertale.models.maskedlm import (
            InstructionTraceTransformerEncoderForMaskedLM,
        )
        from undertale.models.tokenizer import load as load_tokenizer

        if not logging.root.handlers:
            setup_logging()

        config = settings.fetch()
        engine = connect(config["database"], echo=False)
        self.tokenizer = load_tokenizer(config["tokenizer"])
        self.maskedlm = (
            InstructionTraceTransformerEncoderForMaskedLM.load_from_checkpoint(
                config["maskedlm-checkpoint"]
            )
        )
        self.maskedlm.eval()

        logger.info("worker started (pid=%d)", os.getpid())

        try:
            while True:
                with Session(engine) as session:
                    candidate_id = session.scalar(
                        select(Completion.id)
                        .where(Completion.state == int(CompletionState.queued))
                        .order_by(Completion.timestamp)
                        .limit(1)
                    )
                    if candidate_id is None:
                        logger.debug("no queued completions, sleeping")
                        time.sleep(1)
                        continue

                    # Atomically claim the completion. The state guard ensures only
                    # one worker wins if multiple workers find the same candidate.
                    result = session.execute(
                        update(Completion)
                        .where(
                            Completion.id == candidate_id,
                            Completion.state == int(CompletionState.queued),
                        )
                        .values(state=int(CompletionState.running))
                    )
                    session.commit()

                    if result.rowcount == 0:
                        logger.debug(
                            "lost race for completion %d, skipping", candidate_id
                        )
                        continue  # another worker claimed it first

                    logger.info("claimed completion %d", candidate_id)
                    completion = session.get(Completion, candidate_id)
                    try:
                        if completion.type == int(CompletionType.MaskedLM):
                            completion.output = self.complete_maskedlm(completion.input)
                        completion.state = int(CompletionState.complete)
                        logger.info("completed completion %d", completion.id)
                    except Exception as error:
                        logger.exception(
                            "Worker failed on completion %d", completion.id
                        )
                        completion.output = str(error)
                        completion.state = int(CompletionState.complete)
                    session.commit()
        except KeyboardInterrupt:
            logger.info("terminated (pid=%d)", os.getpid())

    def complete_maskedlm(self, input: str) -> str:
        """Run masked language model inference.

        Arguments:
            input: The input string containing [MASK] tokens.

        Returns:
            The input string with [MASK] tokens replaced by predictions.
        """

        from torch import no_grad, tensor

        from undertale.models.tokenizer import TOKEN_PAD

        encoded = self.tokenizer.encode(input)

        tokens = tensor(encoded.ids).unsqueeze(0).to(self.maskedlm.device)
        mask = tensor(encoded.attention_mask).unsqueeze(0).to(self.maskedlm.device)

        with no_grad():
            filled = self.maskedlm.infer(tokens, mask)

        predicted = self.tokenizer.decode(filled.tolist(), skip_special_tokens=False)

        return predicted.replace(TOKEN_PAD, "").strip()
