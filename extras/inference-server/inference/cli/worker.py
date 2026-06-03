from inference.logging import get_logger
from inference.worker import Worker as WorkerProcess

from .base import Command

logger = get_logger(__name__)


class Worker(Command):
    """CLI command to start inference worker processes."""

    name = "worker"
    help = "start inference worker(s)"

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--parallelism",
            type=int,
            default=1,
            help="number of worker processes to spawn (default: 1)",
        )

    def handle(self, arguments):
        logger.info("starting %d worker(s)", arguments.parallelism)

        processes = [WorkerProcess() for _ in range(arguments.parallelism)]

        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            pass

        logger.info("all workers exited")
