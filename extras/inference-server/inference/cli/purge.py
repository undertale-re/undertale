from sqlalchemy.orm import Session

from ..models import Completion, CompletionState, connect
from ..settings import fetch as fetch_settings
from .base import Command

WARNING = (
    "Warning: This command should only be run when there are no active inference\n"
    "workers. Running it while workers are active may cause completions to be\n"
    "processed more than once."
)

PROMPT = "Reset all running completions to queued? (y/N): "


class Purge(Command):
    name = "purge"
    help = "reset running completions to queued to recover from worker failure"

    def add_arguments(self, parser):
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="skip the interactive confirmation prompt",
        )

    def handle(self, arguments):
        print(WARNING)

        if not arguments.confirm:
            response = input(PROMPT)
            if response.strip().lower() != "y":
                print("Aborted.")
                return

        settings = fetch_settings()
        engine = connect(settings["database"])

        with Session(engine) as session:
            records = (
                session.query(Completion)
                .filter_by(state=int(CompletionState.running))
                .all()
            )

            count = len(records)

            if count == 0:
                print("No running completions found. Nothing to do.")
                return

            for record in records:
                record.state = int(CompletionState.queued)

            session.commit()

        print(f"Reset {count} running completion(s) to queued.")
