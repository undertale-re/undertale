from sqlalchemy.orm import Session

from ..exceptions import CommandError
from ..models import Completion, connect
from ..settings import fetch as fetch_settings
from .base import Command

PROMPT = "Delete completion {id}? [y/N]: "


class Delete(Command):
    name = "delete"
    help = "delete a single completion by ID"

    def add_arguments(self, parser):
        parser.add_argument("id", type=int, help="ID of the completion to delete")
        parser.add_argument(
            "-c",
            "--confirm",
            action="store_true",
            help="skip the interactive confirmation prompt",
        )

    def handle(self, arguments):
        settings = fetch_settings()
        engine = connect(settings["database"])

        with Session(engine) as session:
            completion = session.get(Completion, arguments.id)

            if completion is None:
                raise CommandError(f"completion with ID {arguments.id} does not exist")

            if not arguments.confirm:
                response = input(PROMPT.format(id=arguments.id))
                if response.strip().lower() != "y":
                    print("Aborted.")
                    return

            session.delete(completion)
            session.commit()

        print(f"Deleted completion {arguments.id}.")
