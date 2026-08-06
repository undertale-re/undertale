from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..exceptions import CommandError
from ..models import Completion, CompletionState, CompletionType, User, connect
from ..settings import fetch as fetch_settings
from .base import Command


class Submit(Command):
    name = "submit"
    help = "submit a completion to be consumed by a worker"

    def add_arguments(self, parser):
        parser.add_argument(
            "-u",
            "--username",
            required=True,
            help="username to submit the completion as",
        )
        parser.add_argument(
            "-t",
            "--type",
            required=True,
            choices=list(CompletionType.__members__),
            help="type of completion to submit",
        )
        parser.add_argument("input", help="the input text to submit")

    def handle(self, arguments):
        settings = fetch_settings()
        engine = connect(settings["database"])

        type = CompletionType[arguments.type]

        with Session(engine) as session:
            user = (
                session.query(User).filter(User.username == arguments.username).first()
            )

            if user is None:
                raise CommandError(f"user '{arguments.username}' does not exist")

            completion = Completion(
                user=user,
                type=int(type),
                input=arguments.input,
                timestamp=datetime.now(UTC),
                state=int(CompletionState.queued),
            )
            session.add(completion)
            session.commit()

            username = completion.user.username
            timestamp = completion.timestamp.isoformat()
            completiontype = CompletionType(completion.type).name
            print("Submitted:")
            print(f"  \033[1m{username}\033[0m  {completion.id}  {timestamp}")
            print(f"    type:  {completiontype}")
            print(f"    input: {completion.input}")
