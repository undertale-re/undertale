from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Completion, CompletionRating, CompletionState, User, connect
from ..settings import fetch as fetch_settings
from .base import Command


class Completions(Command):
    name = "completions"
    help = "list completions from the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "-l",
            "--limit",
            type=int,
            default=10,
            help="maximum number of completions to show (default: 10)",
        )
        parser.add_argument("-u", "--user", help="filter by username")
        parser.add_argument("-d", "--date", help="filter by date (YYYY-MM-DD)")
        parser.add_argument(
            "-i", "--input", help="filter by input (case-insensitive substring)"
        )

    def handle(self, arguments):
        settings = fetch_settings()
        engine = connect(settings["database"])

        with Session(engine) as session:
            query = (
                session.query(Completion)
                .join(User, Completion.user_id == User.id)
                .order_by(Completion.timestamp.asc())
            )

            if arguments.user:
                query = query.filter(User.username == arguments.user)

            if arguments.date:
                day = date.fromisoformat(arguments.date)
                query = query.filter(func.date(Completion.timestamp) == day)

            if arguments.input:
                query = query.filter(Completion.input.ilike(f"%{arguments.input}%"))

            query = query.limit(arguments.limit)

            print("Completions:")
            for completion in query.all():
                state = CompletionState(completion.state).name
                username = completion.user.username
                timestamp = completion.timestamp.isoformat()
                print(
                    f"  \033[1m{username}\033[0m  {completion.id}  {timestamp}  [{state}]"
                )
                print(f"    input:  {completion.input}")
                print(f"    output: {completion.output}")
                if completion.rating is not None:
                    rating = CompletionRating(completion.rating).name
                    print(f"    rating: {rating}")
                if completion.comments is not None:
                    print(f"    comments: {completion.comments}")
