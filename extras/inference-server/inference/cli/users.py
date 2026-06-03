from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from ..models import Completion, User, connect
from ..settings import fetch as fetch_settings
from .base import Command


class Users(Command):
    name = "users"
    help = "list users in the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "-s",
            "--sorted",
            action="store_true",
            help="sort by completion count descending instead of alphabetically",
        )

    def handle(self, arguments):
        settings = fetch_settings()
        engine = connect(settings["database"])

        print("Users:")

        with Session(engine) as session:
            with_feedback = func.sum(
                case(
                    (
                        or_(
                            Completion.rating.isnot(None),
                            Completion.comments.isnot(None),
                        ),
                        1,
                    ),
                    else_=0,
                )
            )
            total = func.count(Completion.id)

            query = (
                session.query(User.username, User.admin, total, with_feedback)
                .outerjoin(Completion, Completion.user_id == User.id)
                .group_by(User.id)
            )

            if arguments.sorted:
                query = query.order_by(total.desc())
            else:
                query = query.order_by(User.username)

            for username, admin, count, feedback in query.all():
                suffix = " (admin)" if admin else ""
                print(
                    f"  \033[1m{username}\033[0m (completions: {count} [{feedback}]){suffix}"
                )
