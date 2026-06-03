from sqlalchemy.orm import Session

from ..exceptions import CommandError
from ..models import User, connect
from ..settings import fetch as fetch_settings
from .base import Command


class Admin(Command):
    name = "admin"
    help = "set or unset the admin flag on a user"

    def add_arguments(self, parser):
        parser.add_argument("username", help="username of the user to modify")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--promote", action="store_true", help="grant admin privileges"
        )
        group.add_argument(
            "--demote", action="store_true", help="revoke admin privileges"
        )

    def handle(self, arguments):
        settings = fetch_settings()
        engine = connect(settings["database"])

        with Session(engine) as session:
            user = session.query(User).filter_by(username=arguments.username).first()

            if user is None:
                raise CommandError(
                    f"No user with username '{arguments.username}' exists."
                )

            user.admin = arguments.promote
            session.commit()

            if arguments.promote:
                print(f"{arguments.username} promoted to admin")
            else:
                print(f"{arguments.username} demoted from admin")
