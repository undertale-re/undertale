from ..models import connect, migrate
from ..settings import fetch as fetch_settings
from .base import Command


class Migrate(Command):
    name = "migrate"
    help = "migrate the configured database"

    def handle(self, arguments):
        """"""

        settings = fetch_settings()
        engine = connect(settings["database"])
        migrate(engine)
