from ..models import connect, destroy
from ..settings import fetch as fetch_settings
from .base import Command


class Destroy(Command):
    name = "destroy"
    help = "destroy the configured database"

    def handle(self, arguments):
        """"""

        settings = fetch_settings()
        engine = connect(settings["database"])
        destroy(engine)
