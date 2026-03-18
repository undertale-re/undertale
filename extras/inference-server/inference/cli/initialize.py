from ..settings import initialize
from .base import Command


class Initialize(Command):
    name = "initialize"
    help = "initialize configuration"

    def handle(self, arguments):
        """"""

        initialize()
