from .admin import Admin
from .authenticate import Authenticate
from .base import Command, build_parser
from .completions import Completions
from .destroy import Destroy
from .export import Export
from .initialize import Initialize
from .migrate import Migrate
from .purge import Purge
from .users import Users
from .worker import Worker

__commands__ = [
    Initialize,
    Migrate,
    Destroy,
    Admin,
    Authenticate,
    Purge,
    Users,
    Completions,
    Export,
    Worker,
]

__all__ = ["Command", "build_parser"]
