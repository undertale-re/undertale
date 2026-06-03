from .admin import Admin
from .authenticate import Authenticate
from .base import Command, build_parser
from .completions import Completions
from .delete import Delete
from .destroy import Destroy
from .export import Export
from .initialize import Initialize
from .migrate import Migrate
from .purge import Purge
from .submit import Submit
from .users import Users
from .worker import Worker

__commands__ = [
    Initialize,
    Migrate,
    Destroy,
    Admin,
    Authenticate,
    Purge,
    Delete,
    Users,
    Completions,
    Submit,
    Export,
    Worker,
]

__all__ = ["Command", "build_parser"]
