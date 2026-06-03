from enum import IntEnum
from os import remove
from os.path import abspath, expanduser

from sqlalchemy import (
    URL,
    Boolean,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, relationship

from .logging import get_logger

logger = get_logger(__name__)


class Model(DeclarativeBase):
    """Base class for all models."""


class fields:
    """Namespace for custom, shared field factories."""

    @staticmethod
    def id():
        return mapped_column(Integer, primary_key=True)

    @staticmethod
    def string(unique=False):
        return mapped_column(String(256), nullable=False, unique=unique)


class User(Model):
    """An individual user account."""

    __tablename__ = "user"

    id = fields.id()

    username = fields.string(unique=True)
    admin = mapped_column(Boolean, default=False, nullable=False)

    completions = relationship("Completion", back_populates="user")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(username={self.username})"


class CompletionType(IntEnum):
    MaskedLM = 0


class CompletionState(IntEnum):
    queued = 0
    running = 1
    complete = 2


class Completion(Model):
    """A text completion."""

    __tablename__ = "completion"

    id = fields.id()

    user_id = mapped_column(ForeignKey("user.id"), nullable=False)
    user = relationship("User", back_populates="completions")

    type = mapped_column(Integer, nullable=False)
    input = mapped_column(Text, nullable=False)
    output = mapped_column(Text, nullable=True)
    timestamp = mapped_column(DateTime, nullable=False)
    state = mapped_column(Integer, default=CompletionState.queued, nullable=False)

    rating = mapped_column(Integer, nullable=True)
    comments = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(user={self.user_id},timestamp={self.timestamp})"
        )


class CompletionRating(IntEnum):
    bad = 0
    fair = 1
    average = 2
    good = 3
    great = 4


def connect(database: str, echo: bool = False) -> Engine:
    """Connect to a database.

    Arguments:
        database: The name of the database (path to database file or
            ``:memory:``).
        echo: Enable statement logging if ``True``.

    Returns:
        An engine object connected to the given database.
    """

    url = URL.create("sqlite", database=database)
    options = {"timeout": 3600}

    return create_engine(url, echo=echo, connect_args=options)


def migrate(engine: Engine) -> None:
    """Initialize the database on the given engine.

    Emits the DDL necessary to build the database schema - this should only be
    called on an empty database.

    Arguments:
        engine: The engine to use.
    """

    Model.metadata.create_all(engine)

    logger.info(f"migrated database on {engine}")


def destroy(engine: Engine) -> None:
    """Destroy the database on the given engine.

    Emits the DDL necessary to destroy the database and drop all data.

    Warning:
        Use with caution! This will delete the entire database and remove all
        tables and data.

    Arguments:
        engine: The engine to use.
    """

    logger.warning(f"destroying database on {engine}")

    Model.metadata.drop_all(engine)

    if engine.url.drivername == "sqlite" and engine.url.database != ":memory:":
        remove(abspath(expanduser(engine.url.database)))


__all__ = [
    "Model",
    "User",
    "CompletionType",
    "CompletionState",
    "Completion",
    "CompletionRating",
    "connect",
    "migrate",
    "destroy",
]
