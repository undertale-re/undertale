import logging
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Completion, CompletionRating, CompletionType, User, connect
from ..settings import fetch as fetch_settings
from .base import Command

logger = logging.getLogger(__name__)


class Export(Command):
    name = "export"
    help = "export completions from the database to a parquet file"

    def add_arguments(self, parser):
        parser.add_argument("output", help="path to write the .parquet file")
        parser.add_argument(
            "-d",
            "--start-date",
            help="exclude completions before this date (YYYY-MM-DD)",
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

            if arguments.start_date:
                start = date.fromisoformat(arguments.start_date)
                query = query.filter(Completion.timestamp >= start)

            rows = [
                {
                    "timestamp": completion.timestamp,
                    "username": completion.user.username,
                    "type": CompletionType(completion.type).name,
                    "input": completion.input,
                    "output": completion.output,
                    "rating": (
                        CompletionRating(completion.rating).name
                        if completion.rating is not None
                        else None
                    ),
                    "comments": completion.comments,
                }
                for completion in query.all()
            ]

        pd.DataFrame(rows).to_parquet(arguments.output, index=False)  # noqa: UT001
        logger.info(f"exported {len(rows)} completions to {arguments.output}")
