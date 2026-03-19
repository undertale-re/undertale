from sqlalchemy.orm import Session

from ..models import User, connect
from ..settings import fetch as fetch_settings
from .base import Command


class Authenticate(Command):
    name = "authenticate"
    help = "generate a token for a user, creating them if they do not exist"

    def add_arguments(self, parser):
        parser.add_argument("username", help="username to authenticate as")

    def handle(self, arguments):
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token

        settings = fetch_settings()
        engine = connect(settings["database"])

        with Session(engine) as session:
            user = session.query(User).filter_by(username=arguments.username).first()
            if user is None:
                user = User(username=arguments.username)
                session.add(user)
                session.commit()

        application = Flask(__name__)
        application.config["JWT_SECRET_KEY"] = settings["jwtsecret"]
        JWTManager(application)

        with application.app_context():
            token = create_access_token(identity=arguments.username)

        print(token)
