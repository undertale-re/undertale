from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

from flask import Flask, abort, g, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from ldap3 import AUTO_BIND_NONE, SIMPLE, Connection, Server
from ldap3.core.exceptions import LDAPBindError, LDAPException
from sqlalchemy.orm import Session, joinedload
from werkzeug.middleware.proxy_fix import ProxyFix

from .logging import get_logger
from .models import (
    Completion,
    CompletionRating,
    CompletionState,
    CompletionType,
    User,
    connect,
)
from .settings import fetch as fetch_settings
from .text import sanitize

logger = get_logger(__name__)


def require_fields(data: Optional[Dict], *fields: str) -> None:
    if data is None:
        abort(400)
        return
    for field in fields:
        if field not in data:
            abort(400)


def serialize_completion(completion: Completion) -> Dict[str, Any]:
    return {
        "id": completion.id,
        "input": completion.input,
        "output": completion.output,
        "timestamp": completion.timestamp.isoformat() + "Z",
        "completed": completion.state == int(CompletionState.complete),
        "username": completion.user.username,
        "rating": completion.rating,
        "comments": completion.comments,
        "state": int(completion.state),
    }


def create_app() -> Flask:
    application = Flask(__name__)

    application.wsgi_app = ProxyFix(
        application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )

    settings = fetch_settings()
    application.config["ENGINE"] = connect(settings["database"])
    application.config["JWT_SECRET_KEY"] = settings["jwtsecret"]
    application.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=14)
    application.config["LDAP_HOST"] = settings["ldaphost"]
    application.config["LDAP_PORT"] = settings["ldapport"]
    application.config["LDAP_DOMAIN"] = settings["ldapdomain"]

    if settings["jwtsecret"] == "secret":
        logger.warning(
            "JWTSecret is set to its default value; set a strong secret before deploying"
        )

    JWTManager(application)

    @application.before_request
    def before_request():
        g.session = Session(application.config["ENGINE"])

    @application.teardown_request
    def teardown_request(exception):
        session = g.pop("session", None)
        if session is not None:
            session.close()

    @application.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "not found"}), 404

    @application.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "bad request"}), 400

    @application.errorhandler(401)
    def unauthorized(error):
        return jsonify({"error": "unauthorized"}), 401

    @application.route("/login/", methods=["POST"])
    def login():
        data = request.get_json(silent=True)
        require_fields(data, "username", "password")

        username = data["username"]
        password = data["password"]
        ldap_domain = application.config["LDAP_DOMAIN"]
        upn = f"{username}@{ldap_domain}"

        try:
            server = Server(
                application.config["LDAP_HOST"],
                port=application.config["LDAP_PORT"],
                use_ssl=True,
            )
            with Connection(
                server,
                user=upn,
                password=password,
                authentication=SIMPLE,
                auto_bind=AUTO_BIND_NONE,
            ):
                pass
        except LDAPBindError:
            abort(401)
        except LDAPException:
            logger.exception("LDAP error during authentication")
            abort(500)

        user = g.session.query(User).filter_by(username=username).first()
        if user is None:
            user = User(username=username)
            g.session.add(user)
            g.session.commit()

        return jsonify({"token": create_access_token(identity=username)})

    @application.route("/")
    @jwt_required()
    def index():
        prefix = request.script_root.rstrip("/")
        return jsonify(
            {
                "endpoints": [
                    f"GET {prefix}/",
                    f"POST {prefix}/login/",
                    f"GET {prefix}/maskedlm/completion/",
                    f"POST {prefix}/maskedlm/completion/",
                    f"GET {prefix}/maskedlm/completion/<id>/",
                    f"DELETE {prefix}/maskedlm/completion/<id>/",
                    f"POST {prefix}/maskedlm/completion/<id>/feedback/",
                ]
            }
        )

    @application.route("/maskedlm/completion/", methods=["GET"])
    @jwt_required()
    def list_completions():
        completions = (
            g.session.query(Completion)
            .options(joinedload(Completion.user))
            .filter_by(type=int(CompletionType.MaskedLM))
            .all()
        )
        return jsonify([serialize_completion(c) for c in completions])

    @application.route("/maskedlm/completion/", methods=["POST"])
    @jwt_required()
    def create_completion():
        data = request.get_json(silent=True)
        require_fields(data, "input")

        user = g.session.query(User).filter_by(username=get_jwt_identity()).first()
        if user is None:
            abort(401)

        completion = Completion(
            user=user,
            type=int(CompletionType.MaskedLM),
            input=sanitize(data["input"]),
            timestamp=datetime.now(UTC),
            state=int(CompletionState.queued),
        )
        g.session.add(completion)
        g.session.commit()

        logger.info(f"created completion {completion.id} for user {user.username}")

        return jsonify(serialize_completion(completion)), 201

    @application.route("/maskedlm/completion/<int:completion_id>/", methods=["GET"])
    @jwt_required()
    def get_completion(completion_id: int):
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)
        return jsonify(serialize_completion(completion))

    @application.route("/maskedlm/completion/<int:completion_id>/", methods=["DELETE"])
    @jwt_required()
    def delete_completion(completion_id: int):
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)

        g.session.delete(completion)
        g.session.commit()

        logger.info(f"deleted completion {completion_id}")

        return "", 204

    @application.route(
        "/maskedlm/completion/<int:completion_id>/feedback/", methods=["POST"]
    )
    @jwt_required()
    def upsert_feedback(completion_id: int):
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)

        data = request.get_json(silent=True)
        require_fields(data, "rating")

        rating = data["rating"]
        valid_ratings = [int(r) for r in CompletionRating]
        if rating not in valid_ratings:
            abort(400)

        comments = data.get("comments")

        completion.rating = rating
        completion.comments = comments
        g.session.commit()

        return jsonify({"rating": completion.rating, "comments": completion.comments})

    return application


app = create_app()

__all__ = ["app"]
