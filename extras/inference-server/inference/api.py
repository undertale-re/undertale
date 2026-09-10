import functools
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, Optional, ParamSpec, TypeVar

from flask import Flask, abort, current_app, g, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    verify_jwt_in_request,
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


ANONYMOUS = "default"
"""Username that all requests run as when authentication is disabled."""


P = ParamSpec("P")
T = TypeVar("T")


def authenticated(function: Callable[P, T]) -> Callable[P, T]:
    """Require a valid JWT, unless authentication is disabled."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if current_app.config["AUTHENTICATION"]:
            verify_jwt_in_request()
        return function(*args, **kwargs)

    return wrapper


def require_fields(data: Optional[Dict], *fields: str) -> None:
    if data is None:
        abort(400)
        return
    for field in fields:
        if field not in data:
            abort(400)


def current_user(session: Session) -> User:
    if not current_app.config["AUTHENTICATION"]:
        user = session.query(User).filter_by(username=ANONYMOUS).first()
        if user is None:
            user = User(username=ANONYMOUS, admin=True)
            session.add(user)
            session.commit()
        elif not user.admin:
            user.admin = True
            session.commit()
        return user

    user = session.query(User).filter_by(username=get_jwt_identity()).first()
    if user is None:
        abort(401)
    return user


def serialize_completion(completion: Completion) -> Dict[str, Any]:
    return {
        "id": completion.id,
        "input": completion.input,
        "output": completion.output,
        "timestamp": completion.timestamp.isoformat() + "Z",
        "completed": completion.state
        in (int(CompletionState.complete), int(CompletionState.failed)),
        "failed": completion.state == int(CompletionState.failed),
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
    application.config["AUTHENTICATION"] = settings["authentication"]

    if settings["authentication"]:
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
    else:
        logger.warning(
            f"authentication is disabled; all requests run as admin user {ANONYMOUS!r}"
        )

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

    @application.errorhandler(403)
    def forbidden(error):
        return jsonify({"error": "forbidden"}), 403

    @application.route("/login/", methods=["POST"])
    def login():
        if not application.config["AUTHENTICATION"]:
            abort(404)

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
            ) as connection:
                if not connection.bind():
                    abort(401)
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

        return jsonify(
            {"token": create_access_token(identity=username), "admin": user.admin}
        )

    @application.route("/")
    @authenticated
    def index():
        prefix = request.script_root.rstrip("/")
        return jsonify(
            {
                "authentication": application.config["AUTHENTICATION"],
                "endpoints": [
                    f"GET {prefix}/",
                    f"POST {prefix}/login/",
                    f"GET {prefix}/maskedlm/completion/",
                    f"POST {prefix}/maskedlm/completion/",
                    f"GET {prefix}/maskedlm/completion/<id>/",
                    f"DELETE {prefix}/maskedlm/completion/<id>/",
                    f"POST {prefix}/maskedlm/completion/<id>/feedback/",
                    f"GET {prefix}/fnaming/completion/",
                    f"POST {prefix}/fnaming/completion/",
                    f"GET {prefix}/fnaming/completion/<id>/",
                    f"DELETE {prefix}/fnaming/completion/<id>/",
                    f"POST {prefix}/fnaming/completion/<id>/feedback/",
                ],
            }
        )

    @application.route("/maskedlm/completion/", methods=["GET"])
    @authenticated
    def list_completions():
        user = current_user(g.session)
        query = (
            g.session.query(Completion)
            .options(joinedload(Completion.user))
            .filter_by(type=int(CompletionType.MaskedLM))
        )
        if not user.admin:
            query = query.filter_by(user_id=user.id)
        completions = query.order_by(Completion.timestamp.desc()).all()
        return jsonify([serialize_completion(c) for c in completions])

    @application.route("/maskedlm/completion/", methods=["POST"])
    @authenticated
    def create_completion():
        data = request.get_json(silent=True)
        require_fields(data, "input")

        user = current_user(g.session)

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
    @authenticated
    def get_completion(completion_id: int):
        user = current_user(g.session)
        completion = (
            g.session.query(Completion)
            .options(joinedload(Completion.user))
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)
        if not user.admin and completion.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence
        return jsonify(serialize_completion(completion))

    @application.route("/maskedlm/completion/<int:completion_id>/", methods=["DELETE"])
    @authenticated
    def delete_completion(completion_id: int):
        user = current_user(g.session)
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)
        if not user.admin and completion.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence

        g.session.delete(completion)
        g.session.commit()

        logger.info(f"deleted completion {completion_id}")

        return "", 204

    @application.route(
        "/maskedlm/completion/<int:completion_id>/feedback/", methods=["POST"]
    )
    @authenticated
    def upsert_feedback(completion_id: int):
        user = current_user(g.session)
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.MaskedLM))
            .first()
        )
        if completion is None:
            abort(404)
        if not user.admin and completion.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence

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

    @application.route("/fnaming/completion/", methods=["GET"])
    @authenticated
    def list_namings():
        user = current_user(g.session)
        query = (
            g.session.query(Completion)
            .options(joinedload(Completion.user))
            .filter_by(type=int(CompletionType.FunctionNaming))
        )
        if not user.admin:
            query = query.filter_by(user_id=user.id)
        completions = query.order_by(Completion.timestamp.desc()).all()
        return jsonify([serialize_completion(c) for c in completions])

    @application.route("/fnaming/completion/", methods=["POST"])
    @authenticated
    def name_function():
        data = request.get_json(silent=True)
        require_fields(data, "input")

        user = current_user(g.session)

        naming = Completion(
            user=user,
            type=int(CompletionType.FunctionNaming),
            input=sanitize(data["input"]),
            timestamp=datetime.now(UTC),
            state=int(CompletionState.queued),
        )
        g.session.add(naming)
        g.session.commit()

        logger.info(f"created function naming {naming.id} for user {user.username}")

        return jsonify(serialize_completion(naming)), 201

    @application.route("/fnaming/completion/<int:completion_id>/", methods=["GET"])
    @authenticated
    def get_naming(completion_id: int):
        user = current_user(g.session)
        naming = (
            g.session.query(Completion)
            .options(joinedload(Completion.user))
            .filter_by(id=completion_id, type=int(CompletionType.FunctionNaming))
            .first()
        )
        if naming is None:
            abort(404)
        if not user.admin and naming.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence
        return jsonify(serialize_completion(naming))

    @application.route("/fnaming/completion/<int:completion_id>/", methods=["DELETE"])
    @authenticated
    def delete_naming(completion_id: int):
        user = current_user(g.session)
        naming = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.FunctionNaming))
            .first()
        )
        if naming is None:
            abort(404)
        if not user.admin and naming.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence

        g.session.delete(naming)
        g.session.commit()

        logger.info(f"deleted naming {completion_id}")

        return "", 204

    @application.route(
        "/fnaming/completion/<int:completion_id>/feedback/", methods=["POST"]
    )
    @authenticated
    def upsert_feedback_fnaming(completion_id: int):
        user = current_user(g.session)
        completion = (
            g.session.query(Completion)
            .filter_by(id=completion_id, type=int(CompletionType.FunctionNaming))
            .first()
        )
        if completion is None:
            abort(404)
        if not user.admin and completion.user_id != user.id:
            abort(404)  # 404 not 403, to avoid leaking existence

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
