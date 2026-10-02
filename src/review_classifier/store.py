"""Persistence of prediction metadata and labelled feedback.

Every prediction is recorded with the model version that produced it, so a
prediction can always be traced to exact weights, and feedback can later be
joined back to it to measure how the deployed model performs.

Design choices:

* **No review text is stored with predictions**, only its SHA-256 and length,
  because reviews can contain personal data. Text is stored only when a client
  sends it with feedback, and only if it hashes to the predicted text, so a
  labelled example is always the exact text the model saw.
* **Serving does not depend on the database.** If a write fails the prediction is
  still returned, marked ``recorded: false``. After a connection failure the
  store stops trying for ``retry_after`` seconds, so a database outage costs
  each request nothing instead of a connection timeout.
* **The schema is owned by migrations** (``python -m review_classifier.db
  upgrade``). The service checks the schema version at startup and refuses to
  start against a database that was not migrated to the version it expects.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

logger = logging.getLogger("review_classifier")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# ---- schema (mirrors migrations/versions; migrations are the source of truth) ----

metadata = sa.MetaData()

predictions = sa.Table(
    "predictions",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    ),
    sa.Column("request_id", sa.String(128), nullable=False),
    sa.Column("endpoint", sa.String(32), nullable=False),
    sa.Column("batch_index", sa.Integer, nullable=False),
    sa.Column("batch_size", sa.Integer, nullable=False),
    sa.Column("model_id", sa.String(512), nullable=False),
    sa.Column("model_version", sa.String(128), nullable=False),
    sa.Column("task", sa.String(64), nullable=False),
    sa.Column("label", sa.String(128), nullable=False),
    sa.Column("score", sa.Float, nullable=False),
    sa.Column("scores", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
    sa.Column("text_sha256", sa.String(64), nullable=False),
    sa.Column("text_chars", sa.Integer, nullable=False),
    sa.Column("inference_ms", sa.Float, nullable=False),
)

feedback = sa.Table(
    "feedback",
    metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
    sa.Column(
        "prediction_id",
        sa.Uuid,
        sa.ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    ),
    sa.Column("label", sa.String(128), nullable=False),
    sa.Column("text", sa.Text, nullable=True),
)


# ---- helpers ---------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """Hash of the text exactly as the client sent it, before preprocessing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_database_url(url: str) -> str:
    """Use the psycopg 3 driver for plain postgresql:// URLs (the usual form in docs)."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def redact(url: str) -> str:
    return make_url(normalize_database_url(url)).render_as_string(hide_password=True)


def create_db_engine(url: str) -> Engine:
    url = normalize_database_url(url)
    backend = make_url(url).get_backend_name()
    if backend == "postgresql":
        connect_args: dict[str, Any] = {
            # Fail fast when the database is unreachable or a statement hangs.
            "connect_timeout": 3,
            "options": "-c statement_timeout=5000",
        }
        return sa.create_engine(
            url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args=connect_args
        )
    if backend == "sqlite":
        return sa.create_engine(url, connect_args={"check_same_thread": False})
    return sa.create_engine(url, pool_pre_ping=True)


def head_revision() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    head = ScriptDirectory.from_config(config).get_current_head()
    assert head is not None
    return head


def current_revision(engine: Engine) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


# ---- errors ----------------------------------------------------------------------


class SchemaMismatchError(RuntimeError):
    """The database schema is not at the version this code expects."""


class StoreUnavailableError(RuntimeError):
    """The database cannot be reached right now."""


class PredictionNotFoundError(LookupError):
    pass


class FeedbackExistsError(RuntimeError):
    pass


class InvalidFeedbackError(ValueError):
    def __init__(self, field: str, kind: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.kind = kind


# ---- store -----------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    id: uuid.UUID
    request_id: str
    endpoint: str
    batch_index: int
    batch_size: int
    model_id: str
    model_version: str
    task: str
    label: str
    score: float
    scores: dict[str, float]
    text_sha256: str
    text_chars: int
    inference_ms: float


@dataclass(frozen=True)
class FeedbackResult:
    id: int
    prediction_id: uuid.UUID
    label: str
    predicted_label: str
    text_stored: bool

    @property
    def model_was_correct(self) -> bool:
        return self.label == self.predicted_label


class PredictionStore:
    def __init__(
        self,
        url: str,
        *,
        retry_after: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = redact(url)
        self.engine = create_db_engine(url)
        self._retry_after = retry_after
        self._clock = clock
        self._lock = threading.Lock()
        self._unavailable_until = 0.0

    # -- availability -----------------------------------------------------------

    @property
    def available(self) -> bool:
        with self._lock:
            return self._clock() >= self._unavailable_until

    def _mark_unavailable(self, exc: Exception) -> None:
        with self._lock:
            already_down = self._clock() < self._unavailable_until
            self._unavailable_until = self._clock() + self._retry_after
        if not already_down:
            logger.error(
                "database unavailable (%s); predictions are not recorded, retrying in %.0fs",
                type(exc).__name__,
                self._retry_after,
            )

    def _mark_available(self) -> None:
        with self._lock:
            was_down = self._unavailable_until > 0.0
            self._unavailable_until = 0.0
        if was_down:
            logger.info("database available again; recording resumed")

    @property
    def status(self) -> str:
        return "ok" if self.available else "unavailable"

    def check_schema(self) -> None:
        """Called at startup. Raises SchemaMismatchError; tolerates an unreachable database."""
        try:
            current = current_revision(self.engine)
        except OperationalError as exc:
            self._mark_unavailable(exc)
            return
        expected = head_revision()
        if current != expected:
            raise SchemaMismatchError(
                f"Database schema is at revision {current!r} but this version of the service "
                f"needs {expected!r}. Run: python -m review_classifier.db upgrade"
            )
        logger.info("database ready at %s (schema %s)", self.url, current)

    # -- writes -----------------------------------------------------------------

    def record(self, rows: Sequence[PredictionRecord]) -> bool:
        """Insert prediction rows. Returns False, never raises, if they could not be stored."""
        if not rows:
            return True
        if not self.available:
            return False
        try:
            with self.engine.begin() as connection:
                connection.execute(predictions.insert(), [row.__dict__ for row in rows])
        except OperationalError as exc:
            self._mark_unavailable(exc)
            return False
        except SQLAlchemyError:
            logger.exception("could not record %d predictions", len(rows))
            return False
        self._mark_available()
        return True

    def add_feedback(
        self, prediction_id: uuid.UUID, label: str, text: str | None = None
    ) -> FeedbackResult:
        if not self.available:
            raise StoreUnavailableError("The database is unavailable.")
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    sa.select(
                        predictions.c.label, predictions.c.scores, predictions.c.text_sha256
                    ).where(predictions.c.id == prediction_id)
                ).one_or_none()
                if row is None:
                    raise PredictionNotFoundError(str(prediction_id))
                if label not in row.scores:
                    raise InvalidFeedbackError(
                        "label",
                        "invalid_label",
                        f"Label must be one of {sorted(row.scores)} for this prediction.",
                    )
                if text is not None and text_sha256(text) != row.text_sha256:
                    raise InvalidFeedbackError(
                        "text",
                        "text_mismatch",
                        "Text does not match the text of this prediction.",
                    )
                feedback_id = connection.execute(
                    feedback.insert()
                    .values(prediction_id=prediction_id, label=label, text=text)
                    .returning(feedback.c.id)
                ).scalar_one()
        except IntegrityError as exc:
            raise FeedbackExistsError(str(prediction_id)) from exc
        except OperationalError as exc:
            self._mark_unavailable(exc)
            raise StoreUnavailableError("The database is unavailable.") from exc
        self._mark_available()
        return FeedbackResult(
            id=int(feedback_id),
            prediction_id=prediction_id,
            label=label,
            predicted_label=row.label,
            text_stored=text is not None,
        )

    def close(self) -> None:
        self.engine.dispose()
