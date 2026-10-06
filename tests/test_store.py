"""Database layer, migrations and the API's use of them.

Every test runs against SQLite and, when TEST_DATABASE_URL points at a
PostgreSQL database, against PostgreSQL too (CI does both). Each test starts
from an empty database migrated to the latest schema.
"""

from __future__ import annotations

import logging
import os
import uuid

import pytest

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from review_classifier import store as store_module  # noqa: E402
from review_classifier.api import create_app  # noqa: E402
from review_classifier.config import Settings  # noqa: E402
from review_classifier.db import downgrade, upgrade  # noqa: E402
from review_classifier.store import (  # noqa: E402
    FeedbackExistsError,
    InvalidFeedbackError,
    PredictionNotFoundError,
    PredictionRecord,
    PredictionStore,
    SchemaMismatchError,
    StoreUnavailableError,
    create_db_engine,
    current_revision,
    head_revision,
    normalize_database_url,
    redact,
    text_sha256,
)

from .scrape import parse, value  # noqa: E402

UNREACHABLE = "postgresql://user:secret@127.0.0.1:1/nowhere"


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def database_url(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'test.db'}"
    else:
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL is not set")
        pytest.importorskip("psycopg")
    engine = create_db_engine(url)
    downgrade(engine, "base")
    upgrade(engine)
    engine.dispose()
    yield url
    engine = create_db_engine(url)
    downgrade(engine, "base")
    engine.dispose()


@pytest.fixture
def store(database_url):
    s = PredictionStore(database_url)
    yield s
    s.close()


def _record(**overrides) -> PredictionRecord:
    values = dict(
        id=uuid.uuid4(),
        request_id="req-1",
        endpoint="predict",
        batch_index=0,
        batch_size=1,
        model_id="someone/model",
        model_version="0123456789abcdef0123456789abcdef01234567",
        task="sarcasm",
        label="not_sarcastic",
        score=0.8,
        scores={"not_sarcastic": 0.8, "sarcastic": 0.2},
        text_sha256=text_sha256("Oh great, another delay"),
        text_chars=len("Oh great, another delay"),
        inference_ms=12.5,
    )
    values.update(overrides)
    return PredictionRecord(**values)


def _rows(url: str, table: str) -> list[dict]:
    # Typed reads, so UUID and JSON columns decode the same on SQLite and PostgreSQL.
    engine = create_db_engine(url)
    try:
        with engine.connect() as connection:
            result = connection.execute(sa.select(store_module.metadata.tables[table]))
            return [dict(row._mapping) for row in result]
    finally:
        engine.dispose()


# ---- helpers -----------------------------------------------------------------------


def test_postgres_urls_use_psycopg_and_passwords_are_redacted():
    assert normalize_database_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_database_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"
    assert "secret" not in redact(UNREACHABLE)


def test_settings_never_show_the_database_password():
    settings = Settings.from_env({"DATABASE_URL": UNREACHABLE})
    assert settings.database_url == UNREACHABLE
    assert "secret" not in repr(settings)


# ---- migrations ----------------------------------------------------------------------


def test_migrations_create_and_remove_the_schema(database_url):
    engine = create_db_engine(database_url)
    try:
        assert current_revision(engine) == head_revision()
        assert {"predictions", "feedback"} <= set(sa.inspect(engine).get_table_names())
        downgrade(engine, "base")
        assert current_revision(engine) is None
        assert not {"predictions", "feedback"} & set(sa.inspect(engine).get_table_names())
        upgrade(engine)  # and back again
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


def test_schema_check_rejects_an_unmigrated_database(tmp_path):
    s = PredictionStore(f"sqlite:///{tmp_path / 'empty.db'}")
    with pytest.raises(SchemaMismatchError, match="db upgrade"):
        s.check_schema()


def test_schema_check_accepts_a_migrated_database(store):
    store.check_schema()
    assert store.status == "ok"


# ---- store ---------------------------------------------------------------------------


def test_records_predictions_without_text(store, database_url):
    rows = [_record(batch_index=i, batch_size=2) for i in range(2)]
    assert store.record(rows) is True
    stored = _rows(database_url, "predictions")
    assert {r["id"] for r in stored} == {r.id for r in rows}
    first = stored[0]
    assert first["model_version"] == rows[0].model_version
    assert first["scores"] == {"not_sarcastic": 0.8, "sarcastic": 0.2}
    assert first["created_at"] is not None
    assert "text" not in first


def test_feedback_is_stored_and_compared_with_the_prediction(store, database_url):
    row = _record()
    store.record([row])
    result = store.add_feedback(row.id, "sarcastic")
    assert result.predicted_label == "not_sarcastic"
    assert result.model_was_correct is False
    assert result.text_stored is False
    [stored] = _rows(database_url, "feedback")
    assert stored["prediction_id"] == row.id
    assert stored["label"] == "sarcastic"
    assert stored["text"] is None


def test_feedback_can_keep_the_exact_text(store, database_url):
    row = _record()
    store.record([row])
    result = store.add_feedback(row.id, "sarcastic", text="Oh great, another delay")
    assert result.text_stored is True
    assert _rows(database_url, "feedback")[0]["text"] == "Oh great, another delay"


def test_feedback_rejects_text_that_was_not_classified(store):
    row = _record()
    store.record([row])
    with pytest.raises(InvalidFeedbackError) as error:
        store.add_feedback(row.id, "sarcastic", text="Oh great, another delay!")
    assert error.value.kind == "text_mismatch"


def test_feedback_rejects_labels_the_model_does_not_have(store):
    row = _record()
    store.record([row])
    with pytest.raises(InvalidFeedbackError) as error:
        store.add_feedback(row.id, "positive")
    assert error.value.kind == "invalid_label"


def test_feedback_for_an_unknown_prediction(store):
    with pytest.raises(PredictionNotFoundError):
        store.add_feedback(uuid.uuid4(), "sarcastic")


def test_feedback_is_recorded_once_per_prediction(store, database_url):
    row = _record()
    store.record([row])
    store.add_feedback(row.id, "sarcastic")
    with pytest.raises(FeedbackExistsError):
        store.add_feedback(row.id, "not_sarcastic")
    assert len(_rows(database_url, "feedback")) == 1


# ---- outages ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FlakyEngine:
    """Wraps an engine; raises OperationalError on begin() while ``down`` is set."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.down = False
        self.attempts = 0

    def begin(self):
        self.attempts += 1
        if self.down:
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))
        return self.engine.begin()

    def __getattr__(self, name):
        return getattr(self.engine, name)


def test_outage_pauses_recording_then_recovers(database_url, caplog):
    clock = Clock()
    s = PredictionStore(database_url, retry_after=30, clock=clock)
    flaky = FlakyEngine(s.engine)
    s.engine = flaky
    try:
        flaky.down = True
        with caplog.at_level(logging.ERROR, logger="review_classifier"):
            assert s.record([_record()]) is False
        assert s.status == "unavailable"
        assert "database unavailable" in caplog.text

        # Within the retry window the database is not even tried.
        attempts = flaky.attempts
        clock.now += 10
        assert s.record([_record()]) is False
        with pytest.raises(StoreUnavailableError):
            s.add_feedback(uuid.uuid4(), "sarcastic")
        assert flaky.attempts == attempts

        # After the window it is tried again, and recording resumes.
        flaky.down = False
        clock.now += 30
        assert s.record([_record()]) is True
        assert s.status == "ok"
    finally:
        flaky.engine.dispose()


def test_unreachable_database_at_startup_is_tolerated():
    pytest.importorskip("psycopg")
    s = PredictionStore(UNREACHABLE)
    s.check_schema()  # does not raise
    assert s.status == "unavailable"
    assert s.record([_record()]) is False
    s.close()


# ---- through the API ----------------------------------------------------------------


@pytest.fixture
def api(database_url):
    settings = Settings(model_backend="dummy", database_url=database_url, max_text_chars=200)
    with TestClient(create_app(settings)) as client:
        yield client


def test_ready_reports_the_database(api):
    assert api.get("/ready").json() == {"status": "ready", "database": "ok"}


def test_predictions_are_recorded_with_their_model_version(api, database_url):
    text = "I love it, the best purchase"
    response = api.post("/v1/predict", json={"text": text}, headers={"X-Request-ID": "abc-1"})
    body = response.json()
    assert body["recorded"] is True
    [row] = _rows(database_url, "predictions")
    assert str(row["id"]) == body["prediction"]["id"]
    assert row["request_id"] == "abc-1"
    assert row["endpoint"] == "predict"
    assert row["model_version"] == body["model"]["version"]
    assert row["label"] == body["prediction"]["label"]
    assert row["text_sha256"] == text_sha256(text)
    assert row["text_chars"] == len(text)


def test_batch_predictions_are_recorded_in_order(api, database_url):
    texts = ["love it", "awful", "a box"]
    body = api.post("/v1/predict/batch", json={"texts": texts}).json()
    assert body["recorded"] is True
    rows = {str(r["id"]): r for r in _rows(database_url, "predictions")}
    for i, prediction in enumerate(body["predictions"]):
        row = rows[prediction["id"]]
        assert (row["batch_index"], row["batch_size"]) == (i, 3)
        assert row["endpoint"] == "predict_batch"
        assert row["text_sha256"] == text_sha256(texts[i])


def test_review_text_is_never_stored_with_predictions(api, database_url):
    secret = "Ring me on 07700 900123, Jane Doe"
    api.post("/v1/predict", json={"text": secret})
    for row in _rows(database_url, "predictions"):
        assert all(secret not in str(value) for value in row.values())


def test_feedback_round_trip(api):
    text = "Oh brilliant, it broke again"
    prediction = api.post("/v1/predict", json={"text": text}).json()["prediction"]
    response = api.post(
        "/v1/feedback",
        json={"prediction_id": prediction["id"], "label": "negative", "text": text},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["prediction_id"] == prediction["id"]
    assert body["predicted_label"] == prediction["label"]
    assert body["model_was_correct"] == (prediction["label"] == "negative")
    assert body["text_stored"] is True

    again = api.post("/v1/feedback", json={"prediction_id": prediction["id"], "label": "positive"})
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "feedback_exists"


def test_recording_and_feedback_are_measured(api):
    first = api.post("/v1/predict", json={"text": "I love it"}).json()["prediction"]
    second = api.post("/v1/predict", json={"text": "Terrible, broke at once"}).json()["prediction"]
    api.post("/v1/feedback", json={"prediction_id": first["id"], "label": first["label"]})
    wrong = "negative" if second["label"] != "negative" else "positive"
    api.post("/v1/feedback", json={"prediction_id": second["id"], "label": wrong})
    samples = parse(api.get("/metrics").text)
    assert value(samples, "db_up") == 1
    assert value(samples, "db_prediction_records_total", outcome="recorded") == 2
    assert value(samples, "feedback_total", model_correct="true") == 1
    assert value(samples, "feedback_total", model_correct="false") == 1


@pytest.mark.parametrize(
    ("payload", "status", "code", "detail_type"),
    [
        ({"label": "sarcastic"}, 422, "invalid_request", "invalid_label"),
        (
            {"label": "negative", "text": "not what was sent"},
            422,
            "invalid_request",
            "text_mismatch",
        ),
        ({"label": "negative", "text": "x" * 201}, 422, "invalid_request", "text_too_long"),
        ({"label": ""}, 422, "invalid_request", "string_too_short"),
    ],
)
def test_feedback_validation(api, payload, status, code, detail_type):
    prediction_id = api.post("/v1/predict", json={"text": "great"}).json()["prediction"]["id"]
    response = api.post("/v1/feedback", json={"prediction_id": prediction_id, **payload})
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["details"][0]["type"] == detail_type


def test_feedback_for_an_unknown_prediction_via_api(api):
    response = api.post("/v1/feedback", json={"prediction_id": str(uuid.uuid4()), "label": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "prediction_not_found"


def test_feedback_rejects_malformed_ids(api):
    response = api.post("/v1/feedback", json={"prediction_id": "not-a-uuid", "label": "negative"})
    assert response.status_code == 422


def test_without_a_database_predictions_are_served_but_not_recorded():
    with TestClient(create_app(Settings(model_backend="dummy"))) as client:
        body = client.post("/v1/predict", json={"text": "great"}).json()
        assert body["recorded"] is False
        uuid.UUID(body["prediction"]["id"])  # ids are still issued
        response = client.post(
            "/v1/feedback", json={"prediction_id": body["prediction"]["id"], "label": "positive"}
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "feedback_unavailable"


def test_service_runs_while_the_database_is_down():
    pytest.importorskip("psycopg")
    settings = Settings(model_backend="dummy", database_url=UNREACHABLE)
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").json() == {"status": "ready", "database": "unavailable"}
        response = client.post("/v1/predict", json={"text": "great"})
        assert response.status_code == 200
        assert response.json()["recorded"] is False
        samples = parse(client.get("/metrics").text)
        assert value(samples, "db_up") == 0
        assert value(samples, "db_prediction_records_total", outcome="not_recorded") == 1
        feedback = client.post(
            "/v1/feedback",
            json={"prediction_id": response.json()["prediction"]["id"], "label": "positive"},
        )
    assert feedback.status_code == 503
    assert feedback.json()["error"]["code"] == "database_unavailable"


def test_service_refuses_to_start_on_an_unmigrated_database(tmp_path):
    settings = Settings(model_backend="dummy", database_url=f"sqlite:///{tmp_path / 'new.db'}")
    with pytest.raises(SchemaMismatchError), TestClient(create_app(settings)):
        pass
