"""API integration tests, run against the dummy backend and injected fakes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import REQUEST_ID_HEADER, create_app
from review_classifier.config import Settings

from .fakes import FailingPredictor, RecordingPredictor


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    return body


# ---- ops endpoints -------------------------------------------------------------


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_once_model_is_loaded(client):
    assert client.get("/ready").json() == {"status": "ready", "database": "disabled"}


def test_not_ready_before_model_loads(settings):
    # Without the context manager the lifespan never runs, so no model is loaded.
    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 200
    assert_error(client.get("/ready"), 503, "model_not_ready")
    assert_error(client.post("/v1/predict", json={"text": "great"}), 503, "model_not_ready")


def test_model_info(client, settings):
    body = client.get("/v1/model").json()
    assert body["backend"] == "dummy"
    assert body["version"] == "dummy-1"
    assert body["origin"] == "builtin"
    assert body["requested_revision"] is None
    assert body["preprocessing"] == {
        "normalize_unicode": True,
        "normalize_whitespace": True,
        "replace_urls": True,
        "replace_mentions": True,
        # The dummy backend has no serving_config.json, so defaults apply.
        "source": dict.fromkeys(
            ["normalize_unicode", "normalize_whitespace", "replace_urls", "replace_mentions"],
            "default",
        ),
    }
    assert body["labels"] == ["negative", "neutral", "positive"]
    assert body["limits"] == {
        "max_text_chars": settings.max_text_chars,
        "max_batch_size": settings.max_batch_size,
    }


# ---- single prediction -------------------------------------------------------


def test_predict_returns_label_scores_and_model_version(client):
    response = client.post("/v1/predict", json={"text": "Great product, love it"})
    assert response.status_code == 200
    body = response.json()

    assert body["model"] == {"id": "keyword-baseline", "version": "dummy-1"}
    assert body["prediction"]["label"] == "positive"
    scores = body["prediction"]["scores"]
    assert set(scores) == {"negative", "neutral", "positive"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert body["prediction"]["score"] == scores["positive"]
    assert body["inference_ms"] >= 0
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_text_is_normalised_before_reaching_the_model(settings):
    fake = RecordingPredictor()
    with TestClient(create_app(settings, predictor=fake)) as client:
        client.post(
            "/v1/predict", json={"text": "  @shop  \uff27\uff32\uff25\uff21\uff34  https://x.co/a "}
        )
    assert fake.seen == ["@user GREAT http"]


def test_preprocessing_follows_configuration():
    settings = Settings(
        model_backend="dummy", replace_urls=False, replace_mentions=False, normalize_unicode=False
    )
    fake = RecordingPredictor()
    with TestClient(create_app(settings, predictor=fake)) as client:
        client.post("/v1/predict", json={"text": "  @shop  \uff27  https://x.co/a "})
        reported = client.get("/v1/model").json()["preprocessing"]
    assert fake.seen == ["@shop \uff27 https://x.co/a"]
    assert reported["source"] == {
        "normalize_unicode": "env",
        "normalize_whitespace": "default",
        "replace_urls": "env",
        "replace_mentions": "env",
    }


def test_raw_text_mode_passes_text_through_unchanged():
    settings = Settings(
        model_backend="dummy",
        normalize_unicode=False,
        normalize_whitespace=False,
        replace_urls=False,
        replace_mentions=False,
    )
    fake = RecordingPredictor()
    text = "  Yeah nah,  brilliant\n@mate https://x.co \uff27 "
    with TestClient(create_app(settings, predictor=fake)) as client:
        assert client.post("/v1/predict", json={"text": text}).status_code == 200
        # Whitespace-only text is still rejected, whatever the preprocessing.
        assert client.post("/v1/predict", json={"text": " \n "}).status_code == 422
    assert fake.seen == [text]


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "\x00"])
def test_predict_rejects_blank_text(client, text):
    body = assert_error(client.post("/v1/predict", json={"text": text}), 422, "invalid_request")
    assert body["error"]["details"] == [
        {"loc": ["body", "text"], "msg": "Text is empty after normalisation.", "type": "text_empty"}
    ]


def test_predict_rejects_text_over_the_limit(client, settings):
    text = "a" * (settings.max_text_chars + 1)
    body = assert_error(client.post("/v1/predict", json={"text": text}), 422, "invalid_request")
    assert body["error"]["details"][0]["type"] == "text_too_long"


def test_predict_accepts_text_at_the_limit(client, settings):
    response = client.post("/v1/predict", json={"text": "a" * settings.max_text_chars})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": 42}, {"text": None}, {"text": "ok", "unexpected": 1}, ["not", "an", "object"]],
)
def test_predict_rejects_malformed_bodies(client, payload):
    assert_error(client.post("/v1/predict", json=payload), 422, "invalid_request")


def test_predict_rejects_invalid_json(client):
    response = client.post(
        "/v1/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert_error(response, 422, "invalid_request")


def test_validation_errors_do_not_echo_input(client):
    body = client.post("/v1/predict", json={"text": 12345}).json()
    for detail in body["error"]["details"]:
        assert set(detail) == {"loc", "msg", "type"}


# ---- batch prediction ----------------------------------------------------------


def test_batch_preserves_order(client):
    texts = ["love it", "awful", "a box", "best purchase"]
    response = client.post("/v1/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    labels = [p["label"] for p in response.json()["predictions"]]
    assert labels == ["positive", "negative", "neutral", "positive"]


def test_batch_rejects_empty_list(client):
    assert_error(client.post("/v1/predict/batch", json={"texts": []}), 422, "invalid_request")


def test_batch_rejects_more_than_the_limit(client, settings):
    texts = ["good"] * (settings.max_batch_size + 1)
    assert_error(client.post("/v1/predict/batch", json={"texts": texts}), 422, "batch_too_large")


def test_batch_reports_every_invalid_item_by_index(client, settings):
    texts = ["fine", "  ", "a" * (settings.max_text_chars + 1), "good"]
    body = assert_error(
        client.post("/v1/predict/batch", json={"texts": texts}), 422, "invalid_request"
    )
    found = [(d["loc"], d["type"]) for d in body["error"]["details"]]
    assert found == [
        (["body", "texts", 1], "text_empty"),
        (["body", "texts", 2], "text_too_long"),
    ]


def test_batch_rejects_non_string_items(client):
    response = client.post("/v1/predict/batch", json={"texts": ["ok", 3]})
    assert_error(response, 422, "invalid_request")


# ---- request IDs and errors ------------------------------------------------------


def test_request_id_is_generated_when_absent(client):
    response = client.get("/health")
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_valid_client_request_id_is_propagated(client):
    response = client.post(
        "/v1/predict", json={"text": "good"}, headers={REQUEST_ID_HEADER: "client-req_42.a"}
    )
    assert response.headers[REQUEST_ID_HEADER] == "client-req_42.a"
    assert response.json()["request_id"] == "client-req_42.a"


@pytest.mark.parametrize("bad_id", ["has spaces", "x" * 129, "semi;colon", "slash/inside"])
def test_unsafe_client_request_id_is_replaced(client, bad_id):
    response = client.get("/health", headers={REQUEST_ID_HEADER: bad_id})
    assert response.headers[REQUEST_ID_HEADER] != bad_id
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_unknown_route_uses_error_envelope(client):
    assert_error(client.get("/v1/nope"), 404, "not_found")


def test_wrong_method_uses_error_envelope(client):
    assert_error(client.get("/v1/predict"), 405, "method_not_allowed")


def test_inference_failure_returns_500_without_leaking_details(settings, caplog):
    with TestClient(create_app(settings, predictor=FailingPredictor())) as client:
        response = client.post("/v1/predict", json={"text": "good"})
    body = assert_error(response, 500, "internal_error")
    assert "secret" not in response.text
    assert body["error"]["details"] == []
    # The traceback is logged server-side with the request ID for debugging.
    assert any(body["request_id"] in record.getMessage() for record in caplog.records)


def test_review_text_is_not_logged(client, caplog):
    caplog.set_level("INFO", logger="review_classifier")
    client.post("/v1/predict", json={"text": "my private review mentioning Jane Doe"})
    assert caplog.records, "expected an access log line"
    assert all("Jane Doe" not in record.getMessage() for record in caplog.records)
