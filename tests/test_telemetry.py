"""Prometheus metrics: names, labels and values the dashboards and alerts rely on."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings
from review_classifier.predictors import DummyPredictor, Prediction

from .fakes import FailingPredictor
from .live_server import free_port, live_server
from .scrape import parse, value


def scrape(client: TestClient) -> dict:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return parse(response.text)


def test_model_state_is_exported(client):
    samples = scrape(client)
    assert value(samples, "model_ready") == 1
    assert (
        value(
            samples,
            "model_info",
            backend="dummy",
            model_id="keyword-baseline",
            version="dummy-1",
            task="sentiment",
            device="cpu",
        )
        == 1
    )
    assert value(samples, "model_max_concurrent_inferences") == 1
    # No database configured: no db_up series rather than a misleading 0.
    assert not [key for key in samples if key[0] == "db_up"]


def test_model_not_ready_before_it_loads(settings):
    client = TestClient(create_app(settings))  # lifespan not run
    assert value(scrape(client), "model_ready") == 0


def test_requests_are_counted_by_route_template_and_status(client):
    client.post("/v1/predict", json={"text": "I love it"})
    client.post("/v1/predict", json={"text": ""})  # 422
    client.get("/v1/predictions/123")  # no such route
    client.get("/metrics")  # scrapes are not counted
    samples = scrape(client)

    def requests(method: str, route: str, status: str) -> float | None:
        return value(samples, "http_requests_total", method=method, route=route, status=status)

    assert requests("POST", "/v1/predict", "200") == 1
    assert requests("POST", "/v1/predict", "422") == 1
    assert requests("GET", "unmatched", "404") == 1
    assert not [k for k in samples if dict(k[1]).get("route") == "/metrics"]
    assert (
        value(samples, "http_request_duration_seconds_count", method="POST", route="/v1/predict")
        == 2
    )
    assert value(samples, "http_requests_in_progress") == 1  # this scrape


def test_unhandled_errors_are_counted_as_500(settings):
    app = create_app(settings, predictor=FailingPredictor())
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/v1/predict", json={"text": "hi"}).status_code == 500
        samples = scrape(client)
    assert (
        value(samples, "http_requests_total", method="POST", route="/v1/predict", status="500") == 1
    )


def test_inference_and_predictions_are_measured(client):
    client.post("/v1/predict", json={"text": "I love it"})
    client.post("/v1/predict/batch", json={"texts": ["awful", "great", "superb"]})
    samples = scrape(client)
    assert value(samples, "model_inference_duration_seconds_count", endpoint="predict") == 1
    assert value(samples, "model_inference_duration_seconds_count", endpoint="predict_batch") == 1
    assert value(samples, "model_batch_texts_count") == 2
    assert value(samples, "model_batch_texts_sum") == 4
    assert value(samples, "model_input_chars_count") == 4
    assert value(samples, "model_input_chars_sum") == len("I love it") + 5 + 5 + 6
    assert value(samples, "model_inference_queue_seconds_count") == 2
    labels = {
        dict(k[1])["label"]: v for k, v in samples.items() if k[0] == "model_predictions_total"
    }
    assert sum(labels.values()) == 4
    assert labels.get("negative") == 1
    assert value(samples, "model_prediction_confidence_count") == 4
    assert value(samples, "db_prediction_records_total", outcome="disabled") == 4


def test_process_metrics_are_included(client):
    samples = scrape(client)
    names = {name for name, _ in samples}
    assert "python_info" in names
    if "process_resident_memory_bytes" in names:  # Linux only
        assert value(samples, "process_resident_memory_bytes") > 0


def test_metrics_can_move_to_their_own_port(settings):
    port = free_port()
    app = create_app(Settings(model_backend="dummy", metrics_port=port))
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404  # not on the API port
        client.post("/v1/predict", json={"text": "great"})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            samples = parse(r.read().decode())
        assert (
            value(samples, "http_requests_total", method="POST", route="/v1/predict", status="200")
            == 1
        )
    # Stopped with the app.
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2)


# ---- inference concurrency ------------------------------------------------------------


class SlowPredictor(DummyPredictor):
    """Takes a while per call and records how many calls overlapped."""

    def __init__(self, seconds: float = 0.2) -> None:
        super().__init__()
        self._seconds = seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self._seconds)
            return super().predict(texts)
        finally:
            with self._lock:
                self.active -= 1


def _post_concurrently(url: str, n: int) -> list[int]:
    def post(_: int) -> int:
        request = urllib.request.Request(
            f"{url}/v1/predict",
            data=b'{"text": "great"}',
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.status

    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(post, range(n)))


@pytest.mark.parametrize(("limit", "expected_max"), [(1, 1), (2, 2), (None, 4)])
def test_inference_concurrency_limit(limit, expected_max):
    predictor = SlowPredictor()
    # Without batching: this measures the plain concurrency limit.
    settings = Settings(
        model_backend="dummy", max_concurrent_inferences=limit, batch_requests=False
    )
    app = create_app(settings, predictor=predictor)
    with live_server(app) as url:
        assert _post_concurrently(url, 4) == [200] * 4
        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
            samples = parse(r.read().decode())
    assert predictor.max_active == expected_max
    if limit == 1:
        # Three requests queued behind the first; the last waited for three passes.
        assert value(samples, "model_inference_queue_seconds_sum") >= 0.2 * (1 + 2 + 3) * 0.9
