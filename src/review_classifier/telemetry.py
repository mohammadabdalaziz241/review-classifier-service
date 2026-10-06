"""Prometheus metrics: what the service is doing, how fast, and what it predicts.

Each app gets its own registry, so several apps in one process (tests) do not
share counters. Metrics are served either on the API at ``/metrics`` or, when
``METRICS_PORT`` is set, on a separate port that is not published to the
internet (the deployment does this).

Metrics are per process: with WORKERS > 1 every worker reports its own, so run
one worker per container, as the image does, when scraping them.

Naming: ``http_*`` for traffic, ``model_*`` for inference and predictions,
``db_*`` for the database, ``feedback_total`` for labelled feedback. Labels are
bounded (route templates, not paths; model labels, not texts), so a client
cannot create unbounded series.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import prometheus_client
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

if TYPE_CHECKING:
    from wsgiref.simple_server import WSGIServer

    from .predictors import Prediction, Predictor
    from .store import PredictionStore

# The *_created series double the output and no dashboard here uses them.
prometheus_client.disable_created_metrics()

__all__ = ["CONTENT_TYPE_LATEST", "Telemetry"]

# Seconds. Fine-grained around the 10-500 ms where single predictions land on CPU.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0,
)  # fmt: skip
QUEUE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
CHARS_BUCKETS = (25, 50, 100, 200, 400, 800, 1600, 3200)
CONFIDENCE_BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)

# Requests that do not match a route share one label value.
UNMATCHED_ROUTE = "unmatched"


def _inference_threads() -> int | None:
    """Intra-op threads PyTorch uses for one forward pass, if torch is loaded."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return int(torch.get_num_threads())
    except Exception:  # pragma: no cover - defensive: never fail a scrape
        return None


class _StateCollector(Collector):
    """Gauges read at scrape time from the app's current state."""

    def __init__(
        self,
        predictor: Callable[[], Predictor | None],
        store: Callable[[], PredictionStore | None],
        max_concurrent: int | None,
    ) -> None:
        self._predictor = predictor
        self._store = store
        self._max_concurrent = max_concurrent

    def collect(self) -> Iterable[GaugeMetricFamily]:
        predictor = self._predictor()
        yield GaugeMetricFamily(
            "model_ready", "1 when the model is loaded and requests can be served.",
            value=0 if predictor is None else 1,
        )  # fmt: skip
        info = GaugeMetricFamily(
            "model_info",
            "The loaded model; the value is always 1.",
            labels=["backend", "model_id", "version", "task", "device"],
        )
        if predictor is not None:
            i = predictor.info
            info.add_metric([i.backend, i.model_id, i.version, i.task, i.device], 1)
        yield info

        threads = _inference_threads()
        if threads is not None:
            yield GaugeMetricFamily(
                "model_inference_threads", "PyTorch intra-op threads per forward pass.",
                value=threads,
            )  # fmt: skip
        yield GaugeMetricFamily(
            "model_max_concurrent_inferences",
            "Forward passes allowed at once (MAX_CONCURRENT_INFERENCES); 0 means unlimited.",
            value=self._max_concurrent or 0,
        )

        store = self._store()
        if store is not None:  # absent when the service runs without a database
            yield GaugeMetricFamily(
                "db_up", "1 when the database accepts writes, 0 while it is unavailable.",
                value=1 if store.available else 0,
            )  # fmt: skip


class Telemetry:
    """The service's metrics, registered on a registry of its own."""

    def __init__(
        self,
        predictor: Callable[[], Predictor | None] = lambda: None,
        store: Callable[[], PredictionStore | None] = lambda: None,
        max_concurrent_inferences: int | None = None,
    ) -> None:
        self.registry = registry = CollectorRegistry()
        ProcessCollector(registry=registry)  # memory, CPU, open files (Linux)
        PlatformCollector(registry=registry)  # Python version
        registry.register(_StateCollector(predictor, store, max_concurrent_inferences))

        self.http_requests = Counter(
            "http_requests",
            "HTTP requests handled, by route template and status code.",
            ["method", "route", "status"],
            registry=registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "Time from receiving a request to sending its response.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self.http_in_progress = Gauge(
            "http_requests_in_progress", "Requests being handled right now.", registry=registry
        )
        self.inference_duration = Histogram(
            "model_inference_duration_seconds",
            "Model time for one request (all its texts), excluding the wait for a slot.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self.inference_queue = Histogram(
            "model_inference_queue_seconds",
            "Time a request waited for a free inference slot (MAX_CONCURRENT_INFERENCES).",
            buckets=QUEUE_BUCKETS,
            registry=registry,
        )
        self.batch_texts = Histogram(
            "model_batch_texts",
            "Texts per inference call.",
            buckets=BATCH_BUCKETS,
            registry=registry,
        )
        self.input_chars = Histogram(
            "model_input_chars",
            "Length of each classified text, in characters.",
            buckets=CHARS_BUCKETS,
            registry=registry,
        )
        self.predictions = Counter(
            "model_predictions",
            "Predictions served, by predicted label. A shift in the mix can signal drift.",
            ["label"],
            registry=registry,
        )
        self.confidence = Histogram(
            "model_prediction_confidence",
            "Probability the model gave its predicted label.",
            buckets=CONFIDENCE_BUCKETS,
            registry=registry,
        )
        self.records = Counter(
            "db_prediction_records",
            "Predictions by whether they were stored: recorded, not_recorded "
            "(database unavailable) or disabled (no database configured).",
            ["outcome"],
            registry=registry,
        )
        self.feedback = Counter(
            "feedback",
            "Labelled feedback received, by whether the model's prediction was correct.",
            ["model_correct"],
            registry=registry,
        )
        # Start these series at 0, so increase() counts the first event too.
        for outcome in ("recorded", "not_recorded", "disabled"):
            self.records.labels(outcome)
        for correct in ("true", "false"):
            self.feedback.labels(correct)
        self._server: WSGIServer | None = None
        self._lock = threading.Lock()

    # -- recording ------------------------------------------------------------------

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method, route, str(status)).inc()
        self.http_duration.labels(method, route).observe(seconds)

    def observe_inference(
        self,
        endpoint: str,
        texts: Sequence[str],
        predictions: Sequence[Prediction],
        seconds: float,
    ) -> None:
        self.inference_duration.labels(endpoint).observe(seconds)
        self.batch_texts.observe(len(texts))
        for text in texts:
            self.input_chars.observe(len(text))
        for prediction in predictions:
            self.predictions.labels(prediction.label).inc()
            self.confidence.observe(prediction.score)

    def observe_records(self, count: int, outcome: str) -> None:
        self.records.labels(outcome).inc(count)

    def observe_feedback(self, model_was_correct: bool) -> None:
        self.feedback.labels(str(model_was_correct).lower()).inc()

    # -- exposition -----------------------------------------------------------------

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def serve(self, port: int, addr: str = "0.0.0.0") -> None:
        """Serve the metrics on their own port, in a background thread."""
        with self._lock:
            if self._server is None:
                self._server, _ = prometheus_client.start_http_server(
                    port, addr=addr, registry=self.registry
                )

    def stop(self) -> None:
        with self._lock:
            server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

    @property
    def serving_port(self) -> int | None:
        server = self._server
        return None if server is None else int(server.server_address[1])
