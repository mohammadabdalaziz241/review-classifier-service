"""Dynamic batching: unit tests of the batcher, then through the API."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.batching import DynamicBatcher
from review_classifier.config import Settings
from review_classifier.predictors import DummyPredictor, Prediction

from .live_server import live_server
from .scrape import parse, value


class EchoModel:
    """Labels each text with itself, so tests can check every answer reaches its caller.

    With ``gated=True`` every pass blocks until the test calls ``release()``, so tests
    control exactly which requests are queued while a pass runs (no timing guesses).
    """

    def __init__(self, gated: bool = False, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[list[str]] = []
        self.active = 0
        self.max_active = 0
        self.entered = threading.Event()
        self._gate = threading.Event()
        if not gated:
            self._gate.set()
        self._lock = threading.Lock()

    def release(self) -> None:
        self._gate.set()

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        with self._lock:
            self.calls.append(list(texts))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            self._gate.wait(10)
            if self.fail_on in texts:
                raise RuntimeError("model failure")
            return [Prediction(label=t, score=1.0, scores={t: 1.0}) for t in texts]
        finally:
            with self._lock:
                self.active -= 1


def run(coroutine):
    return asyncio.run(coroutine)


async def _with_batcher(model, body, **kwargs):
    batcher = DynamicBatcher(model.predict, **kwargs)
    await batcher.start()
    try:
        return await body(batcher)
    finally:
        model.release()
        await batcher.stop()


async def _first_pass_running(batcher, model, text: str = "first"):
    """Start a pass and return once the model is inside it."""
    task = asyncio.create_task(batcher.submit([text]))
    assert await asyncio.to_thread(model.entered.wait, 10)
    return task


async def _queue(batcher, *jobs: list[str]):
    """Submit jobs and let them reach the queue, in order."""
    tasks = []
    for texts in jobs:
        tasks.append(asyncio.create_task(batcher.submit(texts)))
        await asyncio.sleep(0)
    return tasks


def labels(result) -> list[str]:
    return [p.label for p in result.predictions]


def test_a_lone_request_runs_at_once_and_alone():
    model = EchoModel()

    async def body(batcher):
        started = time.perf_counter()
        result = await batcher.submit(["a"])
        return result, time.perf_counter() - started

    result, elapsed = run(_with_batcher(model, body))
    assert labels(result) == ["a"]
    assert result.batch_texts == 1
    assert elapsed < 1
    assert model.calls == [["a"]]


def test_requests_waiting_for_the_model_share_the_next_pass():
    model = EchoModel(gated=True)

    async def body(batcher):
        first = await _first_pass_running(batcher, model)
        rest = await _queue(batcher, *[[f"t{i}"] for i in range(5)])
        model.release()
        return await first, await asyncio.gather(*rest)

    first, rest = run(_with_batcher(model, body, max_texts=16))
    assert model.calls == [["first"], ["t0", "t1", "t2", "t3", "t4"]]
    assert labels(first) == ["first"] and first.batch_texts == 1
    for i, result in enumerate(rest):
        assert labels(result) == [f"t{i}"]  # each caller gets its own answer
        assert result.batch_texts == 5


def test_batches_are_capped_and_keep_arrival_order():
    model = EchoModel(gated=True)

    async def body(batcher):
        first = await _first_pass_running(batcher, model, "x")
        jobs = await _queue(batcher, *[[f"{i}a", f"{i}b"] for i in range(5)])
        model.release()
        await first
        return await asyncio.gather(*jobs)

    results = run(_with_batcher(model, body, max_texts=4))
    # Requests are never split: 2 + 2 texts per pass, then the remainder.
    assert model.calls == [["x"], ["0a", "0b", "1a", "1b"], ["2a", "2b", "3a", "3b"], ["4a", "4b"]]
    assert [labels(r) for r in results] == [[f"{i}a", f"{i}b"] for i in range(5)]


def test_a_request_larger_than_the_cap_runs_alone():
    model = EchoModel()
    texts = [str(i) for i in range(10)]
    result = run(_with_batcher(model, lambda b: b.submit(texts), max_texts=4))
    assert labels(result) == texts
    assert model.calls == [texts]


def test_a_failed_pass_fails_its_requests_only():
    model = EchoModel(gated=True, fail_on="bad")

    async def body(batcher):
        first = await _first_pass_running(batcher, model, "ok")
        shared = await _queue(batcher, ["bad"], ["victim"])
        model.release()
        outcomes = await asyncio.gather(first, *shared, return_exceptions=True)
        after = await batcher.submit(["later"])  # the batcher keeps working
        return outcomes, after

    (first, bad, victim), after = run(_with_batcher(model, body))
    assert labels(first) == ["ok"]
    assert isinstance(bad, RuntimeError) and isinstance(victim, RuntimeError)
    assert bad is not victim  # separate exceptions keep tracebacks apart
    assert isinstance(bad.__cause__, RuntimeError)
    assert labels(after) == ["later"]


def test_a_failing_callback_does_not_stop_the_batcher():
    model = EchoModel()

    def broken_callback(texts: int, seconds: float) -> None:
        raise ValueError("metrics broke")

    async def body(batcher):
        return await batcher.submit(["a"]), await batcher.submit(["b"])

    first, second = run(_with_batcher(model, body, on_pass=broken_callback))
    assert labels(first) == ["a"] and labels(second) == ["b"]


def test_requests_whose_caller_left_are_skipped():
    model = EchoModel(gated=True)

    async def body(batcher):
        first = await _first_pass_running(batcher, model)
        gone, kept = await _queue(batcher, ["gone"], ["kept"])
        gone.cancel()
        await asyncio.sleep(0)
        model.release()
        await first
        return await kept

    kept = run(_with_batcher(model, body))
    assert labels(kept) == ["kept"]
    assert all("gone" not in call for call in model.calls)


def test_waiting_collects_more_texts():
    model = EchoModel()

    async def body(batcher):
        first, second = await _queue(batcher, ["a"], ["b"])
        return await first, await second

    first, second = run(_with_batcher(model, body, max_wait=0.3))
    assert model.calls == [["a", "b"]]
    assert first.batch_texts == second.batch_texts == 2


def test_waiting_counts_from_arrival_not_from_each_pass():
    model = EchoModel(gated=True)

    async def body(batcher):
        first = await _first_pass_running(batcher, model)  # waits max_wait, then runs
        [second] = await _queue(batcher, ["second"])
        await asyncio.sleep(0.25)  # second has now waited longer than max_wait
        model.release()
        await first
        started = time.perf_counter()
        result = await second
        return result, time.perf_counter() - started

    result, after_release = run(_with_batcher(model, body, max_wait=0.2))
    assert labels(result) == ["second"]
    assert after_release < 0.15  # no second max_wait


def test_workers_run_passes_in_parallel():
    model = EchoModel(gated=True)

    async def body(batcher):
        # One text per pass, so the two requests cannot share one.
        tasks = await _queue(batcher, ["0"], ["1"])
        for _ in range(100):
            if model.active == 2:
                break
            await asyncio.sleep(0.02)
        model.release()
        return await asyncio.gather(*tasks)

    run(_with_batcher(model, body, workers=2, max_texts=1))
    assert model.max_active == 2


def test_stop_fails_queued_and_in_flight_requests():
    model = EchoModel(gated=True)

    async def body():
        batcher = DynamicBatcher(model.predict)
        await batcher.start()
        in_flight = await _first_pass_running(batcher, model)
        [queued] = await _queue(batcher, ["queued"])
        await batcher.stop()
        model.release()
        for task in (in_flight, queued):
            with pytest.raises(RuntimeError, match="shutting down"):
                await asyncio.wait_for(task, 5)

    run(body())


def test_submit_needs_a_running_batcher():
    with pytest.raises(RuntimeError, match="not running"):
        run(DynamicBatcher(EchoModel().predict).submit(["a"]))


@pytest.mark.parametrize("kwargs", [{"max_texts": 0}, {"workers": 0}, {"max_wait": -1.0}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        DynamicBatcher(EchoModel().predict, **kwargs)


# ---- through the API ------------------------------------------------------------------


class SlowDummy(DummyPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.calls.append(len(texts))
        time.sleep(0.15)
        return super().predict(texts)


def test_api_batches_concurrent_requests():
    predictor = SlowDummy()
    settings = Settings(model_backend="dummy", batch_requests=True, batch_max_texts=8)
    texts = ["I love it", "awful, broken", "it arrived", "best ever", "terrible", "fine"] * 2

    def post(text: str) -> dict:
        request = urllib.request.Request(
            f"{url}/v1/predict",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as r:
            return json.loads(r.read())

    with live_server(create_app(settings, predictor=predictor)) as url:
        with ThreadPoolExecutor(len(texts)) as pool:
            answers = list(pool.map(post, texts))
        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
            samples = parse(r.read().decode())
        with urllib.request.urlopen(f"{url}/v1/model", timeout=5) as r:
            runtime = json.loads(r.read())["runtime"]

    # Every caller got the answer for its own text...
    expected = DummyPredictor().predict(texts)
    assert [a["prediction"]["label"] for a in answers] == [p.label for p in expected]
    # ...from fewer, larger passes.
    assert sum(predictor.calls) == len(texts)
    assert len(predictor.calls) < len(texts)
    assert max(predictor.calls) > 1 and max(predictor.calls) <= 8
    assert value(samples, "model_dynamic_batching") == 1
    assert value(samples, "model_batch_texts_count") == len(predictor.calls)
    assert value(samples, "model_batch_texts_sum") == len(texts)
    assert value(samples, "model_inference_queue_seconds_count") == len(texts)
    assert runtime["batch_requests"] is True
    assert runtime["batch_max_texts"] == 8


def test_api_batching_is_on_by_default(client):
    runtime = client.get("/v1/model").json()["runtime"]
    assert runtime["batch_requests"] is True
    assert (runtime["batch_max_texts"], runtime["batch_wait_ms"]) == (16, 0)
    assert value(parse(client.get("/metrics").text), "model_dynamic_batching") == 1


def test_api_batching_can_be_turned_off():
    with TestClient(create_app(Settings(model_backend="dummy", batch_requests=False))) as c:
        runtime = c.get("/v1/model").json()["runtime"]
        assert runtime == {
            "inference_threads": runtime["inference_threads"],
            "batch_requests": False,
            "batch_max_texts": None,
            "batch_wait_ms": None,
        }
        assert value(parse(c.get("/metrics").text), "model_dynamic_batching") == 0
        assert c.post("/v1/predict", json={"text": "I love it"}).status_code == 200


def test_api_with_batching_answers_single_and_batch_requests():
    settings = Settings(model_backend="dummy", batch_requests=True)
    with TestClient(create_app(settings)) as client:
        single = client.post("/v1/predict", json={"text": "I love it"})
        batch = client.post("/v1/predict/batch", json={"texts": ["awful", "great"]})
    assert single.status_code == 200 and single.json()["prediction"]["label"] == "positive"
    assert [p["label"] for p in batch.json()["predictions"]] == ["negative", "positive"]
