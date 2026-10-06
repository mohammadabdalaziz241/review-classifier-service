"""The benchmark tool, run briefly against a real server with the dummy backend."""

from __future__ import annotations

import argparse
import json
import time

import pytest

from review_classifier import benchmark
from review_classifier.api import create_app
from review_classifier.benchmark import Scenario, make_texts, parse_scenario, percentile
from review_classifier.config import Settings

from .live_server import free_port, live_server


def test_texts_are_deterministic_and_sized():
    assert make_texts("short") == make_texts("short")
    assert make_texts("short", seed=1) != make_texts("short")
    short, long = make_texts("short"), make_texts("long")
    assert all(15 <= len(t.split()) <= 40 for t in short)
    assert all(180 <= len(t.split()) <= 230 for t in long)
    # Within the service's default 2,000-character limit.
    assert max(map(len, long)) < 2000
    with pytest.raises(ValueError):
        make_texts("medium")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("short-c4", Scenario("short-c4", "short", 4)),
        ("long-c1", Scenario("long-c1", "long", 1)),
        ("batch16-short-c2", Scenario("batch16-short-c2", "short", 2, batch=16)),
    ],
)
def test_parse_scenario(spec, expected):
    assert parse_scenario(spec) == expected
    assert expected.path == ("/v1/predict" if expected.batch == 1 else "/v1/predict/batch")


@pytest.mark.parametrize("spec", ["short", "medium-c1", "short-4", "short-c0", "batchx-short-c1"])
def test_parse_scenario_rejects_nonsense(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_scenario(spec)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5.0], 99) == 5.0
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 100


def test_benchmark_against_a_live_server(tmp_path, capsys):
    app = create_app(Settings(model_backend="dummy", max_batch_size=16))
    with live_server(app) as url:
        out = tmp_path / "bench.json"
        code = benchmark.main(
            [
                "--url", url,
                "--metrics-url", f"{url}/metrics",
                "--duration", "1",
                "--warmup", "0.2",
                "--scenario", "short-c2",
                "--scenario", "batch4-long-c1",
                "--label", "test",
                "--out", str(out),
                "--fail-on-errors",
            ]
        )  # fmt: skip
    assert code == 0
    report = json.loads(out.read_text())
    assert report["label"] == "test"
    assert report["conditions"]["model"]["backend"] == "dummy"
    assert report["conditions"]["server"]["max_concurrent_inferences"] == 1
    assert report["conditions"]["texts_mean_chars"]["long"] > 1000
    short, batch = report["scenarios"]
    assert short["scenario"]["path"] == "/v1/predict"
    assert batch["scenario"]["path"] == "/v1/predict/batch"
    for result in report["scenarios"]:
        assert result["requests"] > 0
        assert result["errors"] == 0
        assert result["latency_ms"]["p50"] > 0
        assert result["server_ms"]["p50"] is not None
        assert result["inference_ms"]["p50"] is not None
        assert result["latency_ms"]["p50"] <= result["latency_ms"]["p99"]
    assert batch["texts_per_s"] == pytest.approx(batch["requests_per_s"] * 4, rel=0.01)
    table = capsys.readouterr().out
    assert "| short-c2 | 2 | 1 |" in table
    assert "| batch4-long-c1 | 1 | 4 |" in table


def test_errors_are_counted_not_hidden(tmp_path):
    # Batches above the server's limit are rejected with 422.
    app = create_app(Settings(model_backend="dummy", max_batch_size=2))
    with live_server(app) as url:
        report = benchmark.run(url, [parse_scenario("batch4-short-c1")], duration=0.5, warmup=0)
        code = benchmark.main(
            ["--url", url, "--duration", "0.3", "--warmup", "0",
             "--scenario", "batch4-short-c1", "--fail-on-errors"]
        )  # fmt: skip
    [result] = report.scenarios
    assert result.requests > 0
    assert result.error_rate == 1.0
    assert result.error_kinds == {"HTTP 422": result.requests}
    assert result.requests_per_s == 0
    assert code == 1


def test_unreachable_service():
    assert benchmark.main(["--url", f"http://127.0.0.1:{free_port()}", "--duration", "1"]) == 2


def test_wait_gives_up_after_the_timeout():
    started = time.monotonic()
    code = benchmark.main(
        ["--url", f"http://127.0.0.1:{free_port()}", "--duration", "1", "--wait", "2"]
    )
    assert code == 2
    assert 1.5 <= time.monotonic() - started < 10
