"""Measure the running service: latency, throughput, error rate, memory and CPU.

    python -m review_classifier.benchmark --url http://127.0.0.1:8000 \\
        --metrics-url http://127.0.0.1:9000/metrics --out benchmark.json

Each scenario is a closed loop: ``concurrency`` clients each send a request,
wait for the answer and send the next, for ``--duration`` seconds after a
``--warmup`` that is not counted. Latency is measured three ways:

* **end-to-end**, by the client: what a caller experiences on this network;
* **server**, the API's ``X-Process-Time-Ms`` header: the same request without
  the network, including any wait for a free inference slot;
* **inference**, the ``inference_ms`` the API reports: the model alone.

With ``--metrics-url`` it also reads the server's memory and CPU from its
Prometheus metrics. The conditions (model, limits, threads, machine, text
lengths) are recorded with the results, so a run can be repeated and compared.

The texts are generated deterministically (no dataset download): "short" ones
of about 20 words and "long" ones of about 200 words, which RoBERTa's tokenizer
turns into roughly 30 and 250 tokens, near the 256-token truncation length.

Uses only the standard library, so it runs inside the service image.
Exit codes: 0 done, 1 a scenario had errors and --fail-on-errors was given,
2 the service could not be reached.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import platform
import random
import resource
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ---- texts ------------------------------------------------------------------------

_OPENERS = [
    "I bought this last month",
    "We ordered it for the family",
    "Picked one up on sale",
    "After a week of daily use",
    "My partner chose this one",
    "Ordered online and it arrived quickly",
    "Third one I have owned",
    "Got it as a birthday present",
]
_VERDICTS = [
    "and honestly it is brilliant",
    "and it broke within days",
    "and it does exactly what it says",
    "and it was a complete waste of money",
    "and I would happily buy it again",
    "and the quality is shockingly poor",
    "and it is fine, nothing special",
    "oh great, another thing that stopped working",
]
_DETAILS = [
    "The battery lasts far longer than the old one, even with heavy use.",
    "Customer service took three weeks to reply and then closed the ticket.",
    "Setup was simple and the instructions were clear enough for anyone.",
    "The colour is nothing like the photos on the website, which is disappointing.",
    "It feels sturdy, the buttons are responsive and the finish looks premium.",
    "Delivery was late, the box was crushed and one part was missing entirely.",
    "For the price you really cannot complain about what you get here.",
    "It makes a strange rattling noise whenever it gets warm after an hour.",
    "My kids use it every day and it has survived being dropped more than once.",
    "The app keeps logging me out and the updates make it slower each time.",
    "Compared with the more expensive brands it holds up surprisingly well.",
    "Absolutely loved being charged twice and then told it was my fault.",
    "It is lighter than expected, which makes it easy to carry around town.",
    "The smell when you first unpack it is strong but fades after a day or two.",
    "Cleaning it is a nightmare because of all the small gaps and corners.",
    "Would recommend to friends, although the cable could be a bit longer.",
]

TEXT_KINDS = ("short", "long")


def make_texts(kind: str, n: int = 64, seed: int = 0) -> list[str]:
    """``n`` review-like texts. Same arguments, same texts, on every machine."""
    if kind not in TEXT_KINDS:
        raise ValueError(f"text kind must be one of {TEXT_KINDS}, got {kind!r}")
    rng = random.Random(f"{kind}-{seed}")
    texts = []
    for _ in range(n):
        text = f"{rng.choice(_OPENERS)} {rng.choice(_VERDICTS)}."
        if kind == "short":
            text += " " + rng.choice(_DETAILS)
        else:
            details = []
            while len(" ".join(details).split()) < 180:
                details.append(rng.choice(_DETAILS))
            text += " " + " ".join(details)
        texts.append(text)
    return texts


# ---- scenarios --------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    texts: str  # "short" or "long"
    concurrency: int
    batch: int = 1  # 1: POST /v1/predict; more: POST /v1/predict/batch

    @property
    def path(self) -> str:
        return "/v1/predict" if self.batch == 1 else "/v1/predict/batch"


DEFAULT_SCENARIOS = (
    Scenario("short-c1", "short", 1),
    Scenario("short-c4", "short", 4),
    Scenario("short-c16", "short", 16),
    Scenario("long-c1", "long", 1),
    Scenario("long-c4", "long", 4),
    Scenario("batch16-short-c1", "short", 1, batch=16),
)


def parse_scenario(spec: str) -> Scenario:
    """``short-c4``, ``long-c1`` or ``batch16-short-c2``."""
    parts = spec.split("-")
    try:
        batch = 1
        if parts[0].startswith("batch"):
            batch = int(parts.pop(0)[len("batch") :])
        kind, conc = parts
        if kind not in TEXT_KINDS or not conc.startswith("c"):
            raise ValueError
        concurrency = int(conc[1:])
        if concurrency < 1 or batch < 1:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"bad scenario {spec!r}: use e.g. short-c4, long-c1 or batch16-short-c1"
        ) from exc
    return Scenario(spec, kind, concurrency, batch)


# ---- measurements ------------------------------------------------------------------


@dataclass
class Sample:
    started: float
    seconds: float
    ok: bool
    server_ms: float | None = None
    inference_ms: float | None = None
    error: str | None = None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks; q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    def r(x: float | None) -> float | None:
        return None if x is None else round(x, 2)

    return {
        "p50": r(percentile(values, 50)),
        "p90": r(percentile(values, 90)),
        "p95": r(percentile(values, 95)),
        "p99": r(percentile(values, 99)),
        "max": r(max(values)) if values else None,
        "mean": r(sum(values) / len(values)) if values else None,
    }


class _Client:
    """One keep-alive HTTP connection, reopened after an error."""

    def __init__(self, url: str, timeout: float) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must start with http:// or https://, got {url!r}")
        self._https = parsed.scheme == "https"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port
        self._prefix = parsed.path.rstrip("/")
        self._timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._connection is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            self._connection = cls(self._host, self._port, timeout=self._timeout)
        return self._connection

    def post(self, path: str, body: bytes) -> Sample:
        started = time.perf_counter()
        try:
            connection = self._connect()
            connection.request(
                "POST", self._prefix + path, body, {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            payload = response.read()
            seconds = time.perf_counter() - started
        except (OSError, http.client.HTTPException) as exc:
            self.close()
            return Sample(started, time.perf_counter() - started, False, error=type(exc).__name__)
        server = response.getheader("X-Process-Time-Ms")
        if response.status != 200:
            return Sample(started, seconds, False, error=f"HTTP {response.status}")
        try:
            inference = float(json.loads(payload)["inference_ms"])
        except (ValueError, KeyError, TypeError):
            inference = None
        return Sample(started, seconds, True, float(server) if server else None, inference)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class _MetricsReader:
    """Reads process memory and CPU from the server's Prometheus metrics."""

    NAMES = (
        "process_resident_memory_bytes",
        "process_cpu_seconds_total",
        "model_inference_threads",
        "model_max_concurrent_inferences",
    )

    def __init__(self, url: str | None) -> None:
        self.url = url

    def read(self) -> dict[str, float]:
        if not self.url:
            return {}
        try:
            with urllib.request.urlopen(self.url, timeout=5) as response:
                text = response.read().decode()
        except OSError:
            return {}
        values: dict[str, float] = {}
        for line in text.splitlines():
            name, _, rest = line.partition(" ")
            if name in self.NAMES:
                with contextlib.suppress(IndexError, ValueError):
                    values[name] = float(rest.split()[0])
        return values


@dataclass
class ScenarioResult:
    scenario: dict
    duration_s: float
    requests: int
    errors: int
    error_rate: float
    error_kinds: dict[str, int]
    requests_per_s: float
    texts_per_s: float
    latency_ms: dict
    server_ms: dict
    inference_ms: dict
    server_cpu_cores: float | None = None
    server_rss_mb_max: float | None = None
    client_cpu_cores: float | None = None


def run_scenario(
    url: str,
    scenario: Scenario,
    duration: float,
    warmup: float,
    metrics: _MetricsReader,
    timeout: float = 60.0,
) -> ScenarioResult:
    texts = make_texts(scenario.texts)
    counter = iter(range(sys.maxsize))
    counter_lock = threading.Lock()

    def next_body() -> bytes:
        with counter_lock:
            i = next(counter)
        if scenario.batch == 1:
            return json.dumps({"text": texts[i % len(texts)]}).encode()
        batch = [texts[(i * scenario.batch + j) % len(texts)] for j in range(scenario.batch)]
        return json.dumps({"texts": batch}).encode()

    begin = time.perf_counter()
    measure_from = begin + warmup
    stop_at = measure_from + duration
    samples: list[Sample] = []
    samples_lock = threading.Lock()

    def worker() -> None:
        client = _Client(url, timeout)
        local: list[Sample] = []
        try:
            while time.perf_counter() < stop_at:
                sample = client.post(scenario.path, next_body())
                # Count requests that finished inside the window (throughput counts
                # completions; dropping the ones that started earlier would drop the
                # slowest requests).
                if measure_from <= sample.started + sample.seconds <= stop_at:
                    local.append(sample)
        finally:
            client.close()
        with samples_lock:
            samples.extend(local)

    rss: list[float] = []
    sampling = threading.Event()

    def sample_memory() -> None:
        while not sampling.wait(1.0):
            value = metrics.read().get("process_resident_memory_bytes")
            if value is not None and time.perf_counter() >= measure_from:
                rss.append(value)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(scenario.concurrency)]
    sampler = threading.Thread(target=sample_memory, daemon=True)
    for t in threads:
        t.start()
    sampler.start()

    # CPU used by the server and by this client during the measured window.
    while time.perf_counter() < measure_from:
        time.sleep(0.01)
    before = metrics.read()
    client_before = _client_cpu()
    window_start = time.perf_counter()
    for t in threads:
        t.join()
    window = time.perf_counter() - window_start
    after = metrics.read()
    client_after = _client_cpu()
    sampling.set()
    sampler.join()

    ok = [s for s in samples if s.ok]
    errors = [s for s in samples if not s.ok]
    kinds: dict[str, int] = {}
    for s in errors:
        kinds[s.error or "error"] = kinds.get(s.error or "error", 0) + 1
    cpu = None
    if "process_cpu_seconds_total" in before and "process_cpu_seconds_total" in after:
        cpu = (after["process_cpu_seconds_total"] - before["process_cpu_seconds_total"]) / window
    if "process_resident_memory_bytes" in after:
        rss.append(after["process_resident_memory_bytes"])
    return ScenarioResult(
        scenario=asdict(scenario) | {"path": scenario.path},
        duration_s=round(duration, 2),
        requests=len(samples),
        errors=len(errors),
        error_rate=round(len(errors) / len(samples), 4) if samples else 0.0,
        error_kinds=kinds,
        requests_per_s=round(len(ok) / duration, 2),
        texts_per_s=round(len(ok) * scenario.batch / duration, 2),
        latency_ms=_summary([s.seconds * 1000 for s in ok]),
        server_ms=_summary([s.server_ms for s in ok if s.server_ms is not None]),
        inference_ms=_summary([s.inference_ms for s in ok if s.inference_ms is not None]),
        server_cpu_cores=None if cpu is None else round(cpu, 2),
        server_rss_mb_max=round(max(rss) / 2**20, 1) if rss else None,
        client_cpu_cores=round((client_after - client_before) / window, 2) if window else None,
    )


def _client_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


# ---- the run ---------------------------------------------------------------------


@dataclass
class Report:
    started_at: str
    label: str | None
    url: str
    conditions: dict
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def conditions(url: str, metrics: _MetricsReader, duration: float, warmup: float) -> dict:
    model = _get_json(url.rstrip("/") + "/v1/model")
    server = metrics.read()
    text_chars = {
        kind: round(sum(map(len, make_texts(kind))) / len(make_texts(kind))) for kind in TEXT_KINDS
    }
    return {
        "model": {
            k: model.get(k)
            for k in ("id", "version", "backend", "task", "device", "max_seq_length")
        },
        "limits": model.get("limits"),
        "server": {
            "inference_threads": server.get("model_inference_threads"),
            "max_concurrent_inferences": server.get("model_max_concurrent_inferences"),
            "metrics": bool(metrics.url),
        },
        "client": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "texts_mean_chars": text_chars,
        "duration_s": duration,
        "warmup_s": warmup,
    }


def markdown(report: Report) -> str:
    def num(x: float | None, digits: int = 0) -> str:
        return "n/a" if x is None else f"{x:.{digits}f}"

    header = (
        "Scenario",
        "Clients",
        "Texts/req",
        "Req/s",
        "Texts/s",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "Server p50",
        "Model p50",
        "Errors",
        "Server CPU",
        "Server RSS MB",
    )
    rows = ["| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
    for r in report.scenarios:
        s = r.scenario
        cells = (
            s["name"],
            str(s["concurrency"]),
            str(s["batch"]),
            num(r.requests_per_s, 1),
            num(r.texts_per_s, 1),
            num(r.latency_ms["p50"]),
            num(r.latency_ms["p95"]),
            num(r.latency_ms["p99"]),
            num(r.server_ms["p50"]),
            num(r.inference_ms["p50"]),
            f"{r.error_rate:.1%}",
            num(r.server_cpu_cores, 2),
            num(r.server_rss_mb_max),
        )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def run(
    url: str,
    scenarios: Sequence[Scenario],
    duration: float,
    warmup: float,
    metrics_url: str | None = None,
    label: str | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> Report:
    metrics = _MetricsReader(metrics_url)
    report = Report(
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label=label,
        url=url,
        conditions=conditions(url, metrics, duration, warmup),
    )
    for scenario in scenarios:
        progress(f"{scenario.name}: {scenario.concurrency} client(s), {duration:g}s")
        result = run_scenario(url, scenario, duration, warmup, metrics)
        progress(
            f"  {result.requests_per_s:.1f} req/s, p50 {result.latency_ms['p50']} ms, "
            f"p95 {result.latency_ms['p95']} ms, errors {result.error_rate:.1%}"
        )
        report.scenarios.append(result)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m review_classifier.benchmark",
        description="Load-test the running service and report latency, throughput and errors.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--metrics-url", help="Prometheus metrics URL of the server, for memory and CPU"
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds per scenario")
    parser.add_argument("--warmup", type=float, default=5.0, help="Uncounted seconds first")
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        type=parse_scenario,
        help="e.g. short-c4, long-c1, batch16-short-c1 (repeatable; default: the standard set)",
    )
    parser.add_argument("--label", help="Describes where this ran, e.g. 'AWS c7i-flex.large'")
    parser.add_argument("--out", help="Write the full report as JSON here ('-' for stdout)")
    parser.add_argument(
        "--fail-on-errors", action="store_true", help="Exit 1 if any request failed"
    )
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.warmup < 0:
        parser.error("--duration must be positive and --warmup not negative")

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        _get_json(args.url.rstrip("/") + "/ready")
    except OSError as exc:
        print(f"error: the service at {args.url} is not ready: {exc}", file=sys.stderr)
        return 2
    report = run(
        args.url,
        args.scenarios or DEFAULT_SCENARIOS,
        args.duration,
        args.warmup,
        args.metrics_url,
        args.label,
        progress,
    )
    document = json.dumps(report.to_json(), indent=2)
    if args.out == "-":
        print(document)
    else:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(document + "\n")
        print(markdown(report))
    if args.fail_on_errors and any(r.errors for r in report.scenarios):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
