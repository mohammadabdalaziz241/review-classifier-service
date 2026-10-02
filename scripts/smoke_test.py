"""End-to-end check of a running service, e.g. the Docker Compose stack.

    python scripts/smoke_test.py --url http://127.0.0.1:8000 --expect-database

Waits for /ready, then exercises every endpoint the way a client would and
checks the answers: predictions, batch order, error envelopes, and, with
--expect-database, that predictions are recorded and feedback round-trips.
Standard library only, so it runs anywhere without installing anything.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


class CheckFailed(AssertionError):
    pass


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-Request-ID": f"smoke-{uuid.uuid4().hex}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)
    print(f"  ok  {message}")


def wait_until_ready(base: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = call(base, "GET", "/ready")
            if status == 200:
                return body
            last = f"{status} {body}"
        except (urllib.error.URLError, ConnectionError, OSError) as error:
            last = str(error)
        time.sleep(2)
    raise CheckFailed(f"service not ready after {timeout:.0f}s (last: {last})")


def run(base: str, expect_database: bool, timeout: float) -> None:
    print(f"Waiting for {base}/ready ...")
    ready = wait_until_ready(base, timeout)
    check(ready["status"] == "ready", f"ready: {ready}")

    status, model = call(base, "GET", "/v1/model")
    check(status == 200, f"model {model.get('id')} @ {model.get('version')}")
    labels = model["labels"]

    text = "The staff were lovely and the food was great"
    status, single = call(base, "POST", "/v1/predict", {"text": text})
    check(status == 200, "single prediction")
    prediction = single["prediction"]
    check(prediction["label"] in labels, f"label {prediction['label']!r} is one of {labels}")
    check(abs(sum(prediction["scores"].values()) - 1) < 1e-3, "scores sum to 1")
    check(single["model"]["version"] == model["version"], "response names the served version")

    texts = ["Brilliant, really", "Terrible service", "It was a Tuesday"]
    status, batch = call(base, "POST", "/v1/predict/batch", {"texts": texts})
    check(status == 200 and len(batch["predictions"]) == 3, "batch of three")
    ids = {p["id"] for p in batch["predictions"]}
    check(len(ids) == 3 and prediction["id"] not in ids, "every prediction has its own id")

    status, error = call(base, "POST", "/v1/predict", {"text": "   "})
    check(status == 422 and error["error"]["code"] == "invalid_request", "blank text rejected")

    if not expect_database:
        return

    check(ready.get("database") == "ok", "database ok")
    check(single["recorded"] and batch["recorded"], "predictions recorded")

    other = next(label for label in labels if label != prediction["label"])
    status, feedback = call(
        base,
        "POST",
        "/v1/feedback",
        {"prediction_id": prediction["id"], "label": other, "text": text},
    )
    check(status == 201, "feedback recorded")
    check(feedback["model_was_correct"] is False and feedback["text_stored"], "feedback compared")

    status, again = call(
        base, "POST", "/v1/feedback", {"prediction_id": prediction["id"], "label": other}
    )
    check(
        status == 409 and again["error"]["code"] == "feedback_exists", "duplicate feedback refused"
    )

    status, wrong = call(
        base,
        "POST",
        "/v1/feedback",
        {"prediction_id": batch["predictions"][0]["id"], "label": labels[0], "text": "edited"},
    )
    check(
        status == 422 and wrong["error"]["details"][0]["type"] == "text_mismatch",
        "feedback text must match the classified text",
    )

    status, missing = call(
        base, "POST", "/v1/feedback", {"prediction_id": str(uuid.uuid4()), "label": labels[0]}
    )
    check(
        status == 404 and missing["error"]["code"] == "prediction_not_found",
        "feedback for an unknown prediction",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-database", action="store_true")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds to wait for /ready")
    args = parser.parse_args()
    try:
        run(args.url.rstrip("/"), args.expect_database, args.timeout)
    except CheckFailed as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        return 1
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
