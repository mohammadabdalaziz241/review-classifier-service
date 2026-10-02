"""Check that the running service reproduces a checkpoint's training metrics.

    python -m review_classifier.evaluate --url http://127.0.0.1:8000 \\
        --manifest checkpoints/roberta-sentiment/training_manifest.json

Sends the test split through ``POST /v1/predict/batch`` exactly as a client
would and computes the same metrics as training. If the served macro-F1 and
accuracy match the manifest's test metrics, the serving path (preprocessing,
tokenisation, truncation, label order) reproduces training. A gap points to a
mismatch in one of them.

It also reports how many test texts the service's preprocessing changes, which
should be (almost) none for a model trained on raw text, and fails early if any
text exceeds the service's MAX_TEXT_CHARS.

Exit codes: 0 metrics match, 1 metrics differ, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import classification_metrics
from .preprocessing import preprocess
from .serving_config import PreprocessingConfig

# Pooled RoBERTa test macro-F1 from the project's results/all_results.json, by
# seed. Shown for context only: a retrained model will not reproduce it exactly
# (GPU nondeterminism, library versions). The pass/fail check uses the manifest.
NOTEBOOK_TEST_MACRO_F1 = {
    "sentiment": {42: 0.9028, 123: 0.9011},
    "sarcasm": {42: 0.7001, 123: 0.7173},
}
DEFAULT_TOLERANCE = 0.005


class EvaluationError(RuntimeError):
    """The evaluation could not be carried out."""


@dataclass
class Report:
    model: dict
    n: int
    metrics: dict
    metrics_by_variety: dict = field(default_factory=dict)
    changed_by_preprocessing: list[int] = field(default_factory=list)
    expected: dict | None = None
    differences: dict = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE
    seconds: float = 0.0

    @property
    def passed(self) -> bool | None:
        if self.expected is None:
            return None
        return all(abs(d) <= self.tolerance for d in self.differences.values())

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "n": self.n,
            "seconds": round(self.seconds, 2),
            "metrics": self.metrics,
            "metrics_by_variety": self.metrics_by_variety,
            "changed_by_preprocessing": {
                "count": len(self.changed_by_preprocessing),
                "indices": self.changed_by_preprocessing[:50],
            },
            "expected": self.expected,
            "differences": self.differences,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def evaluate_service(
    client,
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    expected_labels: Sequence[str] | None = None,
    varieties: Sequence[str] | None = None,
    expected_metrics: dict | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    batch_size: int = 32,
) -> Report:
    """Run ``texts`` through the service behind ``client`` (an httpx-style client)."""
    if len(texts) != len(labels):
        raise EvaluationError(f"{len(texts)} texts but {len(labels)} labels")

    response = client.get("/v1/model")
    if response.status_code != 200:
        raise EvaluationError(f"GET /v1/model returned {response.status_code}: {response.text}")
    model = response.json()
    served_labels = model["labels"]
    if expected_labels is not None and list(expected_labels) != served_labels:
        raise EvaluationError(
            f"The service returns labels {served_labels} but the data expects "
            f"{list(expected_labels)} (same order, index = class id). Wrong checkpoint?"
        )

    limit = model["limits"]["max_text_chars"]
    too_long = [i for i, t in enumerate(texts) if len(t) > limit]
    if too_long:
        longest = max(len(texts[i]) for i in too_long)
        raise EvaluationError(
            f"{len(too_long)} texts exceed the service's MAX_TEXT_CHARS={limit} "
            f"(longest: {longest} characters). Restart the service with a higher "
            "MAX_TEXT_CHARS; the model truncates by tokens anyway."
        )

    preprocessing = PreprocessingConfig(
        **{k: model["preprocessing"][k] for k in PreprocessingConfig().as_dict()}
    )
    changed = [i for i, t in enumerate(texts) if preprocess(t, preprocessing) != t]

    batch_size = min(batch_size, model["limits"]["max_batch_size"])
    index = {label: i for i, label in enumerate(served_labels)}
    predictions: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.post(
            "/v1/predict/batch",
            json={"texts": batch},
            headers={"X-Request-ID": f"eval-{start}"},
        )
        if response.status_code != 200:
            raise EvaluationError(
                f"Batch starting at {start} failed with {response.status_code}: {response.text}"
            )
        body = response.json()
        if body["model"]["version"] != model["version"]:
            raise EvaluationError("The model version changed during evaluation.")
        predictions.extend(index[p["label"]] for p in body["predictions"])
    seconds = time.perf_counter() - started

    num_classes = len(served_labels)
    metrics = classification_metrics(list(labels), predictions, num_classes=num_classes)
    by_variety = {}
    if varieties is not None:
        for variety in sorted(set(varieties)):
            idx = [i for i, v in enumerate(varieties) if v == variety]
            by_variety[variety] = classification_metrics(
                [labels[i] for i in idx], [predictions[i] for i in idx], num_classes=num_classes
            )

    report = Report(
        model={k: model[k] for k in ("id", "version", "task", "labels", "max_seq_length")}
        | {"preprocessing": model["preprocessing"]},
        n=len(texts),
        metrics=metrics,
        metrics_by_variety=by_variety,
        changed_by_preprocessing=changed,
        expected=expected_metrics,
        tolerance=tolerance,
        seconds=seconds,
    )
    if expected_metrics is not None:
        report.differences = {
            key: metrics[key] - expected_metrics[key]
            for key in ("macro_f1", "accuracy")
            if key in expected_metrics
        }
    return report


def _print_summary(report: Report, task: str | None, seed: int | None = None) -> None:
    m = report.metrics
    model = report.model
    print(f"Model    {model['id']} @ {model['version']}")
    print(f"Task     {model['task']}  labels={model['labels']}")
    print(f"         max_seq_length={model['max_seq_length']}")
    active = [k for k, v in model["preprocessing"].items() if v is True]
    print(f"Preproc  {', '.join(active) or 'none (raw text)'}")
    changed = len(report.changed_by_preprocessing)
    print(f"         {changed} of {report.n} texts changed by preprocessing")
    print(
        f"Served   n={report.n} accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
        f"class_1_f1={m['class_1_f1']:.4f}  ({report.seconds:.1f}s)"
    )
    for variety, vm in report.metrics_by_variety.items():
        print(f"         {variety:8s} n={vm['n']:<5d} macro_f1={vm['macro_f1']:.4f}")
    if report.expected is not None:
        e = report.expected
        print(f"Training accuracy={e['accuracy']:.4f} macro_f1={e['macro_f1']:.4f}")
        diffs = "  ".join(f"{k} {v:+.4f}" for k, v in report.differences.items())
        verdict = "MATCH" if report.passed else "MISMATCH"
        print(f"Result   {verdict} (tolerance ±{report.tolerance})  {diffs}")
    reference = NOTEBOOK_TEST_MACRO_F1.get(task or "", {})
    if reference:
        runs = ", ".join(f"seed {s}: {v:.4f}" for s, v in reference.items())
        note = f" (this checkpoint: seed {seed})" if seed is not None else ""
        print(f"Context  notebook pooled RoBERTa test macro-F1 {runs}{note}")


def main(argv: list[str] | None = None) -> int:
    import httpx

    from .train import DEFAULT_DATASET, TASKS, load_splits

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", type=Path, help="training_manifest.json of the served model")
    parser.add_argument("--task", choices=sorted(TASKS), help="Default: from the manifest")
    parser.add_argument("--dataset", help=f"Default: from the manifest, else {DEFAULT_DATASET}")
    parser.add_argument("--dataset-revision", help="Default: the manifest's dataset version")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--report", type=Path, help="Write the full report as JSON here")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    task = args.task or (manifest or {}).get("task")
    if task is None:
        parser.error("--task is required without --manifest")
    spec = TASKS[task]

    dataset = args.dataset or (manifest["dataset"]["id"] if manifest else DEFAULT_DATASET)
    revision = args.dataset_revision
    if revision is None and manifest and dataset == manifest["dataset"]["id"]:
        version = manifest["dataset"]["version"]
        revision = None if version.startswith("sha256:") else version
    expected = None
    if manifest and args.split == "test":
        expected = manifest["test_metrics"]

    try:
        splits, _ = load_splits(dataset, revision)
        split = splits[args.split]
        texts = list(split["text"])
        labels = [int(v) for v in split[spec.column]]
        varieties = list(split["variety"]) if "variety" in split.column_names else None
        with httpx.Client(base_url=args.url, timeout=120) as client:
            report = evaluate_service(
                client,
                texts,
                labels,
                expected_labels=spec.labels,
                varieties=varieties,
                expected_metrics=expected,
                tolerance=args.tolerance,
                batch_size=args.batch_size,
            )
    except (EvaluationError, httpx.HTTPError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2

    seed = manifest["recipe"].get("seed") if manifest else None
    _print_summary(report, task, seed)
    if args.report:
        args.report.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"Report   {args.report}")
    return 1 if report.passed is False else 0


if __name__ == "__main__":
    sys.exit(main())
