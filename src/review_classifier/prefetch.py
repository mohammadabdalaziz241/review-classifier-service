"""Fetch the configured model ahead of time and prove it loads offline.

    MODEL_ID=user/model MODEL_REVISION=<commit> python -m review_classifier.prefetch

Used when building the Docker image: the model is downloaded into the image's
Hugging Face cache, then loaded again with the network switched off and run on
a test sentence. A model that would fail in production (missing files, wrong
format, multi-label head) fails the image build instead.

Reads the same environment variables as the service. Prints a JSON summary.
"""

from __future__ import annotations

import json
import os
import re
import sys

from .config import Settings
from .model_source import resolve_model_source

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def prefetch(settings: Settings) -> dict:
    if settings.model_backend == "dummy":
        return {"backend": "dummy", "model": None, "note": "dummy backend: nothing to fetch"}

    # 1. Download (Hub) or hash (local directory) while the network is available.
    source = resolve_model_source(settings.model_id, settings.model_revision)

    # 2. Load it the way the service will in the container: offline, CPU.
    os.environ["HF_HUB_OFFLINE"] = "1"
    from .predictors import HFPredictor

    predictor = HFPredictor(
        settings.model_id,
        revision=settings.model_revision,
        task=settings.task,
        device="cpu",
        max_seq_length=settings.max_seq_length,
    )
    [prediction] = predictor.predict(["This is a quick check that the model works."])
    info = predictor.info
    if info.version != source.version:
        raise RuntimeError(f"Offline load found version {info.version}, expected {source.version}")

    summary = {
        "backend": "hf",
        "model": info.model_id,
        "version": info.version,
        "origin": info.origin,
        "labels": list(info.labels),
        "task": info.task,
        "max_seq_length": info.max_seq_length,
        "check_prediction": prediction.label,
    }
    if info.origin == "hub" and not _COMMIT.match(settings.model_revision or ""):
        summary["warning"] = (
            f"MODEL_REVISION={settings.model_revision or 'main'!r} is not a commit hash; it "
            f"resolved to {info.version}. Pin that commit so rebuilds get the same weights."
        )
    return summary


def main() -> int:
    summary = prefetch(Settings.from_env())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
