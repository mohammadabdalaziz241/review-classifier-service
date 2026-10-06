"""Model-loading and predictor tests.

Tests marked ``hf`` build a tiny BERT classifier in a temporary directory and
load it through the real transformers code path, so they need torch and
transformers but no network access or model download.
"""

from __future__ import annotations

import math
import shutil
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings
from review_classifier.model_source import hash_directory
from review_classifier.predictors import (
    DummyPredictor,
    HFPredictor,
    ModelLoadError,
    create_predictor,
)

from .tiny_model import LABELS, build_tiny_model

# ---- dummy backend ----------------------------------------------------------


def test_dummy_predictions_are_deterministic_and_normalised():
    predictor = DummyPredictor()
    texts = ["great product, love it", "terrible, arrived broken", "it is a box"]
    first = predictor.predict(texts)
    assert first == predictor.predict(texts)
    for prediction in first:
        assert set(prediction.scores) == set(DummyPredictor.LABELS)
        assert math.isclose(sum(prediction.scores.values()), 1.0, rel_tol=1e-9)
        assert prediction.score == max(prediction.scores.values())
    assert [p.label for p in first] == ["positive", "negative", "neutral"]


def test_factory_builds_dummy_backend():
    predictor = create_predictor(Settings(model_backend="dummy", task="sarcasm"))
    assert isinstance(predictor, DummyPredictor)
    assert predictor.info.task == "sarcasm"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown model backend"):
        create_predictor(Settings(model_backend="onnx"))


def test_hf_backend_reports_missing_dependencies(monkeypatch):
    # A None entry in sys.modules makes `import transformers` raise ImportError.
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ModelLoadError, match=r"pip install '\.\[hf\]'"):
        HFPredictor("any/model")


# ---- real transformers backend -------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-classifier")
    build_tiny_model(directory)
    return directory


@pytest.mark.hf
def test_hf_loads_local_model_and_reports_content_hash(tiny_model_dir):
    predictor = HFPredictor(str(tiny_model_dir), device="cpu", max_seq_length=8)
    info = predictor.info
    assert info.backend == "hf"
    assert info.origin == "local"
    assert info.labels == ("negative", "positive")
    assert info.version == hash_directory(tiny_model_dir)
    assert info.version.startswith("sha256:")
    assert info.device == "cpu"


@pytest.mark.hf
def test_hf_hub_model_reports_resolved_commit_not_branch(tiny_model_dir, tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    commit = "0123456789abcdef0123456789abcdef01234567"
    snapshot = tmp_path / "snapshots" / commit
    shutil.copytree(tiny_model_dir, snapshot)
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    predictor = HFPredictor("someone/tiny-model", revision="main", device="cpu")

    assert calls["repo_id"] == "someone/tiny-model"
    assert calls["revision"] == "main"
    assert predictor.info.origin == "hub"
    assert predictor.info.requested_revision == "main"
    assert predictor.info.version == commit
    # Weights really came from the snapshot folder: predictions match the local load.
    local = HFPredictor(str(tiny_model_dir), device="cpu")
    assert predictor.predict(["great"])[0].scores == pytest.approx(
        local.predict(["great"])[0].scores
    )


@pytest.mark.hf
def test_hf_batches_preserve_order_and_count(tiny_model_dir):
    predictor = HFPredictor(str(tiny_model_dir), device="cpu", batch_size=2, max_seq_length=8)
    texts = ["great product", "bad", "it", "great great", "bad product"]
    batched = predictor.predict(texts)
    one_by_one = [predictor.predict([t])[0] for t in texts]

    assert len(batched) == len(texts)
    for a, b in zip(batched, one_by_one, strict=True):
        assert a.label == b.label
        for label in LABELS.values():
            # Padding must not change results.
            assert math.isclose(a.scores[label], b.scores[label], abs_tol=1e-5)
        assert math.isclose(sum(a.scores.values()), 1.0, abs_tol=1e-5)


@pytest.mark.hf
def test_hf_truncates_long_input(tiny_model_dir):
    predictor = HFPredictor(str(tiny_model_dir), device="cpu", max_seq_length=8)
    [prediction] = predictor.predict(["great product " * 500])
    assert prediction.label in LABELS.values()


@pytest.mark.hf
def test_hf_rejects_missing_model(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(ModelLoadError, match="does not exist"):
        HFPredictor(str(tmp_path / "does-not-exist"), device="cpu")


@pytest.mark.hf
def test_hf_rejects_multi_label_model(tmp_path):
    build_tiny_model(tmp_path, multi_label=True)
    with pytest.raises(ModelLoadError, match="multi-label"):
        HFPredictor(str(tmp_path), device="cpu")


@pytest.mark.hf
def test_service_serves_real_model_end_to_end(tiny_model_dir):
    settings = Settings(model_backend="hf", model_id=str(tiny_model_dir), device="cpu")
    with TestClient(create_app(settings)) as client:
        info = client.get("/v1/model").json()
        assert info["backend"] == "hf"
        assert info["labels"] == ["negative", "positive"]

        response = client.post("/v1/predict/batch", json={"texts": ["great product", "bad"]})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2


@pytest.mark.hf
def test_service_does_not_start_when_model_fails_to_load(tmp_path):
    pytest.importorskip("transformers")
    settings = Settings(model_backend="hf", model_id=str(tmp_path / "missing"), device="cpu")
    with pytest.raises(ModelLoadError), TestClient(create_app(settings)):
        pass


@pytest.mark.hf
def test_inference_threads_setting(tiny_model_dir):
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        settings = Settings(
            model_backend="hf", model_id=str(tiny_model_dir), device="cpu", inference_threads=1
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/v1/model").json()["runtime"]["inference_threads"] == 1

        # Applied in the thread that runs the pass, even one that already used torch.
        predictor = HFPredictor(str(tiny_model_dir), device="cpu", threads=1)
        seen = {}

        def worker():
            torch.set_num_threads(2)
            predictor.predict(["great"])
            seen["threads"] = torch.get_num_threads()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert seen["threads"] == 1
    finally:
        torch.set_num_threads(before)


@pytest.mark.hf
def test_batched_requests_get_the_same_scores_as_single_ones(tiny_model_dir):
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from .live_server import live_server

    texts = ["great product", "bad", "it was great", "the food was bad", "love it"] * 3
    reference = HFPredictor(str(tiny_model_dir), device="cpu").predict(texts)
    settings = Settings(
        model_backend="hf", model_id=str(tiny_model_dir), device="cpu", batch_requests=True
    )

    with live_server(create_app(settings)) as url:

        def post(text: str) -> dict:
            request = urllib.request.Request(
                f"{url}/v1/predict",
                data=json.dumps({"text": text}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as r:
                return json.loads(r.read())["prediction"]

        with ThreadPoolExecutor(8) as pool:
            served = list(pool.map(post, texts))

    for got, want in zip(served, reference, strict=True):
        assert got["label"] == want.label
        for label, score in want.scores.items():
            assert math.isclose(got["scores"][label], score, abs_tol=1e-5)
