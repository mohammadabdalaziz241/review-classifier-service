"""The image-build model step: fetch, then load offline."""

from __future__ import annotations

import shutil

import pytest

from review_classifier.config import Settings
from review_classifier.model_source import hash_directory
from review_classifier.prefetch import prefetch

from .tiny_model import build_tiny_model

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def restore_offline_flag(monkeypatch):
    # prefetch() switches the process offline; undo that after each test.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


def test_dummy_backend_needs_nothing():
    assert prefetch(Settings(model_backend="dummy"))["backend"] == "dummy"


@pytest.mark.hf
def test_local_checkpoint_is_verified(tmp_path):
    build_tiny_model(tmp_path)
    summary = prefetch(Settings(model_id=str(tmp_path)))
    assert summary["origin"] == "local"
    assert summary["version"] == hash_directory(tmp_path)
    assert summary["labels"] == ["negative", "positive"]
    assert summary["check_prediction"] in {"negative", "positive"}
    assert "warning" not in summary


@pytest.mark.hf
def test_hub_model_is_downloaded_then_loaded_offline(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    build_tiny_model(model_dir)
    snapshot = tmp_path / "snapshots" / COMMIT
    shutil.copytree(model_dir, snapshot)
    calls = []

    def fake_snapshot_download(**kwargs):
        import os

        calls.append(os.environ.get("HF_HUB_OFFLINE"))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    summary = prefetch(Settings(model_id="someone/model", model_revision="main"))

    assert summary["origin"] == "hub"
    assert summary["version"] == COMMIT
    # First call downloads with the network on; the second loads with it off.
    assert calls == [None, "1"]
    assert "Pin that commit" in summary["warning"]


@pytest.mark.hf
def test_pinned_hub_model_has_no_warning(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    snapshot = tmp_path / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    build_tiny_model(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(snapshot))
    summary = prefetch(Settings(model_id="someone/model", model_revision=COMMIT))
    assert "warning" not in summary
