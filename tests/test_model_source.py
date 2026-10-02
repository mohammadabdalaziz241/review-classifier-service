"""Model version resolution. These tests need no torch and no network."""

from __future__ import annotations

import pytest

from review_classifier.model_source import (
    HUB_ALLOW_PATTERNS,
    ModelSourceError,
    hash_directory,
    resolve_model_source,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def model_dir(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text('{"num_labels": 2}')
    (directory / "model.safetensors").write_bytes(b"\x00\x01weights")
    return directory


def test_hash_is_stable(model_dir):
    assert hash_directory(model_dir) == hash_directory(model_dir)
    assert len(hash_directory(model_dir)) == len("sha256:") + 64


def test_hash_changes_when_weights_change(model_dir):
    before = hash_directory(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"\x00\x02weights")
    assert hash_directory(model_dir) != before


def test_hash_changes_when_a_file_is_renamed(model_dir):
    before = hash_directory(model_dir)
    (model_dir / "model.safetensors").rename(model_dir / "other.safetensors")
    assert hash_directory(model_dir) != before


def test_hash_ignores_hidden_files(model_dir):
    before = hash_directory(model_dir)
    (model_dir / ".cache").mkdir()
    (model_dir / ".cache" / "lock").write_text("x")
    (model_dir / ".DS_Store").write_text("x")
    assert hash_directory(model_dir) == before


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(ModelSourceError, match="no files"):
        hash_directory(tmp_path)


def test_local_directory_is_identified_by_content(model_dir):
    source = resolve_model_source(str(model_dir), revision="main")
    assert source.origin == "local"
    assert source.path == model_dir
    assert source.version == hash_directory(model_dir)
    # The requested revision is recorded but never used as the version.
    assert source.requested_revision == "main"


@pytest.mark.parametrize("path", ["/no/such/model", "./missing", "~/missing-model", "a/b/c"])
def test_missing_local_path_is_not_sent_to_the_hub(path):
    with pytest.raises(ModelSourceError, match="does not exist"):
        resolve_model_source(path)


def test_hub_model_reports_resolved_commit(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    snapshot = tmp_path / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    source = resolve_model_source("someone/model", revision="main")

    assert source.version == COMMIT
    assert source.origin == "hub"
    assert source.path == snapshot
    assert source.requested_revision == "main"
    assert seen == {
        "repo_id": "someone/model",
        "revision": "main",
        "allow_patterns": list(HUB_ALLOW_PATTERNS),
        "local_files_only": False,
    }


def test_offline_mode_uses_the_local_cache_only(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    snapshot = tmp_path / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(snapshot)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert resolve_model_source("someone/model", revision=COMMIT).version == COMMIT
    assert seen["local_files_only"] is True


def test_hub_snapshot_without_commit_hash_is_rejected(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(tmp_path / "main"))
    with pytest.raises(ModelSourceError, match="commit hash"):
        resolve_model_source("someone/model")


def test_hub_download_failure_is_reported(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def fail(**_):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(ModelSourceError, match="401 Unauthorized"):
        resolve_model_source("someone/private-model", revision="v2")
