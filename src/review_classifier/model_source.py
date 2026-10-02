"""Resolve a model reference to files on disk and an exact, immutable version.

A version must identify the weights that were actually loaded, so branch
names ("main") and placeholders ("local") are never reported:

* Hub models are downloaded into a snapshot folder named after the resolved
  commit hash and loaded from that folder. The reported version is that commit
  hash, so it always matches the loaded files, even if the branch moves later.
* Local directories are identified by a SHA-256 hash of their contents.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CHUNK = 1024 * 1024

# Files needed to run a PyTorch sequence classifier. TensorFlow, Flax, ONNX and
# other exports in the same repository are skipped.
HUB_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "pytorch_model*.bin",
)


def hub_offline() -> bool:
    """True when HF_HUB_OFFLINE is set, e.g. in a container with the model pre-cached."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class ModelSourceError(RuntimeError):
    """The model reference could not be resolved to an exact version."""


@dataclass(frozen=True)
class ModelSource:
    path: Path
    version: str  # commit hash for Hub models, "sha256:<hex>" for local directories
    origin: str  # "hub" or "local"
    requested_revision: str | None


def hash_directory(directory: Path) -> str:
    """SHA-256 over every file's relative path and contents, in sorted order.

    Hidden files and folders (e.g. ``.git``, ``.cache``) are excluded, so the
    hash changes when the model files change and only then.
    """
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )
    if not files:
        raise ModelSourceError(f"Model directory {directory} contains no files.")
    for file in files:
        digest.update(file.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _looks_like_path(model_id: str) -> bool:
    return model_id.startswith(("/", "./", "../", "~")) or model_id.count("/") > 1


def resolve_model_source(model_id: str, revision: str | None = None) -> ModelSource:
    local = Path(model_id).expanduser()
    if not local.is_dir() and _looks_like_path(model_id):
        raise ModelSourceError(f"Model directory {local} does not exist.")
    if local.is_dir():
        return ModelSource(
            path=local,
            version=hash_directory(local),
            origin="local",
            requested_revision=revision,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # installed with transformers
        raise ModelSourceError("huggingface_hub is required to load Hub models.") from exc

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=list(HUB_ALLOW_PATTERNS),
                # Offline, use the local cache only; pin MODEL_REVISION to a commit hash.
                local_files_only=hub_offline(),
            )
        )
    except Exception as exc:  # network, auth, unknown repo or revision
        raise ModelSourceError(
            f"Could not download {model_id!r} at revision {revision or 'main'!r}: {exc}"
        ) from exc

    commit = snapshot.name
    if not _COMMIT_SHA.match(commit):
        raise ModelSourceError(
            f"Could not determine the commit hash for {model_id!r} (snapshot folder {snapshot})."
        )
    return ModelSource(path=snapshot, version=commit, origin="hub", requested_revision=revision)
