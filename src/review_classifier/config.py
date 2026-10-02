"""Service configuration, read from environment variables.

Every setting has a default so the service starts without any configuration,
and every value is validated at startup so a bad deployment fails fast instead
of misbehaving on the first request.

Settings that describe how a model must be served (task, truncation length,
preprocessing) default to ``None``, meaning "not set here": the checkpoint's
serving_config.json then decides (see serving_config.py).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .serving_config import PREPROCESSING_ENV, PREPROCESSING_KEYS

VALID_BACKENDS = ("hf", "dummy")
VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def _int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str) -> bool | None:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple[str, ...]) -> str:
    value = env.get(name, "").strip().lower() or default
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    # Which predictor to load: "hf" (transformers model) or "dummy" (keyword
    # baseline used for tests and for running the API without torch installed).
    model_backend: str = "hf"
    # Hugging Face Hub id or local directory of a sequence-classification model.
    model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Branch, tag or commit to download. The reported version is always the
    # resolved commit hash; pin a commit here so every deployment gets the same weights.
    model_revision: str | None = None
    device: str = "auto"

    # Where to record predictions and feedback, e.g.
    # postgresql://user:password@host:5432/db. Unset: nothing is recorded.
    # Excluded from repr so the password never ends up in logs or tracebacks.
    database_url: str | None = field(default=None, repr=False)

    # Request limits.
    max_text_chars: int = 2000
    max_batch_size: int = 32
    inference_batch_size: int = 16

    # Serving contract. None = use the checkpoint's serving_config.json, then a default.
    # The task is a descriptive label; it never changes what the model predicts.
    task: str | None = None
    max_seq_length: int | None = None
    normalize_unicode: bool | None = None
    normalize_whitespace: bool | None = None
    replace_urls: bool | None = None
    replace_mentions: bool | None = None

    def preprocessing_overrides(self) -> dict[str, bool | None]:
        return {key: getattr(self, key) for key in PREPROCESSING_KEYS}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            model_backend=_choice(env, "MODEL_BACKEND", defaults.model_backend, VALID_BACKENDS),
            model_id=env.get("MODEL_ID", "").strip() or defaults.model_id,
            model_revision=env.get("MODEL_REVISION", "").strip() or None,
            device=_choice(env, "DEVICE", defaults.device, VALID_DEVICES),
            database_url=env.get("DATABASE_URL", "").strip() or None,
            max_text_chars=_int(env, "MAX_TEXT_CHARS", defaults.max_text_chars),
            max_batch_size=_int(env, "MAX_BATCH_SIZE", defaults.max_batch_size),
            inference_batch_size=_int(env, "INFERENCE_BATCH_SIZE", defaults.inference_batch_size),
            task=env.get("MODEL_TASK", "").strip() or None,
            max_seq_length=_int(env, "MAX_SEQ_LENGTH", None),
            **{key: _bool(env, PREPROCESSING_ENV[key]) for key in PREPROCESSING_KEYS},
        )
