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


def _concurrency(env: Mapping[str, str], default: int | None) -> int | None:
    """MAX_CONCURRENT_INFERENCES: a positive integer, or 0 / "unlimited" for no limit."""
    raw = env.get("MAX_CONCURRENT_INFERENCES", "").strip().lower()
    if raw in ("0", "unlimited"):
        return None
    return _int(env, "MAX_CONCURRENT_INFERENCES", default)


def _bool_or(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _bool(env, name)
    return default if value is None else value


def _non_negative(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if raw == "0":
        return 0
    value = _int(env, name, default)
    return default if value is None else value


def _port(env: Mapping[str, str], name: str) -> int | None:
    value = _int(env, name, None)
    if value is not None and value > 65535:
        raise ValueError(f"{name} must be a TCP port (1-65535), got {value}")
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
    # Forward passes allowed at once; further requests wait. On a CPU, concurrent
    # passes compete for the same cores, so 1 keeps latency predictable. None: unlimited.
    max_concurrent_inferences: int | None = 1
    # PyTorch threads per forward pass. None: PyTorch's default, one per physical core
    # (on a 2-vCPU cloud instance with hyperthreading, that is 1).
    inference_threads: int | None = None
    # Dynamic batching: requests that arrive while the model is busy share its next
    # forward pass, up to batch_max_texts texts (see batching.py). On by default: on the
    # AWS instance it raised throughput under load by 79% at no cost to a lone request.
    batch_requests: bool = True
    batch_max_texts: int = 16
    # Milliseconds a pass may wait to collect more texts; 0 never waits.
    batch_wait_ms: int = 0

    # Serve Prometheus metrics on this port instead of at /metrics on the API port,
    # so they can stay off the public interface. None: /metrics on the API.
    metrics_port: int | None = None

    # Serving contract. None = use the checkpoint's serving_config.json, then a default.
    # The task is a descriptive label; it never changes what the model predicts.
    task: str | None = None
    max_seq_length: int | None = None
    normalize_unicode: bool | None = None
    normalize_whitespace: bool | None = None
    replace_urls: bool | None = None
    replace_mentions: bool | None = None

    def __post_init__(self) -> None:
        if self.batch_requests and self.max_concurrent_inferences is None:
            raise ValueError(
                "BATCH_REQUESTS needs a limit on concurrent passes: set "
                "MAX_CONCURRENT_INFERENCES to 1 or more (the batcher runs that many passes), "
                "or BATCH_REQUESTS=false."
            )

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
            max_concurrent_inferences=_concurrency(env, defaults.max_concurrent_inferences),
            inference_threads=_int(env, "INFERENCE_THREADS", defaults.inference_threads),
            batch_requests=_bool_or(env, "BATCH_REQUESTS", defaults.batch_requests),
            batch_max_texts=_int(env, "BATCH_MAX_TEXTS", defaults.batch_max_texts),
            batch_wait_ms=_non_negative(env, "BATCH_WAIT_MS", defaults.batch_wait_ms),
            metrics_port=_port(env, "METRICS_PORT"),
            task=env.get("MODEL_TASK", "").strip() or None,
            max_seq_length=_int(env, "MAX_SEQ_LENGTH", None),
            **{key: _bool(env, PREPROCESSING_ENV[key]) for key in PREPROCESSING_KEYS},
        )
