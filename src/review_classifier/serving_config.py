"""The serving contract a checkpoint carries with it.

A checkpoint trained by ``review_classifier.train`` includes a
``serving_config.json`` that records how it must be served: its task, its
truncation length and the text preprocessing used during training. The service
applies it automatically, so the serving pipeline matches training without
anyone having to copy settings by hand.

Precedence, per setting: environment variable > checkpoint > built-in default.
An environment variable that contradicts the checkpoint is allowed (useful for
experiments) but logged as a warning, because predictions will then no longer
be comparable with the training metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger("review_classifier")

SERVING_CONFIG_FILE = "serving_config.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    replace_urls: bool = True
    replace_mentions: bool = True

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


# Used when neither the environment nor the checkpoint says anything: the
# convention of Twitter-trained RoBERTa models such as the default model.
BUILTIN_PREPROCESSING = PreprocessingConfig()
# Raw text, as passed to the tokenizer in the project notebook.
RAW_TEXT = PreprocessingConfig(
    normalize_unicode=False, normalize_whitespace=False, replace_urls=False, replace_mentions=False
)
PREPROCESSING_KEYS = tuple(f.name for f in fields(PreprocessingConfig))
PREPROCESSING_ENV = {key: f"PREPROCESS_{key.upper()}" for key in PREPROCESSING_KEYS}


class ServingConfigError(ValueError):
    """The checkpoint's serving_config.json is malformed."""


@dataclass(frozen=True)
class CheckpointServingConfig:
    task: str | None
    max_seq_length: int | None
    preprocessing: dict[str, bool]  # only the keys the checkpoint specifies

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "max_seq_length": self.max_seq_length,
            "preprocessing": dict(self.preprocessing),
        }


def write_serving_config(directory: Path, config: CheckpointServingConfig) -> Path:
    path = Path(directory) / SERVING_CONFIG_FILE
    path.write_text(json.dumps(config.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def read_serving_config(directory: Path) -> CheckpointServingConfig | None:
    path = Path(directory) / SERVING_CONFIG_FILE
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServingConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ServingConfigError(f"{path} must contain a JSON object")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ServingConfigError(f"{path}: unsupported schema_version {version!r}")

    task = raw.get("task")
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ServingConfigError(f"{path}: 'task' must be a non-empty string")

    max_seq_length = raw.get("max_seq_length")
    valid_length = (
        isinstance(max_seq_length, int)
        and not isinstance(max_seq_length, bool)
        and max_seq_length > 0
    )
    if max_seq_length is not None and not valid_length:
        raise ServingConfigError(f"{path}: 'max_seq_length' must be a positive integer")

    preprocessing = raw.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        raise ServingConfigError(f"{path}: 'preprocessing' must be an object")
    unknown = set(preprocessing) - set(PREPROCESSING_KEYS)
    if unknown:
        raise ServingConfigError(
            f"{path}: unknown preprocessing keys {sorted(unknown)}; "
            f"expected a subset of {list(PREPROCESSING_KEYS)}"
        )
    for key, value in preprocessing.items():
        if not isinstance(value, bool):
            raise ServingConfigError(f"{path}: preprocessing.{key} must be true or false")

    return CheckpointServingConfig(
        task=task, max_seq_length=max_seq_length, preprocessing=dict(preprocessing)
    )


def resolve_preprocessing(
    overrides: dict[str, bool | None], checkpoint: CheckpointServingConfig | None
) -> tuple[PreprocessingConfig, dict[str, str]]:
    """Combine environment overrides, the checkpoint and defaults.

    Returns the effective configuration and, per setting, where it came from:
    ``"env"``, ``"checkpoint"`` or ``"default"``.
    """
    from_checkpoint = checkpoint.preprocessing if checkpoint else {}
    values: dict[str, bool] = {}
    sources: dict[str, str] = {}
    for key in PREPROCESSING_KEYS:
        env_value = overrides.get(key)
        if env_value is not None:
            values[key], sources[key] = env_value, "env"
            trained = from_checkpoint.get(key)
            if trained is not None and trained != env_value:
                logger.warning(
                    "%s=%s overrides the checkpoint, which was trained with %s=%s; "
                    "predictions will not be comparable with its training metrics.",
                    PREPROCESSING_ENV[key],
                    str(env_value).lower(),
                    key,
                    str(trained).lower(),
                )
        elif key in from_checkpoint:
            values[key], sources[key] = from_checkpoint[key], "checkpoint"
        else:
            values[key], sources[key] = getattr(BUILTIN_PREPROCESSING, key), "default"
    return PreprocessingConfig(**values), sources
