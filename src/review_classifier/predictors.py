"""Model backends.

The API depends only on the ``Predictor`` protocol, so the transformers model
can be swapped for a fake in tests, or for another backend (e.g. an ONNX or
quantised model) later, without touching the HTTP layer.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .model_source import ModelSourceError, resolve_model_source
from .serving_config import CheckpointServingConfig, ServingConfigError, read_serving_config

logger = logging.getLogger("review_classifier")

UNSPECIFIED_TASK = "unspecified"
# Truncation length when neither the settings, the checkpoint nor the tokenizer gives one.
FALLBACK_MAX_SEQ_LENGTH = 512


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded. The service refuses to start."""


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    model_id: str
    # Immutable identifier of the loaded weights: a Hub commit hash or a
    # "sha256:..." content hash. Never a branch name.
    version: str
    origin: str  # "hub", "local" or "builtin"
    requested_revision: str | None
    task: str
    labels: tuple[str, ...]
    device: str
    max_seq_length: int
    # The checkpoint's own serving contract, if it ships one.
    serving_config: CheckpointServingConfig | None = None


class Predictor(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _to_prediction(labels: Sequence[str], probs: Sequence[float]) -> Prediction:
    scores = dict(zip(labels, probs, strict=True))
    best = max(range(len(labels)), key=lambda i: probs[i])
    return Prediction(label=labels[best], score=probs[best], scores=scores)


class DummyPredictor:
    """Deterministic keyword baseline.

    It exists so the API, its validation and its tests can run without torch or
    a model download. It is not a sentiment model and must not be deployed.
    """

    LABELS = ("negative", "neutral", "positive")
    _POSITIVE = frozenset({"good", "great", "love", "excellent", "amazing", "happy", "best"})
    _NEGATIVE = frozenset({"bad", "terrible", "hate", "awful", "worst", "broken", "poor"})
    _TOKEN_RE = re.compile(r"[a-z']+")

    def __init__(self, task: str | None = None, max_seq_length: int | None = None) -> None:
        self._info = ModelInfo(
            backend="dummy",
            model_id="keyword-baseline",
            version="dummy-1",
            origin="builtin",
            requested_revision=None,
            task=task or "sentiment",
            labels=self.LABELS,
            device="cpu",
            max_seq_length=max_seq_length or FALLBACK_MAX_SEQ_LENGTH,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        results = []
        for text in texts:
            tokens = self._TOKEN_RE.findall(text.lower())
            pos = sum(t in self._POSITIVE for t in tokens)
            neg = sum(t in self._NEGATIVE for t in tokens)
            # Logits: neutral wins when there is no signal either way.
            logits = [float(neg), 0.5, float(pos)]
            results.append(_to_prediction(self.LABELS, _softmax(logits)))
        return results


class HFPredictor:
    """Single-label sequence classifier loaded with Hugging Face transformers."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        task: str | None = None,
        device: str = "auto",
        max_seq_length: int | None = None,
        batch_size: int = 16,
    ) -> None:
        """Load a checkpoint.

        ``task`` and ``max_seq_length`` override the checkpoint's
        serving_config.json; leave them as None to use what the checkpoint declares.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "The 'hf' backend needs torch and transformers. "
                "Install them with: pip install '.[hf]'"
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(device)
        self._batch_size = batch_size

        try:
            source = resolve_model_source(model_id, revision)
        except ModelSourceError as exc:
            raise ModelLoadError(str(exc)) from exc

        # Load from the resolved folder, never by name, so the files loaded are
        # exactly the ones the reported version identifies.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(source.path)
            self._model = AutoModelForSequenceClassification.from_pretrained(source.path)
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"Could not load model {model_id!r}: {exc}") from exc

        try:
            serving = read_serving_config(source.path)
        except ServingConfigError as exc:
            raise ModelLoadError(str(exc)) from exc

        config = self._model.config
        self._max_seq_length = (
            max_seq_length
            or (serving.max_seq_length if serving else None)
            or self._tokenizer_limit(config)
        )
        trained_length = serving.max_seq_length if serving else None
        if max_seq_length and trained_length and max_seq_length != trained_length:
            logger.warning(
                "MAX_SEQ_LENGTH=%d overrides the checkpoint, which was trained with "
                "max_seq_length=%d; predictions will not be comparable with its "
                "training metrics.",
                max_seq_length,
                trained_length,
            )
        if config.problem_type == "multi_label_classification":
            raise ModelLoadError(
                f"{model_id!r} is a multi-label model; this service expects single-label "
                "classification (softmax over mutually exclusive labels)."
            )

        self._model.to(self._device)
        self._model.eval()

        labels = tuple(config.id2label[i] for i in range(config.num_labels))
        self._info = ModelInfo(
            backend="hf",
            model_id=model_id,
            version=source.version,
            origin=source.origin,
            requested_revision=source.requested_revision,
            task=task or (serving.task if serving else None) or UNSPECIFIED_TASK,
            labels=labels,
            device=str(self._device),
            max_seq_length=self._max_seq_length,
            serving_config=serving,
        )

    def _tokenizer_limit(self, config) -> int:
        # Tokenizers without a configured limit report a huge sentinel value.
        limit = self._tokenizer.model_max_length
        if not isinstance(limit, int) or limit > 100_000:
            limit = FALLBACK_MAX_SEQ_LENGTH
        positions = getattr(config, "max_position_embeddings", None)
        return min(limit, positions) if positions else limit

    def _resolve_device(self, device: str):
        torch = self._torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("DEVICE=cuda was requested but CUDA is not available.")
        return torch.device(device)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        torch = self._torch
        results: list[Prediction] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self._max_seq_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            results.extend(_to_prediction(self._info.labels, row) for row in probs)
        return results


def create_predictor(settings: Settings) -> Predictor:
    if settings.model_backend == "dummy":
        return DummyPredictor(task=settings.task, max_seq_length=settings.max_seq_length)
    if settings.model_backend == "hf":
        return HFPredictor(
            settings.model_id,
            revision=settings.model_revision,
            task=settings.task,
            device=settings.device,
            max_seq_length=settings.max_seq_length,
            batch_size=settings.inference_batch_size,
        )
    raise ValueError(f"Unknown model backend: {settings.model_backend!r}")
