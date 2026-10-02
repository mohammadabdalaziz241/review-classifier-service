"""Fake predictors for tests."""

from __future__ import annotations

from collections.abc import Sequence

from review_classifier.predictors import DummyPredictor, ModelInfo, Prediction


class FailingPredictor(DummyPredictor):
    """Simulates a model that raises during inference."""

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        raise RuntimeError("CUDA out of memory (secret internal detail)")


class RecordingPredictor:
    """Records exactly what text reaches the model."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self._info = ModelInfo(
            backend="fake",
            model_id="recording",
            version="test-1",
            origin="builtin",
            requested_revision=None,
            task="sentiment",
            labels=("negative", "positive"),
            device="cpu",
            max_seq_length=16,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.seen.extend(texts)
        return [
            Prediction(label="positive", score=0.9, scores={"negative": 0.1, "positive": 0.9})
            for _ in texts
        ]
