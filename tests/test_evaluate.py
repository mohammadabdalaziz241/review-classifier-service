"""Evaluation logic against fake predictors: no torch, no dataset download."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings
from review_classifier.evaluate import EvaluationError, evaluate_service
from review_classifier.predictors import Prediction

from .fakes import RecordingPredictor


class KeywordPredictor(RecordingPredictor):
    """Says "positive" if the text contains "good", else "negative"."""

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.seen.extend(texts)
        out = []
        for text in texts:
            label = "positive" if "good" in text else "negative"
            p = 0.9 if label == "positive" else 0.1
            out.append(Prediction(label, max(p, 1 - p), {"negative": 1 - p, "positive": p}))
        return out


TEXTS = ["good food", "good service", "bad food", "awful", "good?", "terrible"]
LABELS = [1, 1, 0, 0, 0, 0]  # "good?" is labelled negative, so one error
VARIETIES = ["en-UK", "en-AU", "en-UK", "en-IN", "en-IN", "en-AU"]


@pytest.fixture
def service():
    predictor = KeywordPredictor()
    settings = Settings(model_backend="dummy", max_batch_size=4, max_text_chars=50)
    with TestClient(create_app(settings, predictor=predictor)) as client:
        yield client, predictor


def test_computes_metrics_through_the_api(service):
    client, predictor = service
    report = evaluate_service(
        client, TEXTS, LABELS, expected_labels=("negative", "positive"), varieties=VARIETIES
    )
    assert report.n == 6
    assert report.metrics["accuracy"] == pytest.approx(5 / 6)
    assert report.metrics["confusion_matrix"] == [[3, 1], [0, 2]]
    assert set(report.metrics_by_variety) == {"en-AU", "en-IN", "en-UK"}
    assert report.metrics_by_variety["en-UK"]["accuracy"] == 1.0
    assert report.passed is None  # nothing to compare against
    # Every text went through the API, in batches no larger than the service allows.
    assert predictor.seen == TEXTS


def test_matching_training_metrics_pass(service):
    client, _ = service
    expected = {"accuracy": 5 / 6, "macro_f1": 0.8285714285714285}
    report = evaluate_service(client, TEXTS, LABELS, expected_metrics=expected)
    assert report.passed is True
    assert report.differences["macro_f1"] == pytest.approx(0, abs=1e-9)


def test_different_training_metrics_fail(service):
    client, _ = service
    expected = {"accuracy": 1.0, "macro_f1": 1.0}
    report = evaluate_service(client, TEXTS, LABELS, expected_metrics=expected)
    assert report.passed is False
    assert report.differences["accuracy"] == pytest.approx(5 / 6 - 1)
    assert report.to_json()["passed"] is False


def test_wrong_label_order_is_caught(service):
    client, _ = service
    with pytest.raises(EvaluationError, match="same order"):
        evaluate_service(client, TEXTS, LABELS, expected_labels=("positive", "negative"))


def test_texts_over_the_service_limit_are_reported_up_front(service):
    client, predictor = service
    with pytest.raises(EvaluationError, match="MAX_TEXT_CHARS=50"):
        evaluate_service(client, [*TEXTS, "x" * 51], [*LABELS, 0])
    assert predictor.seen == []  # failed before sending anything


def test_reports_texts_changed_by_preprocessing(service):
    client, _ = service
    texts = ["good food", "good  food", "@someone good", "bad"]
    report = evaluate_service(client, texts, [1, 1, 1, 0])
    # Default preprocessing collapses spaces and replaces mentions.
    assert report.changed_by_preprocessing == [1, 2]
