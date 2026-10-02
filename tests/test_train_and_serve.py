"""Train -> serve -> evaluate, end to end, offline.

A tiny BERT stands in for roberta-base and a synthetic dataset with the
BESSTIE-CW-26 columns stands in for the real data, so the real training,
serving and evaluation code paths run in seconds on a CPU without downloads.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings
from review_classifier.evaluate import evaluate_service
from review_classifier.predictors import HFPredictor, ModelLoadError
from review_classifier.serving_config import RAW_TEXT

from .tiny_model import build_tiny_model

pytestmark = pytest.mark.hf

POSITIVE = ["great product", "love it", "great service", "love the food", "great food"]
NEGATIVE = ["bad product", "awful service", "bad food", "the food was awful", "awful it was"]
VARIETIES = ["en-UK", "en-AU", "en-IN"]


def _rows(n_repeats: int, offset: int) -> dict[str, list]:
    rows = {"text": [], "Sentiment": [], "Sarcasm": [], "variety": [], "source": []}
    for r in range(n_repeats):
        for i, text in enumerate(POSITIVE + NEGATIVE):
            positive = i < len(POSITIVE)
            # Raw text on purpose: double spaces and a trailing newline.
            rows["text"].append(f"{text}  !\n" if (i + r) % 3 == 0 else text)
            rows["Sentiment"].append(int(positive))
            rows["Sarcasm"].append(int((i + r + offset) % 4 == 0))
            rows["variety"].append(VARIETIES[(i + r + offset) % 3])
            rows["source"].append("Google")
    return rows


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    datasets = pytest.importorskip("datasets")
    directory = tmp_path_factory.mktemp("besstie-like")
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(_rows(4, 0)),
            "validation": datasets.Dataset.from_dict(_rows(2, 1)),
            "test": datasets.Dataset.from_dict(_rows(2, 2)),
        }
    ).save_to_disk(str(directory))
    return directory


@pytest.fixture(scope="module")
def base_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-base")
    build_tiny_model(directory)
    return directory


@pytest.fixture(scope="module")
def trained(tmp_path_factory, dataset_dir, base_model_dir):
    pytest.importorskip("accelerate")
    from review_classifier.train import Recipe, train

    output = tmp_path_factory.mktemp("out") / "roberta-sentiment"
    manifest = train(
        task="sentiment",
        output=output,
        dataset=str(dataset_dir),
        base_model=str(base_model_dir),
        max_length=24,
        recipe=Recipe(epochs=20, learning_rate=5e-3, train_batch_size=8, eval_batch_size=8),
        log=lambda *_: None,
    )
    return output, manifest


def test_checkpoint_contains_weights_contract_and_manifest(trained):
    output, manifest = trained
    for name in ("config.json", "serving_config.json", "training_manifest.json", "README.md"):
        assert (output / name).is_file(), name
    assert any(output.glob("*.safetensors"))

    serving = json.loads((output / "serving_config.json").read_text())
    assert serving == {
        "schema_version": 1,
        "task": "sentiment",
        "max_seq_length": 24,
        "preprocessing": RAW_TEXT.as_dict(),
    }

    config = json.loads((output / "config.json").read_text())
    assert config["id2label"] == {"0": "negative", "1": "positive"}

    assert manifest["dataset"]["version"].startswith("sha256:")
    assert manifest["base_model"]["version"].startswith("sha256:")
    assert manifest["dataset"]["sizes"] == {"train": 40, "validation": 20, "test": 20}
    assert manifest["recipe"]["preprocessing"] == "none (raw text to tokenizer)"
    assert set(manifest["test_metrics_by_variety"]) == set(VARIETIES)
    # Scratch checkpoints are cleaned up.
    assert not (output.parent / f".{output.name}-runs").exists()


def test_service_applies_the_checkpoint_contract(trained):
    output, _ = trained
    settings = Settings(model_backend="hf", model_id=str(output), device="cpu")
    with TestClient(create_app(settings)) as client:
        info = client.get("/v1/model").json()
    assert info["task"] == "sentiment"
    assert info["labels"] == ["negative", "positive"]
    assert info["max_seq_length"] == 24
    assert info["preprocessing"]["source"] == dict.fromkeys(RAW_TEXT.as_dict(), "checkpoint")
    assert all(info["preprocessing"][k] is False for k in RAW_TEXT.as_dict())


def test_served_predictions_reproduce_training_metrics(trained, dataset_dir):
    import datasets

    output, manifest = trained
    test = datasets.load_from_disk(str(dataset_dir))["test"]
    settings = Settings(model_backend="hf", model_id=str(output), device="cpu")
    with TestClient(create_app(settings)) as client:
        report = evaluate_service(
            client,
            list(test["text"]),
            [int(v) for v in test["Sentiment"]],
            expected_labels=("negative", "positive"),
            varieties=list(test["variety"]),
            expected_metrics=manifest["test_metrics"],
            tolerance=1e-9,
        )
    assert report.passed, report.differences
    # The parity check means something only if the model predicts both classes.
    predicted_per_class = [
        sum(col) for col in zip(*report.metrics["confusion_matrix"], strict=True)
    ]
    assert all(count > 0 for count in predicted_per_class), predicted_per_class
    assert report.changed_by_preprocessing == []  # raw text reaches the model unchanged
    assert report.metrics["confusion_matrix"] == manifest["test_metrics"]["confusion_matrix"]


def test_preprocessing_mismatch_is_visible_in_evaluation(trained, dataset_dir, caplog):
    import datasets

    output, _ = trained
    test = datasets.load_from_disk(str(dataset_dir))["test"]
    settings = Settings(
        model_backend="hf", model_id=str(output), device="cpu", normalize_whitespace=True
    )
    with (
        caplog.at_level(logging.WARNING, logger="review_classifier"),
        TestClient(create_app(settings)) as client,
    ):
        report = evaluate_service(client, list(test["text"]), [int(v) for v in test["Sentiment"]])
    assert "overrides the checkpoint" in caplog.text
    assert len(report.changed_by_preprocessing) > 0


def test_overriding_truncation_length_warns(trained, caplog):
    output, _ = trained
    with caplog.at_level(logging.WARNING, logger="review_classifier"):
        predictor = HFPredictor(str(output), device="cpu", max_seq_length=16)
    assert predictor.info.max_seq_length == 16
    assert "MAX_SEQ_LENGTH=16 overrides the checkpoint" in caplog.text


def test_malformed_serving_config_stops_the_service(trained, tmp_path):
    import shutil

    output, _ = trained
    broken = tmp_path / "broken"
    shutil.copytree(output, broken)
    (broken / "serving_config.json").write_text('{"schema_version": 1, "max_seq_length": -1}')
    with pytest.raises(ModelLoadError, match="max_seq_length"):
        HFPredictor(str(broken), device="cpu")


def test_refuses_to_overwrite_existing_output(trained, dataset_dir, base_model_dir):
    from review_classifier.train import train

    output, _ = trained
    with pytest.raises(FileExistsError, match="--overwrite"):
        train(
            task="sentiment",
            output=output,
            dataset=str(dataset_dir),
            base_model=str(base_model_dir),
        )
