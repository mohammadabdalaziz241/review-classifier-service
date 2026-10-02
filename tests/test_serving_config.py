import json
import logging

import pytest

from review_classifier.serving_config import (
    BUILTIN_PREPROCESSING,
    RAW_TEXT,
    CheckpointServingConfig,
    ServingConfigError,
    read_serving_config,
    resolve_preprocessing,
    write_serving_config,
)

NO_OVERRIDES = dict.fromkeys(BUILTIN_PREPROCESSING.as_dict())


def _write(directory, payload):
    (directory / "serving_config.json").write_text(json.dumps(payload))


def test_round_trip(tmp_path):
    config = CheckpointServingConfig(
        task="sarcasm", max_seq_length=384, preprocessing=RAW_TEXT.as_dict()
    )
    write_serving_config(tmp_path, config)
    assert read_serving_config(tmp_path) == config


def test_missing_file_means_no_contract(tmp_path):
    assert read_serving_config(tmp_path) is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": 1, "task": ""}, "task"),
        ({"schema_version": 1, "max_seq_length": 0}, "max_seq_length"),
        ({"schema_version": 1, "max_seq_length": True}, "max_seq_length"),
        ({"schema_version": 1, "preprocessing": {"lowercase": True}}, "unknown preprocessing"),
        ({"schema_version": 1, "preprocessing": {"replace_urls": "no"}}, "true or false"),
        ([1, 2], "JSON object"),
    ],
)
def test_malformed_files_are_rejected(tmp_path, payload, message):
    _write(tmp_path, payload)
    with pytest.raises(ServingConfigError, match=message):
        read_serving_config(tmp_path)


def test_invalid_json_is_rejected(tmp_path):
    (tmp_path / "serving_config.json").write_text("{nope")
    with pytest.raises(ServingConfigError, match="not valid JSON"):
        read_serving_config(tmp_path)


def test_defaults_apply_without_checkpoint_or_env():
    config, source = resolve_preprocessing(NO_OVERRIDES, None)
    assert config == BUILTIN_PREPROCESSING
    assert set(source.values()) == {"default"}


def test_checkpoint_overrides_defaults():
    checkpoint = CheckpointServingConfig(None, None, RAW_TEXT.as_dict())
    config, source = resolve_preprocessing(NO_OVERRIDES, checkpoint)
    assert config == RAW_TEXT
    assert set(source.values()) == {"checkpoint"}


def test_partial_checkpoint_falls_back_to_defaults_per_key():
    checkpoint = CheckpointServingConfig(None, None, {"replace_urls": False})
    config, source = resolve_preprocessing(NO_OVERRIDES, checkpoint)
    assert config.replace_urls is False
    assert config.replace_mentions is True
    assert source["replace_urls"] == "checkpoint"
    assert source["replace_mentions"] == "default"


def test_env_overrides_checkpoint_and_warns(caplog):
    checkpoint = CheckpointServingConfig(None, None, RAW_TEXT.as_dict())
    overrides = NO_OVERRIDES | {"normalize_whitespace": True}
    with caplog.at_level(logging.WARNING, logger="review_classifier"):
        config, source = resolve_preprocessing(overrides, checkpoint)
    assert config.normalize_whitespace is True
    assert source["normalize_whitespace"] == "env"
    assert "PREPROCESS_NORMALIZE_WHITESPACE=true overrides the checkpoint" in caplog.text


def test_env_agreeing_with_checkpoint_does_not_warn(caplog):
    checkpoint = CheckpointServingConfig(None, None, RAW_TEXT.as_dict())
    with caplog.at_level(logging.WARNING, logger="review_classifier"):
        resolve_preprocessing(NO_OVERRIDES | {"replace_urls": False}, checkpoint)
    assert caplog.text == ""
