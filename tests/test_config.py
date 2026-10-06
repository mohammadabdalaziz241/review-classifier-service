import pytest

from review_classifier.config import Settings


def test_defaults_without_environment():
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.model_backend == "hf"


def test_reads_environment():
    settings = Settings.from_env(
        {
            "MODEL_BACKEND": "DUMMY",
            "MODEL_ID": "user/my-model",
            "MODEL_REVISION": "abc123",
            "MODEL_TASK": "sarcasm",
            "DEVICE": "cpu",
            "MAX_BATCH_SIZE": "8",
            "MAX_TEXT_CHARS": "500",
            "PREPROCESS_REPLACE_URLS": "false",
            "PREPROCESS_REPLACE_MENTIONS": "0",
            "PREPROCESS_NORMALIZE_UNICODE": "no",
            "PREPROCESS_NORMALIZE_WHITESPACE": "on",
            "MAX_SEQ_LENGTH": "256",
        }
    )
    assert settings.model_backend == "dummy"
    assert settings.model_id == "user/my-model"
    assert settings.model_revision == "abc123"
    assert settings.task == "sarcasm"
    assert settings.max_batch_size == 8
    assert settings.max_text_chars == 500
    assert settings.replace_urls is False
    assert settings.replace_mentions is False
    assert settings.normalize_unicode is False
    assert settings.normalize_whitespace is True
    assert settings.max_seq_length == 256


@pytest.mark.parametrize(("raw", "expected"), [("", 1), ("2", 2), ("0", None), ("unlimited", None)])
def test_inference_concurrency(raw, expected):
    env = {"MAX_CONCURRENT_INFERENCES": raw, "BATCH_REQUESTS": "false"}
    assert Settings.from_env(env).max_concurrent_inferences == expected


def test_inference_runtime_settings():
    defaults = Settings.from_env({})
    assert defaults.inference_threads is None
    assert defaults.batch_requests is True
    assert (defaults.batch_max_texts, defaults.batch_wait_ms) == (16, 0)
    settings = Settings.from_env(
        {
            "INFERENCE_THREADS": "2",
            "BATCH_REQUESTS": "true",
            "BATCH_MAX_TEXTS": "8",
            "BATCH_WAIT_MS": "0",
        }
    )
    assert settings.inference_threads == 2
    assert settings.batch_requests is True
    assert (settings.batch_max_texts, settings.batch_wait_ms) == (8, 0)
    assert Settings.from_env({"BATCH_WAIT_MS": "5"}).batch_wait_ms == 5


def test_metrics_port():
    assert Settings.from_env({}).metrics_port is None
    assert Settings.from_env({"METRICS_PORT": "9000"}).metrics_port == 9000


def test_serving_contract_is_unset_by_default():
    # None means "let the checkpoint decide", so nothing is forced on it.
    settings = Settings.from_env({})
    assert settings.task is None
    assert settings.max_seq_length is None
    assert set(settings.preprocessing_overrides().values()) == {None}


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"MAX_BATCH_SIZE": "lots"}, "must be an integer"),
        ({"MAX_BATCH_SIZE": "0"}, "positive"),
        ({"MODEL_BACKEND": "onnx"}, "MODEL_BACKEND must be one of"),
        ({"DEVICE": "tpu"}, "DEVICE must be one of"),
        ({"PREPROCESS_REPLACE_URLS": "maybe"}, "must be true or false"),
        ({"METRICS_PORT": "70000"}, "must be a TCP port"),
        ({"MAX_CONCURRENT_INFERENCES": "-1"}, "positive"),
        ({"INFERENCE_THREADS": "0"}, "positive"),
        ({"BATCH_REQUESTS": "sometimes"}, "must be true or false"),
        ({"BATCH_MAX_TEXTS": "0"}, "positive"),
        ({"BATCH_WAIT_MS": "-5"}, "positive"),
        (
            {"BATCH_REQUESTS": "true", "MAX_CONCURRENT_INFERENCES": "unlimited"},
            "BATCH_REQUESTS needs a limit",
        ),
    ],
)
def test_invalid_values_fail_fast(env, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env(env)
