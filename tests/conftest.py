from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings


@pytest.fixture
def settings() -> Settings:
    # Small limits so limit tests stay readable.
    return Settings(model_backend="dummy", max_text_chars=200, max_batch_size=4)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # The context manager runs the lifespan, which loads the predictor.
    with TestClient(create_app(settings)) as test_client:
        yield test_client
