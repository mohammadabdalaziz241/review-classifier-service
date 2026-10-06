"""Request and response models. These define the public API contract."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        description="Review text to classify.", examples=["Brilliant, absolutely love it"]
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(
        min_length=1,
        description="Review texts to classify. Results are returned in the same order.",
        examples=[["Great value for money", "Arrived broken, awful packaging"]],
    )


class PredictionOut(BaseModel):
    id: uuid.UUID = Field(description="Identifies this prediction, e.g. to send feedback on it.")
    label: str
    score: float = Field(description="Probability of the predicted label.")
    scores: dict[str, float] = Field(description="Probability of every label.")


class ModelRef(BaseModel):
    id: str
    version: str


class PredictResponse(BaseModel):
    request_id: str
    model: ModelRef
    prediction: PredictionOut
    inference_ms: float
    recorded: bool = Field(description="Whether the prediction was stored in the database.")


class BatchPredictResponse(BaseModel):
    request_id: str
    model: ModelRef
    predictions: list[PredictionOut]
    inference_ms: float
    recorded: bool = Field(description="Whether the predictions were stored in the database.")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: uuid.UUID
    label: str = Field(min_length=1, description="The correct label, e.g. 'sarcastic'.")
    text: str | None = Field(
        default=None,
        description=(
            "Optional: the review text, to keep it as a labelled example. It must be exactly "
            "the text that was classified; only predictions' hashes are stored otherwise."
        ),
    )


class FeedbackResponse(BaseModel):
    id: int
    prediction_id: uuid.UUID
    label: str
    predicted_label: str
    model_was_correct: bool
    text_stored: bool


class Limits(BaseModel):
    max_text_chars: int
    max_batch_size: int
    max_concurrent_inferences: int | None = Field(
        description="Forward passes run at once; further requests wait. null: unlimited."
    )


class Runtime(BaseModel):
    inference_threads: int | None = Field(
        description="PyTorch threads per forward pass; null when torch is not loaded."
    )
    batch_requests: bool = Field(
        description="Requests waiting for the model share forward passes (dynamic batching)."
    )
    batch_max_texts: int | None = Field(description="Most texts per shared pass.")
    batch_wait_ms: int | None = Field(description="Time a pass may wait to collect texts.")


class Preprocessing(BaseModel):
    normalize_unicode: bool
    normalize_whitespace: bool
    replace_urls: bool
    replace_mentions: bool
    source: dict[str, str] = Field(
        description="Where each setting came from: env, checkpoint or default."
    )


class ModelInfoResponse(BaseModel):
    backend: str
    id: str
    version: str = Field(description="Hub commit hash or sha256 content hash of the weights.")
    origin: str
    requested_revision: str | None
    task: str = Field(description="Descriptive label from configuration or the checkpoint.")
    labels: list[str]
    device: str
    max_seq_length: int
    preprocessing: Preprocessing
    limits: Limits
    runtime: Runtime


class StatusResponse(BaseModel):
    status: str
    database: str | None = Field(
        default=None, description="ok, unavailable, or disabled (no DATABASE_URL)."
    )


class ErrorDetail(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None
