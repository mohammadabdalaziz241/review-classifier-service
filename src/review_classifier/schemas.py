"""Request and response models. These define the public API contract."""

from __future__ import annotations

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


class BatchPredictResponse(BaseModel):
    request_id: str
    model: ModelRef
    predictions: list[PredictionOut]
    inference_ms: float


class Limits(BaseModel):
    max_text_chars: int
    max_batch_size: int


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


class StatusResponse(BaseModel):
    status: str


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
