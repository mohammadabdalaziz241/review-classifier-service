"""HTTP layer: routes, validation, error envelope, request IDs and access logs.

Run with:
    uvicorn review_classifier.api:create_app --factory
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import Settings
from .predictors import Prediction, Predictor, create_predictor
from .preprocessing import preprocess
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    Limits,
    ModelInfoResponse,
    ModelRef,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    Preprocessing,
    StatusResponse,
)
from .serving_config import resolve_preprocessing

logger = logging.getLogger("review_classifier")

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SCORE_DECIMALS = 6


class APIError(Exception):
    """An error with a stable machine-readable code, rendered as the error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse.model_validate(
        {
            "error": {"code": code, "message": message, "details": details or []},
            "request_id": getattr(request.state, "request_id", None),
        }
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def _validation_details(errors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep only loc/msg/type: pydantic's raw errors can echo the input back and
    # contain objects that are not JSON-serialisable.
    return [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in errors
    ]


def _out(prediction: Prediction) -> PredictionOut:
    return PredictionOut(
        label=prediction.label,
        score=round(prediction.score, _SCORE_DECIMALS),
        scores={k: round(v, _SCORE_DECIMALS) for k, v in prediction.scores.items()},
    )


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Build the application.

    ``predictor`` can be injected (tests do this); otherwise it is created from
    ``settings`` at startup. A model that fails to load stops the service from
    starting, so an orchestrator sees a failed deployment rather than a running
    container that answers every request with an error.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        started = time.perf_counter()
        app.state.predictor = predictor or create_predictor(settings)
        info = app.state.predictor.info
        # Preprocessing must match training: env overrides, else the checkpoint's
        # serving_config.json, else the built-in default.
        app.state.preprocessing, app.state.preprocessing_source = resolve_preprocessing(
            settings.preprocessing_overrides(), info.serving_config
        )
        logger.info(
            "model loaded backend=%s id=%s version=%s labels=%s device=%s in %.1fs",
            info.backend,
            info.model_id,
            info.version,
            ",".join(info.labels),
            info.device,
            time.perf_counter() - started,
        )
        logger.info(
            "serving task=%s max_seq_length=%d preprocessing=%s",
            info.task,
            info.max_seq_length,
            " ".join(
                f"{key}={str(value).lower()}({app.state.preprocessing_source[key]})"
                for key, value in app.state.preprocessing.as_dict().items()
            ),
        )
        if info.serving_config is None and info.backend == "hf":
            logger.warning(
                "The checkpoint has no serving_config.json, so built-in preprocessing defaults "
                "apply. Confirm they match how the model was trained."
            )
        if info.backend == "dummy":
            logger.warning("Running the dummy keyword backend: predictions are not a real model.")
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Review Classifier Service",
        version=__version__,
        description="Sentiment and sarcasm classification for English reviews.",
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    app.state.settings = settings
    app.state.predictor = None
    app.state.preprocessing, app.state.preprocessing_source = resolve_preprocessing(
        settings.preprocessing_overrides(), None
    )

    # ---- middleware ---------------------------------------------------------

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log the traceback, but never send internal details to the client.
            logger.exception("unhandled error request_id=%s", request_id)
            response = _error_response(
                request, 500, "internal_error", "An unexpected error occurred."
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        # Request text is deliberately not logged: reviews can contain personal data.
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ---- error handlers -----------------------------------------------------

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return _error_response(
            request,
            422,
            "invalid_request",
            "The request body is invalid.",
            _validation_details(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error_response(
            request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None)
        )

    # ---- helpers ------------------------------------------------------------

    def get_predictor(request: Request) -> Predictor:
        current = request.app.state.predictor
        if current is None:
            raise APIError(503, "model_not_ready", "The model is not loaded yet.")
        return current

    def prepare(request: Request, texts: Sequence[str], field: str, indexed: bool) -> list[str]:
        """Preprocess texts and enforce limits, reporting every bad item at once."""
        config = request.app.state.preprocessing
        cleaned: list[str] = []
        details: list[dict[str, Any]] = []
        for i, raw in enumerate(texts):
            loc: list[str | int] = ["body", field, i] if indexed else ["body", field]
            if len(raw) > settings.max_text_chars:
                details.append(
                    {
                        "loc": loc,
                        "msg": f"Text is longer than {settings.max_text_chars} characters.",
                        "type": "text_too_long",
                    }
                )
                continue
            text = preprocess(raw, config)
            if not text.strip():
                details.append(
                    {"loc": loc, "msg": "Text is empty after normalisation.", "type": "text_empty"}
                )
                continue
            cleaned.append(text)
        if details:
            raise APIError(422, "invalid_request", "The request body is invalid.", details)
        return cleaned

    def model_ref(model: Predictor) -> ModelRef:
        return ModelRef(id=model.info.model_id, version=model.info.version)

    def run(model: Predictor, texts: list[str]) -> tuple[list[Prediction], float]:
        started = time.perf_counter()
        predictions = model.predict(texts)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if len(predictions) != len(texts):
            raise RuntimeError(f"Predictor returned {len(predictions)} results for {len(texts)}")
        return predictions, elapsed_ms

    # ---- routes -------------------------------------------------------------
    # Inference routes are sync functions: FastAPI runs them in a worker thread,
    # so a slow forward pass does not block the event loop.

    @app.get("/health", response_model=StatusResponse, tags=["ops"])
    def health() -> StatusResponse:
        """Liveness: the process is up and serving HTTP."""
        return StatusResponse(status="ok")

    @app.get("/ready", response_model=StatusResponse, tags=["ops"])
    def ready(request: Request) -> StatusResponse:
        """Readiness: the model is loaded and requests can be served."""
        get_predictor(request)
        return StatusResponse(status="ready")

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request) -> ModelInfoResponse:
        info = get_predictor(request).info
        return ModelInfoResponse(
            backend=info.backend,
            id=info.model_id,
            version=info.version,
            origin=info.origin,
            requested_revision=info.requested_revision,
            task=info.task,
            labels=list(info.labels),
            device=info.device,
            max_seq_length=info.max_seq_length,
            preprocessing=Preprocessing(
                **request.app.state.preprocessing.as_dict(),
                source=request.app.state.preprocessing_source,
            ),
            limits=Limits(
                max_text_chars=settings.max_text_chars, max_batch_size=settings.max_batch_size
            ),
        )

    @app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model = get_predictor(request)
        texts = prepare(request, [body.text], "text", indexed=False)
        predictions, elapsed_ms = run(model, texts)
        return PredictResponse(
            request_id=request.state.request_id,
            model=model_ref(model),
            prediction=_out(predictions[0]),
            inference_ms=elapsed_ms,
        )

    @app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
    def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        model = get_predictor(request)
        if len(body.texts) > settings.max_batch_size:
            raise APIError(
                422,
                "batch_too_large",
                f"A batch may contain at most {settings.max_batch_size} texts.",
                [
                    {
                        "loc": ["body", "texts"],
                        "msg": f"Got {len(body.texts)} texts; limit is {settings.max_batch_size}.",
                        "type": "batch_too_large",
                    }
                ],
            )
        texts = prepare(request, body.texts, "texts", indexed=True)
        predictions, elapsed_ms = run(model, texts)
        return BatchPredictResponse(
            request_id=request.state.request_id,
            model=model_ref(model),
            predictions=[_out(p) for p in predictions],
            inference_ms=elapsed_ms,
        )

    return app
