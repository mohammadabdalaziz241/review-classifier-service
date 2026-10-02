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
from typing import TYPE_CHECKING, Any

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
    FeedbackRequest,
    FeedbackResponse,
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

if TYPE_CHECKING:  # the database layer is optional: pip install '.[db]'
    from .store import PredictionStore

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


def _out(prediction: Prediction, prediction_id: uuid.UUID) -> PredictionOut:
    return PredictionOut(
        id=prediction_id,
        label=prediction.label,
        score=round(prediction.score, _SCORE_DECIMALS),
        scores={k: round(v, _SCORE_DECIMALS) for k, v in prediction.scores.items()},
    )


def _open_store(settings: Settings) -> PredictionStore | None:
    if not settings.database_url:
        return None
    try:
        from .store import PredictionStore
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set but the database extra is not installed: pip install '.[db]'"
        ) from exc
    return PredictionStore(settings.database_url)


def create_app(
    settings: Settings | None = None,
    predictor: Predictor | None = None,
    store: PredictionStore | None = None,
) -> FastAPI:
    """Build the application.

    ``predictor`` and ``store`` can be injected (tests do this); otherwise they
    are created from ``settings`` at startup. A model that fails to load, or a
    database whose schema does not match this code, stops the service from
    starting, so an orchestrator sees a failed deployment rather than a running
    container that answers every request with an error. An unreachable database
    does not: predictions are still served, just not recorded.
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

        app.state.store = store or _open_store(settings)
        if app.state.store is None:
            logger.info("DATABASE_URL not set: predictions are not recorded")
        else:
            app.state.store.check_schema()
        try:
            yield
        finally:
            if app.state.store is not None:
                app.state.store.close()
            app.state.store = None
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
    app.state.store = None
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

    def record(
        request: Request,
        model: Predictor,
        endpoint: str,
        raw_texts: Sequence[str],
        predictions: Sequence[Prediction],
        elapsed_ms: float,
    ) -> tuple[list[uuid.UUID], bool]:
        """Give each prediction an id and store it if a database is configured."""
        ids = [uuid.uuid4() for _ in predictions]
        current_store = request.app.state.store
        if current_store is None:
            return ids, False
        from .store import PredictionRecord, text_sha256

        info = model.info
        rows = [
            PredictionRecord(
                id=prediction_id,
                request_id=request.state.request_id,
                endpoint=endpoint,
                batch_index=i,
                batch_size=len(predictions),
                model_id=info.model_id,
                model_version=info.version,
                task=info.task,
                label=p.label,
                score=p.score,
                scores=dict(p.scores),
                text_sha256=text_sha256(raw),
                text_chars=len(raw),
                inference_ms=elapsed_ms,
            )
            for i, (prediction_id, raw, p) in enumerate(
                zip(ids, raw_texts, predictions, strict=True)
            )
        ]
        return ids, current_store.record(rows)

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

    @app.get(
        "/health", response_model=StatusResponse, response_model_exclude_none=True, tags=["ops"]
    )
    def health() -> StatusResponse:
        """Liveness: the process is up and serving HTTP."""
        return StatusResponse(status="ok")

    @app.get("/ready", response_model=StatusResponse, tags=["ops"])
    def ready(request: Request) -> StatusResponse:
        """Readiness: the model is loaded and requests can be served.

        The database is reported but does not affect readiness: predictions are
        served while it is down, they are just not recorded.
        """
        get_predictor(request)
        current_store = request.app.state.store
        database = "disabled" if current_store is None else current_store.status
        return StatusResponse(status="ready", database=database)

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
        ids, recorded = record(request, model, "predict", [body.text], predictions, elapsed_ms)
        return PredictResponse(
            request_id=request.state.request_id,
            model=model_ref(model),
            prediction=_out(predictions[0], ids[0]),
            inference_ms=elapsed_ms,
            recorded=recorded,
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
        ids, recorded = record(request, model, "predict_batch", body.texts, predictions, elapsed_ms)
        return BatchPredictResponse(
            request_id=request.state.request_id,
            model=model_ref(model),
            predictions=[_out(p, i) for p, i in zip(predictions, ids, strict=True)],
            inference_ms=elapsed_ms,
            recorded=recorded,
        )

    @app.post(
        "/v1/feedback",
        response_model=FeedbackResponse,
        status_code=201,
        tags=["feedback"],
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def add_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
        """Record the correct label for a recorded prediction."""
        current_store = request.app.state.store
        if current_store is None:
            raise APIError(
                503, "feedback_unavailable", "Feedback needs a database; DATABASE_URL is not set."
            )
        if body.text is not None and len(body.text) > settings.max_text_chars:
            raise APIError(
                422,
                "invalid_request",
                "The request body is invalid.",
                [
                    {
                        "loc": ["body", "text"],
                        "msg": f"Text is longer than {settings.max_text_chars} characters.",
                        "type": "text_too_long",
                    }
                ],
            )
        from .store import (
            FeedbackExistsError,
            InvalidFeedbackError,
            PredictionNotFoundError,
            StoreUnavailableError,
        )

        try:
            result = current_store.add_feedback(body.prediction_id, body.label, body.text)
        except PredictionNotFoundError as exc:
            raise APIError(
                404, "prediction_not_found", "No recorded prediction has this id."
            ) from exc
        except FeedbackExistsError as exc:
            raise APIError(
                409, "feedback_exists", "Feedback for this prediction was already recorded."
            ) from exc
        except InvalidFeedbackError as exc:
            raise APIError(
                422,
                "invalid_request",
                "The request body is invalid.",
                [{"loc": ["body", exc.field], "msg": str(exc), "type": exc.kind}],
            ) from exc
        except StoreUnavailableError as exc:
            raise APIError(
                503, "database_unavailable", "The database is unavailable; try again later."
            ) from exc
        return FeedbackResponse(
            id=result.id,
            prediction_id=result.prediction_id,
            label=result.label,
            predicted_label=result.predicted_label,
            model_was_correct=result.model_was_correct,
            text_stored=result.text_stored,
        )

    return app
