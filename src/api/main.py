"""Tasks 6 & 8 — FastAPI model-serving app with logging and Prometheus metrics.

Endpoints
  GET  /            -> service banner
  GET  /health      -> liveness/readiness probe (used by Kubernetes)
  GET  /metrics     -> Prometheus exposition format (added by the instrumentator)
  POST /predict     -> JSON in, {prediction, label, confidence, ...} out

Observability
  * Every request is logged (method, path, latency, status) via a middleware.
  * A custom Counter (``heart_predictions_total{outcome=...}``) tracks how many
    positive vs negative predictions have been served — useful for spotting
    prediction drift on a Grafana dashboard.
  * prometheus-fastapi-instrumentator exposes standard latency/throughput
    metrics at /metrics.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import (
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH

# --------------------------------------------------------------------------- #
# Logging — structured-ish line logs that show up in `kubectl logs`.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")

# Custom business metric: prediction outcomes.
PREDICTION_COUNTER = Counter(
    "heart_predictions_total",
    "Total predictions served, labelled by outcome",
    ["outcome"],
)

# Module-level model holder, populated at startup.
_STATE: dict = {"model": None, "model_name": "unknown"}


def _load_model() -> None:
    """Load the trained pipeline + metadata into module state."""
    if not Path(MODEL_PATH).exists():
        logger.warning("Model file %s not found — /predict will 503.", MODEL_PATH)
        return
    _STATE["model"] = joblib.load(MODEL_PATH)
    if Path(METADATA_PATH).exists():
        meta = json.loads(Path(METADATA_PATH).read_text())
        _STATE["model_name"] = meta.get("model_name", "unknown")
    logger.info("Loaded model '%s' from %s", _STATE["model_name"], MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup (not per request)."""
    _load_model()
    yield
    _STATE.clear()


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart-disease risk from patient clinical features.",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics for Prometheus.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its latency and status code."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _STATE["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=_STATE["model_name"] if loaded else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    """Return a heart-disease prediction with a confidence score."""
    model = _STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a one-row DataFrame with the exact training column order.
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]

    try:
        proba_disease = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # pragma: no cover
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    prediction = int(proba_disease >= 0.5)
    confidence = proba_disease if prediction == 1 else 1.0 - proba_disease
    outcome = "disease" if prediction == 1 else "no_disease"
    PREDICTION_COUNTER.labels(outcome=outcome).inc()
    logger.info("prediction=%s p_disease=%.4f", prediction, proba_disease)

    return PredictionResponse(
        prediction=prediction,
        label="Heart disease" if prediction == 1 else "No heart disease",
        confidence=round(confidence, 4),
        probability_disease=round(proba_disease, 4),
        model_name=_STATE["model_name"],
    )
