"""FastAPI inference service with request logging and Prometheus metrics."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from heart_mlops.data import FEATURE_COLUMNS

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_mlops.api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/heart_disease_model.joblib"))
REQUEST_COUNT = Counter(
    "heart_api_requests_total", "HTTP requests received by the heart disease API",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds", "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total", "Predictions by predicted class", ["predicted_class"]
)

class PatientFeatures(BaseModel):
    """Input schema using the 13 UCI Cleveland predictor columns."""

    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(ge=0, le=350)
    chol: float = Field(ge=0, le=1200)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=0, le=8)

app = FastAPI(
    title="Heart Disease MLOps Demo API",
    description=(
        "Educational binary classifier built for an MLOps assignment. "
        "It is not a diagnostic or treatment tool."
    ),
    version="1.0.0",
)
app.state.model = None

def _get_model():
    model = getattr(app.state, "model", None)
    if model is not None:
        return model
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {MODEL_PATH}. Run the training step first.",
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # raise safe public error; detailed failure stays in logs
        logger.exception("Could not load model artifact")
        raise HTTPException(status_code=503, detail="Model artifact could not be loaded.") from exc
    app.state.model = model
    return model

@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    route = request.url.path
    REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, route).observe(elapsed)
    logger.info(
        "http_request method=%s path=%s status=%s duration_ms=%.2f",
        request.method, route, response.status_code, elapsed * 1000,
    )
    return response

@app.get("/health", tags=["operations"])
def health():
    return {"status": "ok", "service": "heart-disease-api"}

@app.get("/ready", tags=["operations"])
def readiness():
    _get_model()
    return {"status": "ready"}

@app.get("/metrics", tags=["operations"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", tags=["inference"])
def predict(patient: PatientFeatures):
    model = _get_model()
    payload = patient.model_dump()
    row = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    positive_probability = float(probabilities[1])
    confidence = float(max(probabilities))
    PREDICTION_COUNT.labels(str(prediction)).inc()
    label = "heart_disease_present" if prediction == 1 else "heart_disease_absent"
    return {
        "prediction": prediction,
        "prediction_label": label,
        "probability_heart_disease": round(positive_probability, 6),
        "confidence": round(confidence, 6),
        "model_version": "1.0.0",
        "disclaimer": "Educational output only; not for medical diagnosis or treatment.",
    }
