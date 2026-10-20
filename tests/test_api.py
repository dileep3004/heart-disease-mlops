import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from heart_mlops import api
from heart_mlops.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from heart_mlops.modeling import build_pipeline

def _fitted_model():
    rng = np.random.default_rng(11)
    rows = 80
    data = {column: rng.normal(0, 1, rows) for column in NUMERIC_COLUMNS}
    levels = {
        "sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2],
        "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7],
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.choice(levels[column], rows)
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    target = (frame["age"] > frame["age"].median()).astype(int)
    model = build_pipeline("logistic_regression")
    return model.fit(frame, target)

def test_health_and_metrics_endpoints():
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "heart_api_requests_total" in metrics_response.text

def test_predict_returns_probability_and_disclaimer(monkeypatch):
    monkeypatch.setattr(api.app.state, "model", _fitted_model())
    client = TestClient(api.app)
    sample = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3,
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability_heart_disease"] <= 1
    assert "not for medical diagnosis" in body["disclaimer"]

def test_predict_rejects_missing_required_field():
    client = TestClient(api.app)
    response = client.post("/predict", json={"age": 50})
    assert response.status_code == 422
