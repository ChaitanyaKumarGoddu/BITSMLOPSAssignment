"""Integration tests for the FastAPI service (Task 5).

We inject the fitted test pipeline into the app's module state so the API can
be exercised without a pre-trained ``model.pkl`` on disk.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def _client(trained_pipeline) -> TestClient:
    main._STATE["model"] = trained_pipeline
    main._STATE["model_name"] = "logistic_regression_test"
    return TestClient(main.app)


def test_health_ok(trained_pipeline):
    client = _client(trained_pipeline)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True


def test_predict_returns_valid_schema(trained_pipeline):
    client = _client(trained_pipeline)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in ("Heart disease", "No heart disease")


def test_predict_rejects_out_of_range(trained_pipeline):
    client = _client(trained_pipeline)
    bad = dict(VALID_PAYLOAD, age=-5)  # violates ge=0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field(trained_pipeline):
    client = _client(trained_pipeline)
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "chol"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_503_when_model_missing():
    main._STATE["model"] = None
    client = TestClient(main.app)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503


def test_metrics_endpoint_exposed(trained_pipeline):
    client = _client(trained_pipeline)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "heart_predictions_total" in r.text or "http_request" in r.text
