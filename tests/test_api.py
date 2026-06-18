"""Tests des endpoints de l'API FastAPI (via TestClient)."""
from __future__ import annotations

VALID_PAYLOAD = dict(
    age=52, trestbps=120, chol=240, thalach=150, oldpeak=1.0, ca=0,
    sex=1, cp=2, fbs=0, restecg=1, exang=0, slope=2, thal=3,
)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_label_and_proba(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_payload(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # erreur de validation Pydantic


def test_model_info_exposes_expected_keys(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_name", "model_alias", "model_source", "features"):
        assert key in body
    assert "numeric" in body["features"]
    assert "categorical" in body["features"]
