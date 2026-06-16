"""API FastAPI pour la prédiction de maladie cardiaque.

Usage :
    uvicorn heart.api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from heart.config import (
    API_URL,
    CATEGORICAL_FEATURES,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from heart.evaluation import load_model_from_uri

logger = logging.getLogger(__name__)


class Features(BaseModel):
    age: float = Field(..., example=52)
    trestbps: float = Field(..., example=120)
    chol: float = Field(..., example=240)
    thalach: float = Field(..., example=150)
    oldpeak: float = Field(..., example=1.0)
    ca: float = Field(..., example=0)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=2)
    fbs: int = Field(..., example=0)
    restecg: int = Field(..., example=1)
    exang: int = Field(..., example=0)
    slope: int = Field(..., example=2)
    thal: int = Field(..., example=3)

app = FastAPI(
    title="Heart Disease Classifier API",
    description="Prédit la présence d'une maladie cardiaque à partir de données cliniques.",
    version="0.1.0",
)

pipeline: Pipeline | None = None
model_source: str | None = None


def _load_pipeline() -> tuple[Pipeline, str]:
    try:
        model, resolved_uri = load_model_from_uri(model_name=MODEL_NAME, alias=MODEL_ALIAS)
        logger.info("Modèle chargé depuis MLflow: %s", resolved_uri)
        return model, resolved_uri
    except Exception as exc:
        logger.warning("Chargement MLflow impossible (%s), fallback local.", exc)

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if not isinstance(model, Pipeline):
            raise TypeError(f"Le modèle local '{MODEL_PATH}' n'est pas un Pipeline scikit-learn.")
        logger.info("Modèle chargé depuis le disque: %s", MODEL_PATH)
        return model, str(MODEL_PATH)

    raise RuntimeError("Aucun modèle disponible (ni MLflow, ni fichier local).")


@app.on_event("startup")
def startup_load_model() -> None:
    global pipeline, model_source
    pipeline, model_source = _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(features: Features) -> dict:
    if pipeline is None:
        raise RuntimeError("Modèle non chargé.")

    df = pd.DataFrame([features.model_dump()])
    proba = float(pipeline.predict_proba(df)[0, 1])
    label = int(pipeline.predict(df)[0])
    return {"label": label, "probability": round(proba, 4)}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_source": model_source,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "api_url": API_URL,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
