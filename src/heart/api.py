"""API FastAPI — à compléter (TP S12).

Usage :
    uvicorn churn.api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Heart Disease Classifier API",
    description="Prédit la présence d'une maladie cardiaque à partir de données cliniques.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# TODO (S12-1) : définir le schéma d'entrée Pydantic
# Les champs doivent correspondre exactement à NUMERIC_FEATURES + CATEGORICAL_FEATURES
# définis dans config.py.
#
# from pydantic import BaseModel, Field
#
# class Features(BaseModel):
#     age:      float = Field(..., example=52)
#     trestbps: float = Field(..., example=120)
#     chol:     float = Field(..., example=240)
#     thalach:  float = Field(..., example=150)
#     oldpeak:  float = Field(..., example=1.0)
#     ca:       float = Field(..., example=0)
#     sex:      int   = Field(..., example=1)
#     cp:       int   = Field(..., example=2)
#     fbs:      int   = Field(..., example=0)
#     restecg:  int   = Field(..., example=1)
#     exang:    int   = Field(..., example=0)
#     slope:    int   = Field(..., example=2)
#     thal:     int   = Field(..., example=3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S12-2) : charger le modèle au démarrage (lifespan ou module-level)
#
# import joblib
# from churn.config import MODEL_PATH
#
# pipeline = joblib.load(MODEL_PATH)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S12-3) : implémenter la route POST /predict
#
# import pandas as pd
#
# @app.post("/predict")
# def predict(features: Features):
#     df = pd.DataFrame([features.model_dump()])
#     proba = pipeline.predict_proba(df)[0, 1]
#     label = int(pipeline.predict(df)[0])
#     return {"label": label, "probability": round(float(proba), 4)}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S12-4) : implémenter la route GET /health
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
# ---------------------------------------------------------------------------
