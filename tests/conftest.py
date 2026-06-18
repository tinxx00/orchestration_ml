"""Fixtures partagées pour les tests."""
from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from heart.data import load_data
from heart.train import build_model


@pytest.fixture(scope="session")
def dataframe():
    """Jeu de données Heart Disease UCI chargé une seule fois."""
    return load_data()


@pytest.fixture(scope="session")
def trained_pipeline(dataframe) -> Pipeline:
    """Pipeline baseline entraîné sur l'ensemble des données (rapide)."""
    from heart.config import TARGET

    x = dataframe.drop(columns=[TARGET])
    y = dataframe[TARGET]
    model = build_model()
    model.fit(x, y)
    return model


@pytest.fixture
def client(trained_pipeline, monkeypatch):
    """TestClient FastAPI avec un modèle entraîné injecté (sans MLflow)."""
    from fastapi.testclient import TestClient

    import heart.api as api

    # On évite tout appel MLflow/disque : le startup utilise notre pipeline.
    monkeypatch.setattr(api, "_load_pipeline", lambda: (trained_pipeline, "test"))

    with TestClient(api.app) as test_client:
        yield test_client
