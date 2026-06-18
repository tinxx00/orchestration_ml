"""DAG Airflow - trafic de previsions quotidien (S17).

Seance 17 - TP Airflow (suite)
    Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a
    10h, on echantillonne 20 lignes du jeu de donnees et on les envoie en
    POST /predict. Cela simule un flux de previsions en production (chaque
    appel est journalise par l'API) et alimente la boucle de feedback.

Prerequis : l'API doit etre joignable via `API_URL` (defaut `http://api:8000`
en docker). En pratique : lancer la stack qui expose l'API en plus d'Airflow.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Nombre de previsions envoyees a chaque execution.
N_PREDICTIONS = 20

# En contexte docker/Airflow, l'API est joignable via le nom de service "api".
API_URL = os.getenv("API_URL", "http://api:8000")

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Echantillonner N_PREDICTIONS lignes et les envoyer a l'API /predict."""
    import httpx

    from heart.config import TARGET
    from heart.data import load_data

    # On retire la colonne cible : l'API ne recoit que les features.
    features = load_data().drop(columns=[TARGET])

    # (S17-6) Echantillonner N_PREDICTIONS lignes au hasard.
    sample = features.sample(n=N_PREDICTIONS)

    # (S17-7) Ouvrir un client httpx sur API_URL, verifier /health, puis
    # envoyer chaque ligne en POST /predict.
    sent = 0
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            # to_json() garantit des types JSON natifs (pas de numpy).
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            sent += 1

    logger.info("%d previsions envoyees a %s", sent, API_URL)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    # (S17-8) Planifier tous les jours a 10h.
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
