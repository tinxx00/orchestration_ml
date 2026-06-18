"""DAG Airflow - pipeline de re-entrainement du modele (S17).

Seance 17 - TP Airflow
    Pipeline simple : preparation des donnees -> entrainement -> controle
    qualite.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# f1 minimal du modele entraine pour que le pipeline soit considere comme reussi.
QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    # (S17-1) Charger le jeu de donnees pour s'assurer qu'il est disponible
    # et exploitable avant l'entrainement.
    from heart.data import load_data

    df = load_data()
    logger.info("Donnees pretes : %d lignes, %d colonnes", len(df), df.shape[1])


def task_train(**context) -> None:
    # (S17-2) Entrainer le modele et pousser le f1 dans XCom.
    from heart.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine : f1=%.4f", metrics["f1"])


def task_check_quality(**context) -> None:
    # (S17-3) Recuperer le f1 et echouer si en dessous du seuil.
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None:
        raise ValueError("Aucun f1 recupere depuis la tache d'entrainement.")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : f1={f1:.4f} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Controle qualite OK : f1=%.4f >= %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    # (S17-4) Tous les lundis a 3h du matin.
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # (S17-5) Ordre d'execution du pipeline.
    prepare >> train_task >> check
