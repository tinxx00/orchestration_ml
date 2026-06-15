"""DAG Airflow de ré-entraînement automatique — à compléter (TP S17).

Ce DAG planifie le re-entraînement du modèle de classification cardiaque.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# TODO (S17-1) : importer les modules Airflow nécessaires
#
# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S17-2) : définir les arguments par défaut et le DAG
#
# default_args = {
#     "owner": "mlops",
#     "retries": 1,
#     "retry_delay": timedelta(minutes=5),
# }
#
# with DAG(
#     dag_id="heart_disease_retrain",
#     description="Ré-entraîne le classifieur de maladies cardiaques",
#     schedule_interval="@weekly",   # ex. toutes les semaines
#     start_date=datetime(2026, 6, 1),
#     catchup=False,
#     default_args=default_args,
#     tags=["mlops", "heart-disease"],
# ) as dag:
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S17-3) : définir les tâches du pipeline
#
#     # Tâche 1 : préparation des données
#     prepare_data = BashOperator(
#         task_id="prepare_data",
#         bash_command="PYTHONPATH=/app python -m churn.data",
#     )
#
#     # Tâche 2 : entraînement du modèle
#     train_model = BashOperator(
#         task_id="train_model",
#         bash_command="PYTHONPATH=/app python -m churn.train",
#     )
#
#     # Tâche 3 : promotion du modèle si les métriques sont suffisantes
#     promote_model = BashOperator(
#         task_id="promote_model",
#         bash_command="PYTHONPATH=/app python -m churn.promote",
#     )
#
#     # Ordre d'exécution
#     prepare_data >> train_model >> promote_model
# ---------------------------------------------------------------------------
