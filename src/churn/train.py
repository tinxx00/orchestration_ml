"""Baseline LogisticRegression + MLflow — à compléter (TP S5).

Usage :
    python -m churn.train
    python -m churn.train --c 0.1 --max-iter 500
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn.config import (
    C,
    MAX_ITER,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODEL_NAME,
    RANDOM_STATE,
)
from churn.data import get_splits, load_data
from churn.features import build_preprocessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne la baseline LogisticRegression.")
    p.add_argument("--c", type=float, default=C, help="Paramètre de régularisation C")
    p.add_argument("--max-iter", type=int, default=MAX_ITER, help="Nombre max d'itérations")
    return p.parse_args()


def train(c: float, max_iter: int) -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(C=c, max_iter=max_iter, random_state=RANDOM_STATE)),
    ])

    # -----------------------------------------------------------------------
    # TODO (S5-1) : configurer le tracking URI et l'experiment MLflow
    # import mlflow
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # TODO (S5-2) : ouvrir un run MLflow et logger params + métriques + modèle
    # with mlflow.start_run():
    #     mlflow.log_params({"C": c, "max_iter": max_iter})
    #     pipeline.fit(X_train, y_train)
    #     ...
    #     mlflow.log_metrics({"f1": f1, "roc_auc": auc})
    #     mlflow.sklearn.log_model(pipeline, artifact_path="model",
    #                              registered_model_name=MODEL_NAME)
    # -----------------------------------------------------------------------

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"f1={f1:.4f}  roc_auc={auc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[OK] Modèle sauvegardé → {MODEL_PATH}")


if __name__ == "__main__":
    args = parse_args()
    train(c=args.c, max_iter=args.max_iter)
