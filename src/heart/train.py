"""Entrainement du modele de classification (baseline).

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele SANS suivi d'experience MLflow.
    Votre mission (S5) : instrumenter cet entrainement via les TODO.
"""
from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from heart.config import (
    C,
    MAX_ITER,
    MODELS_DIR,
    RANDOM_STATE,
)
from heart.data import get_splits, load_data
from heart.features import build_preprocessor
from heart.tracking import log_dataset, setup_experiment


def build_model(c: float = C, max_iter: int = MAX_ITER) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                LogisticRegression(C=c, max_iter=max_iter, random_state=RANDOM_STATE),
            ),
        ]
    )


def train(c: float = C, max_iter: int = MAX_ITER) -> dict[str, float]:
    df = load_data()
    x_train, x_test, y_train, y_test = get_splits(df)

    setup_experiment()

    with mlflow.start_run(run_name=f"logreg-c{c}"):
        log_dataset(df, context="training", name="heart_disease_uci")
        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "model.joblib"
        joblib.dump(model, model_path)
        print(f"[OK] Modèle sauvegardé → {model_path}")
        return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne la baseline LogisticRegression.")
    p.add_argument("--c", type=float, default=C, help="Paramètre de régularisation C")
    p.add_argument("--max-iter", type=int, default=MAX_ITER, help="Nombre max d'itérations")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(c=args.c, max_iter=args.max_iter)
