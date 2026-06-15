"""Optimisation Optuna + Model Registry — à compléter (TP S6).

Usage :
    python -m churn.train_optuna
    python -m churn.train_optuna --n-trials 50 --cv 5
"""
from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# TODO (S6-1) : importer optuna et les modules nécessaires
# import optuna
# import mlflow
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from churn.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_NAME, RANDOM_STATE
# from churn.data import get_splits, load_data
# from churn.features import build_preprocessor
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimisation Optuna.")
    p.add_argument("--n-trials", type=int, default=30, help="Nombre d'essais Optuna")
    p.add_argument("--cv", type=int, default=5, help="Nombre de folds de validation croisée")
    return p.parse_args()


def objective(trial, X_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna — à compléter (TP S6-2)."""
    # TODO (S6-2) : définir l'espace de recherche (trial.suggest_*) et
    #               évaluer le pipeline avec cross_val_score
    # Exemple :
    # n_estimators = trial.suggest_int("n_estimators", 50, 300)
    # max_depth    = trial.suggest_int("max_depth", 2, 10)
    # pipeline = Pipeline([
    #     ("preprocessor", build_preprocessor()),
    #     ("model", RandomForestClassifier(n_estimators=n_estimators,
    #                                      max_depth=max_depth,
    #                                      random_state=RANDOM_STATE)),
    # ])
    # scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    # return scores.mean()
    raise NotImplementedError("TODO (S6-2)")


def train_optuna(n_trials: int, cv: int) -> None:
    """Lance l'optimisation Optuna — à compléter (TP S6-3)."""
    # TODO (S6-3) : créer une étude Optuna et lancer l'optimisation
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # df = load_data()
    # X_train, X_test, y_train, y_test = get_splits(df)
    #
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, X_train, y_train, cv),
    #                n_trials=n_trials)
    #
    # TODO (S6-4) : réentraîner avec les meilleurs params et enregistrer
    #               dans le Model Registry MLflow
    raise NotImplementedError("TODO (S6-3)")


if __name__ == "__main__":
    args = parse_args()
    train_optuna(n_trials=args.n_trials, cv=args.cv)
