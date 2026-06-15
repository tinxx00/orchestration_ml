"""Comparaison de modèles GridSearchCV + SHAP — à compléter (TP S7).

Usage :
    python -m churn.train_models
    python -m churn.train_models --cv 5 --scoring roc_auc
"""
from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# TODO (S7-1) : importer les classifieurs, GridSearchCV et mlflow
# import mlflow
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from churn.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODELS_DIR, RANDOM_STATE
# from churn.data import get_splits, load_data
# from churn.evaluation import shap_summary
# from churn.features import build_preprocessor
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparaison de modèles (GridSearchCV + SHAP).")
    p.add_argument("--cv", type=int, default=5, help="Nombre de folds de validation croisée")
    p.add_argument("--scoring", type=str, default="roc_auc", help="Métrique de scoring")
    return p.parse_args()


def train_models(cv: int, scoring: str) -> None:
    """Compare RF / XGBoost / LightGBM via GridSearchCV — à compléter (TP S7)."""
    # TODO (S7-2) : définir les modèles et leurs grilles de paramètres
    # models = {
    #     "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE), {...}),
    #     "XGBoost":      (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"), {...}),
    #     "LightGBM":     (LGBMClassifier(random_state=RANDOM_STATE, verbose=-1), {...}),
    # }

    # TODO (S7-3) : pour chaque modèle, lancer GridSearchCV et logger dans MLflow
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # df = load_data()
    # X_train, X_test, y_train, y_test = get_splits(df)
    # best_pipeline, best_score = None, -1
    # for name, (clf, param_grid) in models.items():
    #     with mlflow.start_run(run_name=name):
    #         pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", clf)])
    #         gs = GridSearchCV(pipeline, {"model__" + k: v for k, v in param_grid.items()},
    #                           cv=cv, scoring=scoring, n_jobs=-1)
    #         gs.fit(X_train, y_train)
    #         mlflow.log_params(gs.best_params_)
    #         mlflow.log_metric(scoring, gs.best_score_)
    #         if gs.best_score_ > best_score:
    #             best_score, best_pipeline = gs.best_score_, gs.best_estimator_

    # TODO (S7-4) : générer le SHAP summary plot du meilleur modèle
    # shap_summary(best_pipeline, X_test)
    raise NotImplementedError("TODO (S7)")


if __name__ == "__main__":
    args = parse_args()
    train_models(cv=args.cv, scoring=args.scoring)
