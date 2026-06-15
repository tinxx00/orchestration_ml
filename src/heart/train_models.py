"""Comparaison de modèles (GridSearchCV) + SHAP + MLflow (TP S7)."""
from __future__ import annotations

import argparse
import logging

import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from heart.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODELS_DIR,
    RANDOM_STATE,
)
from heart.data import get_splits, load_data
from heart.evaluation import shap_summary
from heart.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparaison de modèles (GridSearchCV + SHAP + MLflow).")
    p.add_argument("--cv", type=int, default=5, help="Nombre de folds de validation croisée")
    p.add_argument("--scoring", type=str, default="roc_auc", help="Métrique de scoring")
    p.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Désactive le tracking MLflow",
    )
    p.set_defaults(use_mlflow=True)
    return p.parse_args()


def _build_models() -> dict:
    return {
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.1, 0.01],
            },
        ),
        "lightgbm": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            {
                "model__n_estimators": [100, 200],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.1, 0.01],
            },
        ),
    }


def train_models(cv: int, scoring: str, use_mlflow: bool = True) -> None:
    df = load_data()
    x_train, x_test, y_train, y_test = get_splits(df)

    models = _build_models()

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info("MLflow tracking: %s / experiment=%s", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)

    best_name = ""
    best_score = -1.0
    best_pipeline = None

    parent_ctx = mlflow.start_run(run_name="compare-models") if use_mlflow else None
    try:
        if use_mlflow:
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)

        for name, (clf, grid) in models.items():
            logger.info("Optimisation de %s", name)

            child_ctx = mlflow.start_run(run_name=name, nested=True) if use_mlflow else None
            try:
                pipeline = Pipeline([
                    ("preprocessor", build_preprocessor()),
                    ("model", clf),
                ])
                gs = GridSearchCV(
                    estimator=pipeline,
                    param_grid=grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(x_train, y_train)

                best = gs.best_estimator_
                proba = best.predict_proba(x_test)[:, 1]
                preds = (proba >= 0.5).astype(int)

                f1 = float(f1_score(y_test, preds))
                auc = float(roc_auc_score(y_test, proba))
                cv_best = float(gs.best_score_)

                logger.info(
                    "%s -> cv_%s=%.4f  f1=%.4f  roc_auc=%.4f",
                    name,
                    scoring,
                    cv_best,
                    f1,
                    auc,
                )

                if use_mlflow:
                    mlflow.log_params(gs.best_params_)
                    mlflow.log_metrics(
                        {
                            f"cv_{scoring}": cv_best,
                            "f1": f1,
                            "roc_auc": auc,
                        }
                    )
                    mlflow.sklearn.log_model(
                        sk_model=best,
                        name="model",
                        registered_model_name=MODEL_NAME if auc > best_score else None,
                    )

                if auc > best_score:
                    best_name = name
                    best_score = auc
                    best_pipeline = best
            finally:
                if child_ctx is not None:
                    mlflow.end_run()

        if best_pipeline is None:
            raise RuntimeError("Aucun modèle entraîné")

        logger.info("Meilleur modèle: %s (roc_auc=%.4f)", best_name, best_score)

        shap_summary(best_pipeline, x_test)
        if use_mlflow:
            mlflow.log_artifact(str(MODELS_DIR / "shap_summary.png"))
            mlflow.set_tag("best_model", best_name)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MODELS_DIR / "model.joblib"
        joblib.dump(best_pipeline, output_path)
        logger.info("Modèle sauvegardé dans %s", output_path)
    finally:
        if parent_ctx is not None:
            mlflow.end_run()


def main() -> None:
    args = parse_args()
    train_models(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
