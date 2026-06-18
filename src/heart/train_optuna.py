"""Optimisation d'hyperparametres avec Optuna.

Seance 6 - TP Optuna
    Ce module optimise les hyperparametres de trois familles de modeles
    (Random Forest, XGBoost, LightGBM) avec Optuna (sampler TPE), compare
    leurs performances et persiste le meilleur dans `models/model.joblib`.
    Completez les TODO (S6-n).

Chaque famille est suivie dans MLflow (un run par famille, imbrique sous un
run parent) et la meilleure est enregistree dans le Model Registry.

Lancement :
    python -m heart.train_optuna
    python -m heart.train_optuna --n-trials 50 --cv 3
    python -m heart.train_optuna --no-mlflow   # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from mlflow.models import infer_signature
from optuna import samplers
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from heart.config import MODEL_ALIAS, MODEL_NAME, MODELS_DIR, RANDOM_STATE
from heart.data import get_splits, load_data
from heart.evaluation import evaluate_pipeline, load_model_from_uri, log_shap_summary
from heart.features import build_preprocessor
from heart.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification d'une famille de modeles a optimiser avec Optuna.

    Parameters
    ----------
    name : str
        Identifiant lisible de la famille de modeles.
    suggest_params : Callable[[optuna.Trial], dict]
        Construit un jeu d'hyperparametres pour un essai donne.
    build_estimator : Callable[[dict], ClassifierMixin]
        Construit l'estimateur scikit-learn a partir d'un jeu d'hyperparametres.
    """

    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des familles de modeles a optimiser.

    Returns
    -------
    list of ModelSpec
        Random Forest, XGBoost et LightGBM avec leurs espaces de recherche.
    """
    return [
        ModelSpec(
            name="random_forest",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            },
            build_estimator=lambda params: cast(
                ClassifierMixin,
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params),
            ),
        ),
        ModelSpec(
            name="xgboost",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
            build_estimator=lambda params: cast(
                ClassifierMixin,
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                    **params,
                ),
            ),
        ),
        ModelSpec(
            name="lightgbm",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            },
            build_estimator=lambda params: cast(
                ClassifierMixin,
                LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params),
            ),
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le preprocessing et un classifieur dans un pipeline.

    Parameters
    ----------
    estimator : ClassifierMixin
        Classifieur place en derniere etape (``clf``).

    Returns
    -------
    Pipeline
        Pipeline scikit-learn pret a etre optimise.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def objective(trial: optuna.Trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : score moyen de validation croisee.

    Parameters
    ----------
    trial : optuna.Trial
        Essai courant.
    spec : ModelSpec
        Famille de modeles optimisee.
    x_train, y_train : array-like
        Donnees d'entrainement.
    cv : int
        Nombre de plis de validation croisee.

    Returns
    -------
    float
        Score ROC AUC moyen sur les plis de validation croisee (a maximiser).
    """
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(scores.mean())


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer l'etude Optuna pour une famille de modeles.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles a optimiser.
    x_train, y_train : array-like
        Donnees d'entrainement.
    n_trials : int
        Nombre d'essais a evaluer.
    cv : int
        Nombre de plis de validation croisee passe a `objective`.

    Returns
    -------
    optuna.Study
        Etude Optuna une fois l'optimisation terminee.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(lambda trial: objective(trial, spec, x_train, y_train, cv), n_trials=n_trials)
    return study


@dataclass
class FamilyResult:
    """Resultat d'optimisation d'une famille de modeles.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles optimisee.
    study : optuna.Study
        Etude Optuna terminee.
    best_pipeline : Pipeline
        Pipeline reentraine avec les meilleurs hyperparametres.
    test_roc_auc : float
        ROC AUC sur le jeu de test.
    preds : np.ndarray
        Predictions (classes) sur le jeu de test.
    """

    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    """Optimiser une famille de modeles avec Optuna et l'evaluer sur le test.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles a optimiser.
    x_train, y_train : array-like
        Donnees d'entrainement.
    x_test, y_test : array-like
        Donnees de test pour l'evaluation finale.
    n_trials : int
        Nombre d'essais Optuna.
    cv : int
        Nombre de plis de validation croisee.

    Returns
    -------
    FamilyResult
        Meilleur pipeline et metriques associees.
    """
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f  test_roc_auc=%.3f  params=%s",
        spec.name,
        study.best_value,
        test_roc_auc,
        study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    """Logger une famille de modeles dans un run MLflow imbrique.

    Parameters
    ----------
    result : FamilyResult
        Resultat a tracer (etude Optuna, pipeline, metriques).
    x_test : pandas.DataFrame
        Jeu de test, utilise pour inferer la signature et un exemple d'entree.
    y_test : array-like
        Cibles du jeu de test, utilisees pour la matrice de confusion et le
        rapport de classification.
    n_trials : int
        Nombre d'essais Optuna (loggue comme parametre).
    cv : int
        Nombre de plis de validation croisee (loggue comme parametre).
    register_as : str, optional
        Si fourni, enregistre le modele dans le Model Registry sous ce nom.
    """
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                if trial.params:
                    mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", float(trial.value))

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
            serialization_format="cloudpickle",
        )

        pipeline_for_evaluation = result.best_pipeline
        resolved_model_uri = model_info.model_uri

        if register_as and model_info.registered_model_version:
            client = mlflow.MlflowClient()
            registered_version = int(model_info.registered_model_version)
            client.set_registered_model_alias(register_as, MODEL_ALIAS, str(registered_version))
            resolved_model_uri = f"models:/{register_as}@{MODEL_ALIAS}"
            pipeline_for_evaluation, resolved_model_uri = load_model_from_uri(
                model_uri=resolved_model_uri
            )
            mlflow.log_param("registered_model_uri", resolved_model_uri)
            registry_metrics = evaluate_pipeline(pipeline_for_evaluation, x_test, y_test)
            mlflow.log_metrics({f"registry_{key}": value for key, value in registry_metrics.items()})
            describe_registered_version(
                name=register_as,
                version=registered_version,
                result=result,
                n_trials=n_trials,
                cv=cv,
            )
        else:
            mlflow.log_param("logged_model_uri", resolved_model_uri)

        log_shap_summary(pipeline_for_evaluation, x_test, name=result.spec.name)


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    """Documenter une version enregistree dans le Model Registry.

    Ajoute une description (algorithme, hyperparametres, metriques) et des
    tags (famille de modele, methode de recherche, scores) sur la version du
    modele afin de pouvoir comparer les versions sans rouvrir le run MLflow.

    Parameters
    ----------
    name : str
        Nom du modele enregistre dans le registry.
    version : int
        Version enregistree a documenter.
    result : FamilyResult
        Resultat d'optimisation associe a cette version.
    n_trials : int
        Nombre d'essais Optuna par famille.
    cv : int
        Nombre de plis de validation croisee.
    """
    client = mlflow.MlflowClient()
    description = (
        f"Famille: {result.spec.name}\n"
        f"Search: Optuna TPE\n"
        f"n_trials: {n_trials}\n"
        f"cv: {cv}\n"
        f"best_params: {result.study.best_params}\n"
        f"cv_roc_auc: {result.study.best_value:.4f}\n"
        f"test_roc_auc: {result.test_roc_auc:.4f}"
    )
    client.update_model_version(name=name, version=str(version), description=description)
    client.set_model_version_tag(name, str(version), "model_family", result.spec.name)
    client.set_model_version_tag(name, str(version), "search_method", "optuna_tpe")
    client.set_model_version_tag(name, str(version), "n_trials", str(n_trials))
    client.set_model_version_tag(name, str(version), "cv", str(cv))
    client.set_model_version_tag(name, str(version), "cv_roc_auc", f"{result.study.best_value:.4f}")
    client.set_model_version_tag(name, str(version), "test_roc_auc", f"{result.test_roc_auc:.4f}")


def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    """Optimiser RF / XGBoost / LightGBM avec Optuna et sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib``. Lorsque ``use_mlflow`` est actif, chaque famille
    est suivie dans un run MLflow imbrique sous un run parent
    ``optuna-compare``, et la meilleure est enregistree dans le Model Registry
    sous ``MODEL_NAME``.

    Parameters
    ----------
    n_trials : int, optional
        Nombre d'essais Optuna par famille de modeles, par defaut 30.
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    use_mlflow : bool, optional
        Active le suivi MLflow, par defaut True.

    Returns
    -------
    list of FamilyResult
        Resultats tries du meilleur au moins bon (ROC AUC de test decroissant).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = get_splits(df)

    if use_mlflow:
        setup_experiment()
        logger.info("Suivi MLflow activé")

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            log_dataset(df, context="training", name="heart_disease_uci")
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(
                    result, x_test, y_test, n_trials, cv, register_as=register_as
                )
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODELS_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODELS_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre d'essais Optuna par famille de modeles"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
