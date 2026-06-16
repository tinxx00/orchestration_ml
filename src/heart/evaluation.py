"""Outils d'evaluation partages et chargement du dernier modele MLflow."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import shap
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from heart.config import MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME, MODELS_DIR
from heart.data import get_splits, load_data

logger = logging.getLogger(__name__)


def _get_classifier(pipeline: Pipeline):
    clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("model")
    if clf is None:
        raise ValueError("Le pipeline doit contenir une etape 'clf' ou 'model'.")
    return clf


def _create_shap_figure(
    pipeline: Pipeline,
    x_test,
    name: str | None = None,
    max_display: int = 10,
    max_samples: int = 200,
):
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = _get_classifier(pipeline)

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    sample = transformed[:max_samples]

    try:
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.LinearExplainer(clf, transformed)
            shap_values = explainer.shap_values(sample)
    except Exception:  # pragma: no cover - depend du type de modele/support SHAP
        logger.warning("SHAP indisponible pour %s, artefact ignore", name or type(clf).__name__)
        return None

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    if name:
        fig.suptitle(f"Importance des variables (SHAP) : {name}")
    fig.tight_layout()
    return fig


def shap_summary(
    pipeline: Pipeline,
    x_test,
    name: str | None = None,
    output_path: Path | None = None,
    max_display: int = 10,
    max_samples: int = 200,
) -> Path | None:
    """Sauvegarder un summary plot SHAP sur disque."""
    figure = _create_shap_figure(
        pipeline,
        x_test,
        name=name,
        max_display=max_display,
        max_samples=max_samples,
    )
    if figure is None:
        return None

    destination = output_path or (MODELS_DIR / "shap_summary.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("SHAP summary sauvegarde dans %s", destination)
    return destination


def log_shap_summary(
    pipeline: Pipeline,
    x_test,
    name: str,
    max_display: int = 10,
    max_samples: int = 200,
    artifact_path: str = "shap_summary.png",
) -> None:
    """Logger un summary plot SHAP comme artefact MLflow."""
    figure = _create_shap_figure(
        pipeline,
        x_test,
        name=name,
        max_display=max_display,
        max_samples=max_samples,
    )
    if figure is None:
        return

    mlflow.log_figure(figure, artifact_path)
    plt.close(figure)


def resolve_model_uri(
    model_name: str = MODEL_NAME,
    alias: str | None = MODEL_ALIAS,
    version: str | int | None = None,
) -> str:
    """Resoudre une URI MLflow pour un modele enregistre.

    Priorite:
    1. version explicite -> ``models:/name/version``
    2. alias -> ``models:/name@alias``
    3. fallback sur la derniere version numerique en registry
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if version is not None:
        return f"models:/{model_name}/{version}"

    if alias:
        try:
            client.get_model_version_by_alias(model_name, alias)
            return f"models:/{model_name}@{alias}"
        except MlflowException:
            logger.info(
                "Alias '%s' introuvable pour '%s', fallback sur la derniere version.",
                alias,
                model_name,
            )

    versions = list(client.search_model_versions(filter_string=f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(
            f"Aucune version trouvee dans le Model Registry pour '{model_name}'."
        )

    latest = max(versions, key=lambda item: int(item.version))
    return f"models:/{model_name}/{latest.version}"


def load_model_from_uri(
    model_uri: str | None = None,
    *,
    model_name: str = MODEL_NAME,
    alias: str | None = MODEL_ALIAS,
    version: str | int | None = None,
) -> tuple[Pipeline, str]:
    """Charger un pipeline scikit-learn depuis MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    resolved_uri = model_uri or resolve_model_uri(model_name=model_name, alias=alias, version=version)
    model = mlflow.sklearn.load_model(resolved_uri)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Le modele charge depuis '{resolved_uri}' n'est pas un Pipeline scikit-learn.")
    return model, resolved_uri


def evaluate_pipeline(pipeline: Pipeline, x_test, y_test) -> dict[str, float]:
    """Calculer les metriques standards de classification binaire."""
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def evaluate_registered_model(
    model_uri: str | None = None,
    *,
    model_name: str = MODEL_NAME,
    alias: str | None = MODEL_ALIAS,
    version: str | int | None = None,
) -> tuple[dict[str, float], str, str]:
    """Charger le dernier modele du registry et l'evaluer sur le split de test."""
    df = load_data()
    _, x_test, _, y_test = get_splits(df)
    pipeline, resolved_uri = load_model_from_uri(
        model_uri=model_uri,
        model_name=model_name,
        alias=alias,
        version=version,
    )

    metrics = evaluate_pipeline(pipeline, x_test, y_test)
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred)
    return metrics, resolved_uri, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalue le dernier modele enregistre dans MLflow.")
    parser.add_argument("--model-uri", type=str, default=None, help="URI MLflow explicite a charger")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Nom du modele enregistre")
    parser.add_argument(
        "--alias",
        type=str,
        default=MODEL_ALIAS,
        help="Alias du modele a privilegier avant fallback sur la derniere version",
    )
    parser.add_argument("--version", type=str, default=None, help="Version explicite du modele")
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Desactive la generation du summary plot SHAP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data()
    _, x_test, _, y_test = get_splits(df)
    pipeline, resolved_uri = load_model_from_uri(
        model_uri=args.model_uri,
        model_name=args.model_name,
        alias=args.alias,
        version=args.version,
    )
    metrics = evaluate_pipeline(pipeline, x_test, y_test)

    print(f"model_uri={resolved_uri}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"roc_auc={metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, pipeline.predict(x_test)))

    if not args.skip_shap:
        output_path = shap_summary(pipeline, x_test, name=args.model_name)
        if output_path is not None:
            print(f"shap_summary={output_path}")


if __name__ == "__main__":
    main()
