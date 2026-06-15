"""Évaluation et explicabilité SHAP — générique, ne pas modifier."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR


def shap_summary(pipeline: Pipeline, X_test, max_display: int = 10) -> None:
    """Génère et sauvegarde un SHAP summary bar plot pour le pipeline entraîné.

    Supporte les modèles arborescents (TreeExplainer) et linéaires (LinearExplainer).
    Le fichier est sauvegardé dans models/shap_summary.png.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_test)

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            # classification binaire : prendre les valeurs de la classe positive
            shap_values = shap_values[1]
    else:
        # modèle linéaire (LogisticRegression, etc.)
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    output_path = MODELS_DIR / "shap_summary.png"
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] summary sauvegardé → {output_path}")
