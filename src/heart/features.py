"""Pré-processing générique — ColumnTransformer piloté par config.py.

Ne pas modifier : brancher un nouveau dataset passe uniquement par config.py.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from heart.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construit le ColumnTransformer adapté aux features définies dans config.py.

    - Numériques  : imputation médiane + StandardScaler.
    - Catégorielles : imputation mode + OrdinalEncoder (unknown → -1).
    - Si CATEGORICAL_FEATURES est vide, la branche catégorielle est omise.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers: list = [("num", numeric_pipeline, NUMERIC_FEATURES)]

    if CATEGORICAL_FEATURES:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ])
        transformers.append(("cat", categorical_pipeline, CATEGORICAL_FEATURES))

    return ColumnTransformer(transformers=transformers, remainder="drop")
