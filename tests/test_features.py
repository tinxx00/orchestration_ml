"""Tests du préprocesseur de features."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer

from heart.config import NUMERIC_FEATURES, TARGET
from heart.data import load_data
from heart.features import build_preprocessor


def test_build_preprocessor_type():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)


def test_preprocessor_fit_transform_keeps_rows():
    df = load_data()
    x = df.drop(columns=[TARGET])
    pre = build_preprocessor()
    transformed = pre.fit_transform(x)
    assert transformed.shape[0] == len(df)


def test_preprocessor_output_has_no_nan():
    df = load_data()
    x = df.drop(columns=[TARGET])
    pre = build_preprocessor()
    transformed = pre.fit_transform(x)
    # l'imputer doit avoir supprimé toutes les valeurs manquantes
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_numeric_features_are_scaled():
    df = load_data()
    x = df.drop(columns=[TARGET])
    pre = build_preprocessor()
    pre.fit_transform(x)
    # le bloc numérique doit couvrir toutes les colonnes numériques déclarées
    num_cols = pre.transformers_[0][2]
    assert list(num_cols) == NUMERIC_FEATURES
