"""Tests du chargement et de la découpe des données."""
from __future__ import annotations

import pandas as pd

from heart.config import TARGET
from heart.data import get_splits, get_X_y, load_data


def test_load_data_returns_non_empty_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_target_column_present_and_binary():
    df = load_data()
    assert TARGET in df.columns
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_get_X_y_excludes_target():
    df = load_data()
    x, y = get_X_y(df)
    assert TARGET not in x.columns
    assert y.name == TARGET
    assert len(x) == len(y) == len(df)


def test_get_splits_shapes_are_consistent():
    df = load_data()
    x_train, x_test, y_train, y_test = get_splits(df)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(df)
    # le test doit être strictement plus petit que le train
    assert 0 < len(x_test) < len(x_train)
