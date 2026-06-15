"""Chargement du CSV et split train/test — générique, ne pas modifier."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from heart.config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_data() -> pd.DataFrame:
    """Charge le CSV défini dans config.DATA_PATH.

    Utilise encoding='utf-8-sig' pour gérer le BOM éventuel
    produit par Excel (ex. heart.csv.xls → heart.csv).
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features de la cible définie dans config.TARGET."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Retourne (X_train, X_test, y_train, y_test) avec stratification."""
    X, y = get_X_y(df)
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
