"""Tests du schéma Pydantic de prédiction."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from heart.api import Features

VALID_PAYLOAD = dict(
    age=52, trestbps=120, chol=240, thalach=150, oldpeak=1.0, ca=0,
    sex=1, cp=2, fbs=0, restecg=1, exang=0, slope=2, thal=3,
)


def test_valid_payload_parses():
    feats = Features(**VALID_PAYLOAD)
    assert feats.age == 52
    assert feats.sex == 1


def test_missing_field_raises():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    with pytest.raises(ValidationError):
        Features(**payload)


def test_non_numeric_value_raises():
    payload = {**VALID_PAYLOAD, "chol": "not-a-number"}
    with pytest.raises(ValidationError):
        Features(**payload)
