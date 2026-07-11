"""Shared pytest fixtures.

Tests must run offline in CI, so instead of downloading the real dataset we
synthesise a small frame with the same schema (including a few '?'/NaN values
and duplicate rows) and train a tiny pipeline on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, RAW_COLUMNS, TARGET
from src.features.pipeline import build_preprocessor


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small raw-style frame: all-object dtype, '?' markers, one duplicate."""
    rng = np.random.default_rng(0)
    n = 60
    data = {
        "age": rng.integers(35, 70, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(100, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
        "num": rng.integers(0, 5, n),  # 0-4 severity, binarised later
    }
    df = pd.DataFrame(data, columns=RAW_COLUMNS).astype(object)
    # Inject missing markers the way the real UCI file does.
    df.loc[3, "ca"] = "?"
    df.loc[7, "thal"] = "?"
    # Inject an exact duplicate row to exercise de-duplication.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


@pytest.fixture
def trained_pipeline(raw_frame) -> Pipeline:
    """A fitted preprocessor+LogReg pipeline built from the synthetic frame."""
    from src.data.preprocess import clean

    clean_df = clean(raw_frame)
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET]
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)
    return pipe
