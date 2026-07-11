"""Unit tests for the data cleaning / preprocessing logic (Task 5)."""

from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET
from src.data.preprocess import clean


def test_clean_binarises_target(raw_frame):
    out = clean(raw_frame)
    # Target must be strictly 0/1 after binarising the 0-4 severity column.
    assert set(out[TARGET].unique()).issubset({0, 1})
    assert "num" not in out.columns


def test_clean_coerces_missing_markers_to_nan(raw_frame):
    out = clean(raw_frame)
    # The '?' entries we injected must become real NaNs (not the string '?').
    assert out["ca"].isna().sum() >= 1
    assert out["thal"].isna().sum() >= 1
    assert not (out == "?").any().any()


def test_clean_drops_duplicates(raw_frame):
    out = clean(raw_frame)
    # We appended one exact duplicate row; it must be gone.
    assert out.duplicated().sum() == 0


def test_clean_column_contract(raw_frame):
    out = clean(raw_frame)
    # Column order/contract the rest of the pipeline relies on.
    assert list(out.columns) == FEATURE_COLUMNS + [TARGET]


def test_clean_numeric_dtypes(raw_frame):
    out = clean(raw_frame)
    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(out[col]), f"{col} not numeric"
