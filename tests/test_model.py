"""Unit tests for the feature pipeline and model behaviour (Task 5)."""

from __future__ import annotations

import numpy as np

from src.config import FEATURE_COLUMNS
from src.data.preprocess import clean
from src.features.pipeline import build_preprocessor
from src.models.evaluate import compute_metrics


def test_preprocessor_removes_nans(raw_frame):
    df = clean(raw_frame)
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[FEATURE_COLUMNS])
    # After imputation there must be no missing values in the feature matrix.
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_preprocessor_output_row_count(raw_frame):
    df = clean(raw_frame)
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[FEATURE_COLUMNS])
    assert transformed.shape[0] == len(df)


def test_pipeline_predicts_binary(trained_pipeline, raw_frame):
    df = clean(raw_frame)
    preds = trained_pipeline.predict(df[FEATURE_COLUMNS])
    assert set(np.unique(preds)).issubset({0, 1})


def test_pipeline_proba_in_range(trained_pipeline, raw_frame):
    df = clean(raw_frame)
    proba = trained_pipeline.predict_proba(df[FEATURE_COLUMNS])
    assert proba.shape[1] == 2
    assert ((proba >= 0) & (proba <= 1)).all()


def test_compute_metrics_keys():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.2, 0.9, 0.1, 0.4, 0.8]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= metrics.keys()
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_pipeline_is_reusable_single_row(trained_pipeline, raw_frame):
    """A single-row inference (as the API does) must work end to end."""
    df = clean(raw_frame)
    one = df[FEATURE_COLUMNS].iloc[[0]]
    proba = trained_pipeline.predict_proba(one)
    assert proba.shape == (1, 2)
