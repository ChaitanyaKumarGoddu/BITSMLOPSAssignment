"""Task 1 — Cleaning & preprocessing (the deterministic, non-learned part).

This module performs the transformations that do **not** need to be fitted on
the training data and therefore can run before the train/test split:

  * coerce columns to numeric (the raw file stores everything as strings once
    the '?' markers appear),
  * binarise the multi-class ``num`` target (0 = no disease, 1-4 = disease)
    into a 0/1 ``target`` column,
  * drop exact duplicate rows.

Missing-value imputation and scaling are intentionally NOT done here — those
are *learned* steps and live in the sklearn ``ColumnTransformer`` pipeline
(see ``src/features/pipeline.py``) so they are fit only on training folds and
reused identically at inference time. That separation is what keeps the whole
project reproducible and leak-free.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_CSV,
    RAW_COLUMNS,
    TARGET,
    ensure_dirs,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame with numeric features and a binary target.

    The input is the raw dataframe as loaded from ``data/raw`` (all object
    dtype, '?' already converted to NaN by the download step).
    """
    df = df.copy()

    # Every column in this dataset is numeric in meaning; coerce and let
    # unparoseable entries ('?') become NaN for the imputer to handle later.
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarise the target: UCI encodes severity 0-4; the assignment asks for a
    # binary presence/absence classifier.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Remove exact duplicates (a handful appear in the Cleveland set).
    df = df.drop_duplicates().reset_index(drop=True)

    # Keep a stable column order: features first, target last.
    df = df[FEATURE_COLUMNS + [TARGET]]
    return df


def load_clean(raw_csv=None) -> pd.DataFrame:
    """Load the raw CSV and return the cleaned frame (does not write to disk)."""
    from src.config import RAW_CSV

    path = raw_csv or RAW_CSV
    raw = pd.read_csv(path)
    return clean(raw)


def build_processed_csv() -> pd.DataFrame:
    """Clean the raw CSV and persist the result to ``PROCESSED_CSV``."""
    ensure_dirs()
    df = load_clean()
    df.to_csv(PROCESSED_CSV, index=False)
    print(
        f"[preprocess] Wrote {len(df)} rows -> {PROCESSED_CSV} "
        f"(target balance: {df[TARGET].mean():.3f} positive)"
    )
    return df


if __name__ == "__main__":
    build_processed_csv()
