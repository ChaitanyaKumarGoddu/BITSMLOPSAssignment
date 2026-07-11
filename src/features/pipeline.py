"""Task 2 & 4 — Reusable preprocessing pipeline (fit at training, reused at inference).

The ``ColumnTransformer`` here is the single source of truth for feature
engineering. It is fitted on the training split only, serialised together with
the estimator into one sklearn ``Pipeline``, and reloaded verbatim by the API.
Because the same object does the imputing/scaling/encoding in both places,
there is zero training/serving skew.

Design:
  * numeric features  -> median imputation + standard scaling
  * categorical feats -> most-frequent imputation + one-hot encoding
    (``handle_unknown="ignore"`` so an unseen category at inference does not
    crash the service).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construct the (unfitted) feature-engineering ColumnTransformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor
