"""Tasks 2, 3, 4 — Model development, experiment tracking, and packaging.

What this script does, end to end:
  1. Loads the cleaned dataset (running download + preprocess if needed).
  2. Stratified train/test split.
  3. Trains THREE candidate classifiers, each as a full sklearn Pipeline
     (preprocessor + estimator) so preprocessing is learned inside CV:
       - Logistic Regression
       - Random Forest
       - XGBoost
     Hyper-parameters are tuned with GridSearchCV (5-fold, ROC-AUC scoring).
  4. Logs params, cross-val + test metrics, ROC curve, confusion matrix, and
     the fitted model to MLflow — one nested run per model.
  5. Selects the best model by test ROC-AUC and saves it as ``models/model.pkl``
     plus a ``model_metadata.json`` describing the winner and its metrics.

Run:
    python -m src.models.train
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METADATA_PATH,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    PROCESSED_CSV,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    ensure_dirs,
)
from src.features.pipeline import build_preprocessor
from src.models.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _load_dataset() -> pd.DataFrame:
    """Load the processed CSV, regenerating it from raw if it is missing."""
    if not PROCESSED_CSV.exists():
        print("[train] Processed data not found; building it now ...")
        from src.data.download import download
        from src.data.preprocess import build_processed_csv

        if not (PROCESSED_CSV.parent.parent / "raw" / "heart_disease_raw.csv").exists():
            download()
        build_processed_csv()
    return pd.read_csv(PROCESSED_CSV)


def _candidate_models() -> dict[str, dict]:
    """Return the model zoo: estimator + hyper-parameter grid for each."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            # penalty is fixed at the default L2; sklearn 1.9 deprecated the
            # explicit `penalty` kwarg, so we tune only C and the solver.
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        },
        "xgboost": {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
            ),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        },
    }


def train() -> dict:
    """Run the full training + tracking + packaging pipeline."""
    ensure_dirs()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Create the experiment with an explicit local artifact location on first
    # run; subsequent runs just attach to it.
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = _load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"[train] Train={len(X_train)}  Test={len(X_test)}  "
        f"positive-rate={y.mean():.3f}"
    )

    results: list[dict] = []
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"model-selection-{run_stamp}"):
        mlflow.set_tag("dataset", "UCI Heart Disease (Cleveland)")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("cv_folds", CV_FOLDS)

        for name, spec in _candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True):
                print(f"\n[train] === {name} — grid search ===")
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor()),
                        ("clf", spec["estimator"]),
                    ]
                )
                search = GridSearchCV(
                    pipe,
                    param_grid=spec["param_grid"],
                    scoring="roc_auc",
                    cv=CV_FOLDS,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                best = search.best_estimator_

                # Cross-validated ROC-AUC on the training set (mean +/- std).
                cv_scores = cross_val_score(
                    best, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc"
                )

                # Held-out test-set metrics.
                y_pred = best.predict(X_test)
                y_proba = best.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
                metrics["cv_roc_auc_std"] = float(cv_scores.std())

                # ---- MLflow logging (params, metrics, artifacts, model) ----
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(metrics)

                cm_path = save_confusion_matrix(
                    y_test,
                    y_pred,
                    FIGURES_DIR / f"cm_{name}.png",
                    f"Confusion Matrix — {name}",
                )
                roc_path = save_roc_curve(
                    y_test,
                    y_proba,
                    FIGURES_DIR / f"roc_{name}.png",
                    f"ROC Curve — {name}",
                )
                mlflow.log_artifact(str(cm_path), artifact_path="plots")
                mlflow.log_artifact(str(roc_path), artifact_path="plots")
                # Use cloudpickle (not the new skops default, which rejects
                # numpy dtypes inside a fitted ColumnTransformer).
                mlflow.sklearn.log_model(
                    best, name="model", serialization_format="cloudpickle"
                )

                print(
                    f"[train] {name}: test_roc_auc={metrics['roc_auc']:.4f} "
                    f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                    f"cv_auc={metrics['cv_roc_auc_mean']:.4f}"
                )
                results.append(
                    {
                        "name": name,
                        "estimator": best,
                        "params": search.best_params_,
                        "metrics": metrics,
                    }
                )

        # ---- Select the winner by test ROC-AUC ----
        best_result = max(results, key=lambda r: r["metrics"]["roc_auc"])
        mlflow.set_tag("best_model", best_result["name"])
        mlflow.log_metric("best_test_roc_auc", best_result["metrics"]["roc_auc"])

    # ---- Task 4: package the winning pipeline for reuse ----
    joblib.dump(best_result["estimator"], MODEL_PATH)
    metadata = {
        "model_name": best_result["name"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_params": best_result["params"],
        "metrics": best_result["metrics"],
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "sklearn_pipeline": True,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\n[train] ===== SUMMARY =====")
    summary = pd.DataFrame(
        [{"model": r["name"], **r["metrics"]} for r in results]
    ).set_index("model")
    print(summary.round(4).to_string())
    print(f"\n[train] Winner: {best_result['name']}")
    print(f"[train] Saved model    -> {MODEL_PATH}")
    print(f"[train] Saved metadata -> {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train()
