"""Central configuration: paths, feature schema, and reproducibility settings.

Keeping every path and column list in one place means the preprocessing
pipeline, the training script, the API and the tests all agree on the same
contract. Change the schema here once and the whole project follows.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# --------------------------------------------------------------------------- #
# Filesystem layout (all relative to the repository root)
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_CSV: Path = RAW_DATA_DIR / "heart_disease_raw.csv"
PROCESSED_CSV: Path = PROCESSED_DATA_DIR / "heart_disease_clean.csv"
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# MLflow — SQLite tracking backend (the file store is deprecated in MLflow 3.x).
# Browse the UI with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI: str = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION: str = (ROOT_DIR / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT_NAME: str = "heart-disease-classification"

# --------------------------------------------------------------------------- #
# Feature schema for the UCI Heart Disease (Cleveland) dataset
# --------------------------------------------------------------------------- #
# The raw UCI file has no header; these are the canonical column names.
RAW_COLUMNS: list[str] = [
    "age",  # age in years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl (1 true, 0 false)
    "restecg",  # resting electrocardiographic results (0-2)
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise induced angina (1 yes, 0 no)
    "oldpeak",  # ST depression induced by exercise
    "slope",  # slope of the peak exercise ST segment (1-3)
    "ca",  # number of major vessels (0-3) coloured by fluoroscopy
    "thal",  # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",  # diagnosis of heart disease (0-4) -> binarised to `target`
]

TARGET: str = "target"

# Continuous features -> median impute + standard scale
NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete / categorical features -> mode impute + one-hot encode
CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
