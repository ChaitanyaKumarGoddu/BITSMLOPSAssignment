"""Pydantic request/response models for the prediction API.

The field constraints double as input validation: a request with, say, a
negative age or an out-of-range chest-pain code is rejected by FastAPI with a
422 before it ever reaches the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """One patient's clinical measurements — the 13 model input features."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=0, le=300, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels coloured (0-3)")
    thal: int = Field(..., ge=0, le=7, description="3=normal, 6=fixed, 7=reversible")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 67,
                "sex": 1,
                "cp": 4,
                "trestbps": 160,
                "chol": 286,
                "fbs": 0,
                "restecg": 2,
                "thalach": 108,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 3,
                "thal": 7,
            }
        }
    }


class PredictionResponse(BaseModel):
    """The model's verdict for one patient."""

    prediction: int = Field(..., description="0 = no heart disease, 1 = disease")
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(
        ..., description="Model probability of heart disease (class 1)"
    )
    model_name: str = Field(..., description="Name of the serving model")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
