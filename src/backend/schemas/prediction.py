from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from enums import UncertaintyLevel, ValidationVerdict
from schemas.patient import PatientSummary


class FindingItem(BaseModel):
    """Single disease finding with probability and uncertainty level."""
    finding: str = Field(..., examples=["Cardiomegaly"])
    probability: float = Field(..., ge=0, le=1, examples=[0.8234])
    uncertainty: UncertaintyLevel = Field(..., examples=["Low"])
    in_prediction_set: bool


class ModelInfoSummary(BaseModel):
    """Summary of the published model used for a prediction."""
    id: str
    name: str
    version: str
    modality: str
    num_labels: int
    validation_verdict: ValidationVerdict


class PredictionResponse(BaseModel):
    """Prediction result from chest X-ray inference."""
    id: int
    patient_id: int
    image_path: str
    top_finding: str
    top_probability: float
    prediction_set_size: int
    coverage: str = Field(..., examples=["90%"])
    alpha: float
    lamhat: float
    findings: list[FindingItem]
    model_info: Optional[ModelInfoSummary] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionDetailResponse(PredictionResponse):
    """Full prediction detail including patient information."""
    patient: PatientSummary


class HistoryItem(BaseModel):
    """Single item in prediction history."""
    prediction_id: int
    patient_id: int
    mrn: str
    first_name: str
    last_name: str
    top_finding: str
    top_probability: float
    prediction_set_size: int
    created_at: datetime


class HistoryResponse(BaseModel):
    """List of recent predictions."""
    items: list[HistoryItem]
