from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from enums import ModelVisibility


class PublishModelRequest(BaseModel):
    calibration_job_id: str
    name: str = Field(..., min_length=1, max_length=150)
    description: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1, max_length=20)
    modality: str = Field(..., min_length=1, max_length=100)
    intended_use: str = Field(..., min_length=1)
    labels: list[str] = Field(..., min_length=1)
    visibility: ModelVisibility = ModelVisibility.PRIVATE
    consent_agreed: bool = False


class PublishedModelResponse(BaseModel):
    id: str
    calibration_job_id: str
    developer_id: int
    name: str
    description: str
    version: str
    modality: str
    intended_use: str
    artifact_type: str
    labels_json: str
    num_labels: int
    alpha: float
    lamhat: float
    validation_verdict: str
    validation_metrics_json: Optional[str] = None
    visibility: str
    is_active: bool
    consent_given_at: Optional[datetime] = None
    developer_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PublishedModelSummary(BaseModel):
    id: str
    name: str
    description: str
    version: str
    modality: str
    num_labels: int
    alpha: float
    lamhat: float
    validation_verdict: str
    visibility: str
    is_active: bool
    developer_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PublishedModelListResponse(BaseModel):
    models: list[PublishedModelSummary]


class UpdateVisibilityRequest(BaseModel):
    visibility: ModelVisibility
    consent_agreed: bool = False


class ToggleActiveRequest(BaseModel):
    is_active: bool
