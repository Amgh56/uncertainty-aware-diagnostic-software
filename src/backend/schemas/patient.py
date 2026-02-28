from datetime import datetime

from pydantic import BaseModel, Field


class PatientCreateRequest(BaseModel):
    """Create or retrieve a patient by MRN (scoped to the authenticated doctor)."""
    mrn: str = Field(..., examples=["MRN-001"])
    first_name: str = Field(..., examples=["John"])
    last_name: str = Field(..., examples=["Doe"])


class PatientResponse(BaseModel):
    """Patient record."""
    id: int
    mrn: str
    first_name: str
    last_name: str
    created_at: datetime

    class Config:
        from_attributes = True


class PatientSummary(BaseModel):
    """Nested patient info within a prediction detail response."""
    id: int
    mrn: str
    first_name: str
    last_name: str
    created_at: datetime


class PatientListItem(BaseModel):
    """Patient with aggregated prediction statistics."""
    id: int
    mrn: str
    first_name: str
    last_name: str
    prediction_count: int
    last_prediction_at: datetime | None
    last_top_finding: str | None
    last_prediction_id: int | None
    created_at: datetime


class PatientListResponse(BaseModel):
    """List of patients with prediction statistics."""
    patients: list[PatientListItem]
