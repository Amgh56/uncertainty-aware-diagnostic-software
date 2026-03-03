from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    """Returned immediately after a calibration job is created."""
    id: str
    status: str
    model_filename: str
    config_filename: Optional[str] = None
    dataset_filename: str
    alpha: float
    created_at: datetime
    expires_at: datetime

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """Full status for a single calibration job."""
    id: str
    status: str = Field(..., examples=["queued"])
    model_filename: str
    config_filename: Optional[str] = None
    dataset_filename: str
    alpha: float
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: datetime

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """List of calibration jobs for the authenticated developer."""
    jobs: list[JobStatusResponse]
