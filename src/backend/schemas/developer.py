from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from enums import JobStatus


class JobCreateResponse(BaseModel):
    id: str
    status: JobStatus
    model_filename: str
    config_filename: Optional[str] = None
    dataset_filename: str
    alpha: float
    created_at: datetime

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    id: str
    status: JobStatus = Field(..., examples=["queued"])
    model_filename: str
    config_filename: Optional[str] = None
    dataset_filename: str
    alpha: float
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
