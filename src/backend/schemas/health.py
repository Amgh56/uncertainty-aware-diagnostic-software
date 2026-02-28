from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Server health check response."""
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    device: Optional[str] = Field(None, examples=["mps"])
    lamhat: Optional[float]
    alpha: Optional[float]
    diseases: list[str]
