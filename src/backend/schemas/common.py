from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error body returned by all error responses."""
    detail: str = Field(..., examples=["Error description"])
