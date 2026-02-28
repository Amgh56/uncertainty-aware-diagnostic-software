from pydantic import BaseModel, Field


class DoctorRegisterRequest(BaseModel):
    """Register a new doctor account."""
    email: str = Field(..., examples=["doctor@hospital.com"])
    password: str = Field(..., min_length=6, examples=["securepass123"])
    full_name: str = Field(..., examples=["Dr. Jane Smith"])


class DoctorLoginRequest(BaseModel):
    """Login with email and password."""
    email: str = Field(..., examples=["doctor@hospital.com"])
    password: str = Field(..., examples=["securepass123"])


class TokenResponse(BaseModel):
    """JWT access token returned on successful authentication."""
    access_token: str
    token_type: str = "bearer"


class DoctorResponse(BaseModel):
    """Doctor profile information."""
    id: int
    email: str
    full_name: str

    class Config:
        from_attributes = True
