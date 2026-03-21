from pydantic import BaseModel, Field

from enums import UserRole


class DoctorRegisterRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])
    password: str = Field(..., min_length=6, examples=["securepass123"])
    full_name: str = Field(..., examples=["Dr. Jane Smith"])


class DoctorLoginRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])
    password: str = Field(..., examples=["securepass123"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    is_verified: bool = True


class DoctorResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: UserRole = Field(..., examples=["clinician"])
    is_verified: bool = True

    class Config:
        from_attributes = True


class DeveloperRegisterRequest(BaseModel):
    email: str = Field(..., examples=["researcher@uni.ac.uk"])
    password: str = Field(..., min_length=6, examples=["securepass123"])
    full_name: str = Field(..., examples=["Dr. Alice Researcher"])


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])
    token: str = Field(..., examples=["abc123..."])
    timestamp: int = Field(..., examples=[1700000000])
    new_password: str = Field(..., min_length=6, examples=["newSecurePass1"])


class VerifyOtpRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])
    otp: str = Field(..., min_length=6, max_length=6, examples=["123456"])


class ResendOtpRequest(BaseModel):
    email: str = Field(..., examples=["doctor@hospital.com"])
