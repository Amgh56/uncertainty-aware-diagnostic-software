from datetime import datetime

from pydantic import BaseModel


# --- Auth ---


class DoctorRegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str


class DoctorLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DoctorResponse(BaseModel):
    id: int
    email: str
    full_name: str

    class Config:
        from_attributes = True


# --- Patient ---


class PatientCreateRequest(BaseModel):
    mrn: str
    first_name: str
    last_name: str


class PatientResponse(BaseModel):
    id: int
    mrn: str
    first_name: str
    last_name: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Prediction ---


class FindingItem(BaseModel):
    finding: str
    probability: float
    uncertainty: str
    in_prediction_set: bool


class PredictionResponse(BaseModel):
    id: int
    patient_id: int
    image_path: str
    top_finding: str
    top_probability: float
    prediction_set_size: int
    coverage: str
    alpha: float
    lamhat: float
    findings: list[FindingItem]
    created_at: datetime

    class Config:
        from_attributes = True


# --- History ---


class HistoryItem(BaseModel):
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
    items: list[HistoryItem]


# --- Patient list ---


class PatientListItem(BaseModel):
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
    patients: list[PatientListItem]
