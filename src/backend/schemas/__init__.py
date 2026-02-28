"""Pydantic schemas, split by entity. Re-exported here for convenience."""

from schemas.common import ErrorResponse
from schemas.doctor import (
    DoctorLoginRequest,
    DoctorRegisterRequest,
    DoctorResponse,
    TokenResponse,
)
from schemas.patient import (
    PatientCreateRequest,
    PatientListItem,
    PatientListResponse,
    PatientResponse,
    PatientSummary,
)
from schemas.prediction import (
    FindingItem,
    HistoryItem,
    HistoryResponse,
    PredictionDetailResponse,
    PredictionResponse,
)
from schemas.health import HealthResponse

__all__ = [
    "ErrorResponse",
    "DoctorLoginRequest",
    "DoctorRegisterRequest",
    "DoctorResponse",
    "TokenResponse",
    "PatientCreateRequest",
    "PatientListItem",
    "PatientListResponse",
    "PatientResponse",
    "PatientSummary",
    "FindingItem",
    "HistoryItem",
    "HistoryResponse",
    "PredictionDetailResponse",
    "PredictionResponse",
    "HealthResponse",
]
