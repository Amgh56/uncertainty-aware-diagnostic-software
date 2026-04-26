from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import User
from schemas import (
    ErrorResponse,
    PatientCreateRequest,
    PatientListResponse,
    PatientResponse,
)
from services.patient_service import create_or_get_patient, get_patient_predictions, list_patients_with_stats

router = APIRouter(tags=["Patients"])


@router.post(
    "/patients",
    response_model=PatientResponse,
    status_code=201,
    summary="Create or retrieve a patient",
    description="Create a new patient for the authenticated doctor. "
    "If a patient with the same MRN already exists for this doctor, "
    "the existing record is returned instead.",
    responses={
        201: {"description": "Patient created or existing patient returned"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
        422: {"description": "Request body validation error"},
    },
)
def create_patient(
    body: PatientCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return create_or_get_patient(
        body.mrn, body.first_name, body.last_name, current_user, db
    )


@router.get(
    "/patients",
    response_model=PatientListResponse,
    summary="List all patients",
    description="Return all patients for the authenticated doctor, "
    "including prediction count, last prediction date, and top finding.",
    responses={
        200: {"description": "Patient list with prediction statistics"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
    },
)
def list_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_patients_with_stats(current_user, db)


@router.get(
    "/patients/{patient_id}/predictions",
    summary="List all predictions for a patient",
    description="Return every prediction for the given patient, newest first.",
    responses={
        200: {"description": "List of predictions"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
        404: {
            "model": ErrorResponse,
            "description": "Patient not found",
        },
    },
)
def patient_predictions(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_patient_predictions(patient_id, current_user, db)
