"""Prediction route handlers: predict, history, detail."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from auth import get_current_doctor
from database import get_db
from models import Doctor
from schemas import (
    ErrorResponse,
    HistoryResponse,
    PredictionDetailResponse,
    PredictionResponse,
)
from services.prediction_service import (
    create_prediction,
    get_history,
    get_prediction_detail,
    is_supported_upload,
)

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Analyse a chest X-ray",
    description="Upload a chest X-ray image (PNG or JPEG) for a given patient. "
    "The image is stored in Supabase Storage, then passed through the "
    "CheXpert model with conformal prediction to produce uncertainty-aware "
    "findings for 5 diseases: Cardiomegaly, Edema, Consolidation, "
    "Atelectasis, and Pleural Effusion.",
    responses={
        200: {"description": "Prediction result with findings and uncertainty levels"},
        400: {
            "model": ErrorResponse,
            "description": "Invalid file type (must be PNG/JPEG) or empty file",
        },
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
        404: {
            "model": ErrorResponse,
            "description": "Patient not found or does not belong to this doctor",
        },
        500: {
            "model": ErrorResponse,
            "description": "ML inference error",
        },
    },
)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (PNG or JPEG)"),
    patient_id: int = Form(..., description="ID of the patient this X-ray belongs to"),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    if not is_supported_upload(file):
        raise HTTPException(status_code=400, detail="File must be PNG, JPEG")

    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise ValueError("Uploaded file is empty")

        return create_prediction(file, img_bytes, patient_id, current_doctor, db)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference error: {exc}"
        ) from exc


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get recent prediction history",
    description="Return the 10 most recent predictions for the authenticated doctor, "
    "including patient info and top finding.",
    responses={
        200: {"description": "List of recent predictions"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
    },
)
def history(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    return get_history(current_doctor, db)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
    summary="Get prediction details",
    description="Return full details of a specific prediction, "
    "including all disease findings with probabilities, "
    "uncertainty levels, and the associated patient information.",
    responses={
        200: {"description": "Full prediction detail with patient info"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
        404: {
            "model": ErrorResponse,
            "description": "Prediction not found or does not belong to this doctor",
        },
    },
)
def prediction_detail(
    prediction_id: int,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    return get_prediction_detail(prediction_id, current_doctor, db)
