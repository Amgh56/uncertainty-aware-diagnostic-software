"""Business logic for predictions: creation, history, detail retrieval."""

import json

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from models import Doctor, Patient, Prediction
from schemas import HistoryItem, HistoryResponse
from services.ml_service import ml_state


def is_supported_upload(file: UploadFile) -> bool:
    if not file:
        return False
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    if content_type in {"image/png", "image/jpeg"}:
        return True
    if filename.endswith((".png", ".jpg", ".jpeg")):
        return True
    return False


def create_prediction(
    file: UploadFile,
    img_bytes: bytes,
    patient_id: int,
    doctor: Doctor,
    db: Session,
) -> dict:
    """Validate patient, upload image, run inference, persist, return result."""
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.doctor_id == doctor.id)
        .first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    image_url = ml_state.upload_xray(
        img_bytes,
        doctor.id,
        file.filename or "upload.png",
        file.content_type or "image/png",
    )

    findings = ml_state.run_inference(img_bytes)

    prediction_set_size = sum(1 for f in findings if f["in_prediction_set"])
    top = findings[0]

    prediction = Prediction(
        patient_id=patient.id,
        doctor_id=doctor.id,
        image_path=image_url,
        top_finding=top["finding"],
        top_probability=top["probability"],
        prediction_set_size=prediction_set_size,
        coverage=f"{int((1 - ml_state.alpha) * 100)}%",
        alpha=ml_state.alpha,
        lamhat=round(ml_state.lamhat, 6),
        findings_json=json.dumps(findings),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "id": prediction.id,
        "patient_id": patient.id,
        "image_path": image_url,
        "top_finding": top["finding"],
        "top_probability": top["probability"],
        "prediction_set_size": prediction_set_size,
        "coverage": f"{int((1 - ml_state.alpha) * 100)}%",
        "alpha": ml_state.alpha,
        "lamhat": round(ml_state.lamhat, 6),
        "findings": findings,
        "created_at": prediction.created_at.isoformat(),
    }


def get_history(doctor: Doctor, db: Session) -> HistoryResponse:
    rows = (
        db.query(Prediction, Patient)
        .join(Patient, Prediction.patient_id == Patient.id)
        .filter(Prediction.doctor_id == doctor.id)
        .order_by(Prediction.created_at.desc())
        .limit(10)
        .all()
    )

    items = []
    for pred, pat in rows:
        items.append(
            HistoryItem(
                prediction_id=pred.id,
                patient_id=pat.id,
                mrn=pat.mrn,
                first_name=pat.first_name,
                last_name=pat.last_name,
                top_finding=pred.top_finding,
                top_probability=pred.top_probability,
                prediction_set_size=pred.prediction_set_size,
                created_at=pred.created_at,
            )
        )
    return HistoryResponse(items=items)


def get_prediction_detail(
    prediction_id: int, doctor: Doctor, db: Session
) -> dict:
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.doctor_id == doctor.id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    patient = db.query(Patient).filter(Patient.id == prediction.patient_id).first()
    findings = json.loads(prediction.findings_json)

    return {
        "id": prediction.id,
        "patient_id": prediction.patient_id,
        "image_path": prediction.image_path,
        "top_finding": prediction.top_finding,
        "top_probability": prediction.top_probability,
        "prediction_set_size": prediction.prediction_set_size,
        "coverage": prediction.coverage,
        "alpha": prediction.alpha,
        "lamhat": prediction.lamhat,
        "findings": findings,
        "created_at": prediction.created_at.isoformat(),
        "patient": {
            "id": patient.id,
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "created_at": patient.created_at.isoformat(),
        },
    }
