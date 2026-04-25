import json

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from models import User, Patient, Prediction, PublishedModel
from schemas import HistoryItem, HistoryResponse
from services.ml_service import ml_state


# ── Upload Validation ────────────────────────────────────────

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


# ── Prediction Creation ──────────────────────────────────────

def create_prediction(
    file: UploadFile,
    img_bytes: bytes,
    patient_id: int,
    user: User,
    db: Session,
    model_id: str,
) -> dict:
    """Validate patient, upload image, run inference, persist, return result."""
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.user_id == user.id)
        .first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    image_url = ml_state.upload_xray(
        img_bytes,
        user.id,
        file.filename or "upload.png",
        file.content_type or "image/png",
    )

    published_model = (
        db.query(PublishedModel)
        .filter(PublishedModel.id == model_id, PublishedModel.is_active == True)
        .first()
    )
    if not published_model:
        raise HTTPException(
            status_code=404,
            detail="Published model not found or is inactive",
        )

    findings = ml_state.run_inference(img_bytes, published_model)

    prediction_set_size = sum(1 for f in findings if f["in_prediction_set"])
    top = findings[0]

    prediction = Prediction(
        patient_id=patient.id,
        user_id=user.id,
        image_path=image_url,
        top_finding=top["finding"],
        top_probability=top["probability"],
        prediction_set_size=prediction_set_size,
        coverage=f"{int((1 - published_model.alpha) * 100)}%",
        alpha=published_model.alpha,
        lamhat=round(published_model.lamhat, 6),
        findings_json=json.dumps(findings),
        published_model_id=model_id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    model_info = {
        "id": published_model.id,
        "name": published_model.name,
        "version": published_model.version,
        "modality": published_model.modality,
        "num_labels": published_model.num_labels,
        "validation_verdict": published_model.validation_verdict,
    }

    return {
        "id": prediction.id,
        "patient_id": patient.id,
        "image_path": image_url,
        "top_finding": top["finding"],
        "top_probability": top["probability"],
        "prediction_set_size": prediction_set_size,
        "coverage": f"{int((1 - published_model.alpha) * 100)}%",
        "alpha": published_model.alpha,
        "lamhat": round(published_model.lamhat, 6),
        "findings": findings,
        "created_at": prediction.created_at.isoformat(),
        "model_info": model_info,
    }


# ── Prediction History ───────────────────────────────────────

def get_history(user: User, db: Session) -> HistoryResponse:
    rows = (
        db.query(Prediction, Patient)
        .join(Patient, Prediction.patient_id == Patient.id)
        .filter(Prediction.user_id == user.id)
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


# ── Prediction Regeneration ──────────────────────────────────

def regenerate_prediction(
    prediction_id: int,
    new_alpha: float,
    user: User,
    db: Session,
) -> dict:
    """Re-apply a new alpha threshold against stored probabilities and save a new Prediction."""
    if not (0.0 < new_alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be between 0 and 1")

    # Fetch the original prediction scoped to this user
    original = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user.id)
        .first()
    )
    if not original:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not original.published_model_id:
        raise HTTPException(
            status_code=400,
            detail="This prediction has no associated published model and cannot be regenerated",
        )

    pub_model = (
        db.query(PublishedModel)
        .filter(PublishedModel.id == original.published_model_id)
        .first()
    )
    if not pub_model:
        raise HTTPException(status_code=404, detail="Published model not found")

    if not pub_model.validation_sweep_json:
        raise HTTPException(
            status_code=400,
            detail="This model has no validation sweep data; regeneration is not available",
        )

    sweep = json.loads(pub_model.validation_sweep_json)
    closest = min(sweep, key=lambda x: abs(x["alpha"] - new_alpha))
    new_lamhat = closest["lamhat"]

    # Re-apply threshold to stored probabilities
    findings = json.loads(original.findings_json)
    for f in findings:
        f["in_prediction_set"] = f["probability"] >= new_lamhat

    top = findings[0]
    prediction_set_size = sum(1 for f in findings if f["in_prediction_set"])

    new_prediction = Prediction(
        patient_id=original.patient_id,
        user_id=original.user_id,
        image_path=original.image_path,
        top_finding=top["finding"],
        top_probability=top["probability"],
        prediction_set_size=prediction_set_size,
        coverage=f"{int((1 - new_alpha) * 100)}%",
        alpha=new_alpha,
        lamhat=round(new_lamhat, 6),
        findings_json=json.dumps(findings),
        published_model_id=original.published_model_id,
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    patient = db.query(Patient).filter(Patient.id == new_prediction.patient_id).first()

    model_info = {
        "id": pub_model.id,
        "name": pub_model.name,
        "version": pub_model.version,
        "modality": pub_model.modality,
        "num_labels": pub_model.num_labels,
        "validation_verdict": pub_model.validation_verdict,
    }

    return {
        "id": new_prediction.id,
        "patient_id": new_prediction.patient_id,
        "image_path": new_prediction.image_path,
        "top_finding": top["finding"],
        "top_probability": top["probability"],
        "prediction_set_size": prediction_set_size,
        "coverage": f"{int((1 - new_alpha) * 100)}%",
        "alpha": new_alpha,
        "lamhat": round(new_lamhat, 6),
        "findings": findings,
        "created_at": new_prediction.created_at.isoformat(),
        "model_info": model_info,
        "patient": {
            "id": patient.id,
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "created_at": patient.created_at.isoformat(),
        },
    }


# ── Prediction Details ───────────────────────────────────────

def get_prediction_detail(
    prediction_id: int, user: User, db: Session
) -> dict:
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user.id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    patient = db.query(Patient).filter(Patient.id == prediction.patient_id).first()
    findings = json.loads(prediction.findings_json)

    # Get model info if prediction used a published model
    model_info = None
    if prediction.published_model_id:
        pub_model = (
            db.query(PublishedModel)
            .filter(PublishedModel.id == prediction.published_model_id)
            .first()
        )
        if pub_model:
            model_info = {
                "id": pub_model.id,
                "name": pub_model.name,
                "version": pub_model.version,
                "modality": pub_model.modality,
                "num_labels": pub_model.num_labels,
                "validation_verdict": pub_model.validation_verdict,
            }

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
        "model_info": model_info,
        "patient": {
            "id": patient.id,
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "created_at": patient.created_at.isoformat(),
        },
    }
